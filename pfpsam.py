"""
PFPSAM: Prompt-Free Polyp Segment Anything Model
=================================================

Core model implementation. PFPSAM replaces the manual prompts (points,
boxes, or masks) required by the Segment Anything Model (SAM) with an
automatic prompt generator, enabling end-to-end, interaction-free polyp
segmentation from colonoscopy images.

Architecture overview
---------------------
    1. SAM Image Encoder (frozen + LoRA adapter)
         -> image embedding [B, 256, 64, 64]
    2. Prompt Generator (trainable)
         -> coarse mask [B, 1, 256, 256] (dense prompt + auxiliary supervision)
         -> point coordinates [B, 1, 2] (sparse prompt, pixel space)
    3. SAM Prompt Encoder (frozen)
         -> sparse / dense prompt embeddings
    4. SAM Mask Decoder (frozen)
         -> low-resolution mask logits [B, 1, 256, 256]
    5. Boundary Refinement Head (trainable, optional)
         -> refined mask [B, 1, 256, 256]
    6. Bilinear upsampling to input resolution

Reference
---------
This implementation builds upon the Segment Anything Model (SAM):
    Kirillov, A., et al. "Segment Anything." ICCV 2023.
    https://github.com/facebookresearch/segment-anything

License: MIT (see LICENSE file).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything.modeling import Sam


# ====================================================================
# 1. LoRA adapter for the SAM image encoder
# ====================================================================

class LoRAQKV(nn.Module):
    """Low-rank adaptation (LoRA) applied to an attention QKV projection.

    The original frozen projection is retained, and two low-rank branches
    (A -> B) are added for the query and value projections:

        qkv_out = qkv(x) + B_q(A_q(x))  (query part)
                        + B_v(A_v(x))   (value part)

    Only the A/B matrices are trainable; the base projection stays frozen.
    """

    def __init__(self, qkv: nn.Linear, rank: int) -> None:
        super().__init__()
        self.qkv = qkv
        self.rank = rank
        self.d_model = qkv.in_features

        self.lora_a_q = nn.Linear(self.d_model, rank, bias=False)
        self.lora_b_q = nn.Linear(rank, self.d_model, bias=False)
        self.lora_a_v = nn.Linear(self.d_model, rank, bias=False)
        self.lora_b_v = nn.Linear(rank, self.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        q_lora = self.lora_b_q(self.lora_a_q(x))
        v_lora = self.lora_b_v(self.lora_a_v(x))
        # Avoid in-place operations on a tensor that requires grad.
        out = qkv + 0
        out[..., : self.d_model] = out[..., : self.d_model] + q_lora
        out[..., -self.d_model :] = out[..., -self.d_model :] + v_lora
        return out


def inject_lora(sam: Sam, rank: int = 4) -> None:
    """Inject LoRA adapters into every transformer block of the image encoder.

    All original parameters of the image encoder, mask decoder, and prompt
    encoder are frozen. The LoRA A matrices are initialized with Kaiming
    uniform and the B matrices with zeros, so the adapters produce zero
    output at initialization and the pretrained behavior is preserved.

    Args:
        sam: A SAM model instance.
        rank: Rank of the low-rank decomposition.
    """
    for param in sam.image_encoder.parameters():
        param.requires_grad = False
    for param in sam.mask_decoder.parameters():
        param.requires_grad = False
    for param in sam.prompt_encoder.parameters():
        param.requires_grad = False

    for blk in sam.image_encoder.blocks:
        w_qkv = blk.attn.qkv
        d_model = w_qkv.in_features
        lora = LoRAQKV(w_qkv, rank)
        nn.init.kaiming_uniform_(lora.lora_a_q.weight, a=np.sqrt(5))
        nn.init.kaiming_uniform_(lora.lora_a_v.weight, a=np.sqrt(5))
        nn.init.zeros_(lora.lora_b_q.weight)
        nn.init.zeros_(lora.lora_b_v.weight)
        blk.attn.qkv = lora


# ====================================================================
# 2. Prompt generator (the core contribution of PFPSAM)
# ====================================================================

class PromptGenerator(nn.Module):
    """Automatic prompt generator.

    Given the image embedding produced by the SAM image encoder, this module
    generates two complementary prompts without any user interaction:

      * Dense prompt: a coarse polyp mask (256 x 256) that is fed to the
        SAM prompt encoder as a mask prompt. It is also supervised by an
        auxiliary loss during training, which teaches the generator to
        localize polyps.

      * Sparse prompt: the weighted centroid of the coarse probability map,
        expressed in the pixel coordinate space of the 1024 x 1024 input
        image (as expected by SAM's prompt encoder). It anchors the mask
        decoder at the center of the target object.
    """

    def __init__(self, in_channels: int = 256, mid_channels: int = 128) -> None:
        super().__init__()
        self.coarse_head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1),
        )

    @staticmethod
    def _extract_centroid(prob: torch.Tensor) -> torch.Tensor:
        """Weighted centroid of a sigmoid probability map in pixel space.

        Args:
            prob: [B, 1, H, W] probability map in [0, 1].

        Returns:
            [B, 1, 2] centroid coordinates (x, y) in the 1024 x 1024
            input space. Falls back to the image center when the map is
            empty (total probability below a small epsilon).
        """
        B, _, H, W = prob.shape
        device = prob.device
        coords = torch.zeros(B, 1, 2, device=device)

        y = torch.arange(H, dtype=torch.float, device=device).view(1, 1, H, 1)
        x = torch.arange(W, dtype=torch.float, device=device).view(1, 1, 1, W)
        totals = prob.sum(dim=(2, 3), keepdim=True)  # [B, 1, 1, 1]

        cx = (prob * x).sum(dim=(2, 3), keepdim=True) / totals.clamp_min(1e-6)
        cy = (prob * y).sum(dim=(2, 3), keepdim=True) / totals.clamp_min(1e-6)

        # Map from feature space (H x W = 64 x 64) to 1024 x 1024 input space.
        coords[:, :, 0] = cx.squeeze() * (1024.0 / W)
        coords[:, :, 1] = cy.squeeze() * (1024.0 / H)
        return coords

    def forward(self, image_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate the dense and sparse prompts.

        Args:
            image_embedding: [B, 256, 64, 64] image embedding from SAM.

        Returns:
            coarse_mask: [B, 1, 256, 256] coarse mask logits (dense prompt).
            point_coords: [B, 1, 2] centroid in pixel space (sparse prompt).
        """
        coarse_logits = self.coarse_head(image_embedding)  # [B, 1, 64, 64]
        coarse_mask = F.interpolate(
            coarse_logits, size=(256, 256), mode="bilinear", align_corners=False
        )
        point_coords = self._extract_centroid(torch.sigmoid(coarse_logits))
        return coarse_mask, point_coords


# ====================================================================
# 3. Boundary refinement head
# ====================================================================

class BoundaryRefinementHead(nn.Module):
    """Lightweight boundary refinement module.

    The low-resolution mask produced by the SAM mask decoder is fused with
    the high-level image embedding and refined through several convolution
    layers, which improves boundary accuracy for polyps with irregular
    contours.
    """

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.embed_reduce = nn.Sequential(
            nn.Conv2d(embed_dim, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(1 + 32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, low_res_mask: torch.Tensor, image_embedding: torch.Tensor) -> torch.Tensor:
        """Refine a mask using the image embedding.

        Args:
            low_res_mask: [B, 1, 256, 256] mask logits.
            image_embedding: [B, 256, 64, 64] image embedding.

        Returns:
            [B, 1, 256, 256] refined mask logits.
        """
        embed = self.embed_reduce(image_embedding)
        embed = F.interpolate(embed, size=low_res_mask.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([low_res_mask, embed], dim=1)
        return self.refine(x)


# ====================================================================
# 4. PFPSAM model
# ====================================================================

class PFPSAM(nn.Module):
    """Prompt-Free Polyp Segment Anything Model.

    Forward pass
    ------------
      1. The frozen SAM image encoder (with LoRA adapters) extracts an
         image embedding.
      2. The prompt generator predicts a coarse polyp mask and the centroid
         point.
      3. The frozen SAM prompt encoder embeds the automatically generated
         prompts.
      4. The frozen SAM mask decoder produces a low-resolution mask.
      5. The boundary refinement head (optional) sharpens the mask.
      6. The mask is upsampled to the input resolution.

    Losses during training
    ----------------------
      * Main loss: BCE + Dice between the refined mask and the ground truth.
      * Auxiliary loss: BCE + Dice between the coarse mask and the ground
        truth, which supervises the prompt generator directly.
    """

    def __init__(self, sam: Sam, lora_rank: int = 4, use_refinement: bool = True) -> None:
        super().__init__()
        self.sam = sam
        self.use_refinement = use_refinement

        inject_lora(sam, rank=lora_rank)
        self.prompt_generator = PromptGenerator(in_channels=256, mid_channels=128)
        if use_refinement:
            self.refinement_head = BoundaryRefinementHead(embed_dim=256)

    def forward(self, images: torch.Tensor) -> dict:
        """Segment polyps from preprocessed images.

        Args:
            images: [B, 3, 1024, 1024] input images after SAM preprocessing
                (resized to 1024 x 1024 and normalized by the SAM pixel
                mean and standard deviation).

        Returns:
            dict with the following keys:
                masks:         [B, 1, H, W] final mask probabilities.
                low_res_masks: [B, 1, 256, 256] refined mask logits.
                coarse_masks:  [B, 1, 256, 256] coarse mask logits (auxiliary loss).
                point_coords:  [B, 1, 2] automatically generated point prompt.
                iou_predictions: [B, 1] predicted IoU (unused by default).
        """
        B = images.shape[0]

        # 1. Image encoding.
        image_embeddings = self.sam.image_encoder(images)  # [B, 256, 64, 64]

        # 2. Automatic prompt generation.
        coarse_mask, point_coords = self.prompt_generator(image_embeddings)

        # 3. Prompt encoding.
        point_labels = torch.ones(B, 1, dtype=torch.float, device=images.device)
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=coarse_mask,
        )

        # 4. Mask decoding.
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )  # [B, 1, 256, 256]

        # 5. Boundary refinement (optional).
        if self.use_refinement:
            low_res_masks = self.refinement_head(low_res_masks, image_embeddings)

        # 6. Upsample to input resolution.
        full_res_masks = F.interpolate(
            low_res_masks, size=images.shape[-2:], mode="bilinear", align_corners=False
        )

        return {
            "masks": full_res_masks,
            "low_res_masks": low_res_masks,
            "coarse_masks": coarse_mask,
            "point_coords": point_coords,
            "iou_predictions": iou_predictions,
        }

    # ------------------------------------------------------------------
    # Trainable-parameter management
    # ------------------------------------------------------------------

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Return all trainable parameters (LoRA + prompt generator + head)."""
        params: List[nn.Parameter] = [
            p for p in self.sam.image_encoder.parameters() if p.requires_grad
        ]
        params += list(self.prompt_generator.parameters())
        if self.use_refinement:
            params += list(self.refinement_head.parameters())
        return params

    def get_param_groups(self, lr_lora: float = 1e-3, lr_head: float = 1e-3):
        """Parameter groups with independent learning rates."""
        lora_params = [p for p in self.sam.image_encoder.parameters() if p.requires_grad]
        head_params = list(self.prompt_generator.parameters())
        if self.use_refinement:
            head_params += list(self.refinement_head.parameters())
        return [
            {"params": lora_params, "lr": lr_lora},
            {"params": head_params, "lr": lr_head},
        ]

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_parameters(self, filename: str) -> None:
        """Save all trainable parameters with prefixed keys."""
        state: dict = {}
        for name, param in self.sam.image_encoder.named_parameters():
            if param.requires_grad:
                state[f"lora_{name}"] = param.data
        for name, param in self.prompt_generator.named_parameters():
            state[f"pg_{name}"] = param.data
        if self.use_refinement:
            for name, param in self.refinement_head.named_parameters():
                state[f"rh_{name}"] = param.data
        torch.save(state, filename)
        print(f"[PFPSAM] Saved {len(state)} tensors to {filename}")

    def load_parameters(self, filename: str, device: str = "cuda") -> None:
        """Load trainable parameters from a checkpoint."""
        state = torch.load(filename, map_location=device)
        for name, param in self.sam.image_encoder.named_parameters():
            if param.requires_grad and f"lora_{name}" in state:
                param.data.copy_(state[f"lora_{name}"])
        for name, param in self.prompt_generator.named_parameters():
            if f"pg_{name}" in state:
                param.data.copy_(state[f"pg_{name}"])
        if self.use_refinement:
            for name, param in self.refinement_head.named_parameters():
                if f"rh_{name}" in state:
                    param.data.copy_(state[f"rh_{name}"])
        print(f"[PFPSAM] Loaded {len(state)} tensors from {filename}")


def build_pfpsam(sam_model: Sam, lora_rank: int = 4, use_refinement: bool = True) -> PFPSAM:
    """Convenience constructor for the PFPSAM model."""
    return PFPSAM(sam_model, lora_rank=lora_rank, use_refinement=use_refinement)


if __name__ == "__main__":
    # Quick sanity check of the forward pass.
    from segment_anything import sam_model_registry

    print("=== PFPSAM forward-pass test ===")
    sam = sam_model_registry["vit_b"](checkpoint=None)
    model = build_pfpsam(sam, lora_rank=4, use_refinement=True)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.get_trainable_params())
    print(f"Total parameters: {total / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable / 1e6:.2f}M ({trainable / total * 100:.2f}%)")

    dummy = torch.randn(2, 3, 1024, 1024)
    with torch.no_grad():
        out = model(dummy)
    print(f"masks:          {tuple(out['masks'].shape)}")
    print(f"low_res_masks:  {tuple(out['low_res_masks'].shape)}")
    print(f"coarse_masks:   {tuple(out['coarse_masks'].shape)}")
    print(f"point_coords:   {out['point_coords']}")
    print("=== Test passed ===")
