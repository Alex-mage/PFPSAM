"""
PFPSAM inference script
========================

Runs the trained PFPSAM model on a folder of colonoscopy images and saves
the predicted binary masks (and optional overlays).

Usage
-----
    python predict.py --checkpoint /path/to/sam_vit_b_01ec64.pth \
        --restore-model ./output/best_model.pth \
        --input ./test_images --output ./predictions

Outputs
-------
    <output>/masks/<name>.png      binary prediction (0/255)
    <output>/overlays/<name>.png   original image with mask overlay
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from segment_anything import sam_model_registry
from tqdm import tqdm

from pfpsam import PFPSAM


SAM_PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
SAM_PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="PFPSAM inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="SAM pretrained weights")
    parser.add_argument("--restore-model", type=str, required=True, help="PFPSAM trained weights")
    parser.add_argument("--input", type=str, required=True, help="Input image folder or single image")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--model-type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--no-refinement", action="store_true")
    parser.add_argument("--save-overlay", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--im-ext", type=str, default=".jpg")
    return parser.parse_args()


@torch.no_grad()
def predict_image(model, image_np: np.ndarray, device: str) -> np.ndarray:
    """Predict the polyp mask for a single image.

    Args:
        image_np: [H, W, 3] uint8 image.
    Returns:
        [H, W] uint8 binary mask (0/255).
    """
    # Resize to 1024 x 1024 with aspect-ratio padding, matching SAM preprocessing.
    h, w = image_np.shape[:2]
    scale = 1024.0 / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image_np, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((1024, 1024, 3), 128, dtype=np.uint8)
    canvas[:nh, :nw] = resized

    tensor = torch.from_numpy(canvas).permute(2, 0, 1).float().unsqueeze(0).to(device)
    tensor = (tensor - SAM_PIXEL_MEAN.to(device)) / SAM_PIXEL_STD.to(device)

    outputs = model(tensor)
    logits = outputs["masks"]  # [1, 1, 1024, 1024]

    # Crop the padding region and resize back to the original size.
    mask_1024 = (torch.sigmoid(logits) > 0.5).float().cpu()
    mask_crop = mask_1024[0, 0, :nh, :nw]
    mask_orig = F.interpolate(
        mask_crop.unsqueeze(0).unsqueeze(0), size=(h, w), mode="nearest"
    )[0, 0]

    return (mask_orig.numpy() * 255).astype(np.uint8)


def main(args) -> None:
    os.makedirs(os.path.join(args.output, "masks"), exist_ok=True)
    if args.save_overlay:
        os.makedirs(os.path.join(args.output, "overlays"), exist_ok=True)

    print("--- Building PFPSAM ---")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    model = PFPSAM(sam, lora_rank=args.lora_rank, use_refinement=not args.no_refinement)
    model.load_parameters(args.restore_model, device=args.device)
    model.to(device=args.device)
    model.eval()

    # Collect input paths.
    if os.path.isdir(args.input):
        im_paths = sorted(
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if f.lower().endswith(args.im_ext)
        )
    else:
        im_paths = [args.input]

    assert im_paths, f"No {args.im_ext} images found in {args.input}"

    for path in tqdm(im_paths, desc="Segmenting"):
        image_np = np.array(Image.open(path).convert("RGB"))
        mask = predict_image(model, image_np, args.device)

        name = os.path.splitext(os.path.basename(path))[0]
        mask_path = os.path.join(args.output, "masks", f"{name}.png")
        Image.fromarray(mask).save(mask_path)

        if args.save_overlay:
            overlay = image_np.copy()
            color = np.array([30, 144, 255], dtype=np.uint8)  # RGB
            overlay[mask > 0] = (overlay[mask > 0] * 0.5 + color * 0.5).astype(np.uint8)
            Image.fromarray(overlay).save(os.path.join(args.output, "overlays", f"{name}.png"))

    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main(parse_args())
