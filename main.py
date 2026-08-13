"""
PFPSAM: Prompt-Free Polyp Segment Anything Model
=================================================

Training, validation, and evaluation entry point.

Usage
-----
Training:
    python main.py --output ./output \
        --checkpoint /path/to/sam_vit_b_01ec64.pth \
        --dataset kvasirseg --lora_rank 4 --max_epoch_num 50

Evaluation:
    python main.py --output ./output --eval \
        --restore-model ./output/best_model.pth \
        --checkpoint /path/to/sam_vit_b_01ec64.pth \
        --dataset kvasirseg

See README.md for dataset and checkpoint preparation instructions.
"""

import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from segment_anything import sam_model_registry

from pfpsam import PFPSAM
from utils.dataloader import (
    get_im_gt_name_dict,
    create_dataloaders,
    RandomHFlip,
    Resize,
    LargeScaleJitter,
)
import utils.misc as misc
from utils.loss_mask import loss_masks


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def compute_iou(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Batch-average Intersection over Union."""
    assert target.shape[1] == 1, "only one mask per image is supported"
    if preds.shape[2:] != target.shape[2:]:
        preds = F.interpolate(preds, size=target.shape[2:], mode="bilinear", align_corners=False)
    return sum(misc.mask_iou(preds[i], target[i]) for i in range(len(preds))) / len(preds)


def compute_dice(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Batch-average Dice coefficient."""
    assert target.shape[1] == 1, "only one mask per image is supported"
    if preds.shape[2:] != target.shape[2:]:
        preds = F.interpolate(preds, size=target.shape[2:], mode="bilinear", align_corners=False)
    return sum(misc.mask_dice(preds[i], target[i]) for i in range(len(preds))) / len(preds)


def compute_boundary_iou(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Batch-average Boundary IoU (Cheng et al., CVPR 2021)."""
    assert target.shape[1] == 1, "only one mask per image is supported"
    if preds.shape[2:] != target.shape[2:]:
        preds = F.interpolate(preds, size=target.shape[2:], mode="bilinear", align_corners=False)
    return sum(misc.boundary_iou(target[i], preds[i]) for i in range(len(preds))) / len(preds)


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------

def get_args_parser():
    parser = argparse.ArgumentParser("PFPSAM", add_help=False)
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--model-type", type=str, default="vit_b",
        choices=["vit_b", "vit_l", "vit_h"], help="SAM backbone type",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the SAM pretrained checkpoint (see README.md for downloads)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", default=42, type=int)

    # Training hyperparameters
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="LoRA learning rate")
    parser.add_argument("--head_lr", default=1e-3, type=float, help="Prompt-generator/refinement learning rate")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--lr_drop_epoch", default=30, type=int)
    parser.add_argument("--max_epoch_num", default=50, type=int)
    parser.add_argument("--input_size", default=[1024, 1024], nargs=2, type=int)
    parser.add_argument("--batch_size_train", default=4, type=int)
    parser.add_argument("--batch_size_valid", default=1, type=int)
    parser.add_argument("--model_save_fre", default=1, type=int)

    # PFPSAM-specific hyperparameters
    parser.add_argument("--lora_rank", default=4, type=int, help="LoRA rank")
    parser.add_argument("--aux_loss_weight", default=0.3, type=float, help="Weight of the coarse-mask auxiliary loss")
    parser.add_argument("--no_refinement", action="store_true", help="Disable the boundary refinement head")

    # Distributed training
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--find_unused_params", action="store_true")

    # Modes
    parser.add_argument("--eval", action="store_true", help="Run evaluation only")
    parser.add_argument("--restore-model", type=str, help="Checkpoint to load for evaluation")

    # Dataset
    parser.add_argument(
        "--dataset", type=str, default="kvasirseg",
        choices=["kvasirseg", "cvc"], help="Dataset to train/evaluate on",
    )
    parser.add_argument(
        "--data-root", type=str, default="/mydata/chd/",
        help="Root directory containing the prepared datasets (see README.md)",
    )
    parser.add_argument("--exp_name", type=str, default="", help="Experiment sub-directory name")

    return parser.parse_args()


def get_dataset_config(dataset_name: str, data_root: str):
    """Return train/validation dataset configurations.

    Expected directory layout under ``data_root`` (see README.md):
        <data_root>/<Dataset>/train/images
        <data_root>/<Dataset>/train/masks
        <data_root>/<Dataset>/val/images
        <data_root>/<Dataset>/val/masks
    """
    configs = {
        "kvasirseg": {
            "train": {
                "name": "Kvasir-SEG",
                "im_dir": f"{data_root}Kvasir-SEG/train/images",
                "gt_dir": f"{data_root}Kvasir-SEG/train/masks",
                "im_ext": ".jpg", "gt_ext": ".jpg",
            },
            "val": {
                "name": "Kvasir-SEG",
                "im_dir": f"{data_root}Kvasir-SEG/val/images",
                "gt_dir": f"{data_root}Kvasir-SEG/val/masks",
                "im_ext": ".jpg", "gt_ext": ".jpg",
            },
        },
        "cvc": {
            "train": {
                "name": "CVC-ColonDB",
                "im_dir": f"{data_root}CVC-ColonDB/train/images",
                "gt_dir": f"{data_root}CVC-ColonDB/train/masks",
                "im_ext": ".png", "gt_ext": ".png",
            },
            "val": {
                "name": "CVC-ColonDB",
                "im_dir": f"{data_root}CVC-ColonDB/val/images",
                "gt_dir": f"{data_root}CVC-ColonDB/val/masks",
                "im_ext": ".png", "gt_ext": ".png",
            },
        },
    }
    return [configs[dataset_name]["train"]], [configs[dataset_name]["val"]]


# ---------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------

def setup_csv_logger(output_dir: str):
    csv_file = os.path.join(output_dir, "metrics.csv")
    fieldnames = ["epoch", "training_loss", "loss_mask", "loss_dice", "loss_aux",
                  "val_iou", "val_boundary_iou", "val_dice"]
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    return csv_file, fieldnames


def log_to_csv(csv_file: str, fieldnames, data: dict) -> None:
    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(data)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(args) -> None:
    misc.init_distributed_mode(args)
    print("args: " + str(args) + "\n")

    use_refinement = not args.no_refinement
    exp_output = os.path.join(
        args.output,
        args.exp_name if args.exp_name else f"pfpsam_{args.dataset}_{args.model_type}_r{args.lora_rank}",
    )
    os.makedirs(exp_output, exist_ok=True)
    args.output = exp_output

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_datasets, valid_datasets = get_dataset_config(args.dataset, args.data_root)
    csv_file, fieldnames = setup_csv_logger(args.output)

    # --- Data loaders ---
    if not args.eval:
        print("--- Creating training dataloader ---")
        train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
        train_dataloaders, _ = create_dataloaders(
            train_im_gt_list,
            my_transforms=[RandomHFlip(), LargeScaleJitter()],
            batch_size=args.batch_size_train,
            training=True,
        )
        print("Training dataloader created")

    print("--- Creating validation dataloader ---")
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, _ = create_dataloaders(
        valid_im_gt_list,
        my_transforms=[Resize(args.input_size)],
        batch_size=args.batch_size_valid,
        training=False,
    )
    print("Validation dataloader created")

    # --- Build PFPSAM ---
    print("--- Building PFPSAM ---")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    model = PFPSAM(sam, lora_rank=args.lora_rank, use_refinement=use_refinement)
    model = model.to(device=args.device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.get_trainable_params())
    print(f"Total parameters: {total / 1e6:.2f}M, "
          f"trainable: {trainable / 1e6:.2f}M ({trainable / total * 100:.2f}%)")

    if not args.eval:
        print("--- Training mode ---")
        param_groups = model.get_param_groups(lr_lora=args.learning_rate, lr_head=args.head_lr)
        optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
        lr_scheduler.last_epoch = args.start_epoch

        model_for_training = model
        if args.distributed:
            model_for_training = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params
            )

        train(args, model_for_training, model, optimizer, train_dataloaders,
              valid_dataloaders, lr_scheduler, csv_file, fieldnames)
    else:
        print("--- Evaluation mode ---")
        if args.restore_model:
            model.load_parameters(args.restore_model, device=args.device)
            model.eval()
            test_stats = evaluate(args, model, valid_dataloaders)
            if misc.is_main_process():
                log_to_csv(csv_file, fieldnames, {"epoch": "final_eval", **test_stats})
            print(f"Evaluation results: {test_stats}")
        else:
            raise ValueError("--restore-model is required in evaluation mode")


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------

def train(args, model_wrapped, model_raw, optimizer, train_dataloaders,
          valid_dataloaders, lr_scheduler, csv_file, fieldnames) -> None:
    model_wrapped.train()

    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num
    best_iou = 0.0

    for epoch in range(epoch_start, epoch_num):
        print(f"\nEpoch {epoch}/{epoch_num - 1}  LR: {optimizer.param_groups[0]['lr']:.6f}")
        metric_logger = misc.MetricLogger(delimiter="  ")

        if hasattr(train_dataloaders, "batch_sampler") and hasattr(
            train_dataloaders.batch_sampler.sampler, "set_epoch"
        ):
            train_dataloaders.batch_sampler.sampler.set_epoch(epoch)

        for data in metric_logger.log_every(train_dataloaders, 100):
            inputs, labels = data["image"], data["label"]
            if torch.cuda.is_available():
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            # SAM preprocessing: (pixel - mean) / std, matching the original SAM.
            inputs_norm = sam_preprocess(inputs)

            outputs = model_wrapped(inputs_norm)

            # Main loss on the refined mask.
            low_res_masks = outputs["low_res_masks"]
            # Normalize ground-truth masks (0-255) to [0, 1] for the BCE/Dice losses.
            labels_resized = F.interpolate(
                labels.float() / 255.0, size=low_res_masks.shape[-2:],
                mode="bilinear", align_corners=False,
            )
            loss_mask, loss_dice = loss_masks(low_res_masks, labels_resized, len(low_res_masks))
            loss_main = loss_mask + loss_dice

            # Auxiliary loss on the coarse mask (supervises the prompt generator).
            coarse_masks = outputs["coarse_masks"]
            labels_coarse = F.interpolate(
                labels.float() / 255.0, size=coarse_masks.shape[-2:],
                mode="bilinear", align_corners=False,
            )
            loss_aux_mask, loss_aux_dice = loss_masks(coarse_masks, labels_coarse, len(coarse_masks))
            loss_aux = (loss_aux_mask + loss_aux_dice) * args.aux_loss_weight

            loss = loss_main + loss_aux

            loss_dict = {"loss_mask": loss_mask, "loss_dice": loss_dice, "loss_aux": loss_aux}
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            loss_value = sum(loss_dict_reduced.values()).item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_logger.update(training_loss=loss_value, **loss_dict_reduced)

        print(f"Epoch {epoch} finished")
        metric_logger.synchronize_between_processes()
        print(f"Training stats: {metric_logger}")
        train_stats = {k: m.global_avg for k, m in metric_logger.meters.items() if m.count > 0}

        lr_scheduler.step()

        # Validation.
        val_stats = evaluate(args, model_wrapped, valid_dataloaders)
        train_stats.update(val_stats)

        if misc.is_main_process():
            log_to_csv(csv_file, fieldnames, {"epoch": epoch, **train_stats})

            current_iou = val_stats.get("val_iou", 0.0)
            if current_iou > best_iou:
                best_iou = current_iou
                best_path = os.path.join(args.output, "best_model.pth")
                model_raw.save_parameters(best_path)
                print(f"  * New best IoU {best_iou:.4f}; saved to {best_path}")

            if epoch % args.model_save_fre == 0 or epoch == epoch_num - 1:
                epoch_path = os.path.join(args.output, f"epoch{epoch}.pth")
                model_raw.save_parameters(epoch_path)

        model_wrapped.train()

    print(f"\nTraining finished. Best validation IoU: {best_iou:.4f}")


# ---------------------------------------------------------------------
# Validation / evaluation
# ---------------------------------------------------------------------

def evaluate(args, model, valid_dataloaders) -> dict:
    model.eval()
    print("Validating...")
    test_stats: dict = {}

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        print(f"Validation set {k}: {len(valid_dataloader)} samples")

        for data_val in metric_logger.log_every(valid_dataloader, 1000):
            inputs_val, labels_val = data_val["image"], data_val["label"]
            if torch.cuda.is_available():
                inputs_val = inputs_val.cuda()
                labels_val = labels_val.cuda()

            inputs_norm = sam_preprocess(inputs_val)

            with torch.no_grad():
                outputs = model(inputs_norm)

            low_res_masks = outputs["low_res_masks"]
            # Evaluation metrics threshold at 128 for 0-255 masks; keep them unnormalized.
            labels_resized = F.interpolate(
                labels_val.float(), size=low_res_masks.shape[-2:], mode="bilinear", align_corners=False
            )

            iou = compute_iou(low_res_masks, labels_resized)
            boundary_iou = compute_boundary_iou(low_res_masks, labels_resized)
            dice = compute_dice(low_res_masks, labels_resized)

            metric_logger.update(
                val_iou=iou, val_boundary_iou=boundary_iou, val_dice=dice
            )

        metric_logger.synchronize_between_processes()
        print(f"Validation stats: {metric_logger}")
        resstat = {k: m.global_avg for k, m in metric_logger.meters.items() if m.count > 0}
        test_stats.update(resstat)

    return test_stats


# ---------------------------------------------------------------------
# SAM preprocessing
# ---------------------------------------------------------------------

SAM_PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
SAM_PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)


def sam_preprocess(images: torch.Tensor) -> torch.Tensor:
    """Normalize [0, 255] images with the SAM pixel mean and std."""
    mean = SAM_PIXEL_MEAN.to(images.device)
    std = SAM_PIXEL_STD.to(images.device)
    return (images - mean) / std


if __name__ == "__main__":
    args = get_args_parser()
    main(args)
