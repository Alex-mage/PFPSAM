"""
Dataset preparation: split polyp datasets into train/validation partitions.
==========================================================================

The raw polyp segmentation datasets (Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB)
typically ship as a flat folder of images and masks. This script organizes
them into the train/val directory layout expected by ``main.py``:

    <data-root>/<Dataset>/train/images/*.jpg
    <data-root>/<Dataset>/train/masks/*.jpg
    <data-root>/<Dataset>/val/images/*.jpg
    <data-root>/<Dataset>/val/masks/*.jpg

Usage
-----
    python data_split.py --src /path/to/Kvasir-SEG \
        --dst /mydata/chd/Kvasir-SEG --val-ratio 0.2 --seed 42
"""

import argparse
import os
import random
import shutil
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser(description="Split a polyp dataset into train/val.")
    parser.add_argument("--src", type=str, required=True,
                        help="Raw dataset folder containing images/ and masks/ subfolders")
    parser.add_argument("--dst", type=str, required=True,
                        help="Destination root, e.g. /mydata/chd/Kvasir-SEG")
    parser.add_argument("--im-dir", type=str, default="images", help="Image subfolder name")
    parser.add_argument("--gt-dir", type=str, default="masks", help="Mask subfolder name")
    parser.add_argument("--im-ext", type=str, default=".jpg")
    parser.add_argument("--gt-ext", type=str, default=".jpg")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main(args) -> None:
    random.seed(args.seed)

    src_im_dir = os.path.join(args.src, args.im_dir)
    src_gt_dir = os.path.join(args.src, args.gt_dir)

    im_paths = sorted(glob(os.path.join(src_im_dir, f"*{args.im_ext}")))
    assert im_paths, f"No images found in {src_im_dir}"

    # Match masks by base name.
    pairs = []
    for im in im_paths:
        base = os.path.splitext(os.path.basename(im))[0]
        gt = os.path.join(src_gt_dir, base + args.gt_ext)
        if os.path.exists(gt):
            pairs.append((im, gt))
        else:
            print(f"WARNING: mask not found for {im}")

    random.shuffle(pairs)
    num_val = max(1, int(len(pairs) * args.val_ratio))
    val_pairs, train_pairs = pairs[:num_val], pairs[num_val:]
    print(f"Total: {len(pairs)} | Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    for split, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        dst_im = os.path.join(args.dst, split, args.im_dir)
        dst_gt = os.path.join(args.dst, split, args.gt_dir)
        os.makedirs(dst_im, exist_ok=True)
        os.makedirs(dst_gt, exist_ok=True)
        for im, gt in split_pairs:
            shutil.copy2(im, dst_im)
            shutil.copy2(gt, dst_gt)

    print("Done.")


if __name__ == "__main__":
    main(parse_args())
