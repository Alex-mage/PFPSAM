"""
Data loading utilities for PFPSAM.
"""

from __future__ import print_function, division

import os
import random
from copy import deepcopy
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from skimage import io
from skimage.util import img_as_ubyte
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms.functional import normalize


# ---------------------------------------------------------------------
# Dataset listing
# ---------------------------------------------------------------------

def get_im_gt_name_dict(datasets, flag="valid"):
    """Build a list of image/ground-truth path pairs for the given datasets."""
    print("------------------------------", flag, "--------------------------------")
    name_im_gt_list = []

    for i in range(len(datasets)):
        print(f"--->>> {flag} dataset {i}/{len(datasets)} {datasets[i]['name']} <<<---")
        tmp_im_list = glob(datasets[i]["im_dir"] + os.sep + "*" + datasets[i]["im_ext"])
        print(f"-im- {datasets[i]['name']} {datasets[i]['im_dir']}: {len(tmp_im_list)}")

        if datasets[i]["gt_dir"] == "":
            print(f"-gt- {datasets[i]['name']}: No ground truth found")
            tmp_gt_list = []
        else:
            tmp_gt_list = [
                datasets[i]["gt_dir"] + os.sep
                + x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]
                + datasets[i]["gt_ext"]
                for x in tmp_im_list
            ]
            print(f"-gt- {datasets[i]['name']} {datasets[i]['gt_dir']}: {len(tmp_gt_list)}")

        name_im_gt_list.append({
            "dataset_name": datasets[i]["name"],
            "im_path": tmp_im_list,
            "gt_path": tmp_gt_list,
            "im_ext": datasets[i]["im_ext"],
            "gt_ext": datasets[i]["gt_ext"],
        })

    return name_im_gt_list


def create_dataloaders(name_im_gt_list, my_transforms=None, batch_size=1, training=False):
    """Create training (concatenated) or validation dataloaders."""
    if my_transforms is None:
        my_transforms = []

    gos_dataloaders = []
    gos_datasets = []

    if len(name_im_gt_list) == 0:
        return gos_dataloaders, gos_datasets

    num_workers_ = 1
    if batch_size > 1:
        num_workers_ = 2
    if batch_size > 4:
        num_workers_ = 4
    if batch_size > 8:
        num_workers_ = 8

    if training:
        for i in range(len(name_im_gt_list)):
            gos_dataset = OnlineDataset(
                [name_im_gt_list[i]], transform=transforms.Compose(my_transforms)
            )
            gos_datasets.append(gos_dataset)

        gos_dataset = ConcatDataset(gos_datasets)
        sampler = DistributedSampler(gos_dataset)
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler, batch_size, drop_last=True
        )
        dataloader = DataLoader(gos_dataset, batch_sampler=batch_sampler_train,
                                num_workers=num_workers_)

        gos_dataloaders = dataloader
        gos_datasets = gos_dataset
    else:
        for i in range(len(name_im_gt_list)):
            gos_dataset = OnlineDataset(
                [name_im_gt_list[i]],
                transform=transforms.Compose(my_transforms),
                eval_ori_resolution=True,
            )
            sampler = DistributedSampler(gos_dataset, shuffle=False)
            dataloader = DataLoader(gos_dataset, batch_size, sampler=sampler,
                                    drop_last=False, num_workers=num_workers_)
            gos_dataloaders.append(dataloader)
            gos_datasets.append(gos_dataset)

    return gos_dataloaders, gos_datasets


# ---------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------

class RandomHFlip(object):
    """Random horizontal flip with probability ``prob``."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        imidx, image, label, shape = (
            sample["imidx"], sample["image"], sample["label"], sample["shape"]
        )
        if random.random() >= self.prob:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[2])
        return {"imidx": imidx, "image": image, "label": label, "shape": shape}


class Resize(object):
    """Bilinear resize of both image and label."""

    def __init__(self, size=[1024, 1024]):
        self.size = size

    def __call__(self, sample):
        imidx, image, label, shape = (
            sample["imidx"], sample["image"], sample["label"], sample["shape"]
        )
        image = torch.squeeze(F.interpolate(torch.unsqueeze(image, 0), self.size, mode="bilinear"), dim=0)
        label = torch.squeeze(F.interpolate(torch.unsqueeze(label, 0), self.size, mode="bilinear"), dim=0)
        return {"imidx": imidx, "image": image, "label": label, "shape": torch.tensor(self.size)}


class Normalize(object):
    """ImageNet normalization (not used with SAM; kept for completeness)."""

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        imidx, image, label, shape = (
            sample["imidx"], sample["image"], sample["label"], sample["shape"]
        )
        image = normalize(image, self.mean, self.std)
        return {"imidx": imidx, "image": image, "label": label, "shape": shape}


class LargeScaleJitter(object):
    """Large-scale jitter augmentation (Ghiasi et al., 2021).

    Implementation adapted from Copy-Paste:
    https://github.com/gaopengcuhk/Pretrained-Pix2Seq
    """

    def __init__(self, output_size=1024, aug_scale_min=0.1, aug_scale_max=2.0):
        self.desired_size = torch.tensor(output_size)
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

    def __call__(self, sample):
        imidx, image, label, image_size = (
            sample["imidx"], sample["image"], sample["label"], sample["shape"]
        )

        random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
        scaled_size = (random_scale * self.desired_size).round()

        scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
        scaled_size = torch.tensor(image_size, dtype=torch.float32) * scale
        scaled_size = scaled_size.round().long()

        scaled_image = torch.squeeze(
            F.interpolate(torch.unsqueeze(image, 0), scaled_size.tolist(), mode="bilinear"), dim=0
        )
        scaled_label = torch.squeeze(
            F.interpolate(torch.unsqueeze(label, 0), scaled_size.tolist(), mode="bilinear"), dim=0
        )

        # Random crop to the desired size.
        crop_size = (min(self.desired_size, scaled_size[0]), min(self.desired_size, scaled_size[1]))
        margin_h = max(scaled_size[0] - crop_size[0], 0).item()
        margin_w = max(scaled_size[1] - crop_size[1], 0).item()
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0].item()
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1].item()

        scaled_image = scaled_image[:, crop_y1:crop_y2, crop_x1:crop_x2]
        scaled_label = scaled_label[:, crop_y1:crop_y2, crop_x1:crop_x2]

        # Pad to the desired size.
        padding_h = max(self.desired_size - scaled_image.size(1), 0).item()
        padding_w = max(self.desired_size - scaled_image.size(2), 0).item()
        image = F.pad(scaled_image, [0, padding_w, 0, padding_h], value=128)
        label = F.pad(scaled_label, [0, padding_w, 0, padding_h], value=0)

        return {"imidx": imidx, "image": image, "label": label, "shape": torch.tensor(image.shape[-2:])}


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class OnlineDataset(Dataset):
    """Dataset that reads images and ground-truth masks on the fly.

    The image is returned as a float32 [C, H, W] tensor in [0, 255] and the
    mask as a float32 [1, H, W] tensor in [0, 255].
    """

    def __init__(self, name_im_gt_list, transform=None, eval_ori_resolution=False):
        self.transform = transform
        self.dataset = {}

        dataset_names = []
        dt_name_list = []
        im_name_list = []
        im_path_list = []
        gt_path_list = []
        im_ext_list = []
        gt_ext_list = []

        for i in range(len(name_im_gt_list)):
            dataset_names.append(name_im_gt_list[i]["dataset_name"])
            dt_name_list.extend(
                [name_im_gt_list[i]["dataset_name"] for _ in name_im_gt_list[i]["im_path"]]
            )
            im_name_list.extend(
                [x.split(os.sep)[-1].split(name_im_gt_list[i]["im_ext"])[0]
                 for x in name_im_gt_list[i]["im_path"]]
            )
            im_path_list.extend(name_im_gt_list[i]["im_path"])
            gt_path_list.extend(name_im_gt_list[i]["gt_path"])
            im_ext_list.extend(
                [name_im_gt_list[i]["im_ext"] for _ in name_im_gt_list[i]["im_path"]]
            )
            gt_ext_list.extend(
                [name_im_gt_list[i]["gt_ext"] for _ in name_im_gt_list[i]["gt_path"]]
            )

        self.dataset["data_name"] = dt_name_list
        self.dataset["im_name"] = im_name_list
        self.dataset["im_path"] = im_path_list
        self.dataset["ori_im_path"] = deepcopy(im_path_list)
        self.dataset["gt_path"] = gt_path_list
        self.dataset["ori_gt_path"] = deepcopy(gt_path_list)
        self.dataset["im_ext"] = im_ext_list
        self.dataset["gt_ext"] = gt_ext_list

        self.eval_ori_resolution = eval_ori_resolution

    def __len__(self):
        return len(self.dataset["im_path"])

    def __getitem__(self, idx):
        im_path = self.dataset["im_path"][idx]
        gt_path = self.dataset["gt_path"][idx]
        im = io.imread(im_path)
        gt = io.imread(gt_path)

        if len(gt.shape) > 2:
            gt = gt[:, :, 0]
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        if im.shape[2] == 4:
            im = im[:, :, :3]

        im = torch.tensor(im.copy(), dtype=torch.float32)
        im = torch.transpose(torch.transpose(im, 1, 2), 0, 1)  # HWC -> CHW
        gt = img_as_ubyte(gt)
        gt = torch.unsqueeze(torch.tensor(gt, dtype=torch.float32), 0)
        H, W = im.shape[-2:]

        sample = {
            "imidx": torch.from_numpy(np.array(idx)),
            "image": im,
            "label": gt,
            "shape": (int(H), int(W)),
        }

        if self.transform:
            sample = self.transform(sample)

        if self.eval_ori_resolution:
            sample["ori_label"] = gt.type(torch.uint8)
            sample["ori_im_path"] = self.dataset["im_path"][idx]
            sample["ori_gt_path"] = self.dataset["gt_path"][idx]

        return sample
