# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Miscellaneous utilities: distributed-training helpers, metric loggers, and
mask-related metric functions. Most helpers are adapted from the torchvision
references.
"""

import datetime
import os
import random
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


class SmoothedValue(object):
    """Track a series of values and provide smoothed/global averages."""

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        if d.shape[0] == 0:
            return 0
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg,
            max=self.max, value=self.value,
        )


def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """Reduce a dict of tensors across processes."""
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        return {k: v for k, v in zip(names, values)}


class MetricLogger(object):
    """Logger that accumulates metrics and prints them periodically."""

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        return self.delimiter.join(
            f"{name}: {str(meter)}" for name, meter in self.meters.items() if meter.count > 0
        )

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        header = header or ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}",
                "{meters}", "time: {time}", "data: {data}", "max mem: {memory:.0f}",
            ])
        else:
            log_msg = self.delimiter.join([
                header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}",
                "{meters}", "time: {time}", "data: {data}",
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string, meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB,
                    ))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string, meters=str(self),
                        time=str(iter_time), data=str(data_time),
                    ))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


# ---------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------

def setup_for_distributed(is_master: bool) -> None:
    """Suppress printing on non-master processes."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed_mode(args) -> None:
    if "WORLD_SIZE" in os.environ and os.environ["WORLD_SIZE"] != "":
        local_world_size = int(os.environ["WORLD_SIZE"])
        args.world_size = args.world_size * local_world_size
        args.gpu = args.local_rank = int(os.environ["LOCAL_RANK"])
        args.rank = args.rank * local_world_size + args.local_rank
        print(f"world size: {args.world_size}, rank: {args.rank}, "
              f"local rank: {args.local_rank}")
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.local_rank = int(os.environ["SLURM_LOCALID"])
        args.world_size = int(os.environ["SLURM_NPROCS"])
        print(f"world size: {args.world_size}, world rank: {args.rank}, "
              f"local rank: {args.local_rank}")
    else:
        print("Not using distributed mode")
        args.distributed = False
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0
        return

    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


# ---------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------

def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """Compute xyxy bounding boxes around masks [N, H, W]."""
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)
    h, w = masks.shape[-2:]
    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    y, x = torch.meshgrid(y, x)
    x_mask = (masks > 128) * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks > 128), 1e8).flatten(1).min(-1)[0]
    y_mask = (masks > 128) * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks > 128), 1e8).flatten(1).min(-1)[0]
    return torch.stack([x_min, y_min, x_max, y_max], 1)


def masks_sample_points(masks: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Randomly sample ``k`` foreground points per mask [N, H, W]."""
    if masks.numel() == 0:
        return torch.zeros((0, 2), device=masks.device)
    h, w = masks.shape[-2:]
    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    y, x = torch.meshgrid(y, x)
    samples = []
    for b_i in range(len(masks)):
        select_mask = masks[b_i] > 128
        x_idx = torch.masked_select(x, select_mask)
        y_idx = torch.masked_select(y, select_mask)
        perm = torch.randperm(x_idx.size(0))
        idx = perm[:k]
        samples.append(torch.cat((x_idx[idx][:, None], y_idx[idx][:, None]), dim=1))
    return torch.stack(samples)


def masks_noise(masks: torch.Tensor) -> torch.Tensor:
    """Add noise to mask inputs (from Mask Transfiner)."""
    def get_incoherent_mask(input_masks, sfact):
        mask = input_masks.float()
        w, h = input_masks.shape[-1], input_masks.shape[-2]
        mask_small = F.interpolate(mask, (h // sfact, w // sfact), mode="bilinear")
        mask_recover = F.interpolate(mask_small, (h, w), mode="bilinear")
        return (mask - mask_recover).abs().ge(0.01).float()

    gt_masks_vector = masks / 255
    mask_noise = torch.randn(gt_masks_vector.shape, device=gt_masks_vector.device)
    inc_masks = get_incoherent_mask(gt_masks_vector, 8)
    gt_masks_vector = ((gt_masks_vector + mask_noise * inc_masks) > 0.5).float()
    return gt_masks_vector * 255


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def mask_iou(pred_label: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Mask IoU between prediction and ground truth."""
    pred = (pred_label > 0)[0].int()
    gt = (label > 128)[0].int()
    intersection = ((gt * pred) > 0).sum()
    union = ((gt + pred) > 0).sum()
    return intersection / union


def mask_dice(pred_label: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Dice coefficient for a single sample."""
    pred = (pred_label > 0)[0].int()
    gt = (label > 128)[0].int()
    intersection = (pred * gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    if pred_sum + gt_sum == 0:
        return torch.tensor(1.0, device=pred_label.device)
    return (2.0 * intersection) / (pred_sum + gt_sum)


def mask_to_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    """Convert a binary mask to its boundary (Bowen Cheng et al.)."""
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    return mask - mask_erode


def boundary_iou(gt: torch.Tensor, dt: torch.Tensor, dilation_ratio: float = 0.02) -> torch.Tensor:
    """Boundary IoU between two binary masks."""
    device = gt.device
    dt_np = (dt > 0)[0].cpu().byte().numpy()
    gt_np = (gt > 128)[0].cpu().byte().numpy()
    gt_boundary = mask_to_boundary(gt_np, dilation_ratio)
    dt_boundary = mask_to_boundary(dt_np, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    return torch.tensor(intersection / union).float().to(device)
