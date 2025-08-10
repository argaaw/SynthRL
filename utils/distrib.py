# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Torch distributed utilities."""

import typing as tp

import torch


def world_size():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1
    
    
def is_distributed():
    return world_size() > 1


def all_reduce(tensor: torch.Tensor, op=torch.distributed.ReduceOp.SUM):
    if is_distributed():
        return torch.distributed.all_reduce(tensor, op)
    

def average_metrics(metrics: tp.Dict[str, float], count=1.):
    """Average a dictionary of metrics across all workers, using the optional
    `count` as unnormalized weight.
    """
    if not is_distributed():
        return metrics
    keys, values = zip(*metrics.items())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor = torch.tensor(list(values) + [1], device=device, dtype=torch.float32)
    tensor *= count
    all_reduce(tensor)
    averaged = (tensor[:-1] / tensor[-1]).cpu().tolist()
    return dict(zip(keys, averaged))


def get_parallel_devices(main_cuda_device_idx: int):
    if torch.cuda.device_count() == 0:
        raise NotImplementedError()
    elif torch.cuda.device_count() == 1:
        output_device = 'cuda:0'
        device_ids = [0]
    else:
        output_device = torch.device(f'cuda:{main_cuda_device_idx}')
        device_ids = [i for i in range(torch.cuda.device_count()) if i != main_cuda_device_idx]
        device_ids.insert(0, main_cuda_device_idx)

    return output_device, device_ids
