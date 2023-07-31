# encoding: utf-8
import argparse
import logging
import os
import random
import time
from collections import OrderedDict
from contextlib import contextmanager

import torch
import torch.distributed as dist

LOGGER = logging.getLogger(name="main.pt_utils")


def reduce_tensor(tensor, dst=0, op=dist.ReduceOp.SUM, world_size=1):
    tensor = tensor.clone()
    dist.reduce(tensor, dst, op)
    if dist.get_rank() == dst:
        tensor.div_(world_size)

    return tensor


def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    tensor.div_(world_size)

    return tensor


def load_restore_model(model, model_file):
    t_start = time.time()

    if model_file is None:
        return model

    if isinstance(model_file, str):
        state_dict = torch.load(model_file)
        if "model" in state_dict.keys():
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]
        elif "module" in state_dict.keys():
            state_dict = state_dict["module"]
    else:
        state_dict = model_file
    t_ioend = time.time()

    model.load_state_dict(state_dict, strict=True)

    del state_dict
    t_end = time.time()
    LOGGER.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend
        )
    )

    return model


def load_params(model_file):
    if isinstance(model_file, str):
        state_dict = torch.load(model_file)
        if "model" in state_dict.keys():
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]
        elif "module" in state_dict.keys():
            state_dict = state_dict["module"]
    else:
        state_dict = model_file
    return state_dict


def load_model(model, model_file, is_restore=False):
    t_start = time.time()

    if model_file is None:
        return model

    state_dict = load_params(model_file)
    t_ioend = time.time()

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = "module." + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=True)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    del state_dict
    t_end = time.time()
    LOGGER.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend
        )
    )

    return model


def parse_devices(input_devices):
    if input_devices.endswith("*"):
        devices = list(range(torch.cuda.device_count()))
        return devices

    devices = []
    for d in input_devices.split(","):
        if "-" in d:
            start_device, end_device = d.split("-")[0], d.split("-")[1]
            assert start_device != ""
            assert end_device != ""
            start_device, end_device = int(start_device), int(end_device)
            assert start_device < end_device
            assert end_device < torch.cuda.device_count()
            for sd in range(start_device, end_device + 1):
                devices.append(sd)
        else:
            device = int(d)
            assert device < torch.cuda.device_count()
            devices.append(device)

    LOGGER.info("using devices {}".format(", ".join([str(d) for d in devices])))

    return devices


def extant_file(x):
    """
    'Type' for argparse - checks that file exists but does not open.
    """
    if not os.path.exists(x):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x


def link_file(src, target):
    if os.path.isdir(target) or os.path.isfile(target):
        os.system("rm -rf {}".format(target))
    os.system("ln -s {} {}".format(src, target))


def ensure_dir(path):
    if not os.path.isdir(path):
        try:
            sleeptime = random.randint(0, 3)
            time.sleep(sleeptime)
            os.makedirs(path, exist_ok=True)
        except:
            print("conflict !!!")


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_master():
    return get_rank() == 0


@contextmanager
def torch_distributed_zero_first(rank: int):
    """Decorator to make all processes in distributed training wait for each local_master to do something."""
    if is_dist_avail_and_initialized() and rank not in [-1, 0]:
        torch.distributed.barrier()
    # 这里的用法其实就是协程的一种哦。
    yield
    if is_dist_avail_and_initialized() and rank == 0:
        torch.distributed.barrier()
