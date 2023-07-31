import argparse
import logging
import os
import os.path as osp
import time

import mmengine
import torch
import torch.distributed as dist

from utils.pt_utils import (
    ensure_dir,
    extant_file,
    link_file,
    load_model,
    torch_distributed_zero_first,
)

LOGGER = logging.getLogger(name="main")


class State(object):
    def __init__(self):
        self.epoch = 1
        self.iteration = 0
        self.dataloader = None
        self.model = None
        self.optimizer = None

    def register(self, **kwargs):
        for k, v in kwargs.items():
            assert k in ["epoch", "iteration", "dataloader", "model", "optimizer"]
            setattr(self, k, v)


class Engine(object):
    def __init__(self):
        LOGGER.info("PyTorch Version {}".format(torch.__version__))
        self.state = State()
        self.distributed = False

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("config", type=str)
        self.parser.add_argument("--output-root", type=str, default="./outputs")
        self.parser.add_argument("--model-name", type=str, required=True)
        self.parser.add_argument("--continue", type=extant_file, dest="continue_fpath")
        self.parser.add_argument("--info", default="", type=str)
        cmd_args = self.parser.parse_args()

        self.args = mmengine.Config().fromfile(filename=cmd_args.config)
        self.args.merge_from_dict(options=vars(cmd_args))
        LOGGER.info(self.args.pretty_text)

        self.continue_state_object = self.args.continue_fpath
        if "WORLD_SIZE" in os.environ:
            self.distributed = int(os.environ["WORLD_SIZE"]) > 1

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if self.distributed:
            self.world_size = int(os.environ["WORLD_SIZE"])
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(
                backend="nccl", world_size=self.world_size, init_method="env://"
            )
        self.is_master = (self.distributed and (self.local_rank == 0)) or (
            not self.distributed
        )

        self.proj_name = self.construct_proj_name(args=self.args)
        self.proj_root = os.path.join(self.args.output_root, self.proj_name)
        with torch_distributed_zero_first(self.local_rank):
            os.makedirs(self.proj_root, exist_ok=True)

        proj_idx = -1
        for proj_idx, proj_instance in enumerate(sorted(os.listdir(self.proj_root))):
            if int(proj_instance.split("-")[-1]) != proj_idx:
                break
        else:
            proj_idx += 1

        self.proj_path = os.path.join(self.proj_root, f"exp-{proj_idx}")
        self.tb_dir = os.path.join(self.proj_path, "tb")
        self.checkpoint_dir = os.path.join(self.proj_path, "checkpoints")
        self.log_file = os.path.join(self.proj_path, "train.log")
        with torch_distributed_zero_first(self.local_rank):
            os.makedirs(self.proj_path, exist_ok=True)
            os.makedirs(self.tb_dir, exist_ok=True)
            if not os.path.exists(self.log_file):
                open(self.log_file, mode="w").close()

    @staticmethod
    def construct_proj_name(args):
        return "-".join(
            [
                args.model_name,
                f"bs{args.batch_size}"
                f"lr{args.lr}"
                f"wd{args.weight_decay}"
                f"pa{args.group_mode}"
                f"fp16{args.use_fp16}"
                f"sche{args.scheduler}"
                f"info{args.info}",
            ]
        )

    def register_state(self, **kwargs):
        self.state.register(**kwargs)

    def update_iteration(self, epoch, iteration):
        self.state.epoch = epoch
        self.state.iteration = iteration

    def _save_checkpoint(self, path):
        LOGGER.info("Saving checkpoint to file {}".format(path))
        t_start = time.time()

        state_dict = {}

        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in self.state.model.state_dict().items():
            key = k
            if k.split(".")[0] == "module":
                key = k[7:]
            new_state_dict[key] = v
        state_dict["model"] = new_state_dict
        state_dict["optimizer"] = self.state.optimizer.state_dict()
        state_dict["epoch"] = self.state.epoch
        state_dict["iteration"] = self.state.iteration

        t_iobegin = time.time()
        torch.save(state_dict, path)
        del state_dict
        del new_state_dict
        t_end = time.time()
        LOGGER.info(
            "Save checkpoint to file {}, "
            "Time usage:\n\tprepare checkpoint: {}, IO: {}".format(
                path, t_iobegin - t_start, t_end - t_iobegin
            )
        )

    def link_tb(self, source, target):
        ensure_dir(source)
        ensure_dir(target)
        link_file(source, target)

    def save_checkpoint(self, checkpoint_dir):
        ensure_dir(checkpoint_dir)
        current_epoch_checkpoint = osp.join(
            checkpoint_dir, "epoch-{}.pth".format(self.state.epoch)
        )
        self._save_checkpoint(current_epoch_checkpoint)
        last_epoch_checkpoint = osp.join(checkpoint_dir, "epoch-last.pth")
        link_file(current_epoch_checkpoint, last_epoch_checkpoint)

    def restore_checkpoint(self):
        t_start = time.time()
        if self.distributed:
            # load the model on cpu first to avoid GPU RAM surge
            # when loading a model checkpoint
            # tmp = torch.load(self.continue_state_object,
            #                  map_location=lambda storage, loc: storage.cuda(
            #                      self.local_rank))
            tmp = torch.load(
                self.continue_state_object, map_location=torch.device("cpu")
            )
        else:
            tmp = torch.load(self.continue_state_object)
        t_ioend = time.time()
        self.state.model = load_model(self.state.model, tmp["model"], is_restore=True)
        self.state.optimizer.load_state_dict(tmp["optimizer"])
        self.state.epoch = tmp["epoch"] + 1
        self.state.iteration = tmp["iteration"]
        del tmp
        t_end = time.time()
        LOGGER.info(
            "Load checkpoint from file {}, "
            "Time usage:\n\tIO: {}, restore checkpoint: {}".format(
                self.continue_state_object, t_ioend - t_start, t_end - t_ioend
            )
        )

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        torch.cuda.empty_cache()
        if type is not None:
            LOGGER.warning(
                "A exception occurred during Engine initialization, "
                "give up running process"
            )
            return False
