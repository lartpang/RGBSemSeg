import logging
import os
import random

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

import models as model_zoo
from engine.engine import Engine
from utils import pt_utils
from utils.data import Label
from utils.init_func import group_weight
from utils.lr_policy import WarmUpPolyLR
from utils.transforms import (
    generate_random_crop_pos,
    normalize,
    random_crop_pad_to_shape,
)

CLASSES = [
    # name,id,trainId,category,catId,hasInstances,ignoreInEval,color,isDifficult
    Label("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0), None),
    Label("ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0), None),
    Label("rectification border", 2, 255, "void", 0, False, True, (0, 0, 0), None),
    Label("out of roi", 3, 255, "void", 0, False, True, (0, 0, 0), None),
    Label("static", 4, 255, "void", 0, False, True, (0, 0, 0), None),
    Label("dynamic", 5, 255, "void", 0, False, True, (111, 74, 0), None),
    Label("ground", 6, 255, "void", 0, False, True, (81, 0, 81), None),
    Label("road", 7, 0, "flat", 1, False, False, (128, 64, 128), False),
    Label("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232), False),
    Label("parking", 9, 255, "flat", 1, False, True, (250, 170, 160), None),
    Label("rail track", 10, 255, "flat", 1, False, True, (230, 150, 140), None),
    Label("building", 11, 2, "construction", 2, False, False, (70, 70, 70), False),
    Label("wall", 12, 3, "construction", 2, False, False, (102, 102, 156), False),
    Label("fence", 13, 4, "construction", 2, False, False, (190, 153, 153), False),
    Label("guard rail", 14, 255, "construction", 2, False, True, (180, 165, 180), None),
    Label("bridge", 15, 255, "construction", 2, False, True, (150, 100, 100), None),
    Label("tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90), None),
    Label("pole", 17, 5, "object", 3, False, False, (153, 153, 153), True),
    Label("polegroup", 18, 255, "object", 3, False, True, (153, 153, 153), None),
    Label("traffic light", 19, 6, "object", 3, False, False, (250, 170, 30), True),
    Label("traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0), False),
    Label("vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35), False),
    Label("terrain", 22, 9, "nature", 4, False, False, (152, 251, 152), False),
    Label("sky", 23, 10, "sky", 5, False, False, (70, 130, 180), False),
    Label("person", 24, 11, "human", 6, True, False, (220, 20, 60), True),
    Label("rider", 25, 12, "human", 6, True, False, (255, 0, 0), True),
    Label("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142), True),
    Label("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70), True),
    Label("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100), True),
    Label("caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90), None),
    Label("trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110), None),
    Label("train", 31, 16, "vehicle", 7, True, False, (0, 80, 100), True),
    Label("motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230), True),
    Label("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32), True),
    Label("license plate", -1, -1, "vehicle", 7, False, True, (0, 0, 142), None),
]

PALETTE = [
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
]


class RGBDataset(data.Dataset):
    valid_classes = [label for label in CLASSES if not label.ignoreInEval]

    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super().__init__()
        self._split_name = split_name
        self._rgb_path = (
            setting["rgb_root"]
            if isinstance(setting["rgb_root"], (list, tuple))
            else [setting["rgb_root"]]
        )
        self._rgb_format = (
            setting["rgb_format"]
            if isinstance(setting["rgb_format"], (list, tuple))
            else [setting["rgb_format"]]
        )

        self._gt_path = (
            setting["gt_root"]
            if isinstance(setting["gt_root"], (list, tuple))
            else [setting["gt_root"]]
        )
        self._gt_format = (
            setting["gt_format"]
            if isinstance(setting["gt_format"], (list, tuple))
            else [setting["gt_format"]]
        )
        self._transform_gt = setting["transform_gt"]

        self._train_source = (
            setting["train_source"]
            if isinstance(setting["train_source"], (list, tuple))
            else [setting["train_source"]]
        )
        self._eval_source = (
            setting["eval_source"]
            if isinstance(setting["eval_source"], (list, tuple))
            else [setting["eval_source"]]
        )

        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        if self._file_length is not None:
            rgb_path, gt_path, item_name = self._construct_new_file_names(
                self._file_length
            )[index]
        else:
            rgb_path, gt_path, item_name = self._file_names[index]

        # Check the following settings if necessary
        rgb = self._open_image(rgb_path, cv2.COLOR_BGR2RGB)
        gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=np.uint8)
        if self._transform_gt:
            gt = self._gt_transform(gt)

        if self.preprocess is not None:
            rgb, gt = self.preprocess(rgb, gt)

        if self._split_name == "train":
            rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
        return dict(
            data=rgb,
            label=gt,
            fn=str(item_name),
            n=len(self._file_names),
        )

    def _get_file_names(self, split_name):
        assert split_name in ["train", "val"]
        sources = self._train_source
        if split_name == "val":
            sources = self._eval_source

        rgb_gt_pairs = []
        for rgb_dir, rgb_fmt, gt_dir, gt_fmt, s in zip(
                self._rgb_path, self._rgb_format, self._gt_path, self._gt_format, sources
        ):
            if os.path.isfile(s):
                with open(s) as f:
                    files = f.readlines()
                for item in files:
                    item_name = item.strip()
                    rgb_gt_pairs.append(
                        (
                            os.path.join(rgb_dir, item_name + rgb_fmt),
                            os.path.join(gt_dir, item_name + gt_fmt),
                            item_name,
                        )
                    )
            else:
                rgb_dir = os.path.join(rgb_dir, s)
                gt_dir = os.path.join(gt_dir, s)
                assert os.path.isdir(rgb_dir) and os.path.isdir(gt_dir), (
                    rgb_dir,
                    gt_dir,
                )

                rgb_names = [
                    x[: len(rgb_fmt)]
                    for x in os.listdir(rgb_dir)
                    if x.endswith(rgb_fmt)
                ]
                gt_names = [
                    x[: len(gt_fmt)] for x in os.listdir(gt_dir) if x.endswith(gt_fmt)
                ]
                valid_names = sorted(set(rgb_names).intersection(gt_names))
                rgb_gt_pairs.extend(
                    [
                        (
                            os.path.join(rgb_dir, item_name + rgb_fmt),
                            os.path.join(gt_dir, item_name + gt_fmt),
                            item_name,
                        )
                        for item_name in valid_names
                    ]
                )
        return rgb_gt_pairs

    def _construct_new_file_names(self, length):
        assert isinstance(length, int)
        files_len = len(self._file_names)
        new_file_names = self._file_names * (length // files_len)

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[: length % files_len]

        new_file_names += [self._file_names[i] for i in new_indices]
        return new_file_names

    def get_length(self):
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        if filepath.endswith(".npy"):
            img = np.load(filepath)
            img = img - img.min()
            img = img / img.max()
            img = (img * 255).astype(dtype)
        else:
            img = cv2.imread(filepath, mode)
            assert img is not None, filepath
            img = np.array(img, dtype=dtype)
        return img

    @staticmethod
    def _gt_transform(gt):
        new_gt = np.ones_like(gt) * 255
        for label in CLASSES:
            new_gt[gt == label.id] = label.trainId
        return new_gt

    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return "".join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 41
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors


def random_mirror(rgb, gt):
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)
        gt = cv2.flip(gt, 1)
    return rgb, gt


def random_scale(rgb, gt, scales):
    scale = random.choice(scales)
    sh = int(rgb.shape[0] * scale)
    sw = int(rgb.shape[1] * scale)
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    return rgb, gt, scale


class TrainPre:
    def __init__(
        self, norm_mean, norm_std, train_scale_array, image_height, image_width
    ):
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.train_scale_array = train_scale_array
        self.image_height = image_height
        self.image_width = image_width

    def __call__(self, rgb, gt):
        rgb, gt = random_mirror(rgb, gt)
        if self.train_scale_array is not None:
            rgb, gt, scale = random_scale(rgb, gt, self.train_scale_array)

        rgb = normalize(rgb, self.norm_mean, self.norm_std)

        crop_size = (self.image_height, self.image_width)
        crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)

        p_rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)

        p_rgb = p_rgb.transpose(2, 0, 1)
        return p_rgb, p_gt


def get_train_loader(engine, dataset, config):
    data_setting = {
        "rgb_root": config.rgb_root_folder,
        "rgb_format": config.rgb_format,
        "gt_root": config.gt_root_folder,
        "gt_format": config.gt_format,
        "transform_gt": config.gt_transform,
        "train_source": config.train_source,
        "eval_source": config.eval_source,
    }
    train_preprocess = TrainPre(
        norm_mean=config.norm_mean,
        norm_std=config.norm_std,
        train_scale_array=config.train_scale_array,
        image_height=config.image_height,
        image_width=config.image_width,
    )

    train_dataset = dataset(
        data_setting,
        "train",
        train_preprocess,
        config.batch_size * config.niters_per_epoch,
    )

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=is_shuffle,
        pin_memory=True,
        sampler=train_sampler,
    )

    return train_loader, train_sampler


LOGGER = logging.getLogger(name="main")
LOGGER.setLevel(level=logging.DEBUG)
formatter = logging.Formatter(fmt="[%(filename)s] %(message)s")

with Engine() as engine:
    file_handler = logging.FileHandler(engine.log_file, mode="a")
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG if engine.is_master else logging.WARN)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    LOGGER.info(engine.args.pretty_text)

    cudnn.benchmark = True
    seed = engine.args.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader
    train_loader, train_sampler = get_train_loader(
        engine, RGBDataset, config=engine.args
    )
    tb = SummaryWriter(log_dir=engine.tb_dir)

    # config network and criterion
    model = model_zoo.__dict__[engine.args.model_name](
        mid_dim=engine.args.embed_dim, num_classes=len(RGBDataset.valid_classes)
    )
    loss_func = nn.CrossEntropyLoss(
        reduction="mean", ignore_index=engine.args.background
    )

    # group weight and config optimizer
    base_lr = engine.args.lr
    if engine.distributed:
        base_lr = engine.args.lr

    params_list = group_weight(model, base_lr, mode=engine.args.group_mode)
    if engine.args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params_list,
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=engine.args.weight_decay,
        )
    elif engine.args.optimizer == "SGDM":
        optimizer = torch.optim.SGD(
            params_list,
            lr=base_lr,
            momentum=engine.args.momentum,
            weight_decay=engine.args.weight_decay,
        )
    else:
        raise NotImplementedError
    LOGGER.info(optimizer)

    # config lr policy
    total_iteration = engine.args.nepochs * engine.args.niters_per_epoch
    lr_policy = WarmUpPolyLR(
        lr_power=engine.args.lr_power,
        total_iters=total_iteration,
        warmup_steps=engine.args.niters_per_epoch * engine.args.warm_up_epoch,
    )
    lr_policy.init_base_lr_group(optimizer=optimizer)

    if engine.distributed:
        LOGGER.info(".............distributed training.............")
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(
                model,
                device_ids=[engine.local_rank],
                output_device=engine.local_rank,
                find_unused_parameters=False,
            )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()
    scaler = torch.cuda.amp.GradScaler(enabled=engine.args.use_fp16)

    optimizer.zero_grad()
    model.train()
    LOGGER.info("begin trainning:")
    for epoch in range(
            engine.state.epoch, engine.args.nepochs + 1
    ):  # default start from 1
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = "{desc}[{elapsed}<{remaining},{rate_fmt}]"

        dataloader = iter(train_loader)
        sum_loss = 0
        for idx in range(engine.args.niters_per_epoch):
            engine.update_iteration(epoch, idx)
            current_idx = (epoch - 1) * engine.args.niters_per_epoch + idx
            lrs_group = lr_policy.step(optimizer=optimizer, curr_iter=current_idx)

            minibatch = dataloader.next()
            imgs = minibatch["data"]
            gts = minibatch["label"]

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)

            with torch.cuda.amp.autocast(enabled=engine.args.use_fp16):
                logits = model(imgs)
                loss = loss_func(input=logits, target=gts)

                # reduce the whole loss over multi-gpu
                if engine.distributed:
                    reduce_loss = pt_utils.all_reduce_tensor(
                        loss, world_size=engine.world_size
                    )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            lr_str = ",".join([f"{lr:.4e}" for lr in lrs_group])
            print_str = (
                f"TR @ E{epoch}/{engine.args.nepochs} I{idx + 1}/{engine.args.niters_per_epoch}]"
                f" lr={lr_str}"
                f" {tuple(gts.shape)}-{tuple(imgs.shape)}"
            )
            if engine.distributed:
                sum_loss += reduce_loss.item()
                print_str += f" loss={reduce_loss.item():.4f} total_loss={sum_loss / (idx + 1):.4f}"
            else:
                sum_loss += loss
                print_str += f" loss={loss:.4f} total_loss={sum_loss / (idx + 1):.4f}"

            if engine.is_master:
                if current_idx % 10 == 0 or (
                        idx == 0 or idx == (engine.args.niters_per_epoch - 1)
                ):
                    for i, lr in enumerate(lrs_group):
                        tb.add_scalar(f"lr/lr_{i}", lr, current_idx)
                    LOGGER.info(print_str)

        if engine.is_master:
            tb.add_scalar("train_loss", sum_loss / engine.args.niters_per_epoch, epoch)

        if (
                (epoch >= engine.args.checkpoint_start_epoch)
                and (epoch % engine.args.checkpoint_step == 0)
                or (epoch == engine.args.nepochs)
        ):
            if engine.is_master:
                engine.save_checkpoint(engine.checkpoint_dir)
