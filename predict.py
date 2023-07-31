import argparse
import logging
import math
import os
import random

import cv2
import numpy as np
import torch
from mmengine import Config
from timm.models.layers import to_2tuple
from torch.utils import data
from tqdm import tqdm

import models as model_zoo
from constant import CLASSES
from utils import pt_utils
from utils.pt_utils import ensure_dir, load_model
from utils.py_utils import resize
from utils.transforms import normalize, pad_image_to_shape

LOGGER = logging.getLogger("main")


class TestDataset(data.Dataset):
    valid_classes = [label for label in CLASSES if not label.ignoreInEval]

    def __init__(self, rgb_root, format, source, preprocess=None):
        super().__init__()
        self._path = rgb_root if isinstance(rgb_root, (list, tuple)) else [rgb_root]
        self._format = format if isinstance(format, (list, tuple)) else [format]
        self._source = source if isinstance(source, (list, tuple)) else [source]

        self._file_names = self._get_file_names()
        self.preprocess = preprocess

    def __len__(self):
        return len(self._file_names)

    def __getitem__(self, index):
        rgb_path, item_name = self._file_names[index]
        rgb = self._open_image(rgb_path, cv2.COLOR_BGR2RGB)
        return dict(data=rgb, fn=str(item_name), n=len(self._file_names))

    def _get_file_names(self):
        samples = []
        for rgb_dir, rgb_fmt, s in zip(self._path, self._format, self._source):
            if os.path.isfile(s):
                with open(s) as f:
                    files = f.readlines()
                for item in files:
                    item_name = item.strip()
                    samples.append(
                        (os.path.join(rgb_dir, item_name + rgb_fmt), item_name)
                    )
            else:
                rgb_dir = os.path.join(rgb_dir, s)
                assert os.path.isdir(rgb_dir), rgb_dir

                rgb_names = [
                    x[: len(rgb_fmt)]
                    for x in os.listdir(rgb_dir)
                    if x.endswith(rgb_fmt)
                ]
                valid_names = sorted(rgb_names)
                samples.extend(
                    [
                        (os.path.join(rgb_dir, item_name + rgb_fmt), item_name)
                        for item_name in valid_names
                    ]
                )
        return samples

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


class Evaluator:
    def __init__(
        self,
        dataset,
        num_classes,
        norm_mean,
        norm_std,
        network,
        multi_scales,
        is_flip,
        save_path=None,
    ):
        self.eval_time = 0
        self.dataset = dataset
        self.ndata = len(self.dataset)
        self.num_classes = num_classes
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.multi_scales = multi_scales
        self.is_flip = is_flip
        self.network = network

        self.val_func = None

        self.save_path = save_path
        if save_path is not None:
            ensure_dir(save_path)
        random.seed(2023)

    def run(self, model_path):
        assert os.path.isfile(model_path)
        LOGGER.info(f"Loading From: {model_path}")
        self.val_func = load_model(self.network, model_path)

        LOGGER.info(f"Handling {self.ndata} data.")
        for idx in tqdm(range(self.ndata)):
            dd = self.dataset[idx]
            self.func_per_iteration(dd)

    def process_image(self, img, crop_size=None):
        p_img = img

        if img.shape[2] < 3:
            im_b = p_img
            im_g = p_img
            im_r = p_img
            p_img = np.concatenate((im_b, im_g, im_r), axis=2)

        p_img = normalize(p_img, self.norm_mean, self.norm_std)

        if crop_size is not None:
            p_img, margin = pad_image_to_shape(
                p_img, crop_size, cv2.BORDER_CONSTANT, value=0
            )
            p_img = p_img.transpose(2, 0, 1)

            return p_img, margin

        p_img = p_img.transpose(2, 0, 1)

        return p_img

    def sliding_eval(self, img, crop_size, stride_rate):
        crop_size = to_2tuple(crop_size)
        ori_hw = img.shape[:2]
        processed_pred = np.zeros((*ori_hw, self.num_classes))

        for s in self.multi_scales:
            new_h = math.ceil(s * ori_hw[0]) // 32 * 32
            new_w = math.ceil(s * ori_hw[1]) // 32 * 32
            img_scale = resize(img, new_h, new_w, interpolation=cv2.INTER_LINEAR)
            new_h, new_w, _ = img_scale.shape
            processed_pred += self.scale_process(
                img_scale, ori_hw, crop_size, stride_rate
            )

        pred = processed_pred.argmax(2)

        return pred

    def scale_process(self, img, ori_shape, crop_size, stride_rate):
        new_h, new_w, c = img.shape

        if new_w <= crop_size[1] and new_h <= crop_size[0]:
            input_data, margin = self.process_image(img, crop_size)
            score = self.val_func_process(input_data)
            score = score[
                    :,
                    margin[0]: (score.shape[1] - margin[1]),
                    margin[2]: (score.shape[2] - margin[3]),
                    ]
        else:
            stride = (
                int(np.ceil(crop_size[0] * stride_rate)),
                int(np.ceil(crop_size[1] * stride_rate)),
            )
            img_pad, margin = pad_image_to_shape(
                img, crop_size, cv2.BORDER_CONSTANT, value=0
            )

            pad_rows = img_pad.shape[0]
            pad_cols = img_pad.shape[1]
            r_grid = int(np.ceil((pad_rows - crop_size[0]) / stride[0])) + 1
            c_grid = int(np.ceil((pad_cols - crop_size[1]) / stride[1])) + 1
            data_scale = torch.zeros(self.num_classes, pad_rows, pad_cols).cuda()

            for grid_yidx in range(r_grid):
                for grid_xidx in range(c_grid):
                    s_x = grid_xidx * stride[0]
                    s_y = grid_yidx * stride[1]
                    e_x = min(s_x + crop_size[0], pad_cols)
                    e_y = min(s_y + crop_size[1], pad_rows)
                    s_x = max(e_x - crop_size[0], 0)
                    s_y = max(e_y - crop_size[1], 0)
                    img_sub = img_pad[s_y:e_y, s_x:e_x, :]
                    # print(s_y, s_x, e_y, e_x)

                    input_data, tmargin = self.process_image(img_sub, crop_size)
                    temp_score = self.val_func_process(input_data)

                    temp_score = temp_score[
                                 :,
                                 tmargin[0]: (temp_score.shape[1] - tmargin[1]),
                                 tmargin[2]: (temp_score.shape[2] - tmargin[3]),
                                 ]
                    data_scale[:, s_y:e_y, s_x:e_x] += temp_score
            score = data_scale
            score = score[
                    :,
                    margin[0]: (score.shape[1] - margin[1]),
                    margin[2]: (score.shape[2] - margin[3]),
                    ]

        score = score.permute(1, 2, 0)
        data_output = cv2.resize(
            score.cpu().numpy(),
            (ori_shape[1], ori_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        return data_output

    def val_func_process(self, input_data):
        input_data = np.ascontiguousarray(input_data[None, :, :, :], dtype=np.float32)
        input_data = torch.FloatTensor(input_data).cuda()

        with torch.cuda.device(input_data.get_device()):
            self.val_func.eval()
            self.val_func.to(input_data.get_device())
            with torch.no_grad():
                score = self.val_func(input_data)
                score = score[0]

                if self.is_flip:
                    input_data = input_data.flip(-1)
                    score_flip = self.val_func(input_data)
                    score_flip = score_flip[0]
                    score += score_flip.flip(-1)
                # score = torch.exp(score)
                # score = score.data
        return score

    def func_per_iteration(self, data):
        img = data["data"]
        name = data["fn"]
        pred = self.sliding_eval(img, cfg.eval_crop_size, cfg.eval_stride_rate)

        color_root = os.path.join(self.save_path, "class_color")
        index_root = os.path.join(self.save_path, "class_index")
        pt_utils.ensure_dir(color_root)
        pt_utils.ensure_dir(index_root)
        h, w = pred.shape

        color_map = np.zeros((h, w, 3), dtype=np.uint8)
        index_map = np.zeros((h, w), dtype=np.uint8)
        for label in TestDataset.valid_classes:
            color_map[pred == label.trainId] = label.color[::-1]  # RGB->BGR
            index_map[pred == label.trainId] = label.id
        cv2.imwrite(os.path.join(color_root, name + ".png"), color_map)
        cv2.imwrite(os.path.join(index_root, name + ".png"), index_map)


def get_args():
    global parser
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--model-name", required=True, type=str)
    parser.add_argument("--load-from", required=True, type=str)
    parser.add_argument("--image-root", required=True, type=str)
    parser.add_argument("--image-format", required=True, type=str)
    parser.add_argument("--image-source", required=True, type=str)
    parser.add_argument("--save-path", required=True, type=str)
    args = parser.parse_args()

    cfg = Config().fromfile(args.config)
    cfg.merge_from_dict(vars(args))
    return cfg


if __name__ == "__main__":
    cfg = get_args()
    LOGGER = logging.getLogger(name="main")
    LOGGER.setLevel(level=logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(logging.Formatter(fmt="[%(filename)s] %(message)s"))
    LOGGER.addHandler(stream_handler)
    LOGGER.info(cfg.pretty_text)

    network = model_zoo.__dict__[cfg.model_name](
        mid_dim=cfg.embed_dim,
        num_classes=len(TestDataset.valid_classes),
        pretrained=False,
    )
    dataset = TestDataset(
        rgb_root=cfg.image_root,
        format=cfg.image_format,
        source=cfg.image_source,
    )

    with torch.no_grad():
        segmentor = Evaluator(
            dataset=dataset,
            num_classes=len(TestDataset.valid_classes),
            norm_mean=cfg.norm_mean,
            norm_std=cfg.norm_std,
            network=network,
            multi_scales=cfg.eval_scale_array,
            is_flip=cfg.eval_flip,
            save_path=cfg.save_path,
        )
        segmentor.run(cfg.load_from)
