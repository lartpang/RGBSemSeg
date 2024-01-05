import numpy as np

seed = 12345

"""Dataset Path"""
output_root = f"outputs"

rgb_root_folder = [
    "datasets/CityScapes/leftImg8bit_trainvaltest/leftImg8bit",
    "datasets/KITTI_SemanticSegmentation/training/image_2",
]
rgb_format = [
    ".png",
    ".png",
]

gt_root_folder = [
    "datasets/CityScapes/gtFine_trainvaltest/gtFine",
    "datasets/KITTI_SemanticSegmentation/training/semantic",
]
gt_format = [
    "_gtFine_labelIds.png",
    ".png",
]
gt_transform = True

train_source = [
    "train",
    "datasets/KITTI_SemanticSegmentation/training/train.txt",
]
eval_source = []

num_train_imgs = 2975 + 200
num_eval_imgs = 0

"""Image Config"""
background = 255
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

"""Train Config"""
image_height = 352
image_width = 352
model_name = "DualRGBGADFormerSwinB_22K_384"  # Remember change the path below.
embed_dim = 64
optimizer = "AdamW"

batch_size = 8
lr = 0.00001
weight_decay = 0.0001
group_mode = "std"
use_fp16 = True
nepochs = 400

scheduler = "warmup_poly"
# warmup_poly
warm_up_epoch = 10
lr_power = 0.9
# multistep
sche_milestones = [250, 400]  # epoch
sche_gamma = 0.1

train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
momentum = 0.9
niters_per_epoch = num_train_imgs // batch_size * (800 // nepochs)
num_workers = 4

fix_bias = True
bn_eps = 1e-3
bn_momentum = 0.1

"""Eval Config"""
eval_crop_size = [352, 352]  # [height weight]
eval_stride_rate = 2 / 3
eval_flip = False
eval_scale_array = [1.5, 2.0, 2.5]

"""Store Config"""
checkpoint_start_epoch = 250
checkpoint_step = 25
