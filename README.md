# 基于聚合扩散注意力机制的跨媒体互补变换器场景解析算法

## 模型结构图

![image](https://github.com/lartpang/RGBSemSeg/assets/26847524/3f5a4a42-fc90-4dce-95cf-0f69a26ade8c)


## 使用方法

### 安装

系统环境与软件版本

- OS: Ubuntu 18.04 LTS
- CUDA: 10.2
- PyTorch 1.12.1
- Python 3.8.13

安装依赖项：`pip install -r requirements.txt`

### 数据集

KITTI & CityScapes Semantic Segmentation 数据集。

### 模型训练

1. 添加数据文件夹的软链接：`ln -s <存放KITTI和CityScapes数据集的文件夹> datasets` 
2. 调整训练配置文件`config.py`中的数据集路径。
3. 在两块GPU上运行分布式训练，并将训练结果保存至`outputs`文件夹中：

```shell
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 torchrun \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:20746 --nnodes=1 --nproc_per_node=2 \
  train.py config.py --model-name DualRGBGADFormerSwinB_22K_384 --output-root ./outputs
```

### 模型评估

在单块GPU环境中执行如下指令，使用当前的`epoch-400.pth`的权重文件进行推理预测：

```shell
python predict.py config.py \
  --load-from outputs/model/checkpoints/epoch-400.pth \
  --model-name DualRGBGADFormerSwinB_22K_384 \
  --image-root "<image root of testing set>" \
  --image-format "<image format of testing set>" \
  --image-source "<image name list of testing set>" \
  --save-path outputs/test
```
