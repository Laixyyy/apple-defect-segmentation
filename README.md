# 苹果缺陷实例分割实验

本仓库保存基于 YOLOv8-seg 的苹果与苹果缺陷实例分割实验代码、模型权重、配置文件、独立测试指标和实验报告。

## 实验模型与结果

所有模型均以 **160×160** 图像作为输入，FLOPs 均由保存的 `best.pt` 在 `imgsz=160` 下统计。

| 模型 | 结构变化 | 参数量 | FLOPs | 缺陷 Mask mAP50 |
| --- | --- | ---: | ---: | ---: |
| 原生 YOLOv8n-seg | 标准 YOLOv8n-seg | 326 万 | 0.717 G | 0.549 |
| MobileNetV2 骨干 | 替换骨干，保留 YOLO 分割头 | 537 万 | 2.936 G | 0.482 |
| 通道降低四分之一 | YOLOv8n-seg 宽度为 0.1875 | 198 万 | 0.450 G | 0.437 |
| 通道降低一半 | YOLOv8n-seg 宽度为 0.125 | 103 万 | 0.245 G | 0.407 |

通道表示卷积层提取特征的数量。减少通道可降低 FLOPs，但可能损失细小缺陷的表达能力。

## 目录说明

```text
├── cloud_train/      # 训练、测试脚本及模型结构 YAML
├── data/             # 数据集配置模板；图像和标注不上传
├── metrics/          # 四组实验的独立测试指标
├── models/           # 预训练权重和四个 best.pt
├── reports/          # 实验报告
├── scripts/          # 数据下载、转换和检查脚本
└── run_*.ps1         # Windows 本地训练启动脚本
```

## 模型文件

`models/` 中包含原始 `yolov8n-seg.pt`、MobileNetV2 预训练权重，以及四个正式对比实验的最佳权重。全部单文件均小于 GitHub 100 MB 限制。

## 数据集准备

数据集图像与标注不提交到 GitHub。请从 [Apple Quality V4i COCO Segmentation](https://platform.ultralytics.com/jason-zarrilli/datasets/apple-qualityv4icoco-segmentation) 下载，并使用 `scripts/prepare_apple_yolo_seg.py` 转为 YOLO 分割格式。

准备后的目录应为：

```text
data/
  apple_quality_160_yolo_seg/
    data.yaml
    train/images/  train/labels/
    val/images/    val/labels/
    test/images/   test/labels/
```

将 `data/data.yaml` 复制到 `data/apple_quality_160_yolo_seg/data.yaml`。本实验采用训练集 1938 张、验证集 185 张、独立测试集 93 张；类别为 `apple` 和 `apple-defect`。

## 安装环境

先根据自己的 GPU 安装匹配的 CUDA 版 PyTorch，再安装其余依赖：

```powershell
python -m pip install -r cloud_train/requirements.txt
```

检查 GPU 是否可用：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 复现实验

在仓库根目录的 PowerShell 中执行对应脚本。训练结果会保存到 `local_results/`，脚本会拒绝覆盖已有同名实验。

```powershell
.\\run_yolov8n_original_local.ps1
.\\run_mobilenetv2_local.ps1
.\\run_yolov8n_three_quarter_channels_local.ps1
.\\run_yolov8n_half_channels_local.ps1
```

`metrics/` 中保存了报告使用的独立测试结果。注意：不同输入尺寸的 FLOPs 不可直接比较，例如 640×640 下的 FLOPs 不能与本仓库的 160×160 结果混用。
