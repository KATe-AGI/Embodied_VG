# 数据集转换与合并脚本说明

项目提供 YOLO visible-plug segmentation 和夹爪抓取状态检测两类数据集工具。下方的分割转换与合并工具生成：

- `yolo_train/seg`
- `yolo_train/meta`
- 可选 `rgbd_test`

检测数据集的格式和转换命令见文末。

## 目标结构

```text
plug_dataset_xxx/
  yolo_train/
    seg/
      plug_seg.yaml
      images/train/
      images/val/
      labels/train/
      labels/val/
    annotations_standard/
    meta/
      dataset_info.json
      split.csv
      label_summary.csv
      frame_manifest.csv

  rgbd_test/
    color/
      color_*.png
    D2RGB/
      D2RGB_*.png
      D2RGB_*.jpg
      *_d2rgb.npy
    meta/
      frame_manifest.csv
```

## 转换

```bash
python tools/convert_plug_dataset.py \
  --camera-dir plug_camera_YYYYMMDD \
  --annotation-dir plug_annotation_YYYYMMDD \
  --rgbd-test-dir plug_test_YYYYMMDD \
  --output plug_dataset_YYYYMMDD \
  --train-ratio 0.8 \
  --force
```

LabelMe 标注只要求：

```text
plug_grasp_region  polygon
```

该 polygon 表示可见插头区域，用于训练新主线的 YOLO-seg 模型。

## 合并

```bash
python tools/merge_plug_datasets.py \
  --inputs plug_dataset plug_dataset_YYYYMMDD \
  --output plug_dataset_all_YYYYMMDD \
  --parts all \
  --force
```

合并时会重新写入 `yolo_train/seg/plug_seg.yaml`，并把 `path` 指向新的输出目录。

## 夹爪抓取状态检测数据集

类别固定为 `0: gripper_with_plug`、`1: gripper_without_plug`，输入标注使用 LabelMe `rectangle`，输出标签为 YOLO 的 `class x_center y_center width height`，坐标归一化到图像尺寸。

单目录 v01 数据的历史转换工具：

```bash
python tools/convert_v01_detection.py --source 'v01_成功1' --output 'v01_成功1_yolo'
```

该工具按文件名哈希划分数据，训练占比约 80%，要求每张图像恰好有一个检测框。

包含多组已标注数据时使用：

```bash
python tools/convert_gripper_detection.py \
  --source '检测标注（已标注）' \
  --output '检测标注（已标注）_yolo' \
  --seed 42
```

该工具递归扫描 JSON，仅复制同目录下具有对应标注的图像，未标注图像不纳入输出。支持多个检测框；空 JSON 标注生成空标签文件并保留为背景样本。按类别分层分配训练/验证数量，总训练数量取图像总数的 80% 四舍五入。源图像和标注保留，输出目录必须不存在。

输出格式：

```text
检测标注（已标注）_yolo/
  data.yaml
  images/train/
  images/val/
  labels/train/
  labels/val/
  conversion_summary.json
  split_manifest.json
  README.txt
```

所有数据集目录均为本地资产，不上传 GitHub。训练命令和增强配置见项目 README 的“检测数据集与训练”。
