# 数据集转换与合并脚本说明

本文档说明两个独立脚本的用途和用法：

- `convert_plug_dataset.py`：把一次相机采集数据和 LabelMe 标注转换成项目数据集结构。
- `merge_plug_datasets.py`：把多个同结构数据集合并成一个新数据集。

两个脚本都不会修改训练、验证、推理脚本本身。

## 目标数据集结构

脚本生成或合并的数据集遵循当前 `plug_dataset` 的结构：

```text
plug_dataset_xxx/
  yolo_train/
    meta/
      dataset_info.json
      split.csv
      label_summary.csv
      frame_manifest.csv
    annotations_standard/
      color_*.standard.json
    seg/
      plug_seg.yaml
      images/train/
      images/val/
      labels/train/
      labels/val/
    pose/
      plug_pose.yaml
      images/train/
      images/val/
      labels/train/
      labels/val/

  rgbd_test/
    color/
      color_*.png
    D2RGB/
      D2RGB_*.png
      D2RGB_*.jpg
    meta/
      frame_manifest.csv
```

`yolo_train` 用于 YOLO 分割和关键点训练；`rgbd_test` 用于后续 6D 位姿推理测试。

## convert_plug_dataset.py

### 功能

`convert_plug_dataset.py` 将：

- 相机采集目录中的 `color_*.png` 作为图像来源。
- LabelMe 标注目录中的 `color_*.json` 作为标注来源。
- 可选测试采集目录中的 `color_*.png`、`D2RGB_*.png`、`D2RGB_*.jpg` 作为 `rgbd_test` 来源。

生成：

- YOLO segmentation 数据集：`yolo_train/seg`
- YOLO pose 数据集：`yolo_train/pose`
- 标准中间标注：`yolo_train/annotations_standard`
- 元数据表：`yolo_train/meta`
- 可选 RGB-D 测试集：`rgbd_test`

### 默认命令

```bash
python convert_plug_dataset.py --force
```

默认输入输出为：

```text
camera:      plug_camera_20260520
annotation:  plug_annotation_20260520
output:      plug_dataset_20260520
```

默认不生成 `rgbd_test`。

### 生成带测试集的数据集

如果有单独采集的测试数据，例如 `plug_test`：

```bash
python convert_plug_dataset.py --force --rgbd-test-dir plug_test
```

此时脚本只会把测试目录里的：

```text
color_*.png
D2RGB_*.png
D2RGB_*.jpg
```

复制到：

```text
plug_dataset_20260520/rgbd_test/color/
plug_dataset_20260520/rgbd_test/D2RGB/
```

不会把 `depth`、`leftIR`、`Point3D`、`Color3D` 等其他相机原始文件放入 `rgbd_test`。

### 常用参数

```bash
python convert_plug_dataset.py \
  --camera-dir plug_camera_20260520 \
  --annotation-dir plug_annotation_20260520 \
  --rgbd-test-dir plug_test \
  --output plug_dataset_20260520 \
  --train-ratio 0.8 \
  --force
```

参数含义：

- `--camera-dir`：训练数据对应的相机采集目录，要求是扁平文件结构。
- `--annotation-dir`：LabelMe 标注目录，可以包含一个或多个子目录。
- `--rgbd-test-dir`：可选，6D 测试数据对应的相机采集目录。
- `--output`：生成的数据集目录。
- `--train-ratio`：训练集比例，默认 `0.8`。
- `--force`：如果输出目录已存在，先删除再重新生成。

### 划分规则

训练/验证集使用稳定的 MD5 split：

```text
md5(raw_id) / 2^128 < train_ratio  -> train
otherwise                          -> val
```

因此同一批样本、同一个 `train-ratio` 下，每次生成的划分一致。

### 标注要求

LabelMe JSON 中使用以下标签：

```text
plug_grasp_region  polygon
plug_head          point
plug_tail          point
```

规则：

- 有 `plug_grasp_region` 的样本会进入 segmentation 数据集。
- 同时有 `plug_grasp_region`、`plug_head`、`plug_tail` 的样本会进入 pose 数据集。
- 缺少关键点的样本不会进入 pose，但会在 `label_summary.csv` 中记录 `pose_valid=False` 和 `label_error`。

## merge_plug_datasets.py

### 功能

`merge_plug_datasets.py` 用于合并多个同结构数据集，例如：

```text
plug_dataset
plug_dataset_20260520
```

生成：

```text
plug_dataset_all_20260520
```

合并时会保持原有结构，并更新合并后 YOLO YAML 中的 `path`。

### 默认命令

```bash
python merge_plug_datasets.py --force
```

默认等价于：

```bash
python merge_plug_datasets.py \
  --inputs plug_dataset plug_dataset_20260520 \
  --output plug_dataset_all_20260520 \
  --parts all \
  --force
```

### 只合并指定部分

只合并训练数据：

```bash
python merge_plug_datasets.py --force --parts yolo_train
```

只合并测试数据：

```bash
python merge_plug_datasets.py --force --parts rgbd_test
```

同时合并两部分：

```bash
python merge_plug_datasets.py --force --parts yolo_train rgbd_test
```

### 自定义输入输出

```bash
python merge_plug_datasets.py \
  --inputs plug_dataset plug_dataset_20260520 another_dataset \
  --output plug_dataset_all_20260520 \
  --parts all \
  --force
```

### 合并策略

文件合并：

- 相同相对路径且内容一致：跳过并计入 `duplicates_skipped`。
- 相同相对路径但内容不同：直接报错，避免静默覆盖。
- YOLO `.cache` 文件不会被合并。

CSV 合并：

- 按 `raw_id` 去重。
- 同一 `raw_id` 内容完全一致：跳过。
- 同一 `raw_id` 内容不一致：报错。

YAML 处理：

- 不直接沿用输入数据集中的 `plug_seg.yaml` 和 `plug_pose.yaml`。
- 会为输出数据集重新写入 YAML，并把 `path` 指向新的输出目录。

### 输出汇总

脚本运行结束会打印合并情况，例如：

```text
[yolo_train]
  standard_annotations: ...
  seg_train_images: ...
  seg_val_images: ...
  pose_train_images: ...
  pose_val_images: ...
  split_rows: ...

[rgbd_test]
  color_images: ...
  d2rgb_png: ...
  d2rgb_jpg: ...
  manifest_rows: ...
  with_d2rgb: ...
  missing_d2rgb: ...
```

合并后的汇总也会写入：

```text
plug_dataset_all_20260520/yolo_train/meta/dataset_info.json
```

## 推荐工作流

一次新的采集和标注完成后：

```bash
python convert_plug_dataset.py \
  --camera-dir plug_camera_YYYYMMDD \
  --annotation-dir plug_annotation_YYYYMMDD \
  --rgbd-test-dir plug_test_YYYYMMDD \
  --output plug_dataset_YYYYMMDD \
  --train-ratio 0.8 \
  --force
```

然后与已有数据集合并：

```bash
python merge_plug_datasets.py \
  --inputs plug_dataset plug_dataset_YYYYMMDD \
  --output plug_dataset_all_YYYYMMDD \
  --parts all \
  --force
```

如果只想迭代训练集，不合并测试集：

```bash
python merge_plug_datasets.py \
  --inputs plug_dataset plug_dataset_YYYYMMDD \
  --output plug_dataset_all_YYYYMMDD \
  --parts yolo_train \
  --force
```
