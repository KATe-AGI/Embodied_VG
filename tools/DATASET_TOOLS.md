# 数据集转换与合并脚本说明

当前项目只训练 YOLO visible-plug segmentation。数据集工具只生成或合并：

- `yolo_train/seg`
- `yolo_train/meta`
- 可选 `rgbd_test`

不会生成其他任务的数据。

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
