#!/usr/bin/env python3
"""Convert nested LabelMe gripper rectangles with a reproducible 80/20 split."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import shutil

from PIL import Image

from convert_v01_detection import CLASS_NAMES, yolo_line


def convert(source: Path, output: Path, seed: int = 42) -> dict:
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    records = []
    images = {
        p for p in source.rglob('*')
        if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    }
    groups = defaultdict(list)
    names = set()
    for annotation in sorted(source.rglob('*.json')):
        data = json.loads(annotation.read_text(encoding='utf-8'))
        candidates = [p for p in images if p.parent == annotation.parent and p.stem == annotation.stem]
        if len(candidates) != 1:
            raise ValueError(f"Expected one matching image for {annotation}, got {len(candidates)}")
        image = candidates[0]
        with Image.open(image) as decoded:
            width, height = decoded.size
        if (data.get('imageWidth'), data.get('imageHeight')) != (width, height):
            raise ValueError(f"Annotation dimensions differ from image: {annotation}")
        lines = []
        classes = []
        for shape in data.get('shapes') or []:
            label = str(shape.get('label'))
            if label not in CLASS_NAMES or shape.get('shape_type') != 'rectangle':
                raise ValueError(f"Unsupported detection shape in {annotation}: {shape}")
            lines.append(yolo_line(label, shape.get('points') or [], width, height))
            classes.append(label)
        if image.stem in names:
            raise ValueError(f"Duplicate output label name: {image.stem}")
        names.add(image.stem)
        record = {'image': image, 'annotation': annotation, 'lines': lines, 'classes': classes}
        records.append(record)
        groups[tuple(sorted(set(classes)))].append(record)
    if not records:
        raise ValueError('No annotations found')

    # Largest remainder allocation keeps the total split at the nearest 80/20
    # ratio while preserving the distribution of labels, including backgrounds.
    target_train = round(len(records) * 0.8)
    quotas = {key: math.floor(len(group) * 0.8) for key, group in groups.items()}
    remaining = target_train - sum(quotas.values())
    order = sorted(groups, key=lambda key: (-(len(groups[key]) * 0.8 - quotas[key]), key))
    for key in order[:remaining]:
        quotas[key] += 1
    rng = random.Random(seed)
    for key in sorted(groups):
        rng.shuffle(groups[key])
        for index, record in enumerate(groups[key]):
            record['split'] = 'train' if index < quotas[key] else 'val'

    counts = {split: Counter() for split in ('train', 'val')}
    manifest = []
    for split in counts:
        (output / 'images' / split).mkdir(parents=True)
        (output / 'labels' / split).mkdir(parents=True)
    for record in records:
        split, image = record['split'], record['image']
        shutil.copy2(image, output / 'images' / split / image.name)
        label_path = output / 'labels' / split / f'{image.stem}.txt'
        label_path.write_text('\n'.join(record['lines']) + ('\n' if record['lines'] else ''), encoding='utf-8')
        counts[split]['images'] += 1
        counts[split]['background_images'] += int(not record['lines'])
        counts[split].update(f'class_{label}' for label in record['classes'])
        manifest.append({'image': str(image.relative_to(source)),
                         'annotation': str(record['annotation'].relative_to(source)), 'split': split})
    (output / 'data.yaml').write_text(
        'path: .\ntrain: images/train\nval: images/val\n\nnames:\n'
        '  0: gripper_with_plug\n  1: gripper_without_plug\n', encoding='utf-8')
    summary = {'source': str(source), 'seed': seed, 'train_ratio': 0.8,
               'images': len(records), 'excluded_unannotated_images': len(images) - len(records),
               'splits': {split: dict(count) for split, count in counts.items()}}
    (output / 'conversion_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'split_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'README.txt').write_text(
        'YOLO detection dataset from nested LabelMe rectangles.\n'
        'Class 0: gripper_with_plug; class 1: gripper_without_plug.\n'
        'Only images with JSON annotations are included. Empty annotations are background samples.\n'
        f'Stratified 80/20 split, seed {seed}. See conversion_summary.json and split_manifest.json.\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('检测标注（已标注）'))
    parser.add_argument('--output', type=Path, default=Path('检测标注（已标注）_yolo'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.output, args.seed), ensure_ascii=False, indent=2))
