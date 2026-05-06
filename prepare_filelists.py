import os
import random
import numpy as np
from pathlib import Path

# Change these paths if necessary.
DRESDEN_ROOT = Path(r'C:\Harsh\gc_dresden_train')
PLACES_ROOT = Path(r'C:\Harsh\gc_places_train')
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR
TRAIN_OUTPUT = OUTPUT_DIR / 'train_filelist.npy'
VAL_OUTPUT = OUTPUT_DIR / 'val_filelist.npy'

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
MASK_SUFFIXES = {'_mask', '_gt', '_ground_truth'}


def normalize_stem(path: Path):
    stem = path.stem.lower()
    for suffix in MASK_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def collect_pairs(root: Path):
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f'Dataset root not found: {root}')

    files = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    inputs = []
    masks = []

    for p in files:
        name = p.stem.lower()
        if any(name.endswith(suffix) for suffix in MASK_SUFFIXES) or 'mask' in p.parts or 'gt' in p.parts:
            masks.append(p)
        else:
            inputs.append(p)

    input_map = {normalize_stem(p): p for p in inputs}
    mask_map = {normalize_stem(p): p for p in masks}

    pairs = []
    for stem, img_path in input_map.items():
        mask_path = mask_map.get(stem)
        if mask_path is not None:
            pairs.append([str(img_path), str(mask_path)])
        else:
            print(f'WARNING: no matching mask for {img_path}')

    if not pairs:
        raise ValueError(f'No paired images found in {root}. Make sure your dataset contains matching input and mask files with the same basename.')

    return pairs


def main():
    dresden_pairs = collect_pairs(DRESDEN_ROOT)
    places_pairs = collect_pairs(PLACES_ROOT)

    all_pairs = dresden_pairs + places_pairs
    random.shuffle(all_pairs)

    if len(all_pairs) < 1000:
        raise ValueError('Combined dataset contains fewer than 1000 pairs. Check your dataset folder structure.')

    val_size = min(1000, max(1, len(all_pairs) // 50))
    train_pairs = all_pairs[val_size:]
    val_pairs = all_pairs[:val_size]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(TRAIN_OUTPUT, np.asarray(train_pairs, dtype=str))
    np.save(VAL_OUTPUT, np.asarray(val_pairs, dtype=str))

    print(f'Found {len(all_pairs)} total pairs.')
    print(f'Saved {len(train_pairs)} training pairs to: {TRAIN_OUTPUT}')
    print(f'Saved {len(val_pairs)} validation pairs to: {VAL_OUTPUT}')


if __name__ == '__main__':
    main()
