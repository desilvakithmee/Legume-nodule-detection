# Creates a tiled YOLO-detect dataset from an existing dataset:
#   data/images/{train,val} + data/labels/{train,val}
# Output:
#   data_tiled/images/{train,val} + data_tiled/labels/{train,val}
#
# Assumes YOLO bbox labels: class x_center y_center width height 

import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# --------------------------
# CONFIG
# --------------------------
DATA_ROOT = Path(r"C:/Users/11579/Desktop/HyperspectralData/EnhancedData/08212026_Enhanced/data")
OUT_ROOT  = DATA_ROOT.parent / "data_tiled"   

TILE = 720 #higher resolution HSI for 720, low resolution HSI for 320, phone camera for 1024
OVERLAP = 0.20             # 20% overlap
KEEP_PARTIAL = 0.25        # keep box if at least 25% of its area remains in tile
IMG_EXTS = [".jpg", ".jpeg", ".png"]
PREFER_EXT_ORDER = [".jpg", ".jpeg", ".png"]  # prefer jpg if available
# --------------------------


def read_yolo_labels(label_path: Path):
    if not label_path.exists():
        return []
    txt = label_path.read_text().strip()
    if txt == "":
        return []
    out = []
    for line in txt.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:5])
        out.append((cls, x, y, w, h))
    return out


def yolo_to_xyxy(cls, x, y, w, h, W, H):
    x1 = (x - w / 2) * W
    y1 = (y - h / 2) * H
    x2 = (x + w / 2) * W
    y2 = (y + h / 2) * H
    return cls, x1, y1, x2, y2


def xyxy_to_yolo(cls, x1, y1, x2, y2, tile_w, tile_h):
    x1 = float(np.clip(x1, 0, tile_w))
    y1 = float(np.clip(y1, 0, tile_h))
    x2 = float(np.clip(x2, 0, tile_w))
    y2 = float(np.clip(y2, 0, tile_h))

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1 or bh <= 1:
        return None

    xc = (x1 + x2) / 2 / tile_w
    yc = (y1 + y2) / 2 / tile_h
    bw = bw / tile_w
    bh = bh / tile_h
    return (cls, xc, yc, bw, bh)


def tile_coords(W, H, tile, overlap):
    stride = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(W - tile, 0) + 1, stride))
    ys = list(range(0, max(H - tile, 0) + 1, stride))
    if not xs:
        xs = [0]
    if not ys:
        ys = [0]
    if xs[-1] != max(W - tile, 0):
        xs.append(max(W - tile, 0))
    if ys[-1] != max(H - tile, 0):
        ys.append(max(H - tile, 0))
    return xs, ys


def pick_one_image_per_stem(img_dir: Path, lbl_dir: Path):
    lbl_stems = {p.stem for p in lbl_dir.glob("*.txt")}

    images_by_stem = defaultdict(list)
    for ext in IMG_EXTS:
        images_by_stem.update({})  # no-op, keeps defaultdict explicit
        for p in img_dir.glob(f"*{ext}"):
            images_by_stem[p.stem].append(p)
        for p in img_dir.glob(f"*{ext.upper()}"):
            images_by_stem[p.stem].append(p)

    chosen = []
    for stem, paths in images_by_stem.items():
        if stem not in lbl_stems:
            continue  # skip unlabeled images for training tiling
        paths_sorted = sorted(
            paths,
            key=lambda p: PREFER_EXT_ORDER.index(p.suffix.lower())
            if p.suffix.lower() in PREFER_EXT_ORDER else 999
        )
        chosen.append(paths_sorted[0])

    return sorted(chosen)


def process_split(split: str):
    img_in = DATA_ROOT / "images" / split
    lbl_in = DATA_ROOT / "labels" / split

    img_out = OUT_ROOT / "images" / split
    lbl_out = OUT_ROOT / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    images = pick_one_image_per_stem(img_in, lbl_in)
    print(f"[{split}] images found (dedup + labeled-only): {len(images)}")

    tile_count = 0
    empty_tiles = 0

    for img_path in images:
        stem = img_path.stem
        label_path = lbl_in / f"{stem}.txt"
        labels = read_yolo_labels(label_path)

        img = cv2.imread(str(img_path))
        if img is None:
            print("WARNING: cannot read image:", img_path)
            continue

        H, W = img.shape[:2]

        # Convert all boxes to absolute xyxy in full-image coords
        boxes_full = [yolo_to_xyxy(cls, x, y, w, h, W, H) for (cls, x, y, w, h) in labels]

        xs, ys = tile_coords(W, H, TILE, OVERLAP)

        for y0 in ys:
            for x0 in xs:
                tile_img = img[y0:y0 + TILE, x0:x0 + TILE].copy()
                th, tw = tile_img.shape[:2]

                tile_labels = []
                for (cls, x1, y1, x2, y2) in boxes_full:
                    ix1 = max(x1, x0)
                    iy1 = max(y1, y0)
                    ix2 = min(x2, x0 + tw)
                    iy2 = min(y2, y0 + th)

                    inter_w = ix2 - ix1
                    inter_h = iy2 - iy1
                    if inter_w <= 0 or inter_h <= 0:
                        continue

                    inter_area = inter_w * inter_h
                    orig_area = max(1.0, (x2 - x1) * (y2 - y1))

                    if inter_area / orig_area < KEEP_PARTIAL:
                        continue

                    # Convert to tile-local coords
                    tx1 = ix1 - x0
                    ty1 = iy1 - y0
                    tx2 = ix2 - x0
                    ty2 = iy2 - y0

                    yolo_box = xyxy_to_yolo(cls, tx1, ty1, tx2, ty2, tw, th)
                    if yolo_box is not None:
                        tile_labels.append(yolo_box)

                tile_id = f"{stem}_x{x0}_y{y0}"
                out_img_path = img_out / f"{tile_id}.jpg"
                out_lbl_path = lbl_out / f"{tile_id}.txt"

                cv2.imwrite(str(out_img_path), tile_img)

                # Write labels (empty file allowed)
                if tile_labels:
                    with open(out_lbl_path, "w") as f:
                        for cls, xc, yc, bw, bh in tile_labels:
                            f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                else:
                    out_lbl_path.write_text("")
                    empty_tiles += 1

                tile_count += 1

    print(f"[{split}] tiles written: {tile_count}")
    print(f"[{split}] empty-label tiles: {empty_tiles}")


# ---- RUN ----
process_split("train")
process_split("val")

print("\nTiled dataset created at:")
print(" ", OUT_ROOT)
print("\nNext: create dataset_tiled.yaml with path pointing to this OUT_ROOT.")
