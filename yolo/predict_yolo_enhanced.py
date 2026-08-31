#!/usr/bin/env python3

from ultralytics import YOLO

model = YOLO("./runs/combined.new_tiled_yolo11/weights/best.pt") #change model path as needed

sources = [
    "/home/labs/brooks_lab/Hyperspectral/datasets/NdfaPredictionSet/images/S93_Enhanced",
]

for src in sources:
    model.predict(
        source=src,           # <-- STRING, not list
        conf=0.15,
        iou=0.7,    # 0.7 for determinates, 0.4 for indeterminates
        max_det=1000,
        save=True,
        save_txt=True,
        project="predictions",
        name=f"{src.split('/')[-4]}_yolo11_predictions",
        device="cpu",
    )

