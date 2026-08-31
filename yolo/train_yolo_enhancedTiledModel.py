#!/usr/bin/env python3
from ultralytics import YOLO
import torch
import os
from pathlib import Path

print("Ultralytics OK")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))

# -----------------------------
# EDIT THESE PATHS (Linux)
# -----------------------------
# Example:
# DATA_YAML = "/home/<netid>/NoduleYoloWorkspace/datasets/Indeterminate_Enhanced_Model/dataset.yaml"
DATA_YAML = "/home/labs/brooks_lab/Hyperspectral/datasets/combinedNodulesTiled/combinedNodulesTiled.yaml"

# Where to store runs/ outputs (relative or absolute)
PROJECT = "runs"
RUN_NAME = "combined.new_tiled_yolo11"

# -----------------------------
# TRAINING PARAMS
# -----------------------------
MODEL  = "./yolo11n.pt"   # downloaded file should be in the folder - changed yolo11n/s --> yolo11m
IMGSZ  = 720    # increased from 720
EPOCHS = 200     # decreased from 500
PATIENCE = 50    # new addition
BATCH  = 4
RECT = True

# Device logic:
# - If you request GPU in sbatch, set DEVICE=0 (GPU id 0)
# - If CPU job, set DEVICE="cpu"
DEVICE = 0 if torch.cuda.is_available() else "cpu"

print("\nResolved settings:")
print("DATA_YAML:", DATA_YAML)
print("PROJECT:", PROJECT)
print("RUN_NAME:", RUN_NAME)
print("MODEL:", MODEL)
print("IMGSZ:", IMGSZ, "EPOCHS:", EPOCHS, "PATIENCE:", PATIENCE, "BATCH:", BATCH, "DEVICE:", DEVICE)

# Safety checks
if not Path(DATA_YAML).exists():
    raise FileNotFoundError(f"dataset.yaml not found: {DATA_YAML}")

# Train
model = YOLO(MODEL)
results = model.train(
    data=DATA_YAML,
    imgsz=IMGSZ,
    epochs=EPOCHS,
    patience=PATIENCE,
    batch=BATCH,
    rect=RECT,
    device=DEVICE,
    project=PROJECT,
    name=RUN_NAME,
    plots=True
)

print("\nTraining finished.")




