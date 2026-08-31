# Legume Nodule Detection

Automated detection and quantification of legume root nodules from hyperspectral-enhanced images using **YOLOv11**.
Preprint: `https://www.biorxiv.org/content/10.1101/2025.07.25.666867v3` 

## Overview

This repository contains the image-processing and deep learning workflow used to detect legume root nodules from RGB and hyperspectral images. Hyperspectral images are processed to generate enhanced images that improve nodule visualization and detection.

The workflow includes:

1. **Hyperspectral image preprocessing** – extraction of selected wavelengths and generation of enhanced images.
2. **Image tiling** – optional step to increase the number of training images.
3. **YOLOv11 training** – training of models using both enhanced and RGB images.
4. **Nodule prediction** – automated detection and quantification of nodules using trained models.

## Repository Structure

```text
├── preprocessing/
│   ├── hyperspectral_preprocessing.ipynb
│   └── tile.py
│   
├── yolo/
│   ├── train_enhanced.ipynb
│   ├── train_rgb.ipynb
│   ├── predict_enhanced.ipynb
│   └── models/
│       ├── enhanced/
│       └── rgb/
│
├── README.md
└── LICENSE
```

## Getting Started

The preprocessing workflow can be run using `preprocessing/hyperspectral_preprocessing.ipynb`. The notebook uses hyperspectral `.bil`/`.hdr` files to generate enhanced images for nodule detection.
YOLOv11 training and prediction workflows are provided in the `yolo/` directory. Pre-trained model weights and configuration files are provided where applicable.

## Data

A small set of example RGB and hyperspectral images is provided for testing the preprocessing workflow. Full datasets are not included in this repository.

## License

This project is distributed under the [MIT License](LICENSE).
