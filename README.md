# Legume Nodule Detection

Automated detection and quantification of legume root nodules from hyperspectral-enhanced images using **YOLOv11** (Jocher and Qiu, 2024).  
The development and evaluation of this approach are described in our preprint:
[View the preprint on bioRxiv](https://www.biorxiv.org/content/10.1101/2025.07.25.666867v3). 

## Overview

This repository contains the image-processing and deep learning workflow used to detect legume root nodules from RGB and hyperspectral images. Hyperspectral images are processed to generate enhanced images that improve nodule visualization and detection.

The workflow includes:

1. **Hyperspectral image preprocessing** – extraction of selected wavelengths from hyperspectral images to generate enhanced images.
2. **Image tiling** – optional step to increase the number of training images.
3. **YOLOv11 training** – training separate models for enhanced or RGB images.
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
│   └── predict_enhanced.ipynb
│
├── models/
│   ├── enhanced/
│   └── rgb/
│
├── README.md
└── LICENSE
```

## Getting Started

The preprocessing workflow can be run using `preprocessing/hyperspectral_preprocessing.ipynb`. The notebook uses hyperspectral `.bil` files to generate enhanced images for nodule detection.
YOLOv11 training and prediction workflows are provided in the `yolo/` directory. Pre-trained model weights and configuration files are provided where applicable.

## Note
As an alternative and complementary approach, a publicly accessible model for legume nodule detection is also available on the Biodock AI platform:
[**Legume nodule detection**](https://app.biodock.ai/public-models)  
The Biodock platform provides an accessible, no-code approach for applying the nodule detection model to new images.

## License

This project is distributed under the [MIT License](LICENSE).

## Resources

This pipeline has been developed using guidance from the following resources:  
Jocher G, Qiu J (2024) Ultralytics YOLO11.   
Skalski P (2019) Make Sense. 

