# Wildlife Cross-Domain Analysis

### A Cross-Domain and Mixed-Domain Analysis for Endangered Animal Detection and Classification Using Deep Learning Models

This repository contains the research implementation and experimental analysis of endangered animal detection and instance segmentation using deep learning models. The study investigates the impact of domain shift between web-sourced wildlife images and real-world camera-trap images and evaluates mixed-domain learning as a strategy for improving model generalization.

## 📌 Overview

Deep learning models can achieve high accuracy when trained and tested on images from similar environments. However, their performance can significantly decline when applied to unfamiliar environments due to differences in lighting, backgrounds, camera angles, image quality, and environmental conditions.

This research presents a comparative experimental framework to investigate these challenges in automated wildlife monitoring.

The study evaluates six deep learning architectures across three experimental scenarios:

- **Same-Domain Evaluation:** Training and testing models within the same visual domain.
- **Cross-Domain Evaluation:** Training on one domain and testing on a different, unseen domain.
- **Mixed-Domain Training:** Combining web and camera-trap images using different training ratios to improve generalization.

Both object detection and instance segmentation tasks are investigated using the same evaluation methodology.

## 🎯 Research Objectives

- Investigate the impact of domain shift on endangered animal detection and instance segmentation.
- Compare modern YOLO-based and transformer-based detection architectures.
- Evaluate instance segmentation models under different visual domain conditions.
- Analyze bidirectional cross-domain generalization between web and camera-trap datasets.
- Investigate different mixed-domain training configurations.
- Identify effective data-mixing strategies for improving wildlife monitoring performance.

## 🧠 Models Evaluated

### Object Detection

| Model | Architecture |
|---|---|
| YOLOv11-L | YOLO-based object detector |
| YOLO26-L | YOLO-based object detector |
| RT-DETR-L | Transformer-based real-time detector |

### Instance Segmentation

| Model | Architecture |
|---|---|
| YOLOv11-seg | YOLO-based instance segmentation |
| YOLO26-seg | YOLO-based instance segmentation |
| Mask R-CNN | Two-stage instance segmentation |

## 📂 Datasets

The experiments utilize two distinct visual domains:

**1. Endangered Animal Web Dataset**

- Images collected from web sources.
- Contains 10 animal classes.
- Training images: 3,379
- Validation images: 483
- Testing images: 965

**2. Endangered Animal Camera-Trap Dataset**

- Images captured using field camera traps.
- Contains 8 animal classes.
- Training images: 3,012
- Validation images: 374
- Testing images: 375

For fair cross-domain evaluation, eight common animal categories were selected:

1. Bat
2. Indian Crested Porcupine
3. Squirrel
4. Vulture
5. Goat
6. Wild Elephant
7. Rat
8. Wild Boar

All images were resized and normalized to 640 × 640 pixels. Detection annotations were converted into the YOLO text-label format.

**Note:** Dataset files are not included in this repository. Users should obtain the datasets from their respective original sources and comply with their licensing and usage conditions.

## ⚙️ Experimental Methodology

The research follows three major experimental protocols.

### 1. Same-Domain Evaluation

Models are trained and tested using data from the same domain.

- Web → Web
- Camera-Trap → Camera-Trap

This establishes the baseline performance of each architecture.

### 2. Cross-Domain Evaluation

Models are trained on one domain and evaluated on another unseen domain.

- Web → Camera-Trap
- Camera-Trap → Web

This measures the impact of domain shift and evaluates the ability of models to generalize across different visual environments.

### 3. Mixed-Domain Training

Four different training configurations are investigated:

| Setup | Web | Camera-Trap | Evaluation |
|---|---:|---:|---|
| Setup 1 | 80% | 20% | Web test data |
| Setup 2 | 70% | 30% | Unused mixed-domain data |
| Setup 3 | 20% | 80% | Camera-Trap test data |
| Setup 4 | 40% | 60% | Unused mixed-domain data |

The same experimental strategy is applied to both detection and instance segmentation.

## 📊 Experimental Results

### Object Detection

Some notable results from the experiments:

| Evaluation Scenario | Model | mAP@0.5 |
|---|---|---:|
| Same-Domain Camera-Trap | YOLOv11-L | 94.64% |
| Cross-Domain Web → Camera-Trap | YOLOv11-L | 23.97% |
| Mixed Setup 2 | YOLOv11-L | 79.34% |
| Mixed Setup 3 | YOLO26-L | 94.31% |
| Mixed Setup 4 | YOLOv11-L | 83.29% |

### Instance Segmentation

| Evaluation Scenario | Model | mAP@0.5 |
|---|---|---:|
| Same-Domain Web | YOLOv11-seg | 73.30% |
| Same-Domain Camera-Trap | YOLOv11-seg | 93.52% |
| Cross-Domain Web → Camera-Trap | YOLOv11-seg | 21.23% |
| Mixed Setup 2 | YOLOv11-seg | 75.43% |
| Mixed Setup 3 | YOLO26-seg | 91.21% |
| Mixed Setup 4 | YOLOv11-seg | 81.97% |

### Key Findings

- Significant performance degradation was observed when models were evaluated across different visual domains.
- YOLOv11-L achieved 94.64% mAP@0.5 in same-domain camera-trap detection but dropped to 23.97% in the Web-to-Camera-Trap evaluation.
- YOLOv11-seg achieved 93.52% same-domain camera-trap segmentation performance but dropped to 21.23% in the Web-to-Camera-Trap evaluation.
- Mixed-domain training improved generalization on previously unused mixed-domain test data.
- Different model architectures demonstrated different strengths across detection, segmentation, precision, and recall.

*All reported values are taken from the final thesis experimental results. Refer to the thesis for complete tables, evaluation curves, confusion matrices, and qualitative comparisons.*

## 📈 Evaluation Metrics

The following metrics are used to evaluate model performance:

- **mAP@0.5:** Mean Average Precision at an IoU threshold of 0.5.
- **mAP@0.5:0.95:** Mean Average Precision across multiple IoU thresholds.
- **Precision:** Proportion of predicted positive detections that are correct.
- **Recall:** Proportion of ground-truth objects successfully detected.

## 🛠️ Technologies Used

- Python
- PyTorch
- Ultralytics
- RT-DETR
- YOLOv11
- YOLO26
- Mask R-CNN
- OpenCV
- NumPy
- Matplotlib

## 📁 Suggested Repository Structure

```text
wildlife-cross-domain-analysis/
│
├── README.md
│
├── detection/
│   ├── yolo11/
│   ├── yolo26/
│   └── rt_detr/
│
├── segmentation/
│   ├── yolo11_seg/
│   ├── yolo26_seg/
│   └── mask_rcnn/
│
├── datasets/
│   ├── web/
│   ├── camera_trap/
│   └── README.md
│
├── experiments/
│   ├── same_domain/
│   ├── cross_domain/
│   └── mixed_domain/
│
├── evaluation/
│   ├── detection_metrics/
│   ├── segmentation_metrics/
│   └── visualization/
│
├── results/
│   ├── detection/
│   └── segmentation/
│
├── notebooks/
│
├── requirements.txt
│
└── LICENSE
```

*This is a suggested organization for publishing the project. Adjust it to match the actual code and files uploaded to the repository.*

## 🔬 Limitations and Future Work

Although mixed-domain training improves generalization, several challenges remain:

- Limited availability of diverse real-world camera-trap images.
- Performance degradation under significant visual domain shifts.
- Computational requirements of advanced detection and segmentation architectures.
- Need for further class-wise and object-size performance analysis.

Potential future research directions include:

- Feature-space domain-gap analysis using dimensionality reduction.
- Domain adaptation and advanced augmentation techniques.
- Detection-versus-segmentation performance analysis.
- Evaluation on larger and more diverse camera-trap datasets.
- Edge deployment and real-world field validation.

## 👨‍💻 Research Team

**Department of Computer Science and Engineering**  
United International University (UIU)

| Name | Responsibility |
|---|---|
| Md. Noman Hossain | Project Lead |

**Supervisor:** Dr. Md. Motaharul Islam  
Professor, Department of Computer Science and Engineering  
United International University

## 📄 Research Publication / Thesis

**Title:** A Cross-Domain and Mixed-Domain Analysis for Endangered Animal Detection and Classification Using Deep Learning Models.

**Project:** Final Year Design Project (FYDP-3), 2026.

The complete thesis report contains detailed methodology, model architectures, experimental configurations, quantitative results, qualitative comparisons, and future research directions.

## 📜 License

A license has not yet been specified. Please check the dataset and implementation licenses before choosing a license for this repository.

---

⭐ If you find this research useful, consider starring the repository.
