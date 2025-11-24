Markdown# 🛒 Grocery Product Recognition & Smart Billing System

| **Course** | **Academic Year** | **Semester** | **Branch** | **Campus** |
| :---: | :---: | :---: | :---: | :---: |
| Advanced Foundations for Machine Learning | 2025 | 5th Sem | AIML | RR |

---

## 💡 Project Description

This project implements a **Smart Checkout System**  designed to automate and verify the billing process for grocery products. It utilizes a **multi-stage machine learning pipeline** to accurately detect, classify, verify, and count items from a single input image, generating a final itemized bill.

### Core Technologies

The system integrates state-of-the-art models for high accuracy and reliability:

* **Object Detection:** **YOLOv8** (to locate products)
* **Classification:** **EfficientNet-B0** (to identify specific product types)
* **Text Extraction & Verification:** **EasyOCR**
* **Anomaly Detection (Optional):** **Deep SVDD** (One-Class Verification)

### Key Features
* **YOLOv8** object detection on **16 grocery classes**.
* **EfficientNet-B0** classification from YOLO crops.
* **OCR-based verification** using EasyOCR to confirm product identity.
* **IOU-based duplicate removal** for accurate item counting.
* Highly reliable billing logic.
* Optional **Deep SVDD** for SKU-level verification.

---

## 🧑‍💻 Development Team

| **Student** | **Role** |
| :--- | :--- |
| PES1UG23AM210 | 
| PES1UG23AM221 |
| PES1UG23AM237 | 
| PES1UG23AM134 | 

---

## 🚀 Getting Started

### 1. Install Dependencies

Install all required Python libraries using the provided file:

```bash
pip install -r requirements.txt
2. Training ModelsTrain the two primary models required for the pipeline:ModelGoalCommandYOLOv8 Object DetectorDetect and locate items.python train_yolo.pyEfficientNet-B0 ClassifierClassify specific item types.python train_classifier.py3. Running the SystemTaskPurposeCommandYOLO vs Classifier EvaluationTest detection and initial classification stages.python project/utils/test_image.pyRun Full Billing PipelineExecute the full pipeline (Detection, Classification, OCR, Billing).cd project  python final1.pyRun One-Class Deep SVDDExecute the optional SKU verification module.cd occ  python occ.py📁 Project StructureGrocery_Billing_System/
├── project/
│   ├── utils/
│   │   ├── test_image.py     # Evaluation script
│   │   ├── counter.py        # Item counting module
│   │   └── cr.py             # Classification utilities
│   ├── final1.py             # Main script for the full billing pipeline
│   └── best.pt               # Trained YOLOv8 weights
│
├── occ/
│   ├── occ.py                # Deep SVDD training and execution script
│   ├── support_pics/         # Images for one-class training
│   └── pics/                 # Images for testing Deep SVDD
│
├── train_yolo.py             # Script for training YOLOv8
├── train_classifier.py       # Script for training EfficientNet-B0
├── classifier_best.pth       # Trained EfficientNet-B0 weights
├── data.yaml                 # YOLO dataset configuration
├── requirements.txt          # List of project dependencies
└── README.md                 # This file
```

📜 License
For academic use under the Advanced Foundations for Machine Learning course.
