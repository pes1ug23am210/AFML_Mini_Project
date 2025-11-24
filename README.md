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

| **Student** | 
| :--- | 
| PES1UG23AM210 | 
| PES1UG23AM221 |
| PES1UG23AM237 | 
| PES1UG23AM134 | 

---

## 🚀 Getting Started

### 1. Install Dependencies

Install all required Python libraries using the provided file:

```bash
1)pip install -r requirements.txt
2) extract the dataset zip into the model folder
3)run train_classifier.py and train_yolo.py to train the classifier model and yolo model.
4)train occ model according to your liking and paste it inside project/models
5)this repo already has the trained models so just run bill.py for the outputs(input in the folder:bill_image).
     
├── model/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   ├── train/
│   │   ├── images/
│   │   ├── labels/
│   │   └── labels.cache
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   │       └── labels.cache
│   ├── best.pt
│   ├── classifier_best.pth
│   ├── data.yaml
│   ├── test_image.py
│   ├── train_classifier.py
│   └── train_yolo.py
├── one_class_classifier/
│   ├── pics/
│   ├── support_pics/
│   ├── occ.py
│   └── oneclass_model.pth
├── project/
│   ├── bill_image/
│   ├── models/
│   │   └── bingo.pth
│   ├── best.pt
│   ├── bill.py
│   ├── classification.py
│   ├── classifier_best.pth
│   ├── config.py
│   ├── data.yaml
│   ├── detection.py
│   ├── main.py
│   ├── models.py
│   ├── ocr.py
│   ├── pipeline.py
│   └── utils.py
└── requirements.txt

```

📜 License
For academic use under the Advanced Foundations for Machine Learning course.
