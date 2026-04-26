# Smart Traffic Monitoring & Number Plate Detection 🚗🔍

An integrated computer vision system for real-time vehicle counting and automatic number plate detection. This project combines background subtraction for traffic flow analysis and Haar Cascade classifiers for plate identification.

## 🌟 Key Features

- **Automatic Number Plate Detection (ANPR):** Uses optimized Haar Cascades to detect vehicle plates in real-time.
- **Vehicle Counting System:** Implements `MOG2` Background Subtraction and contour analysis to count vehicles crossing a defined virtual line.
- **ROI Extraction:** Ability to capture and save the Region of Interest (ROI) of the detected plate for further OCR processing.
- **Real-time Processing:** Optimized pipeline for low-latency video stream analysis.

## 🛠️ Technical Architecture

### 1. Detection Pipeline
- **Plate Detection:** `Grayscale Conversion` $\rightarrow$ `Haar Cascade Classifier` $\rightarrow$ `Area Filtering`.
- **Vehicle Counting:** `Gaussian Blur` $\rightarrow$ `Background Subtraction` $\rightarrow$ `Morphological Operations (Dilate/Close)` $\rightarrow$ `Contour Analysis`.

### 2. Tech Stack
- **Language:** Python 3.x
- **Library:** OpenCV (Open Source Computer Vision Library)
- **Algorithm:** MOG2 Background Subtractor, Haar Cascades.

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/Ravi-Tiwari-2710/Number-plate-detection.git
pip install opencv-python numpy
```

### Usage
```bash
python main.py
```
- Press **'s'** to save the detected number plate image to the `/exports` folder.
- Press **'q'** or **Enter** to exit.

## 📈 Future Scope
- **OCR Integration:** Implementing Tesseract or EasyOCR to convert detected plate images into text.
- **Speed Estimation:** Calculating vehicle speed based on frame-to-frame displacement.
- **Database Integration:** Linking detected plates with a vehicle registry database for automated tolling.

---
*Developed by [Ravi Tiwari](https://github.com/Ravi-Tiwari-2710)*
