# Smart Traffic Monitoring & Number Plate Intelligence 🚗🔍🚔

An advanced AI-powered surveillance system for real-time vehicle tracking, speed enforcement, and automatic number plate recognition (ANPR) with stolen vehicle detection.

## 🌟 Key Features

- **Automatic Number Plate Recognition (ANPR):** Uses a combination of Haar Cascades for detection and **EasyOCR** for text extraction.
- **Stolen Vehicle Detection (Watchlist):** Integrated matching system that cross-references detected plates against a `watchlist.json` of stolen or missing vehicles.
- **Speed Enforcement System:** Real-time vehicle tracking and speed calculation with automatic plate capture for overspeeding vehicles.
- **Intelligence Alert System:**
    - 🔴 **RED ALERT:** Stolen vehicle detected (Matched with Watchlist).
    - 🟡 **YELLOW ALERT:** Overspeeding vehicle detected.
    - 🟢 **NORMAL:** Standard vehicle tracking and plate reading.
- **ROI Extraction:** Captures high-resolution crops of number plates for forensic evidence.

## 🛠️ Technical Architecture

### 1. The Intelligence Pipeline
`Video Stream` $\rightarrow$ `Vehicle Tracking (Centroid)` $\rightarrow$ `Speed Calculation` $\rightarrow$ `Plate Detection` $\rightarrow$ `OCR Extraction (EasyOCR)` $\rightarrow$ `Watchlist Matching` $\rightarrow$ `Alert Trigger`.

### 2. Tech Stack
- **Core:** Python 3.x
- **Computer Vision:** OpenCV (MOG2 Background Subtraction, Haar Cascades)
- **Deep Learning:** EasyOCR (Optical Character Recognition)
- **Data Handling:** NumPy, JSON (for Watchlist)

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/Ravi-Tiwari-2710/Number-plate-detection.git
pip install opencv-python numpy easyocr
```

### Usage
1. Update `watchlist.json` with the plates you want to track.
2. Run the system:
```bash
python main.py
```

## 📈 Use Cases
- **Crime Prevention:** Automatically flagging stolen vehicles in city traffic.
- **Traffic Safety:** Identifying and logging speeders without manual intervention.
- **Smart Parking:** Automating entry/exit logs via ANPR.

---
*Developed by [Ravi Tiwari](https://github.com/Ravi-Tiwari-2710)*
