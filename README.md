# 🧠 Gas Bottle Detection & Tracking using YOLO

## 📘 Project Context

This project is part of an AI-based monitoring system designed to **detect and track gas bottles moving on a conveyor belt** in real-time.  
Using a fine-tuned YOLO model from [Ultralytics](https://github.com/ultralytics/ultralytics), the solution provides automated detection, labeling, and tracking directly from live video streams or stored footage.

The system supports **on-site deployment** and aims to enhance:
- Quality control  
- Production efficiency  
- Safety monitoring  
- Inventory tracking  

---

## 🧩 Customer Requirements

The customer requested the following capabilities:

- ✅ Real-time **detection and tracking** of gas bottles on conveyor belts.  
- ✅ Clear **bounding boxes and confidence levels** displayed on live video.  
- ✅ Reliable performance under **different lighting and motion conditions**.  
- ✅ Easy-to-run Python scripts for local computers.  
- ✅ Extendable dataset and training workflow for future updates.

---

## ⚙️ General Algorithm Architecture

Below is a simplified view of the YOLO-based tracking pipeline implemented in `track-gas-bottle.py`:

```python
import cv2
import random
from ultralytics import YOLO

# Load the fine-tuned model
yolo = YOLO("models/detection/yolo11x_finetuned_bottles_on_site_v3.pt")

def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

video_path = "videos/14_55_front_cropped.mp4"
videoCap = cv2.VideoCapture(video_path)

while True:
    ret, frame = videoCap.read()
    if not ret:
        break

    results = yolo.track(frame, stream=True)
    for result in results:
        for box in result.boxes:
            if box.conf[0] > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                colour = getColours(cls)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, f"{result.names[cls]} {conf:.2f}",
                            (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, colour, 2)

    cv2.imshow('Tracking gas bottles', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCap.release()
```
### 🧠 Main Components
- Model Loading – YOLO11x fine-tuned for gas bottle detection.
- Inference Loop – Real-time frame processing and detection.
- Visualization Layer – Bounding boxes, confidence, and class labels.
- User Control – Press q to stop the live video feed.
---

## 🏆 Latest Performance Results
The model achieved excellent detection accuracy on photo-based validation datasets.
| Metric | Value | Description |
|:-------|:------:|:------------|
| 🧩 **mAP@0.5** | **0.992** | Mean Average Precision at IoU 0.5 |
| 🎯 **Precision** | ≈ 0.99 | Correct detections among predicted positives |
| 🔍 **Recall** | ≈ 0.98 | True detections among actual positives |
| ⚖️ **F1 Score** | 0.96 @ conf=0.79 | Best balance between precision and recall |

### 📊 Performance Visualization
The chart below summarizes the YOLO model’s performance metrics:
- Precision–Recall Curve: mAP@0.5 = 0.992
- F1–Confidence Curve: Peak F1 = 0.96 at confidence 0.79
- Precision–Confidence & Recall–Confidence Curves: Stable up to ~0.8 confidence

🧪 Performance evaluation was performed on fine-tuned YOLO11x trained with real site photos.

---
## 🚀 How to Run the Tracker
### 1️⃣ Install Dependencies
Make sure you have Python ≥ 3.8 and install the required libraries:
```python
pip install ultralytics opencv-python
```
### 2️⃣ Update Video Source
Before running, open track-gas-bottle.py and update the video path:
```Python
video_path = "videos/your_video_here.mp4"
```
### 💡 To use a webcam instead:
```Python
video_path = 0
```
### 3️⃣ Run the Script
#### 🪟 On Windows
```Bash
python track-gas-bottle.py
```
#### 🍎 On macOS
```Bash
python3 track-gas-bottle.py
```
#### 🐧 On Linux
```Bash
python3 track-gas-bottle.py
```
🎥 Press `q` during execution to quit the tracking window.
