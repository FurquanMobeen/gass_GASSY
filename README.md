# 🧠 Gas Bottle Detection & Tracking using YOLO

## 📘 Project Context

This project is part of an AI-based monitoring system designed to **detect and track gas bottles moving on a conveyor belt** in real-time.  
Using a fine-tuned YOLO model from [Ultralytics](https://github.com/ultralytics/ultralytics), the solution provides automated detection, labeling, and tracking directly from live video streams or stored footage.

The system supports **on-site deployment** and aims to enhance:
- Quality control  
- Production efficiency  
- Safety monitoring  
- Inventory tracking  

### Main algorithms
- `track-gas-bottle.py` : Gas bottle tracking from one source video.
- `track-gas-bottle-with-easyOCR.py` : Gas bottle tracking from one source video with OCR.
- `track-gas-bottle-multiple-videos.py` : Gas bottle tracking from multiple source videos.
- `track-gas-bottle-classification.py` : Gas bottle tracking from multiple source videos with classification (ok, not PrimaGaz, etc..)
- `track-gas-bottle-multiple-videos-pretrained-OCR.py` : Gas bottle tracking from multiple source videos with pretrained OCR.
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

Below is a simplified view of the YOLO-based tracking pipeline implemented in `track-gas-bottle-with-easyOCR.py`:

```python
import cv2
import random
from ultralytics import YOLO
import easyocr

# Load YOLO models
bottle_yolo = YOLO("models/yolo11n_bottles.pt")  # For gas bottle tracking
text_yolo = YOLO("models/yolo11s_extract_tarra_weights.pt")  # For text region detection

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

video_path = "videos/14_55_front_cropped.mp4"
videoCap = cv2.VideoCapture(video_path)

frame_count = 0

while True:
    ret, frame = videoCap.read()
    if not ret:
        break
    # Bottle detection & tracking
    results = bottle_yolo.track(
        frame,
        conf=0.4,
        tracker="bytetrack.yaml",
        persist=True,
        stream=False,
    )
    
    for result in results:
        class_names = result.names
        if not hasattr(result, "boxes") or len(result.boxes) == 0:
            continue
        for box in result.boxes:
            conf = float(box.conf[0]) if hasattr(box, "conf") else 1.0
            if conf < 0.4:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0]) if hasattr(box, "cls") else 0
            class_name = class_names.get(cls, str(cls)) if isinstance(class_names, dict) else str(cls)
            colour = getColours(cls)

            # Draw bottle
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(
                frame,
                f"{class_name} {conf:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                colour,
                2,
            )

            # Text detection within bottle ROI using second YOLO
            bottle_roi = frame[y1:y2, x1:x2]
            if bottle_roi.size == 0:
                continue

            text_results = text_yolo(bottle_roi)
            for text_res in text_results:
                if not hasattr(text_res, "boxes") or len(text_res.boxes) == 0:
                    continue
                for tbox in text_res.boxes:
                    tconf = float(tbox.conf[0]) if hasattr(tbox, "conf") else 1.0
                    if tconf < 0.4:
                        continue

                    tx1, ty1, tx2, ty2 = map(int, tbox.xyxy[0])

                    # Adjust coordinates to full frame
                    gx1, gy1, gx2, gy2 = x1 + tx1, y1 + ty1, x1 + tx2, y1 + ty2

                    # Draw text region box
                    cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)

                    # OCR on text region
                    text_roi = frame[gy1:gy2, gx1:gx2]
                    if text_roi.size == 0:
                        continue
                    ocr_texts = reader.readtext(text_roi, detail=0)
                    detected_text = " ".join(ocr_texts).strip()
                    if detected_text:
                        cv2.putText(
                            frame,
                            detected_text,
                            (gx1, max(gy2 + 15, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 0),
                            2,
                        )
    
    # Display the frame
    cv2.imshow('Tracking gas bottles (with OCR)', frame)
    
    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_count += 1

videoCap.release()
cv2.destroyAllWindows()
```
### 🧠 Main Components
- **Imports**: cv2, random, ultralytics.YOLO, easyocr for vision, detection, and OCR.
- **Model Loading:** bottle_yolo (bottle tracking) and text_yolo (text-region detection) from local models/*.pt.
- **OCR Init**: reader = easyocr.Reader(['en']) initialized once and reused.
- **Utility**: getColours(cls_num) returns a consistent RGB color per class.
- **Video Setup**: video_path, cv2.VideoCapture(video_path), and frame_count initialization.
- **Main Loop**:
  - Frame Read: grabs frames until stream ends.
  - Bottle Detection/Tracking: bottle_yolo.track(..., conf=0.4, tracker="bytetrack.yaml", persist=True) to get boxes.
  - Bottle Overlay: draws bottle bbox and label with confidence.
  - Text Detection in ROI: crops bottle ROI and runs text_yolo(roi) to find text regions.
  - Coordinate Mapping: converts ROI text boxes back to full-frame coordinates.
  - OCR & Overlay: reader.readtext(text_roi, detail=0), concatenates text, overlays near each text bbox.
  - Display & Control: shows the annotated frame (cv2.imshow), breaks on q.
- **Cleanup**: releases the capture and calls cv2.destroyAllWindows().
---

## 🏆 Latest Performance Results
### (New) YOLO11s Model Object detection:
The model achieved excellent detection accuracy on photo-based validation datasets.
| Metric | Value | Description |
|:-------|:------:|:------------|
| 🧩 **mAP@0.5** | **0.9932**  | Mean Average Precision at IoU 0.5 |
| 🎯 **Precision** | **0.9811**  | Correct detections among predicted positives |
| 🔍 **Recall** | **0.9832** | True detections among actual positives |
| ⚖️ **F1 Score** | **0.9821** | Best balance between precision and recall |

### EfficientNet_B0 classification:
- Early stopping successfully prevented overfitting across folds and reduced unnecessary computation. 
- Validation performance was consistently high across folds, demonstrating the model’s stability and ability to generalize.. 
- Fold 2 achieved perfect validation accuracy, showing that the dataset is well-learned by the model.
- Overall, the model demonstrates strong generalization, with an average validation accuracy of 98.63% across 5 folds.
- The loss and accuracy curves provide a clear visual understanding of the learning process and model stability. 

### 📊 Performance Visualization
The chart below summarizes the YOLO model’s performance metrics:
- Precision–Recall Curve: mAP@0.5 = 0.992
- F1–Confidence Curve: Peak F1 = 0.96 at confidence 0.79
- Precision–Confidence & Recall–Confidence Curves: Stable up to ~0.8 confidence

🧪 Performance evaluation was performed on fine-tuned YOLO11x trained with real site photos.

### Multiple videos tracking in `track-gas-bottle-multiple-videos.py`:
We added this in order to apply this algorithm to multiple videos from different angles, making the tracking and classification of gas bottles more accurate.

### Gas bottle tracking with reading Tarra weight and Expiration year `track-gas-bottle-with-easyOCR.py`:
easyOCR does not do well with not-horizontally-placed text, therefore meanwhile it reads the text, but it can not read the text correctly. <br>
For example, instead of "7", it reads ">". Or instead of "1", it reads "I" or "L". <br>
Therefore, in the future, we need to find a way to rotate the text dynamically to improve easyOCR performance, or find a different approach.

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
