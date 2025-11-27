import cv2
import random
from ultralytics import YOLO
import easyocr
from collections import defaultdict, Counter
import numpy as np
import re

# Load YOLO models
bottle_yolo = YOLO("models/best.pt")  # For gas bottle tracking
text_yolo = YOLO("models/yolo11s_extract_tarra_weights.pt")  # For text region detection

ocr_memory = defaultdict(list)
MAX_MEMORY = 20   # how many OCR results to keep per bottle

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

video_path = "frames/videos/14_43_top_cropped.mp4"
videoCap = cv2.VideoCapture(video_path)

frame_count = 0

while True:
    frame_count += 1
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
                    
                    candidate_texts = []

                    # --- Preprocessing for weak cylinder text ---
                    gray = cv2.cvtColor(text_roi, cv2.COLOR_BGR2GRAY)

                    # CLAHE contrast boost
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    enhanced = clahe.apply(gray)

                    # Slight blur to smooth noise
                    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)

                    candidate_texts = []
                    angles = [0, -8, 8, -12, 12]

                    for angle in angles:
                        center = (blur.shape[1]//2, blur.shape[0]//2)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        rotated = cv2.warpAffine(blur, M, (blur.shape[1], blur.shape[0]))
                        
                        # --- Lighting normalization (improves visibility of thin blue text) ---
                        norm = cv2.normalize(rotated, None, 0, 255, cv2.NORM_MINMAX)

                        # --- Gentle high-pass filter to boost thin edges ---
                        hp = cv2.addWeighted(norm, 1.5, cv2.GaussianBlur(norm, (31, 31), 0), -0.5, 0)

                        # Use this as final OCR input
                        ocr_input = hp

                        # OCR directly on blurred + enhanced grayscale
                        ocr = reader.readtext(ocr_input, detail=0, allowlist='0123456789KGkg.')
                        text = " ".join(ocr)
                        candidate_texts.append(text)

                    # Combine all OCR attempts
                    all_text = " ".join(candidate_texts)
                    
                    # 🔵 DEBUG-PRINTIT LISÄTÄÄN TÄHÄN
                    print("\n---- OCR DEBUG ----")
                    print("OCR rotations:", candidate_texts)
                    print("Combined text:", all_text)

                    # Extract weight (e.g., "10.8")
                    weight = re.findall(r"\d{1,2}\.\d", all_text)
                    print("Weight extracted:", weight)

                    # Extract year (e.g., "2033")
                    year = re.findall(r"20\d{2}", all_text)
                    print("Year extracted:", year)
                    print("---------------------\n")

                    detected_text = ""
                    if len(weight) > 0:
                        detected_text += weight[0] + " "
                    if len(year) > 0:
                        detected_text += year[0]
                    
                    if detected_text:
                        # Bottle ID from tracker
                        track_id = int(box.id[0]) if hasattr(box, "id") else None

                        if track_id is not None:
                            # Add detected text to memory
                            ocr_memory[track_id].append(detected_text)

                            # Limit memory size
                            if len(ocr_memory[track_id]) > MAX_MEMORY:
                                ocr_memory[track_id] = ocr_memory[track_id][-MAX_MEMORY:]

                            # Determine the stable output (most common text)
                            stable_text = Counter(ocr_memory[track_id]).most_common(1)[0][0]
                            
                        else:
                            stable_text = detected_text
                            
                        # Draw stable text on the screen
                        cv2.putText(
                            frame,
                            stable_text,
                            (gx1, max(gy2 + 15, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 0),
                            2,
                        )
    
    frame_small = cv2.resize(frame, (640, 360))
    cv2.imshow('Tracking gas bottles (with OCR)', frame_small)
    
    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

videoCap.release()
cv2.destroyAllWindows()