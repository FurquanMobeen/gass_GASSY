import cv2
import random
from ultralytics import YOLO
import easyocr
from collections import defaultdict, Counter
import numpy as np
import re

# Load YOLO models
bottle_yolo = YOLO("models/best.pt")                     # bottle tracking model
text_yolo = YOLO("models/new_yolo11s_extract_tarra_weights.pt")   # model for detecting weight/year text

ocr_memory = defaultdict(list)
MAX_MEMORY = 20   # how many OCR results to store per tracked bottle

# Initialize OCR engine
reader = easyocr.Reader(['en'])

def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

video_path = "videos/14_55_top_cropped.mp4"
videoCap = cv2.VideoCapture(video_path)

frame_count = 0

while True:
    frame_count += 1
    ret, frame = videoCap.read()
    if not ret:
        break

    # Main bottle detection + tracking
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
            colour = getColours(cls)

            # Draw bottle bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, f"{class_names.get(cls, cls)} {conf:.2f}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

            # Extract bottle ROI for text detection
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

                    # Convert local ROI coords → full frame coords
                    gx1, gy1, gx2, gy2 = x1 + tx1, y1 + ty1, x1 + tx2, y1 + ty2

                    # Draw text detection box
                    cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)

                    text_roi = frame[gy1:gy2, gx1:gx2]
                    if text_roi.size == 0:
                        continue

                    # Preprocessing steps for thin stamped text
                    gray = cv2.cvtColor(text_roi, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    enhanced = clahe.apply(gray)
                    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)

                    candidate_texts = []
                    angles = [0, -8, 8, -12, 12]   # small rotations help with skewed bottles

                    for angle in angles:
                        center = (blur.shape[1]//2, blur.shape[0]//2)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        rotated = cv2.warpAffine(blur, M, (blur.shape[1], blur.shape[0]))

                        # Light normalization + mild high-pass to highlight stamped characters
                        norm = cv2.normalize(rotated, None, 0, 255, cv2.NORM_MINMAX)
                        hp = cv2.addWeighted(norm, 1.5, cv2.GaussianBlur(norm, (31, 31), 0), -0.5, 0)

                        # OCR
                        ocr_text = reader.readtext(hp, detail=0, allowlist='0123456789KGkg.')
                        candidate_texts.append(" ".join(ocr_text))

                    # Merge all OCR attempts into one string
                    combined_text = " ".join(candidate_texts)

                    # Extract weight values like "10.8"
                    weight_matches = re.findall(r"\d{1,2}\.\d", combined_text)

                    # Extract year values, typical pattern "20xx"
                    year_matches = re.findall(r"20\d{2}", combined_text)

                    detected_text = ""
                    if len(weight_matches) > 0:
                        detected_text += weight_matches[0] + " "
                    if len(year_matches) > 0:
                        detected_text += year_matches[0]

                    if detected_text:
                        track_id = int(box.id[0]) if hasattr(box, "id") else None

                        if track_id is not None:
                            ocr_memory[track_id].append(detected_text)

                            # keep recent OCR results only
                            if len(ocr_memory[track_id]) > MAX_MEMORY:
                                ocr_memory[track_id] = ocr_memory[track_id][-MAX_MEMORY:]

                            # most common OCR result for stability
                            stable_text = Counter(ocr_memory[track_id]).most_common(1)[0][0]
                        else:
                            stable_text = detected_text

                        # Weight filtering — accept only sane values
                        stable_weight = None
                        stable_year = None

                        w = re.search(r"(\d{1,2}\.\d)", stable_text)
                        if w:
                            value = float(w.group(1))
                            if 8.0 <= value <= 20.0:   # reject out-of-range weights
                                stable_weight = f"{value:.1f}"

                        y = re.search(r"(20\d{2})", stable_text)
                        if y:
                            stable_year = y.group(1)

                        # Draw the final result on screen
                        cv2.putText(frame,
                                    stable_text,
                                    (gx1, max(gy2 + 15, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (255, 255, 0),
                                    2)

    # resize output for display
    frame_small = cv2.resize(frame, (640, 360))
    cv2.imshow('Tracking gas bottles (with OCR)', frame_small)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCap.release()
cv2.destroyAllWindows()
