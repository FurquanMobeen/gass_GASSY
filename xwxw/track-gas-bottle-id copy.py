import cv2
import random
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
import easyocr
import re

# Bottle detection model
yolo = YOLO("models/yolo11n_bottles.pt")

# Text (Tarra Weight / Year) detection model
text_yolo = YOLO("models/new_yolo11s_extract_tarra_weights.pt")

# Classification model (binary: ok / nok)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cls_checkpoint_path = "models/bottle_classifier_fold_2.pth"
checkpoint = torch.load(cls_checkpoint_path, map_location=device)
state_dict = checkpoint.get('state_dict', checkpoint)
classifier = models.efficientnet_b0(weights=None)
classifier.classifier[1] = torch.nn.Linear(classifier.classifier[1].in_features, 2)
classifier.load_state_dict(state_dict, strict=False)
classifier.to(device)
classifier.eval()
classifier_class_names = ['nok', 'ok']

classifier_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def classify_roi(frame, x1, y1, x2, y2):
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None, 0.0
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(roi_rgb)
    tensor = classifier_transforms(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = classifier(tensor)
        probs = torch.softmax(out, dim=1)
        cls_idx = int(torch.argmax(probs, dim=1).item())
        conf = float(torch.max(probs).item())
    return classifier_class_names[cls_idx], conf

# EasyOCR reader
reader = easyocr.Reader(['en'])

def ocr_read(image_bgr):
    if image_bgr.size == 0:
        return ""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Light normalization
    gray = cv2.equalizeHist(gray)
    results = reader.readtext(gray, detail=0, paragraph=False)
    return " ".join(results) if results else ""

def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

def getTrackingColour(track_id):
    """Generate consistent color for each tracking ID"""
    random.seed(int(track_id))
    return tuple(random.randint(0, 255) for _ in range(3))

video_path = "videos/14_55_front_cropped_trimmed.mp4"
videoCap = cv2.VideoCapture(video_path)

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('videos/output/14_55_front_cropped.mp4', fourcc, 20.0,
                      (int(videoCap.get(3)), int(videoCap.get(4))))

frame_count = 0

# Per-ID data store: {id: {status, tarra, year, first_frame}}
id_data = {}

# ID mapping for consecutive numbering (removes gaps)
id_mapping = {}
next_consecutive_id = 0

while True:
    ret, frame = videoCap.read()
    if not ret:
        break

    results = yolo.track(frame, conf=0.4, tracker="bytetrack_custom.yaml", persist=True, stream=True)
    
    for result in results:
        class_names = result.names
        for box in result.boxes:
            if box.conf[0] > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                class_name = class_names[cls]
                conf = float(box.conf[0])
                
                # Get tracking ID if available
                # Classification for this bottle region
                class_label, class_conf = classify_roi(frame, x1, y1, x2, y2)
                class_part = f"{class_label}:{class_conf:.2f}" if class_label else "cls:N/A"

                if box.id is not None:
                    track_id = int(box.id[0])
                    
                    # Map to consecutive ID (no gaps)
                    if track_id not in id_mapping:
                        id_mapping[track_id] = next_consecutive_id
                        next_consecutive_id += 1
                    
                    display_id = id_mapping[track_id]
                    colour = getTrackingColour(display_id)
                    label = f"ID:{display_id} {class_name} {conf:.2f} {class_part}"
                else:
                    colour = getColours(cls)
                    label = f"{class_name} {conf:.2f} {class_part}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
                cv2.putText(frame, label,
                            (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, colour, 3)

                # Fetch or init ID record
                if box.id is not None:
                    track_id = int(box.id[0])
                    if track_id not in id_mapping:
                        id_mapping[track_id] = next_consecutive_id
                        next_consecutive_id += 1
                    display_id = id_mapping[track_id]
                    if display_id not in id_data:
                        id_data[display_id] = {
                            'status': class_label if class_label else 'N/A',
                            'tarra': '',
                            'year': '',
                            'first_frame': frame_count
                        }
                else:
                    # Skip logging for objects without tracking ID
                    display_id = -1

                # Prepare containers for tarra/year in this iteration (will merge into id_data)
                tarra_found = ''
                year_found = ''

                # Run text detection model inside the bottle ROI
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                text_results = text_yolo(roi, conf=0.35, verbose=False)
                for tr in text_results:
                    if not hasattr(tr, 'boxes') or len(tr.boxes) == 0:
                        continue
                    for tbox in tr.boxes:
                        tconf = float(tbox.conf[0]) if hasattr(tbox, 'conf') else 1.0
                        if tconf < 0.35:
                            continue
                        tx1, ty1, tx2, ty2 = map(int, tbox.xyxy[0])
                        # Clamp inside ROI
                        tx1 = max(0, tx1); ty1 = max(0, ty1)
                        tx2 = min(roi.shape[1]-1, tx2); ty2 = min(roi.shape[0]-1, ty2)
                        crop = roi[ty1:ty2, tx1:tx2]
                        raw_text = ocr_read(crop).strip()
                        # Extract tarra (weight) and year if not already found locally
                        if raw_text:
                            norm_text = raw_text.replace(',', '.').upper()
                            if not tarra_found:
                                w_match = re.search(r'\b(\d{1,2}(?:\.\d)?)\s*KG\b', norm_text)
                                if w_match:
                                    try:
                                        wf = float(w_match.group(1))
                                        if 5.0 <= wf <= 30.0:
                                            tarra_found = f"{wf:.1f} KG"
                                    except ValueError:
                                        pass
                            if not year_found:
                                y_match = re.search(r'\b20\d{2}\b', raw_text)
                                if y_match:
                                    y_int = int(y_match.group(0))
                                    if 2010 <= y_int <= 2099:
                                        year_found = str(y_int)
                        # Absolute coords on original frame
                        ax1, ay1, ax2, ay2 = x1 + tx1, y1 + ty1, x1 + tx2, y1 + ty2
                        inner_colour = (0, 255, 255)  # Distinct colour for text boxes
                        cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), inner_colour, 2)
                        display_text = raw_text[:28] + ('…' if len(raw_text) > 28 else '') if raw_text else 'NaN'
                        cv2.putText(frame, display_text, (ax1, max(ay1 - 6, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, inner_colour, 1, cv2.LINE_AA)

                # Update stored record if changes detected
                if display_id != -1:
                    rec = id_data[display_id]
                    # Classification change
                    if class_label and class_label != rec['status']:
                        rec['status'] = class_label
                    # Tarra newly found
                    if tarra_found and not rec['tarra']:
                        rec['tarra'] = tarra_found
                    # Year newly found
                    if year_found and not rec['year']:
                        rec['year'] = year_found
    
    # Write the frame to output video
    out.write(frame)
    frame_count += 1

videoCap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processed {frame_count} frames. Output saved to videos/output/output_tracked2.mp4")

# Write final per-ID summary text file
summary_path = 'videos/output/bottle_detection_summary.txt'
with open(summary_path, 'w', encoding='utf-8') as f:
    for did in sorted(id_data.keys()):
        rec = id_data[did]
        status = rec['status']
        tarra = rec['tarra'] or 'N/A'
        year = rec['year'] or 'N/A'
        # Decision logic (final state)
        decision = 'keep'
        if status == 'nok':
            decision = 'push away'
        else:
            if rec['year'] and rec['year'].isdigit() and int(rec['year']) < 2025:
                decision = 'push away'
        line = f"{did}, {status}, {tarra}, {year}, {decision}"
        f.write(line + '\n')
print(f"Per-ID summary written to {summary_path}")