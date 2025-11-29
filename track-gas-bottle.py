import cv2
import random
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image
import timm
import easyocr
import numpy as np
import re
import os
import csv
from datetime import datetime
from collections import defaultdict, Counter
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Get current year
CURRENT_YEAR = datetime.now().year

# Load YOLO model for detection
yolo = YOLO("models/tank_detection/best.pt")

# Load YOLO model for text detection (tarra weight and year)
text_yolo = YOLO("models/text_detection/new_yolo11s_extract_tarra_weights.pt")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Load EfficientNet / ConvNeXt classifier
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the classifier checkpoint (direct state dict)
checkpoint = torch.load("models/tank_classifier/convnextv2_base_trained.pth", map_location=device)

# Create ConvNeXtV2 model with 2 classes using timm
classifier = timm.create_model('convnextv2_base', pretrained=False, num_classes=2)

# Load the state dict
classifier.load_state_dict(checkpoint)
classifier.to(device)
classifier.eval()

# Define class names for the binary classifier
classifier_class_names = ['nok', 'ok']

# Define transforms for the classifier (adjust these based on your training setup)
classifier_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

def getTrackingColour(track_id):
    random.seed(int(track_id))
    return tuple(random.randint(0, 255) for _ in range(3))

def classify_gas_bottle(frame, x1, y1, x2, y2):
    # Extract the region of interest and guard bounds
    h, w = frame.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    if x2c <= x1c or y2c <= y1c:
        return 0, 0.0
    roi = frame[y1c:y2c, x1c:x2c]
    try:
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    except Exception:
        return 0, 0.0
    pil_image = Image.fromarray(roi_rgb)
    input_tensor = classifier_transforms(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = classifier(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = float(torch.max(probabilities).item())
    return predicted_class, confidence

# ------------------- Updated OCR logic -------------------
ocr_memory = defaultdict(list)
MAX_MEMORY = 20   # how many OCR results to store per tracked bottle

def extract_text_with_ocr(bottle_roi, box_id=None):
    """
    New OCR logic for thin stamped text with rotations and CLAHE preprocessing.
    Stores recent results per tracked bottle for stability.
    Returns: stable_text (str), final_weight (str or None), final_year (str or None)
    """
    if bottle_roi is None or bottle_roi.size == 0:
        return "", None, None

    # Ensure valid ROI
    try:
        h, w = bottle_roi.shape[:2]
        if h == 0 or w == 0:
            return "", None, None
    except Exception:
        return "", None, None

    try:
        text_results = text_yolo(bottle_roi)
    except Exception:
        # If text_yolo fails for some reason, return empty
        return "", None, None

    final_weight = None
    final_year = None
    detected_text_combined = ""

    for text_res in text_results:
        if not hasattr(text_res, "boxes") or len(text_res.boxes) == 0:
            continue

        for tbox in text_res.boxes:
            try:
                tconf = float(tbox.conf[0]) if hasattr(tbox, "conf") and tbox.conf is not None else 1.0
            except Exception:
                tconf = 1.0
            if tconf < 0.4:
                continue

            try:
                tx1, ty1, tx2, ty2 = map(int, tbox.xyxy[0])
            except Exception:
                # skip malformed box
                continue

            # guard coordinates
            tx1c, ty1c = max(0, tx1), max(0, ty1)
            tx2c, ty2c = min(w, tx2), min(h, ty2)
            if tx2c <= tx1c or ty2c <= ty1c:
                continue

            text_roi = bottle_roi[ty1c:ty2c, tx1c:tx2c]
            if text_roi is None or text_roi.size == 0:
                continue

            try:
                # Preprocessing
                gray = cv2.cvtColor(text_roi, cv2.COLOR_BGR2GRAY)
            except Exception:
                continue

            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            blur = cv2.GaussianBlur(enhanced, (3, 3), 0)

            candidate_texts = []
            angles = [0, -8, 8, -12, 12]   # small rotations help with skewed bottles

            for angle in angles:
                center = (blur.shape[1]//2, blur.shape[0]//2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(blur, M, (blur.shape[1], blur.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

                # Light normalization + mild high-pass to highlight stamped characters
                norm = cv2.normalize(rotated, None, 0, 255, cv2.NORM_MINMAX)
                hp = cv2.addWeighted(norm, 1.5, cv2.GaussianBlur(norm, (31, 31), 0), -0.5, 0)

                try:
                    ocr_text = reader.readtext(hp, detail=0, allowlist='0123456789KGkg.,- ')
                except Exception:
                    ocr_text = []
                candidate_texts.append(" ".join(ocr_text))

            combined_text = " ".join(candidate_texts).strip()
            if combined_text:
                detected_text_combined += combined_text + " "

    # Merge OCR attempts across frames for this track id
    stable_text = ""
    if box_id is not None and detected_text_combined.strip():
        try:
            ocr_memory[box_id].append(detected_text_combined.strip())
            if len(ocr_memory[box_id]) > MAX_MEMORY:
                ocr_memory[box_id] = ocr_memory[box_id][-MAX_MEMORY:]
            stable_text = Counter(ocr_memory[box_id]).most_common(1)[0][0]
        except Exception:
            stable_text = detected_text_combined.strip()
    else:
        stable_text = detected_text_combined.strip()

    # Extract weight pattern (e.g., 10.8 or 10,8)
    w_match = re.search(r"(\d{1,2}[.,]\d)", stable_text)
    if w_match:
        try:
            value = float(w_match.group(1).replace(',', '.'))
            if 8.0 <= value <= 20.0:
                final_weight = f"{value:.1f}"
        except Exception:
            final_weight = None

    # Extract year pattern like 20xx
    y_match = re.search(r"(20\d{2})", stable_text)
    if y_match:
        final_year = y_match.group(1)

    return stable_text, final_weight, final_year
# ------------------- End Updated OCR logic -------------------

# video_path = "videos/input/14_43/14_43_back_left_cropped.mp4"
# output_path = 'videos/output/14_43_back_left_cropped.mp4'

video_path = "videos/input/14_43/14_43_back_right_cropped.mp4"
output_path = 'videos/output/14_43_back_right_cropped.mp4'

# video_path = "videos/input/14_43/14_43_front_cropped.mp4"
# output_path = 'videos/output/14_43_front_cropped.mp4'

# video_path = "videos/input/14_43/14_43_top_cropped.mp4"
# output_path = 'videos/output/14_43_top_cropped.mp4'

# Delete existing output file if it exists
if os.path.exists(output_path):
    os.remove(output_path)
    print(f"Deleted existing output file: {output_path}")

videoCap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0,
                      (int(videoCap.get(3)), int(videoCap.get(4))))

frame_count = 0
id_mapping = {}
next_consecutive_id = -2
# Store OCR data bound to each tracking ID
id_ocr_data = {}  # {display_id: {"tarra": "weight", "year": "year", "classification": "ok/nok", "confidence": 0.xx}}

def save_csv_data():
    """Save tracking data to CSV file"""
    csv_output_path = output_path.replace('.mp4', '_tracking_data.csv')
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['ID', 'Classification', 'Confidence', 'Tarra Weight', 'Year']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for display_id in sorted(id_ocr_data.keys()):
            data = id_ocr_data[display_id]
            writer.writerow({
                'ID': display_id,
                'Classification': data['classification'] if data['classification'] else 'N/A',
                'Confidence': f"{data['confidence']:.3f}" if data['confidence'] > 0 else 'N/A',
                'Tarra Weight': data['tarra'] if data['tarra'] else 'N/A',
                'Year': data['year'] if data['year'] else 'N/A'
            })
    
    print(f"\nTracking data saved to {csv_output_path}")
    print(f"Total bottles tracked: {len(id_ocr_data)}")
    return csv_output_path

# ------------------- Optional: Load ground truth from Excel (if available) -------------------
ground_truth_path = "groundtruth.xlsx"
ground_truth = {}
if os.path.exists(ground_truth_path):
    try:
        df_gt = pd.read_excel(ground_truth_path)
        for _, row in df_gt.iterrows():
            try:
                gt_id = int(row['id'])
                gt_label = str(row['Primagaz_status']).strip().lower()
                if gt_label not in ('ok','nok'):
                    # Normalize some variants if present
                    gt_label = 'ok' if 'ok' in gt_label else ('nok' if 'nok' in gt_label else gt_label)
                ground_truth[gt_id] = gt_label
            except Exception:
                continue
    except Exception:
        ground_truth = {}
else:
    ground_truth = {}

try:
    while True:
        ret, frame = videoCap.read()
        if not ret:
            break

        frame_count += 1

        # Perform tracking; lowered conf can help detect more intermittent bottles
        try:
            results = yolo.track(frame, conf=0.4, tracker="bytetrack.yaml", persist=True, stream=True)
        except Exception as e:
            # If tracking call fails for a frame, print and continue to next frame
            print(f"Warning: YOLO.track failed on frame {frame_count}: {e}")
            continue
        
        for result in results:
            class_names = getattr(result, "names", {})
            if not hasattr(result, "boxes") or len(result.boxes) == 0:
                continue

            for box in result.boxes:
                # Wrap per-box processing in try/except so one bad box doesn't crash everything
                try:
                    # Safely get confidence, class, bbox
                    try:
                        conf = float(box.conf[0]) if hasattr(box, "conf") and box.conf is not None else 1.0
                    except Exception:
                        conf = 1.0

                    if conf < 0.4:
                        # ignore low confidence
                        continue

                    try:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                    except Exception:
                        # skip malformed box coordinates
                        continue

                    try:
                        cls = int(box.cls[0]) if hasattr(box, "cls") and box.cls is not None else 0
                    except Exception:
                        cls = 0

                    class_name = class_names.get(cls, cls) if isinstance(class_names, dict) else class_names[cls] if hasattr(class_names, '__getitem__') else cls

                    # Classify the detected gas bottle
                    classifier_class, classifier_conf = classify_gas_bottle(frame, x1, y1, x2, y2)
                    classifier_label = classifier_class_names[classifier_class] if classifier_class < len(classifier_class_names) else f"Class_{classifier_class}"

                    # Extract ROI safely
                    h_frame, w_frame = frame.shape[:2]
                    x1c, y1c = max(0, x1), max(0, y1)
                    x2c, y2c = min(w_frame, x2), min(h_frame, y2)
                    roi = frame[y1c:y2c, x1c:x2c] if x2c > x1c and y2c > y1c else np.zeros((1,1,3), dtype=np.uint8)

                    # SAFE TRACK ID HANDLING: accept tracker id if present, otherwise assign consecutive id
                    track_id = None
                    if hasattr(box, "id") and box.id is not None:
                        try:
                            # box.id could be a tensor/array-like; try to extract first element
                            candidate = box.id[0] if hasattr(box.id, '__len__') else box.id
                            track_id = int(candidate)
                        except Exception:
                            # fallback attempts
                            try:
                                track_id = int(box.id)
                            except Exception:
                                track_id = None

                    if track_id is None:
                        # Assign a new unique id when tracker gives None
                        track_id = next_consecutive_id
                        next_consecutive_id += 1

                    # Map tracker id to display id (keeps existing behaviour)
                    if track_id not in id_mapping:
                        id_mapping[track_id] = next_consecutive_id
                        next_consecutive_id += 1

                    display_id = id_mapping[track_id]

                    # Initialize OCR data storage for this ID if not exists
                    if display_id not in id_ocr_data:
                        id_ocr_data[display_id] = {"tarra": "", "year": "", "classification": "", "confidence": 0.0}

                    # Update classification for this ID (keep highest confidence)
                    if classifier_conf > id_ocr_data[display_id]["confidence"]:
                        id_ocr_data[display_id]["classification"] = classifier_label
                        id_ocr_data[display_id]["confidence"] = classifier_conf

                    colour = getTrackingColour(display_id)
                    label = f"ID:{display_id} {class_name} {conf:.2f}"

                    # ------------------- Use new OCR logic -------------------
                    stable_text, stable_weight, stable_year = extract_text_with_ocr(roi, track_id)
                    if display_id is not None:
                        if stable_weight:
                            id_ocr_data[display_id]["tarra"] = stable_weight
                        if stable_year:
                            # Only update year if it's lower than the stored one, or if not set yet
                            prev_year = id_ocr_data[display_id]["year"]
                            if not prev_year or (prev_year.isdigit() and int(stable_year) < int(prev_year)):
                                id_ocr_data[display_id]["year"] = stable_year
                                if int(stable_year) < CURRENT_YEAR:
                                    id_ocr_data[display_id]["classification"] = "nok (expired)"
                    # ------------------- End OCR integration -------------------

                    # Draw bounding box and labels
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)

                    frame_height, frame_width = frame.shape[:2]

                    # Stack labels vertically above box
                    line_height = 45
                    y_text_line2 = max(line_height * 2, y1 - 10 - line_height)
                    x_text = max(5, min(x1, frame_width - 300))

                    display_classification = classifier_label
                    if display_id is not None and display_id in id_ocr_data:
                        display_classification = id_ocr_data[display_id]["classification"] if id_ocr_data[display_id]["classification"] else classifier_label

                    cv2.putText(frame, f"{display_classification} {classifier_conf:.2f}",
                                (x_text, y_text_line2), cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (0, 255, 0), 3)

                    y_text_line1 = max(line_height, y1 - 10)
                    cv2.putText(frame, label,
                                (x_text, y_text_line1), cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, colour, 3)

                    if display_id is not None and display_id in id_ocr_data:
                        ocr_data = id_ocr_data[display_id]
                        y_offset = min(y2 + 35, frame_height - 50)  # Start below the bottom, stay within frame
                        if ocr_data["tarra"]:
                            cv2.putText(frame, f"Tarra: {ocr_data['tarra']}",
                                        (x_text, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                                        1.0, (0, 255, 255), 2)
                            y_offset = min(y_offset + 40, frame_height - 10)
                        if ocr_data["year"]:
                            cv2.putText(frame, f"Year: {ocr_data['year']}",
                                        (x_text, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                                        1.0, (0, 255, 255), 2)

                except Exception as inner_e:
                    # Log and continue with next box
                    print(f"Warning: error processing box on frame {frame_count}: {inner_e}")
                    continue

        out.write(frame)

except KeyboardInterrupt:
    print("\n\nProcessing interrupted by user. Saving data...")
except Exception as e:
    print(f"\n\nError occurred: {e}. Saving data...")
finally:
    videoCap.release()
    out.release()
    print(f"Video saved to {output_path}")
    csv_path = save_csv_data()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ------------------- Confusion Matrix Computation -------------------
if ground_truth:
    y_true = []
    y_pred = []
    for display_id, data in id_ocr_data.items():
        # ground_truth keys are expected to be integers mapping to your known bottle ids.
        # We keep the original logic: only include display_ids that directly match ground_truth keys.
        if display_id in ground_truth:
            y_true.append(ground_truth[display_id])
            pred_label = data["classification"].lower() if data["classification"] else "nok"
            if "nok" in pred_label:
                pred_label = "nok"
            else:
                pred_label = "ok"
            y_pred.append(pred_label)

    if len(y_true) > 0:
        labels = ["ok", "nok"]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

        # Plot and save heatmap
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix of Gas Bottle Classification")
        # Ensure output folder exists
        out_dir = os.path.dirname(output_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), bbox_inches='tight')
        plt.show()
    else:
        print("\nNo matching ground truth entries found for tracked display IDs. Confusion matrix skipped.")
else:
    print("\nNo ground truth file found or ground truth is empty; skipping confusion matrix computation.")
