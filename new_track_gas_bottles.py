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

# Get current year
CURRENT_YEAR = datetime.now().year

# Load YOLO model for detection
yolo = YOLO("models/best.pt")

# Load YOLO model for text detection (tarra weight and year)
text_yolo = YOLO("models/new_yolo11s_extract_tarra_weights.pt")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Load EfficientNet classifier
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the classifier checkpoint (direct state dict)
checkpoint = torch.load("models/convnextv2_base_trained.pth", map_location=device)

# Create ConvNeXtV2-Large model with 2 classes using timm
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
    # Extract the region of interest
    roi = frame[y1:y2, x1:x2]
    
    # Convert BGR to RGB
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(roi_rgb)
    
    # Apply transforms
    input_tensor = classifier_transforms(pil_image).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = classifier(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = torch.max(probabilities).item()
    
    return predicted_class, confidence

# ------------------- Updated OCR logic -------------------
ocr_memory = defaultdict(list)
MAX_MEMORY = 20   # how many OCR results to store per tracked bottle

def extract_text_with_ocr(bottle_roi, box_id=None):
    """
    New OCR logic for thin stamped text with rotations and CLAHE preprocessing.
    Stores recent results per tracked bottle for stability.
    """
    if bottle_roi.size == 0:
        return "", None, None
    
    text_results = text_yolo(bottle_roi)
    final_weight = None
    final_year = None
    detected_text_combined = ""
    
    for text_res in text_results:
        if not hasattr(text_res, "boxes") or len(text_res.boxes) == 0:
            continue

        for tbox in text_res.boxes:
            tconf = float(tbox.conf[0]) if hasattr(tbox, "conf") else 1.0
            if tconf < 0.4:
                continue

            tx1, ty1, tx2, ty2 = map(int, tbox.xyxy[0])
            text_roi = bottle_roi[ty1:ty2, tx1:tx2]
            if text_roi.size == 0:
                continue

            # Preprocessing
            gray = cv2.cvtColor(text_roi, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            blur = cv2.GaussianBlur(enhanced, (3,3), 0)

            candidate_texts = []
            angles = [0, -8, 8, -12, 12]

            for angle in angles:
                center = (blur.shape[1]//2, blur.shape[0]//2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(blur, M, (blur.shape[1], blur.shape[0]))

                norm = cv2.normalize(rotated, None, 0, 255, cv2.NORM_MINMAX)
                hp = cv2.addWeighted(norm, 1.5, cv2.GaussianBlur(norm, (31, 31), 0), -0.5, 0)

                ocr_text = reader.readtext(hp, detail=0, allowlist='0123456789KGkg.')
                candidate_texts.append(" ".join(ocr_text))

            combined_text = " ".join(candidate_texts)
            detected_text_combined += combined_text + " "

    # Merge OCR results for the tracked bottle
    if box_id is not None and detected_text_combined.strip():
        ocr_memory[box_id].append(detected_text_combined.strip())
        if len(ocr_memory[box_id]) > MAX_MEMORY:
            ocr_memory[box_id] = ocr_memory[box_id][-MAX_MEMORY:]
        stable_text = Counter(ocr_memory[box_id]).most_common(1)[0][0]
    else:
        stable_text = detected_text_combined.strip()

    # Extract weight
    w_match = re.search(r"(\d{1,2}\.\d)", stable_text)
    if w_match:
        value = float(w_match.group(1))
        if 8.0 <= value <= 20.0:
            final_weight = f"{value:.1f}"

    # Extract year
    y_match = re.search(r"(20\d{2})", stable_text)
    if y_match:
        final_year = y_match.group(1)

    return stable_text, final_weight, final_year
# ------------------- End Updated OCR logic -------------------

video_path = "videos/input/14_55_front_cropped.mp4"
output_path = 'videos/output/14_55_front_cropped.mp4'

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

try:
    while True:
        ret, frame = videoCap.read()
        if not ret:
            break

        results = yolo.track(frame, conf=0.4, tracker="bytetrack.yaml", persist=True, stream=True)
        
        for result in results:
            class_names = result.names
            for box in result.boxes:
                if box.conf[0] > 0.4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    class_name = class_names[cls]
                    conf = float(box.conf[0])
                    
                    # Classify the detected gas bottle
                    classifier_class, classifier_conf = classify_gas_bottle(frame, x1, y1, x2, y2)
                    classifier_label = classifier_class_names[classifier_class] if classifier_class < len(classifier_class_names) else f"Class_{classifier_class}"
                    
                roi = frame[y1:y2, x1:x2]

                track_id = int(box.id[0]) if hasattr(box, "id") else None
                if track_id is not None:
                    if track_id not in id_mapping:
                        id_mapping[track_id] = next_consecutive_id
                        next_consecutive_id += 1
                    display_id = id_mapping[track_id]
                    
                    if display_id not in id_ocr_data:
                        id_ocr_data[display_id] = {"tarra": "", "year": "", "classification": "", "confidence": 0.0}

                    if classifier_conf > id_ocr_data[display_id]["confidence"]:
                        id_ocr_data[display_id]["classification"] = classifier_label
                        id_ocr_data[display_id]["confidence"] = classifier_conf

                    colour = getTrackingColour(display_id)
                    label = f"ID:{display_id} {class_name} {conf:.2f}"
                else:
                    colour = getColours(cls)
                    label = f"{class_name} {conf:.2f}"
                    display_id = None

                # ------------------- Use new OCR logic -------------------
                stable_text, stable_weight, stable_year = extract_text_with_ocr(roi, track_id)
                if display_id is not None:
                    if stable_weight:
                        id_ocr_data[display_id]["tarra"] = stable_weight
                    if stable_year:
                        id_ocr_data[display_id]["year"] = stable_year
                        if int(stable_year) < CURRENT_YEAR:
                            id_ocr_data[display_id]["classification"] = "nok (expired)"
                # ------------------- End OCR integration -------------------

                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)

                frame_height, frame_width = frame.shape[:2]
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
                    y_offset = min(y2 + 35, frame_height - 50)
                    if ocr_data["tarra"]:
                        cv2.putText(frame, f"Tarra: {ocr_data['tarra']}",
                                    (x_text, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0, (0, 255, 255), 2)
                        y_offset = min(y_offset + 40, frame_height - 10)
                    if ocr_data["year"]:
                        cv2.putText(frame, f"Year: {ocr_data['year']}",
                                    (x_text, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0, (0, 255, 255), 2)

        out.write(frame)
        frame_count += 1

except KeyboardInterrupt:
    print("\n\nProcessing interrupted by user. Saving data...")
except Exception as e:
    print(f"\n\nError occurred: {e}. Saving data...")
finally:
    videoCap.release()
    out.release()
    print(f"Video saved to {output_path}")
    save_csv_data()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
