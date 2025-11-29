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

# Get current year
CURRENT_YEAR = datetime.now().year

# Load YOLO model for detection
yolo = YOLO("models/tank_detection/best.pt")

# Load YOLO model for text detection (tarra weight and year)
text_yolo = YOLO("models/text_detection/new_yolo11s_extract_tarra_weights.pt")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Load EfficientNet classifier
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the classifier checkpoint (direct state dict)
checkpoint = torch.load("models/tank_classifier/convnextv2_base_trained.pth", map_location=device)

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

def preprocess_for_ocr(crop, enhance_for_year=False):
    """Preprocess image crop for better OCR accuracy"""
    # Convert to grayscale
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    
    # Apply CLAHE for better contrast - stronger for year detection
    clip_limit = 4.0 if enhance_for_year else 3.0
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Denoise - stronger for year detection
    h_value = 12 if enhance_for_year else 10
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=h_value, templateWindowSize=7, searchWindowSize=21)
    
    # Sharpen - stronger for year detection
    if enhance_for_year:
        kernel = np.array([[-1,-1,-1,-1,-1],
                          [-1, 2, 2, 2,-1],
                          [-1, 2, 8, 2,-1],
                          [-1, 2, 2, 2,-1],
                          [-1,-1,-1,-1,-1]]) / 8.0
    else:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # Apply morphological operations for year detection to clean up characters
    if enhance_for_year:
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        sharpened = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel_morph)
    
    # Invert (white text becomes black on white background)
    inverted = cv2.bitwise_not(sharpened)
    
    # Upscale for better OCR - higher scale for years
    scale_factor = 4 if enhance_for_year else 3
    upscaled = cv2.resize(inverted, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    return upscaled

def extract_text_with_ocr(crop, is_year=False, is_weight=False):
    """Extract and clean text from image crop using EasyOCR"""
    # Preprocess the crop with enhanced settings for year or weight if needed
    processed = preprocess_for_ocr(crop, enhance_for_year=(is_year or is_weight))
    
    # Try OCR on processed image with adjusted settings
    if is_year:
        # More strict settings for year detection - only digits
        ocr_results = reader.readtext(processed, detail=1, paragraph=False, 
                                      min_size=10, text_threshold=0.7, 
                                      allowlist='0123456789')
    elif is_weight:
        # Settings for weight detection - digits, decimal point, comma
        ocr_results = reader.readtext(processed, detail=1, paragraph=False, 
                                      min_size=8, text_threshold=0.7, 
                                      allowlist='0123456789.,')
    else:
        ocr_results = reader.readtext(processed, detail=1, paragraph=False, 
                                      min_size=5, text_threshold=0.6)
    
    if not ocr_results:
        return "", 0.0
    
    raw_text = " ".join([res[1] for res in ocr_results])
    avg_conf = sum([res[2] for res in ocr_results]) / len(ocr_results)
    
    # Apply regex to extract structured information
    cleaned_parts = []
    
    # Pattern 1: Weight (e.g., "10.8", "10,8", "70.7") - must be between 10 and 20 kg
    weight_pattern = r'(\d+[.,]\d+)'
    weights = re.findall(weight_pattern, raw_text)
    if weights:
        weight_value = float(weights[0].replace(',', '.'))
        # Only accept weights between 10 and 20 kg
        if 10 <= weight_value <= 20:
            cleaned_parts.append(f"{weight_value:.1f}kg")
    
    # Pattern 2: Year in KG-number format (e.g., "KG-2038", "KG-2036") - must be >= 2025
    kg_pattern = r'KG[-\s]?(\d{4})'
    kg_numbers = re.findall(kg_pattern, raw_text, re.IGNORECASE)
    if kg_numbers:
        year = int(kg_numbers[0])
        # Only accept years >= 2025
        if year >= 2025:
            cleaned_parts.append(f"KG-{year}")
    
    # Pattern 3: P15Y format
    p_pattern = r'P\d{2}Y'
    p_codes = re.findall(p_pattern, raw_text, re.IGNORECASE)
    if p_codes:
        cleaned_parts.append(p_codes[0].upper())
    
    # Also look for standalone 4-digit years (must be >= 2025)
    year_pattern = r'\b(202[5-9]|20[3-9]\d)\b'  # Matches 2025-2099
    standalone_years = re.findall(year_pattern, raw_text)
    if standalone_years and not kg_numbers:  # Only if we didn't find a KG- format
        year = int(standalone_years[0])
        if year >= 2025:
            cleaned_parts.append(f"{year}")
    
    # Only return cleaned text if we found valid data, otherwise return empty string
    cleaned_text = " ".join(cleaned_parts) if cleaned_parts else ""
    
    return cleaned_text, avg_conf

video_path = "videos/input/14_43_back_left_cropped.mp4"
output_path = 'videos/output/14_43_back_left_cropped.mp4'

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

        results = yolo.track(frame, conf=0.4, tracker="bytetrack_custom.yaml", persist=True, stream=True)
        
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
                    # Use the defined classifier class names
                    classifier_label = classifier_class_names[classifier_class] if classifier_class < len(classifier_class_names) else f"Class_{classifier_class}"
                    
                    # Detect text (tarra weight and year) within the detected bottle region
                roi = frame[y1:y2, x1:x2]
                text_results = text_yolo(roi, conf=0.3, verbose=False)
                
                if box.id is not None:
                    track_id = int(box.id[0])
                    
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
                else:
                    colour = getColours(cls)
                    label = f"{class_name} {conf:.2f}"
                    display_id = None
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
                
                # Draw text detection boxes within the bottle region and run OCR
                for text_result in text_results:
                    for text_box in text_result.boxes:
                        if text_box.conf[0] > 0.3:
                            tx1, ty1, tx2, ty2 = map(int, text_box.xyxy[0])
                            # Convert ROI coordinates to frame coordinates
                            abs_tx1, abs_ty1 = x1 + tx1, y1 + ty1
                            abs_tx2, abs_ty2 = x1 + tx2, y1 + ty2
                            
                            text_cls = int(text_box.cls[0])
                            text_class_name = text_result.names[text_cls]
                            text_conf = float(text_box.conf[0])
                            
                            # Extract text crop from ROI
                            text_crop = roi[ty1:ty2, tx1:tx2]
                            
                            # Determine detection type based on class name
                            is_year_detection = "year" in text_class_name.lower() or "recertification" in text_class_name.lower()
                            is_weight_detection = "tara" in text_class_name.lower() or "weight" in text_class_name.lower()
                            
                            # Run OCR on the text region with enhanced processing for years/weights
                            ocr_text, ocr_conf = extract_text_with_ocr(text_crop, is_year=is_year_detection, is_weight=is_weight_detection)
                            
                            # Store OCR data bound to tracking ID (update if better data found)
                            if ocr_text and display_id is not None:
                                # Determine if this is tarra weight or year based on content
                                if "kg" in ocr_text.lower() or any(c.isdigit() and "." in ocr_text for c in ocr_text):
                                    # Likely a weight - extract numeric value
                                    weight_match = re.search(r'(\d+\.?\d*)', ocr_text)
                                    if weight_match:
                                        new_weight = float(weight_match.group(1))
                                        # Update if no weight stored yet, or if new weight is lower
                                        if not id_ocr_data[display_id]["tarra"]:
                                            id_ocr_data[display_id]["tarra"] = ocr_text
                                        else:
                                            # Extract current stored weight value
                                            current_match = re.search(r'(\d+\.?\d*)', id_ocr_data[display_id]["tarra"])
                                            if current_match:
                                                current_weight = float(current_match.group(1))
                                                # Update only if new weight is lower AND difference is more than 10
                                                if new_weight < current_weight and (current_weight - new_weight) > 10:
                                                    id_ocr_data[display_id]["tarra"] = ocr_text
                                elif any(char.isdigit() for char in ocr_text):
                                    # Likely a year - extract numeric value
                                    year_match = re.search(r'(\d{4})', ocr_text)
                                    if year_match:
                                        new_year = int(year_match.group(1))
                                        # Update if no year stored yet, or if new year is lower
                                        if not id_ocr_data[display_id]["year"]:
                                            id_ocr_data[display_id]["year"] = ocr_text
                                            
                                            # Check if year is expired (below current year)
                                            if new_year < CURRENT_YEAR:
                                                id_ocr_data[display_id]["classification"] = "nok (expired)"
                                        else:
                                            # Extract current stored year value
                                            current_match = re.search(r'(\d{4})', id_ocr_data[display_id]["year"])
                                            if current_match:
                                                current_year = int(current_match.group(1))
                                                # Update only if new year is lower AND difference is more than 10
                                                if new_year < current_year and (current_year - new_year) > 10:
                                                    id_ocr_data[display_id]["year"] = ocr_text
                                                    
                                                    # Check if updated year is expired (below current year)
                                                    if new_year < CURRENT_YEAR:
                                                        id_ocr_data[display_id]["classification"] = "nok (expired)"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
                
                # Get frame dimensions
                frame_height, frame_width = frame.shape[:2]
                
                # Stack all labels vertically
                # Labels above the box (working upward)
                line_height = 45
                
                # Calculate text positions with boundary checks
                # Line 2 (classification) - higher up
                y_text_line2 = max(line_height * 2, y1 - 10 - line_height)
                x_text = max(5, min(x1, frame_width - 300))  # Keep within frame bounds
                
                # Get current classification to display
                display_classification = classifier_label
                if display_id is not None and display_id in id_ocr_data:
                    display_classification = id_ocr_data[display_id]["classification"] if id_ocr_data[display_id]["classification"] else classifier_label
                
                # Display classification (line 2 - higher up)
                cv2.putText(frame, f"{display_classification} {classifier_conf:.2f}",
                            (x_text, y_text_line2), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 0), 3)
                
                # Line 1 (ID and detection) - just above box
                y_text_line1 = max(line_height, y1 - 10)
                
                # Display ID and detection (line 1 - just above box)
                cv2.putText(frame, label,
                            (x_text, y_text_line1), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, colour, 3)
                
                # Display OCR data below the box
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
        
        out.write(frame)
        frame_count += 1

except KeyboardInterrupt:
    print("\n\nProcessing interrupted by user. Saving data...")
except Exception as e:
    print(f"\n\nError occurred: {e}. Saving data...")
finally:
    videoCap.release()
    out.release()
    
    # Save tracking data to CSV file
    print(f"Video saved to {output_path}")
    save_csv_data()
    
    # Clean up GPU memory if using CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()