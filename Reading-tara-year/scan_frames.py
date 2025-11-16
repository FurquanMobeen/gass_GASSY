from ultralytics import YOLO
import cv2
import easyocr
import numpy as np
import re
import os
from pathlib import Path

# Load model and OCR
model = YOLO("runs/detect/yolo_training/weights/best.pt")
reader = easyocr.Reader(['en'])

# Get all frame images
image_dir = Path("yolo_dataset/images")
image_files = sorted(image_dir.glob("frame_*.jpg"))

print(f"Found {len(image_files)} frames to scan\n")
print("=" * 80)

interesting_frames = []

for idx, image_path in enumerate(image_files):
    print(f"\rScanning {idx+1}/{len(image_files)}: {image_path.name}", end="", flush=True)
    
    img = cv2.imread(str(image_path))
    
    # Quick enhancement
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_full = clahe.apply(gray_full)
    img_enhanced = cv2.cvtColor(enhanced_full, cv2.COLOR_GRAY2BGR)
    
    # Run YOLO
    results = model(img_enhanced, verbose=False)
    
    for r in results:
        boxes = r.boxes.xyxy
        confs = r.boxes.conf
        
        if len(boxes) == 0:
            continue
            
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, box)
            crop = img_enhanced[y1:y2, x1:x2]
            
            # Quick OCR preprocessing
            if len(crop.shape) == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = crop
            
            inverted = cv2.bitwise_not(gray)
            upscaled = cv2.resize(inverted, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # OCR
            ocr_results = reader.readtext(upscaled, detail=1, paragraph=False)
            
            if ocr_results:
                raw_text = " ".join([res[1] for res in ocr_results])
                avg_conf = sum([res[2] for res in ocr_results]) / len(ocr_results)
                
                # Extract key patterns
                weight_pattern = r'(\d+[.,]\d+)'
                kg_pattern = r'KG[-\s]?(\d{4})'
                
                weights = re.findall(weight_pattern, raw_text)
                kg_numbers = re.findall(kg_pattern, raw_text, re.IGNORECASE)
                
                # Frame is interesting if it has both weight and KG number
                if weights and kg_numbers:
                    interesting_frames.append({
                        'file': image_path.name,
                        'text': raw_text,
                        'weight': weights[0],
                        'kg_number': f"KG-{kg_numbers[0]}",
                        'confidence': avg_conf,
                        'yolo_conf': float(conf)
                    })

print("\n" + "=" * 80)
print(f"\nFound {len(interesting_frames)} interesting frames with good detections:\n")

# Sort by confidence
interesting_frames.sort(key=lambda x: x['confidence'], reverse=True)

# Display top results
for i, frame in enumerate(interesting_frames[:30], 1):
    print(f"{i:2d}. {frame['file']:20s} | {frame['weight']:6s} {frame['kg_number']:10s} | "
          f"OCR: {frame['confidence']:.3f} | YOLO: {frame['yolo_conf']:.2f}")
    print(f"    Raw: {frame['text'][:70]}")

# Save full list to file
with open("interesting_frames.txt", "w") as f:
    f.write(f"Found {len(interesting_frames)} frames with text detections\n")
    f.write("=" * 80 + "\n\n")
    for frame in interesting_frames:
        f.write(f"{frame['file']}: {frame['weight']} {frame['kg_number']} "
                f"(OCR: {frame['confidence']:.3f}, YOLO: {frame['yolo_conf']:.2f})\n")
        f.write(f"  Raw: {frame['text']}\n\n")

print(f"\nFull list saved to: interesting_frames.txt")
print(f"\nRecommended frames to test: {', '.join([f['file'] for f in interesting_frames[:10]])}")
