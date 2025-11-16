from ultralytics import YOLO
import cv2
import pytesseract
import numpy as np
import re

# Configure Tesseract path (update this after installing Tesseract)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load your trained YOLOv11 model
model = YOLO("runs/detect/yolo_training/weights/best.pt")

# Load an image
image_path = "yolo_dataset/images/frame_377.jpg"
img = cv2.imread(image_path)

# Save original image
cv2.imwrite("original_image.jpg", img)

# Enhance the entire image first before YOLO detection
print("Enhancing full image before detection...")

# Convert to grayscale
gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply CLAHE for better contrast across the whole image
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced_full = clahe.apply(gray_full)

# Denoise the full image
denoised_full = cv2.fastNlMeansDenoising(enhanced_full, None, h=10, templateWindowSize=7, searchWindowSize=21)

# Optional: Apply slight sharpening to make text edges clearer
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpened_full = cv2.filter2D(denoised_full, -1, kernel)

# Convert back to BGR for YOLO (it expects 3 channels)
img_enhanced = cv2.cvtColor(sharpened_full, cv2.COLOR_GRAY2BGR)

cv2.imwrite("enhanced_full_image.jpg", img_enhanced)
print("Full image enhancement complete. Saved to enhanced_full_image.jpg")

# Run inference on the ENHANCED image
results = model(img_enhanced)

# Loop over detections
detection_count = 0
for r in results:
    boxes = r.boxes.xyxy  # bounding box coordinates
    confs = r.boxes.conf  # confidence scores
    classes = r.boxes.cls  # class IDs

    for box, conf, cls in zip(boxes, confs, classes):
        x1, y1, x2, y2 = map(int, box)
        # Crop from the enhanced image
        crop = img_enhanced[y1:y2, x1:x2]
        # Also get crop from original for comparison
        crop_original = img[y1:y2, x1:x2]

        # Save original crop for comparison
        cv2.imwrite(f"crop_original_{detection_count}.jpg", crop_original)
        
        # Crop is already enhanced from full image, now do final OCR preprocessing
        # Convert to grayscale (if not already)
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        
        # Invert (white text becomes black on white background)
        inverted = cv2.bitwise_not(gray)
        
        # Upscale for better OCR accuracy
        upscaled = cv2.resize(inverted, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        # Final denoising on the upscaled crop
        denoised = cv2.fastNlMeansDenoising(upscaled, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        cv2.imwrite(f"crop_processed_{detection_count}.jpg", denoised)

        # OCR with pytesseract
        raw_text = pytesseract.image_to_string(denoised, config="--psm 6").strip()
        
        # Apply regex to extract structured information
        # Pattern 1: Weight (e.g., "10.8", "10,8", "70.7")
        weight_pattern = r'(\d+[.,]\d+)'
        weights = re.findall(weight_pattern, raw_text)
        
        # Pattern 2: KG-number format (e.g., "KG-2038", "KG-2036")
        kg_pattern = r'KG[-\s]?(\d{4})'
        kg_numbers = re.findall(kg_pattern, raw_text, re.IGNORECASE)
        
        # Pattern 3: P15Y format
        p_pattern = r'P\d{2}Y'
        p_codes = re.findall(p_pattern, raw_text, re.IGNORECASE)
        
        # Build cleaned output
        cleaned_parts = []
        if weights:
            # Standardize to dot notation
            cleaned_parts.append(weights[0].replace(',', '.'))
        if kg_numbers:
            cleaned_parts.append(f"KG-{kg_numbers[0]}")
        if p_codes:
            cleaned_parts.append(p_codes[0].upper())
        
        cleaned_text = " ".join(cleaned_parts) if cleaned_parts else raw_text
        
        print(f"Detection {detection_count}:")
        print(f"  Raw OCR: '{raw_text}'")
        print(f"  Cleaned: '{cleaned_text}'")
        print(f"  YOLO conf: {conf:.2f}")
        
        detection_count += 1

        # Display the result on ORIGINAL image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, cleaned_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

# Show output
cv2.imshow("Detections", img)
cv2.waitKey(0)
cv2.destroyAllWindows()