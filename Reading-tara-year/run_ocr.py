from ultralytics import YOLO
import cv2
import easyocr
import numpy as np
import re

# Load your trained YOLOv11m model
model = YOLO("runs/detect/yolo_training/weights/best.pt")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # add languages if needed, e.g. ['en', 'nl']

# Load the image
image_path = "yolo_dataset/images/frame_179.jpg"
img = cv2.imread(image_path)

# 29,36,44,46,377,179

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

def combine_boxes(boxes, confs, iou_threshold=0.3, distance_threshold=50):
    """Combine overlapping or nearby boxes"""
    if len(boxes) == 0:
        return [], []
    
    boxes_np = boxes.cpu().numpy()
    confs_np = confs.cpu().numpy()
    
    # Calculate centers and dimensions
    centers = np.column_stack([
        (boxes_np[:, 0] + boxes_np[:, 2]) / 2,
        (boxes_np[:, 1] + boxes_np[:, 3]) / 2
    ])
    
    combined = []
    used = set()
    
    for i in range(len(boxes_np)):
        if i in used:
            continue
        
        current_box = boxes_np[i]
        current_conf = confs_np[i]
        group = [i]
        
        for j in range(i + 1, len(boxes_np)):
            if j in used:
                continue
            
            # Check distance between centers
            distance = np.linalg.norm(centers[i] - centers[j])
            
            # Check IoU (Intersection over Union)
            x1 = max(boxes_np[i][0], boxes_np[j][0])
            y1 = max(boxes_np[i][1], boxes_np[j][1])
            x2 = min(boxes_np[i][2], boxes_np[j][2])
            y2 = min(boxes_np[i][3], boxes_np[j][3])
            
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area_i = (boxes_np[i][2] - boxes_np[i][0]) * (boxes_np[i][3] - boxes_np[i][1])
            area_j = (boxes_np[j][2] - boxes_np[j][0]) * (boxes_np[j][3] - boxes_np[j][1])
            union = area_i + area_j - intersection
            iou = intersection / union if union > 0 else 0
            
            # Combine if overlapping or nearby
            if iou > iou_threshold or distance < distance_threshold:
                group.append(j)
        
        # Mark all boxes in group as used
        for idx in group:
            used.add(idx)
        
        # Create combined box from all boxes in group
        group_boxes = boxes_np[group]
        combined_box = [
            np.min(group_boxes[:, 0]),  # min x1
            np.min(group_boxes[:, 1]),  # min y1
            np.max(group_boxes[:, 2]),  # max x2
            np.max(group_boxes[:, 3])   # max y2
        ]
        
        # Use max confidence from group
        combined_conf = np.max(confs_np[group])
        
        combined.append((combined_box, combined_conf))
    
    if combined:
        combined_boxes = np.array([box for box, _ in combined])
        combined_confs = np.array([conf for _, conf in combined])
        return combined_boxes, combined_confs
    else:
        return np.array([]), np.array([])

processed_crops = []  # Store all preprocessed crops
detection_count = 0

for r in results:
    boxes = r.boxes.xyxy  # bounding boxes
    confs = r.boxes.conf  # confidence
    
    # Combine nearby/overlapping boxes
    print(f"Original detections: {len(boxes)}")
    combined_boxes, combined_confs = combine_boxes(boxes, confs)
    print(f"After combining: {len(combined_boxes)}")
    
    for box, conf in zip(combined_boxes, combined_confs):
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
        
        # Try multiple OCR approaches and pick the best result
        all_ocr_results = []
        
        # Approach 1: Inverted and upscaled (current best)
        ocr_results_1 = reader.readtext(denoised, detail=1, paragraph=False, 
                                        min_size=5, text_threshold=0.6)
        all_ocr_results.extend(ocr_results_1)
        
        # Approach 2: Try on the grayscale upscaled (without inversion)
        gray_upscaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        ocr_results_2 = reader.readtext(gray_upscaled, detail=1, paragraph=False,
                                        min_size=5, text_threshold=0.6)
        all_ocr_results.extend(ocr_results_2)
        
        # Combine and deduplicate results, keep highest confidence versions
        ocr_results = all_ocr_results
        
        if ocr_results:
            raw_text = " ".join([res[1] for res in ocr_results])
            avg_conf = sum([res[2] for res in ocr_results]) / len(ocr_results)
            
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
            print(f"  Confidence: OCR={avg_conf:.3f}, YOLO={conf:.2f}")
            
            detected_text = cleaned_text
        else:
            detected_text = ""
            print(f"Detection {detection_count}: No text detected (YOLO conf: {conf:.2f})")
        
        processed_crops.append({
            'original': crop,
            'gray': gray,
            'brightened': inverted,
            'enhanced': upscaled,
            'binary': denoised,
            'denoised': denoised
        })
        
        detection_count += 1

        # Draw the box and text on the ORIGINAL image for visualization
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, detected_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

print(f"\nSaved {detection_count} preprocessed image sets to disk.")
print("Check files: crop_original_X.jpg, crop_brightened_X.jpg, crop_final_X.jpg")

# Show results
import matplotlib.pyplot as plt

# Create a comprehensive visualization showing all preprocessing steps
if len(processed_crops) > 0:
    # Show processing steps for the first detection
    crop_data = processed_crops[0]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].imshow(cv2.cvtColor(crop_data['original'], cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Original Crop")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(crop_data['gray'], cmap='gray')
    axes[0, 1].set_title("2. Grayscale")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(crop_data['brightened'], cmap='gray')
    axes[0, 2].set_title("3. Brightened (alpha=1.5, beta=50)")
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(crop_data['enhanced'], cmap='gray')
    axes[1, 0].set_title("4. CLAHE Enhanced (Contrast)")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(crop_data['binary'], cmap='gray')
    axes[1, 1].set_title("5. Binary + Inverted")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(crop_data['denoised'], cmap='gray')
    axes[1, 2].set_title("6. Final (Denoised) - Sent to OCR")
    axes[1, 2].axis('off')
    
    plt.suptitle("Image Preprocessing Pipeline", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Show final result with detections
fig2, ax = plt.subplots(1, 1, figsize=(12, 10))
ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax.axis('off')
ax.set_title("YOLO + EasyOCR Detections")
plt.tight_layout()
plt.show()