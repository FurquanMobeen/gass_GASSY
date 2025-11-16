from ultralytics import YOLO
import cv2
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt
import numpy as np

# Initialize YOLO model
model = YOLO("runs/detect/yolo_training/weights/best.pt")

# Initialize PaddleOCR
ocr = PaddleOCR(lang='en', use_angle_cls=True)

# Load image
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

# Run YOLO detection on the ENHANCED image
results = model(img_enhanced)

for r in results:
    boxes = r.boxes.xyxy
    confs = r.boxes.conf
    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = map(int, box)

        crop = img_enhanced[y1:y2, x1:x2]
        
        # Additional crop preprocessing for PaddleOCR
        # Convert to grayscale if needed
        if len(crop.shape) == 3:
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray_crop = crop
        
        # Invert (white text becomes black)
        inverted = cv2.bitwise_not(gray_crop)
        
        # Upscale for better OCR
        upscaled = cv2.resize(inverted, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        # Final denoising
        denoised_crop = cv2.fastNlMeansDenoising(upscaled, None, h=10, templateWindowSize=7, searchWindowSize=21)

        # Run PaddleOCR on the cropped region
        ocr_results = ocr.predict(crop)
        detected_texts = []

        for line in ocr_results[0]:
            # line[1] can be either a tuple (text, score) or just a string
            if isinstance(line[1], tuple):
                text = line[1][0]
            else:
                text = line[1]
            detected_texts.append(text)

        detected_text = " ".join(detected_texts)
        print(f"Detected Text: {detected_text} (Confidence: {conf:.2f})")

        # Draw bounding box and text on the ORIGINAL image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, detected_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Display result using matplotlib
import matplotlib.pyplot as plt

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("YOLO + PaddleOCR Detections")
plt.show()