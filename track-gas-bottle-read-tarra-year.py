import cv2
import random
from ultralytics import YOLO
import easyocr

# Initialize YOLO
yolo = YOLO("models/best.pt")

# Initialize EasyOCR
reader = easyocr.Reader(['en'])  # Add languages if needed

# Function to generate colors for each class
def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

# Open video
video_path = "videos/input/14_43_front_cropped.mp4"
videoCap = cv2.VideoCapture(video_path)

frame_count = 0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('videos/output/output_tracked.mp4', fourcc, 20.0,
                      (int(videoCap.get(3)), int(videoCap.get(4))))

while True:
    ret, frame = videoCap.read()
    if not ret:
        break

    results = yolo.track(frame, stream=True)

    ocr_texts = []
    for result in results:
        class_names = result.names
        for box in result.boxes:
            if box.conf[0] > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                class_name = class_names[cls]
                conf = float(box.conf[0])
                colour = getColours(cls)

                crop = frame[y1:y2, x1:x2]
                ocr_results = reader.readtext(crop)
                detected_texts = [res[1] for res in ocr_results]
                detected_text = " | ".join(detected_texts)
                
                if detected_text:
                    ocr_texts.append(detected_text)

                # Draw bounding box and class name
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}",
                            (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, colour, 2)
    
    # Draw OCR text in a separate panel
    if ocr_texts:
        panel_y_start = 100
        panel_height = 60
        cv2.rectangle(frame, (0, panel_y_start), (frame.shape[1], panel_y_start + panel_height), (0, 0, 0), -1)
        y_offset = panel_y_start + 25
        for i, text in enumerate(ocr_texts):
            cv2.putText(frame, f"Text {i+1}: {text}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_offset += 30
            if y_offset > panel_y_start + panel_height - 5:
                break

    # Write the fully annotated frame once
    out.write(frame)
    frame_count += 1

videoCap.release()
out.release()
cv2.destroyAllWindows()
print("Annotated video saved as videos/output/output_tracked.mp4")
