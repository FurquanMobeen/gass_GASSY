import cv2
import random
import os
from pathlib import Path
from ultralytics import YOLO

# Load model
yolo = YOLO("models/yolo11n_bottles.pt")

def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

video_path = "videos/14_55_top_cropped_trimmed.mp4"
videoCap = cv2.VideoCapture(video_path)

# Get video name without extension
video_name = Path(video_path).stem

# Create output folder for crops
crops_folder = Path("crops") / video_name
crops_folder.mkdir(parents=True, exist_ok=True)

# Setup video writer
# NOTE: The output video will be a TIMELAPSE (only containing the processed frames)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('videos/output/14_43_back_left_fast_summary.mp4', fourcc, 5.0, 
                      (int(videoCap.get(3)), int(videoCap.get(4))))

# Configuration
total_frames = int(videoCap.get(cv2.CAP_PROP_FRAME_COUNT))
CROP_INTERVAL = 20  # Process 1 frame, skip 19

frame_count = 0
crop_count = 0
processed_count = 0

print(f"Starting HIGH SPEED processing. Only checking every {CROP_INTERVAL}th frame...")

while True:
    ret, frame = videoCap.read()
    if not ret:
        break

    # SPEED OPTIMIZATION:
    # If this is not the 20th frame, skip EVERYTHING and continue to next loop
    if frame_count % CROP_INTERVAL != 0:
        frame_count += 1
        continue

    # --- The code below only runs once every 20 frames ---
    
    # Use predict instead of track (tracking breaks when skipping frames)
    results = yolo.predict(frame, stream=True, verbose=False)
    
    has_detections = False
    
    for result in results:
        class_names = result.names
        for box in result.boxes:
            if box.conf[0] > 0.4:
                has_detections = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                class_name = class_names[cls]
                conf = float(box.conf[0])
                colour = getColours(cls)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}",
                            (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, colour, 2)
                
                # Save the crop
                # Ensure coordinates are within frame bounds
                h, w, _ = frame.shape
                y1_c, y2_c = max(0, y1), min(h, y2)
                x1_c, x2_c = max(0, x1), min(w, x2)
                
                cropped_img = frame[y1_c:y2_c, x1_c:x2_c]
                
                if cropped_img.size > 0:
                    crop_filename = crops_folder / f"frame_{frame_count:06d}_crop_{crop_count:04d}.jpg"
                    cv2.imwrite(str(crop_filename), cropped_img)
                    crop_count += 1

    # Only write the processed frame to the video
    out.write(frame)
    processed_count += 1
    
    frame_count += 1
    if frame_count % 100 == 0:
        print(f"Scanned {frame_count}/{total_frames} frames...")

videoCap.release()
out.release()
print(f"\nProcessing complete!")
print(f"Total frames scanned: {frame_count}")
print(f"Actual frames processed with YOLO: {processed_count}")
print(f"Total crops saved: {crop_count}")
print(f"Output video (Timelapse) saved to: videos/output/14_43_back_left_fast_summary.mp4")