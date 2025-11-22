import cv2
import random
import os
from pathlib import Path
from ultralytics import YOLO

yolo = YOLO("models/tank_detection/best.pt")

def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

video_path = "videos/input/14_43_front_cropped.mp4"
videoCap = cv2.VideoCapture(video_path)

# Get video name without extension
video_name = Path(video_path).stem

# Create output folder for crops
crops_folder = Path("crops") / video_name
crops_folder.mkdir(parents=True, exist_ok=True)

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('videos/output/14_43_back_left_cropped.mp4', fourcc, 20.0,
                      (int(videoCap.get(3)), int(videoCap.get(4))))

# Get total number of frames to calculate skip interval
total_frames = int(videoCap.get(cv2.CAP_PROP_FRAME_COUNT))
target_crops = 500
skip_interval = max(1, total_frames // target_crops)  # Calculate how many frames to skip

frame_count = 0
crop_count = 0

while True:
    ret, frame = videoCap.read()
    if not ret:
        break
    results = yolo.track(frame, stream=True)
    
    for result in results:
        class_names = result.names
        for box in result.boxes:
            if box.conf[0] > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                class_name = class_names[cls]
                conf = float(box.conf[0])
                colour = getColours(cls)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}",
                            (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, colour, 2)
                
                # Only save crop every skip_interval frames to reach ~500 total
                if frame_count % skip_interval == 0 and crop_count < target_crops:
                    # Crop the detected region
                    cropped_img = frame[y1:y2, x1:x2]
                    
                    # Save the cropped image
                    crop_filename = crops_folder / f"frame_{frame_count:06d}_crop_{crop_count:04d}.jpg"
                    cv2.imwrite(str(crop_filename), cropped_img)
                    crop_count += 1
    
    # Write the frame to output video
    out.write(frame)
    
    frame_count += 1
    if frame_count % 30 == 0:  # Print progress every 30 frames
        print(f"Processed {frame_count} frames")

videoCap.release()
out.release()
print(f"\nProcessing complete!")
print(f"Output video saved to: videos/output/14_43_back_left_cropped.mp4")
print(f"Total frames in video: {total_frames}")
print(f"Skip interval used: {skip_interval}")
print(f"Total frames processed: {frame_count}")
print(f"Total crops saved: {crop_count}")
print(f"Crops saved in: {crops_folder}")