import cv2
import random
from ultralytics import YOLO

yolo = YOLO("models/tank_detection/best.pt")

def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

def getTrackingColour(track_id):
    """Generate consistent color for each tracking ID"""
    random.seed(int(track_id))
    return tuple(random.randint(0, 255) for _ in range(3))

video_path = "videos/input/14_43_back_left_cropped.mp4"
videoCap = cv2.VideoCapture(video_path)

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('videos/output/14_43_back_left_cropped.mp4', fourcc, 20.0,
                      (int(videoCap.get(3)), int(videoCap.get(4))))

frame_count = 0

# ID mapping for consecutive numbering (removes gaps)
id_mapping = {}
next_consecutive_id = 0

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
                
                # Get tracking ID if available
                if box.id is not None:
                    track_id = int(box.id[0])
                    
                    # Map to consecutive ID (no gaps)
                    if track_id not in id_mapping:
                        id_mapping[track_id] = next_consecutive_id
                        next_consecutive_id += 1
                    
                    display_id = id_mapping[track_id]
                    colour = getTrackingColour(display_id)
                    label = f"ID:{display_id} {class_name} {conf:.2f}"
                else:
                    colour = getColours(cls)
                    label = f"{class_name} {conf:.2f}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
                cv2.putText(frame, label,
                            (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, colour, 3)
    
    # Write the frame to output video
    out.write(frame)
    frame_count += 1

videoCap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processed {frame_count} frames. Output saved to videos/output/output_tracked2.mp4")