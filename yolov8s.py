import cv2
import random
from ultralytics import YOLO

yolo = YOLO("yolov8s.pt")

def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

video_path = "video-2.mp4"
videoCap = cv2.VideoCapture(video_path)

frame_count = 0

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
    
    # Display the frame
    cv2.imshow('YOLO Object Detection', frame)
    
    # Break on 'q' key press or after 20 frames
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_count += 1

videoCap.release()


#I had the problem with google.colab.patches so I had to think of a workaround.