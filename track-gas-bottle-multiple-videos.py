import cv2
import random
from ultralytics import YOLO

yolo = YOLO("models/yolo11x_finetuned_bottles_on_site_v3.pt")

def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

video_paths = [
    "videos/14_55_front_cropped.mp4",
    "videos/14_55_top_cropped.mp4",
    "videos/14_55_back_left_cropped.mp4",
    "videos/14_55_back_right_cropped.mp4"
]

frame_count = 0

video_caps = [cv2.VideoCapture(path) for path in video_paths]
active_streams = [True] * len(video_paths)
while True:
    all_videos_finished = True
    
    # Iterate through all video capture objects
    for i, videoCap in enumerate(video_caps):
        
        # Only process if the stream is considered active
        if active_streams[i]:
            ret, frame = videoCap.read()
            
            if ret:
                all_videos_finished = False # At least one video is still running
                
                # Perform tracking on the current frame
                # Using stream=False here for clarity, though stream=True works too
                results = yolo.track(frame, persist=True, stream=False) 
                
                # --- Drawing Bounding Boxes ---
                for result in results:
                    class_names = result.names
                    for box in result.boxes:
                        # Ensure confidence is above threshold
                        if box.conf[0] > 0.4:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls = int(box.cls[0])
                            class_name = class_names[cls]
                            conf = float(box.conf[0])
                            colour = getColours(cls)
                            
                            # Draw box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                            
                            # Draw label
                            label = f"{class_name} {conf:.2f}"
                            cv2.putText(frame, label,
                                        (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6, colour, 2)

                # Display the frame in a separate window for each video
                window_name = f'Tracking Gas Bottles - Video {i + 1}'
                cv2.imshow(window_name, frame)
            
            else:
                # Video has ended
                active_streams[i] = False
    
    # 3. Break if all videos have finished
    if all_videos_finished:
        print("All video streams have finished.")
        break
    
    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_count += 1

for videoCap in video_caps:
    videoCap.release()