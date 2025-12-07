import cv2
import numpy as np
import random
from ultralytics import YOLO
import easyocr

# Load YOLO models
bottle_yolo = YOLO("models/yolo11x_finetuned_bottles_on_site_v3.pt")  # For gas bottle tracking
text_yolo = YOLO("models/yolo11s_extract_tarra_weights.pt")  # For text region detection

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Video paths
video_paths = [
    "videos/14_55_front_cropped.mp4",
    "videos/14_55_top_cropped.mp4",
    "videos/14_55_back_left_cropped.mp4",
    "videos/14_55_back_right_cropped.mp4"
]

# Helper functions
def getColour(id_num):
    random.seed(int(id_num))
    return tuple(random.randint(180, 255) for _ in range(3))

def create_grid(frames, rows=2, cols=2, width=640, height=360):
    resized = []
    for frame in frames:
        if frame is None:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (width, height))
        resized.append(frame)
    while len(resized) < rows * cols:
        resized.append(np.zeros((height, width, 3), dtype=np.uint8))
    grid_rows = []
    for r in range(rows):
        row_frames = resized[r * cols:(r + 1) * cols]
        grid_rows.append(np.hstack(row_frames))
    return np.vstack(grid_rows)

# Video capture
caps = [cv2.VideoCapture(p) for p in video_paths]
active = [True] * len(video_paths)

# Global ID tracking
next_global_id = 1
bottle_history = {}  # gid -> list of records

frame_count = 0

# Main loop
while any(active):
    frame_count += 1
    frames = []
    results_list = []
    all_videos_finished = True

    # 1. Read frames
    for i, cap in enumerate(caps):
        if active[i]:
            ret, frame = cap.read()
            if ret:
                all_videos_finished = False  # At least one video is still running
                frames.append(frame)
                results = bottle_yolo.track(frame, conf=0.4, tracker="bytetrack.yaml", persist=True, stream=False)
                results_list.append(results)
            else:
                active[i] = False
                frames.append(None)
                results_list.append(None)
        else:
            frames.append(None)
            results_list.append(None)

    # 2. Process detections
    for cam_idx, results in enumerate(results_list):
        frame = frames[cam_idx]
        if frame is None or results is None:
            continue

        for res in results:
            if not hasattr(res, "boxes") or len(res.boxes) == 0:
                continue

            class_names = res.names
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, "conf") else np.ones(len(xyxy))
            clses = res.boxes.cls.cpu().numpy().astype(int) if hasattr(res.boxes, "cls") else np.zeros(len(xyxy))

            for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clses):
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                if conf < 0.4:
                    continue

                # Draw bounding box for the gas bottle
                colour = getColour(cls)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, f"Bottle {conf:.2f}", (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

                # --- Text Detection with Second YOLO Model ---
                # Crop the gas bottle region for text detection
                bottle_roi = frame[y1:y2, x1:x2]
                if bottle_roi.size == 0:
                    continue

                # Run the second YOLO model to detect text regions
                text_results = text_yolo(bottle_roi)

                for text_res in text_results:
                    if not hasattr(text_res, "boxes") or len(text_res.boxes) == 0:
                        continue

                    text_boxes = text_res.boxes.xyxy.cpu().numpy()
                    text_confs = text_res.boxes.conf.cpu().numpy()

                    for (tx1, ty1, tx2, ty2), tconf in zip(text_boxes, text_confs):
                        tx1, ty1, tx2, ty2 = map(int, (tx1, ty1, tx2, ty2))
                        if tconf < 0.4:
                            continue

                        # Adjust text box coordinates relative to the original frame
                        tx1 += x1
                        ty1 += y1
                        tx2 += x1
                        ty2 += y1

                        # Draw bounding box for the text region
                        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)

                        # Crop the text region for OCR
                        text_roi = frame[ty1:ty2, tx1:tx2]
                        if text_roi.size == 0:
                            continue

                        # Run OCR on the cropped text region
                        ocr_result = reader.readtext(text_roi, detail=0)
                        detected_text = " ".join(ocr_result)

                        # Overlay detected OCR text on the frame
                        if detected_text:
                            cv2.putText(frame, detected_text, (tx1, max(ty2 + 15, 20)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)  # Yellow color for OCR text

    # 3. Show all cameras in grid
    grid_frame = create_grid(frames, rows=2, cols=2, width=640, height=360)
    cv2.imshow("All Cameras Grid", grid_frame)

    # Display the frame in a separate window for each video
    for i, frame in enumerate(frames):
        if frame is not None:
            window_name = f'Tracking Gas Bottles - Video {i + 1}'
            cv2.imshow(window_name, frame)

    # 4. Break if all videos have finished
    if all_videos_finished:
        print("All video streams have finished.")
        break

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
for cap in caps:
    cap.release()
cv2.destroyAllWindows()