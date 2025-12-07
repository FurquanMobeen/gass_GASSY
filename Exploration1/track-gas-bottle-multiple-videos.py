import cv2
import numpy as np
import random
from ultralytics import YOLO


# Load YOLO model
yolo = YOLO("models/yolo11x_finetuned_bottles_on_site_v3.pt")

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

def getColours(cls_num):
    cls_num = int(cls_num)  # FIXED: ensure Python int
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

def compute_hist(frame, x1, y1, x2, y2):
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1], None, [16,16], [0,180,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def hist_distance(h1, h2):
    if h1 is None or h2 is None:
        return 0
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)

def create_grid(frames, rows=2, cols=2, width=640, height=360):
    resized = []
    for frame in frames:
        if frame is None:
            frame = np.zeros((height, width,3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (width, height))
        resized.append(frame)
    while len(resized) < rows*cols:
        resized.append(np.zeros((height, width,3), dtype=np.uint8))
    grid_rows = []
    for r in range(rows):
        row_frames = resized[r*cols:(r+1)*cols]
        grid_rows.append(np.hstack(row_frames))
    return np.vstack(grid_rows)


# Video capture
caps = [cv2.VideoCapture(p) for p in video_paths]
active = [True]*len(video_paths)

# Global ID tracking
next_global_id = 1
bottle_history = {}  # gid -> list of records

# Matching thresholds
HIST_THRESH = 0.75
SIZE_THRESH = 0.3

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
                results = yolo.track(frame, conf=0.4, tracker="bytetrack.yaml", persist=True, stream=False)
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
            confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes,"conf") else np.ones(len(xyxy))
            clses = res.boxes.cls.cpu().numpy().astype(int) if hasattr(res.boxes,"cls") else np.zeros(len(xyxy))

            for (x1,y1,x2,y2), conf, cls in zip(xyxy, confs, clses):
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                if conf < 0.4:
                    continue

                w, h = x2-x1, y2-y1
                hist = compute_hist(frame, x1, y1, x2, y2)

                # Match with existing bottles globally
                gid = None
                for old_gid, records in bottle_history.items():
                    for r in records:
                        size_diff = abs(w*h - r['w']*r['h']) / max(w*h, r['w']*r['h'])
                        corr = hist_distance(hist, r['hist'])
                        if corr > HIST_THRESH and size_diff < SIZE_THRESH:
                            gid = old_gid
                            r['hist'] = hist
                            r['w'], r['h'] = w, h
                            r['last_seen'] = frame_count
                            if cam_idx not in r['cams']:
                                r['cams'].append(cam_idx)
                            break
                    if gid is not None:
                        break

                # New bottle
                if gid is None:
                    gid = next_global_id
                    next_global_id += 1
                    bottle_history[gid] = [{
                        'hist': hist,
                        'w': w,
                        'h': h,
                        'last_seen': frame_count,
                        'cams': [cam_idx]
                    }]

                # Draw bounding box & ID (from first script)
                colour = getColour(gid)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 5)

                text = f"ID {gid}"
                font_scale = max(1.2, h / 100)
                thickness = int(max(2, font_scale*2))

                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                text_x = x1 + (w - text_width)//2
                text_y = y1 + text_height + 5

                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, colour, thickness)

                # --- Drawing Bounding Boxes --- (from second script)
                class_name = class_names[cls] if cls in class_names else str(cls)
                conf_label = f"{class_name} {conf:.2f}"
                cls_colour = getColours(cls)
                cv2.rectangle(frame, (x1, y1), (x2, y2), cls_colour, 2)
                cv2.putText(frame, conf_label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cls_colour, 2)

    # 3. Show all cameras in grid
    grid_frame = create_grid(frames, rows=2, cols=2, width=640, height=360)
    cv2.imshow("All Cameras Grid", grid_frame)

    # Display the frame in a separate window for each video (from second script)
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
