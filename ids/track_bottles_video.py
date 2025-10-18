import cv2
import numpy as np
import random
from ultralytics import YOLO

# -----------------------------
# Load trained YOLOv8 model
# -----------------------------
yolo = YOLO("yolov8n.pt")

video_paths = [
    "videos/14_55_front_cropped.mp4",
    "videos/14_55_top_cropped.mp4",
    "videos/14_55_back_left_cropped.mp4",
    "videos/14_55_back_right_cropped.mp4"
]

# -----------------------------
# Helper functions
# -----------------------------
def getColour(id_num):
    random.seed(id_num)
    return tuple(random.randint(0, 255) for _ in range(3))

def compute_hist(frame, x1, y1, x2, y2):
    """Compute a simple HSV histogram for the region inside the box."""
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def hist_distance(h1, h2):
    """Compute correlation between two histograms."""
    if h1 is None or h2 is None:
        return 0
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)

# -----------------------------
# Video setup
# -----------------------------
video_caps = [cv2.VideoCapture(p) for p in video_paths]
active_streams = [True] * len(video_paths)

# -----------------------------
# Global ID management
# -----------------------------
next_global_id = 1
bottle_history = {}  # global_id -> list of recent positions & histograms

# Matching thresholds
CENTER_THRESHOLD = 50  # pixels
SIZE_THRESHOLD = 0.3   # 30% size difference
HIST_THRESHOLD = 0.7   # histogram correlation

# -----------------------------
# Grid display settings
# -----------------------------
DISPLAY_WIDTH = 320
DISPLAY_HEIGHT = 240

def create_grid(frames, rows=2, cols=2):
    """Combine frames into a grid of size rows x cols."""
    resized_frames = []
    for frame in frames:
        if frame is not None:
            small_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        else:
            small_frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
        resized_frames.append(small_frame)

    while len(resized_frames) < rows * cols:
        resized_frames.append(np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8))

    grid_rows = []
    for r in range(rows):
        row_frames = resized_frames[r*cols:(r+1)*cols]
        row = np.hstack(row_frames)
        grid_rows.append(row)
    grid = np.vstack(grid_rows)
    return grid

# -----------------------------
# Main loop
# -----------------------------
while any(active_streams):
    frames = []
    results_per_video = []

    # Read frames from all videos
    for i, cap in enumerate(video_caps):
        if active_streams[i]:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                results = yolo.track(frame, persist=True, stream=False, tracker="bytetrack.yaml", conf=0.4)
                results_per_video.append(results)
            else:
                frames.append(None)
                results_per_video.append(None)
                active_streams[i] = False
        else:
            frames.append(None)
            results_per_video.append(None)

    # -----------------------------
    # Cross-camera global ID assignment
    # -----------------------------
    for vid_idx, results in enumerate(results_per_video):
        frame = frames[vid_idx]
        if frame is None or results is None:
            continue

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            xyxy = getattr(result.boxes, 'xyxy', None)
            ids = getattr(result.boxes, 'id', None)
            if xyxy is None or ids is None:
                continue

            for box, local_id in zip(xyxy, ids):
                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w//2, y1 + h//2
                hist = compute_hist(frame, x1, y1, x2, y2)

                # Match with existing global IDs
                assigned_gid = None
                for gid, entries in bottle_history.items():
                    for entry in entries:
                        px, py, pw, ph, phist = entry
                        center_dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                        size_diff = abs(w*h - pw*ph) / max(w*h, pw*ph)
                        hist_corr = hist_distance(hist, phist)

                        if center_dist < CENTER_THRESHOLD and size_diff < SIZE_THRESHOLD and hist_corr > HIST_THRESHOLD:
                            assigned_gid = gid
                            break
                    if assigned_gid is not None:
                        break

                # Assign new global ID if no match
                if assigned_gid is None:
                    assigned_gid = next_global_id
                    next_global_id += 1
                    bottle_history[assigned_gid] = []

                # Save history
                bottle_history[assigned_gid].append((cx, cy, w, h, hist))
                if len(bottle_history[assigned_gid]) > 5:
                    bottle_history[assigned_gid] = bottle_history[assigned_gid][-5:]

                # Draw box and ID
                colour = getColour(assigned_gid)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, f"ID {assigned_gid}", (x1, max(y1-10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

    # -----------------------------
    # Display all videos in a 2x2 grid
    # -----------------------------
    grid_frame = create_grid(frames, rows=2, cols=2)
    cv2.imshow("All Cameras - 2x2 Grid", grid_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
for cap in video_caps:
    cap.release()
cv2.destroyAllWindows()
