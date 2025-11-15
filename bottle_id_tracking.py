import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import math
import csv

# --------------------------------------------------------------
# VIDEOT
# --------------------------------------------------------------
video_front = "frames/videos/1.mp4"
video_top   = "frames/videos/2.mp4"
video_back  = "frames/videos/3.mp4"

caps = [
    cv2.VideoCapture(video_front),
    cv2.VideoCapture(video_top),
    cv2.VideoCapture(video_back)
]

# --------------------------------------------------------------
# simple color (FRONT-KAMERA)
# --------------------------------------------------------------

def simple_color_name(frame, bbox):
    """Antaa yksinkertaisen värinimen (BLACK, GREEN, BLUE, RED, YELLOW...)."""

    if frame is None or bbox is None:
        return "UNKNOWN"

    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]

    # Clamp
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    # If bbox is too small
    if x2 <= x1 or y2 <= y1:
        return "UNKNOWN"

    # *** Uses only center of the box ***
    bw = x2 - x1
    bh = y2 - y1
    margin_x = int(bw * 0.2)  # 20% off from borders
    margin_y = int(bh * 0.2)

    x1_i = x1 + margin_x
    x2_i = x2 - margin_x
    y1_i = y1 + margin_y
    y2_i = y2 - margin_y

    # checks borders again
    x1_i = max(x1, min(x1_i, x2 - 1))
    x2_i = max(x1_i + 1, min(x2_i, x2))
    y1_i = max(y1, min(y1_i, y2 - 1))
    y2_i = max(y1_i + 1, min(y2_i, y2))

    crop = frame[y1_i:y2_i, x1_i:x2_i]
    if crop.size == 0:
        return "UNKNOWN"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    h_chan = hsv[:, :, 0].astype(np.float32)
    s_chan = hsv[:, :, 1].astype(np.float32)
    v_chan = hsv[:, :, 2].astype(np.float32)

    # uses median(to)
    h_med = float(np.median(h_chan))
    s_med = float(np.median(s_chan))
    v_med = float(np.median(v_chan))



    # Black
    if v_med < 70 and s_med < 80:
        return "BLACK"

    # Gray/white
    if s_med < 40:
        if v_med > 180:
            return "WHITE"
        else:
            return "GREY"

    # Hues
    if h_med < 10 or h_med > 170:
        return "RED"
    elif h_med < 25:
        return "ORANGE"
    elif h_med < 35:
        return "YELLOW"
    elif h_med < 85:
        return "CYAN"
    elif h_med < 170:
        return "BLUE"

    return "UNKNOWN"




# --------------------------------------------------------------
# CSV: ID AND COLOR
# --------------------------------------------------------------
def save_color_to_csv(global_id, color_name, filename="bottle_colors.csv"):
    header = ["global_id", "color"]

    # make file (if not made already)
    try:
        open(filename, "x").write(",".join(header) + "\n")
    except:
        pass  # file already made

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([global_id, color_name])


# --------------------------------------------------------------
# YOLO-MODEL
# --------------------------------------------------------------
model = YOLO("models/yolo11x_finetuned_bottles_on_site_v3.pt")
CONF = 0.40
DISPLAY_W, DISPLAY_H = 480, 270

# --------------------------------------------------------------
# FRONT-CAMERA ID
# (not using YOLO-tracking, because it jumps too much)
# --------------------------------------------------------------
front_stable_id = 1
prev_center = None
FRONT_DISTANCE_THRESHOLD = 100  # px

def get_front_stable_id(frame):
    """
    Etsii suurimman YOLO-boksin, laskee sen keskipisteen,
    vertaa edellisen framen keskipisteeseen.

    Jos etäisyys < threshold → sama pullo
    Jos etäisyys > threshold → uusi pullo (ID++)
    """
    global prev_center, front_stable_id

    results = model(frame, conf=CONF, verbose=False)

    best_box = None
    best_area = 0

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            return None  # no bottle in front camera

        for box in boxes:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_box = (x1, y1, x2, y2)

    if best_box is None:
        return None

    # counts focal point
    x1, y1, x2, y2 = best_box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    center = (cx, cy)

    # first frame
    if prev_center is None:
        prev_center = center
        return front_stable_id

    # distance
    dist = math.dist(center, prev_center)

    if dist < FRONT_DISTANCE_THRESHOLD:
        # same bottle
        prev_center = center
        return front_stable_id
    else:
        # new bottle
        front_stable_id += 1
        prev_center = center
        return front_stable_id


# --------------------------------------------------------------
# TOP/BACK: biggest bbox index
# --------------------------------------------------------------
def biggest_bbox_index(frame):
    results = model(frame, conf=CONF, verbose=False)
    best_idx = None
    best_area = 0

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            return None

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_idx = i

    return best_idx


# --------------------------------------------------------------
# GRID
# --------------------------------------------------------------
def make_grid(frames):
    imgs = []
    for f in frames:
        if f is None:
            f = np.zeros((DISPLAY_H, DISPLAY_W, 3), np.uint8)
        else:
            f = cv2.resize(f, (DISPLAY_W, DISPLAY_H))
        imgs.append(f)
    return np.hstack(imgs)


# --------------------------------------------------------------
# COMBO-LOGIC
# --------------------------------------------------------------
combo_history = deque(maxlen=6)
seen_combos = []
global_bottle_id = 1


# --------------------------------------------------------------
# MAIN LOOP
# --------------------------------------------------------------
while True:
    frames = []
    for cap in caps:
        ret, f = cap.read()
        frames.append(f if ret else None)

    if all(f is None for f in frames):
        break

    # FRONT stable ID
    if frames[0] is not None:
        f_id = get_front_stable_id(frames[0])
    else:
        f_id = None

    # TOP
    if frames[1] is not None:
        t_idx = biggest_bbox_index(frames[1])
    else:
        t_idx = None

    # BACK
    if frames[2] is not None:
        b_idx = biggest_bbox_index(frames[2])
    else:
        b_idx = None

    # COMBO
    combo = [f_id, t_idx, b_idx]

    # Valid combo?
    if any(v is None for v in combo):
        combo_history.clear()
    else:
        combo_history.append(tuple(combo))

        if len(combo_history) == 6 and len(set(combo_history)) == 1:
            stable_combo = combo_history[0]

            # blocks double ids, with same bottle
            front_value = stable_combo[0]
            last_front_value = seen_combos[-1][0] if len(seen_combos) > 0 else None

            if front_value == last_front_value:
                combo_history.clear()
                continue  # same bottle → not new

            if stable_combo not in seen_combos:
                print(f"\nNEW BOTTLE DETECTED → GLOBAL ID {global_bottle_id}")
                print(f"combo = {stable_combo}")
                seen_combos.append(stable_combo)

                # saves the id
                current_id = global_bottle_id
                global_bottle_id += 1

                # ---------------- COLOR EXTRACTION (FRONT) ----------------
                front_bbox = None
                if frames[0] is not None:
                    results_front = model(frames[0], conf=CONF, verbose=False)
                    best_area = 0
                    for r in results_front:
                        if hasattr(r, "boxes"):
                            for box in r.boxes.xyxy.cpu().numpy():
                                x1, y1, x2, y2 = box
                                area = (x2 - x1) * (y2 - y1)
                                if area > best_area:
                                    best_area = area
                                    front_bbox = (x1, y1, x2, y2)

                color_name = simple_color_name(frames[0], front_bbox)
                save_color_to_csv(current_id, color_name)
                print(f"Color for bottle {current_id}: {color_name}")
                # -----------------------------------------------------------

            combo_history.clear()

    # TEXT OVERLAYS
    names = ["FRONT", "TOP", "BACK"]
    for i, f in enumerate(frames):
        if f is not None:
            cv2.putText(f, f"{names[i]}: {combo[i]}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    grid = make_grid(frames)
    cv2.imshow("3-CAMERA BOTTLE DETECTOR", grid)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
