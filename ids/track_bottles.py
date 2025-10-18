import os
import cv2
import math
from ultralytics import YOLO

# --- Paths ---
frames_dir = r"frames/images/front"
output_dir = "output_tracked_images"
os.makedirs(output_dir, exist_ok=True)

# --- Load YOLO model (replace with fine-tuned bottle model later) ---
model = YOLO("yolov8n.pt")

# --- Parameters ---
BOX_COLOR = (139, 0, 0)  # Dark blue
MAX_DISTANCE = 50        # pixels to consider same object
MAX_MISSED = 10          # remove ID if not seen for this many frames
DISPLAY_WIDTH, DISPLAY_HEIGHT = 1280, 720

# --- Persistent ID tracking ---
next_id = 1
active_objects = {}  # ID -> (center, bbox, frames_since_seen)

# --- Helper function ---
def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

# --- Sort frames ---
frame_files = sorted(
    [f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.png'))]
)

# --- Process frames ---
for idx, file_name in enumerate(frame_files):
    frame_path = os.path.join(frames_dir, file_name)
    frame = cv2.imread(frame_path)
    if frame is None:
        continue

    detected_centers = []

    # --- YOLO detection & tracking ---
    results = model.track(frame, persist=True, stream=False)

    # Collect detected bottles and their centers
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            class_name = result.names[cls]
            if class_name != "bottle":
                continue
            conf = float(box.conf[0])
            if conf < 0.4:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center = ((x1 + x2)//2, (y1 + y2)//2)
            detected_centers.append((center, (x1, y1, x2, y2)))

    # --- Match detected objects to existing IDs ---
    assigned_ids = {}
    for center, bbox in detected_centers:
        matched = False
        for obj_id, (old_center, _, _) in active_objects.items():
            if distance(center, old_center) < MAX_DISTANCE:
                assigned_ids[obj_id] = (center, bbox)
                active_objects[obj_id] = (center, bbox, 0)
                matched = True
                break
        if not matched:
            assigned_ids[next_id] = (center, bbox)
            active_objects[next_id] = (center, bbox, 0)
            next_id += 1

    # --- Update frames_since_seen and remove old IDs ---
    to_delete = []
    for obj_id in active_objects:
        if obj_id not in assigned_ids:
            c, bbox, missed = active_objects[obj_id]
            missed += 1
            if missed > MAX_MISSED:
                to_delete.append(obj_id)
            else:
                active_objects[obj_id] = (c, bbox, missed)
    for obj_id in to_delete:
        del active_objects[obj_id]

    # --- Draw boxes & IDs ---
    for obj_id, (center, (x1, y1, x2, y2)) in assigned_ids.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        cv2.putText(frame, f"ID {obj_id}", (x1, max(y1-10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, BOX_COLOR, 2)

    # --- Resize for display & save ---
    display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    cv2.imshow("Tracked Bottles", display_frame)
    cv2.imwrite(os.path.join(output_dir, file_name), frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print(f"\n✅ Tracking complete. Annotated frames saved in: {output_dir}")
