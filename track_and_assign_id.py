import os
import cv2
import random
from ultralytics import YOLO

yolo = YOLO("models/tank_detection/best.pt")

INPUT_DIR = 'videos/input'
OUTPUT_DIR = 'videos/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

def getTrackingColour(track_id):
    random.seed(int(track_id))
    return tuple(random.randint(0, 255) for _ in range(3))

def process_video(video_path, output_path, crop_output_root):
    videoCap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0,
                          (int(videoCap.get(3)), int(videoCap.get(4))))
    frame_count = 0
    id_mapping = {}
    next_consecutive_id = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    crop_output_dir = os.path.join(crop_output_root, video_name)
    os.makedirs(crop_output_dir, exist_ok=True)
    # Only save crops for these display IDs
    allowed_ids = {13, 24, 32, 36, 53, 59, 66, 69, 73, 80, 82, 88, 95, 102, 128, 159}
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
                    save_crop = False
                    if box.id is not None:
                        track_id = int(box.id[0])
                        if track_id not in id_mapping:
                            id_mapping[track_id] = next_consecutive_id
                            next_consecutive_id += 1
                        display_id = id_mapping[track_id]
                        colour = getTrackingColour(display_id)
                        label = f"ID:{display_id} {class_name} {conf:.2f}"
                        crop_id_folder = os.path.join(crop_output_dir, f"id_{display_id}_{class_name}")
                        if display_id in allowed_ids:
                            save_crop = True
                    else:
                        colour = getColours(cls)
                        label = f"{class_name} {conf:.2f}"
                        crop_id_folder = os.path.join(crop_output_dir, f"noid_{class_name}")
                        # Do not save crops for noid
                        save_crop = False
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
                    cv2.putText(frame, label,
                                (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, colour, 3)
                    # Crop and save the detected object only for allowed IDs
                    if save_crop:
                        os.makedirs(crop_id_folder, exist_ok=True)
                        crop = frame[y1:y2, x1:x2]
                        crop_filename = os.path.join(crop_id_folder, f"frame{frame_count:05d}.jpg")
                        cv2.imwrite(crop_filename, crop)
        out.write(frame)
        frame_count += 1
    videoCap.release()
    out.release()
    print(f"Processed {frame_count} frames. Output saved to {output_path}")

def main():
    crop_output_root = 'frames/crops'
    os.makedirs(crop_output_root, exist_ok=True)
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_path = os.path.join(INPUT_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, filename)
            print(f"Processing {filename}...")
            process_video(video_path, output_path, crop_output_root)

if __name__ == "__main__":
    main()
