import os
import cv2
import random
from ultralytics import YOLO

yolo = YOLO("models/tank_detection/best.pt")

INPUT_DIR = 'videos/input/13_44'
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
    # next_consecutive_id = -2 # 14_43
    next_consecutive_id = 1 # 13_44
    # next_consecutive_id = 1 # 14_32
    # next_consecutive_id = 1 # 14_55

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    ok_dir = os.path.join('frames', 'ok')
    nok_dir = os.path.join('frames', 'nok')
    os.makedirs(ok_dir, exist_ok=True)
    os.makedirs(nok_dir, exist_ok=True)
    # Only save crops for these display IDs
    # allowed_ids = {13, 24, 32, 36, 53, 59, 66, 69, 73, 80, 82, 88, 95, 102, 128, 159} # 14_43
    allowed_ids = {30, 43, 72, 83, 88, 110, 122} # 13_44
    # allowed_ids = {10, 16, 23, 24, 35, 54, 64, 76, 84, 143, 144, 146, 159, 167, 172} # 14_32
    # allowed_ids = {3, 7, 15, 22, 35, 55, 69, 73, 82, 83, 89, 97, 107} # 14_55

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
                    crop_id_folder = None
                    crop_filename = None
                    if box.id is not None:
                        track_id = int(box.id[0])
                        if track_id not in id_mapping:
                            id_mapping[track_id] = next_consecutive_id
                            next_consecutive_id += 1
                        display_id = id_mapping[track_id]
                        colour = getTrackingColour(display_id)
                        label = f"ID:{display_id} {class_name} {conf:.2f}"
                        # Save to nok if in allowed_ids, else to ok
                        if display_id in allowed_ids:
                            crop_id_folder = nok_dir
                        else:
                            crop_id_folder = ok_dir
                        save_crop = True
                        crop_filename = os.path.join(
                            crop_id_folder,
                            f"{video_name}_id{display_id}_frame{frame_count:05d}.jpg"
                        )
                    else:
                        colour = getColours(cls)
                        label = f"{class_name} {conf:.2f}"
                        # No ID, save to ok with noid in filename
                        crop_id_folder = ok_dir
                        save_crop = True
                        crop_filename = os.path.join(
                            crop_id_folder,
                            f"{video_name}_noid_frame{frame_count:05d}.jpg"
                        )
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
                    cv2.putText(frame, label,
                                (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, colour, 3)
                    # Crop and save the detected object
                    if save_crop and crop_id_folder and crop_filename:
                        crop = frame[y1:y2, x1:x2]
                        cv2.imwrite(crop_filename, crop)
        out.write(frame)
        frame_count += 1
    videoCap.release()
    out.release()
    print(f"Processed {frame_count} frames. Output saved to {output_path}")

def main():
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_path = os.path.join(INPUT_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, filename)
            print(f"Processing {filename}...")
            process_video(video_path, output_path, None)

if __name__ == "__main__":
    main()
