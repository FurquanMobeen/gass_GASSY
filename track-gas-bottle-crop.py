
import cv2
import random
from pathlib import Path
from ultralytics import YOLO

# Set the folder containing input images
input_folder = Path("crops\ok")  # Change this to your folder
image_paths = sorted(list(input_folder.glob("*.jpg")))


# Create output folder for crops in 'training-data'
output_folder = Path("temporary") / input_folder.name
output_folder.mkdir(parents=True, exist_ok=True)

yolo = YOLO("models/tank_detection/best.pt")

def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

target_crops = 500
total_images = len(image_paths)
skip_interval = max(1, total_images // target_crops) if total_images > 0 else 1

crop_count = 0
for idx, img_path in enumerate(image_paths):
    if crop_count >= target_crops:
        break
    if idx % skip_interval != 0:
        continue
    frame = cv2.imread(str(img_path))
    if frame is None:
        continue
    results = yolo.track(frame, stream=True)
    crop_saved = False
    for result in results:
        class_names = result.names
        for box in result.boxes:
            if box.conf[0] > 0.4 and not crop_saved:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Crop the detected region
                cropped_img = frame[y1:y2, x1:x2]
                crop_filename = output_folder / img_path.name
                cv2.imwrite(str(crop_filename), cropped_img)
                crop_count += 1
                crop_saved = True
                break
        if crop_saved:
            break
    if idx % 30 == 0:
        print(f"Processed {idx+1} images, crops saved: {crop_count}")

print(f"\nProcessing complete!")
print(f"Total images in folder: {total_images}")
print(f"Skip interval used: {skip_interval}")
print(f"Total crops saved: {crop_count}")
print(f"Crops saved in: {output_folder}")