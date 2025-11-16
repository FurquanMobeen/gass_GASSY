import cv2
import random
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image
import torchvision.models as models

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load YOLO model
yolo = YOLO(r"C:\Users\purva\Documents\GitHub\gass_GASSY\models\yolo11x_finetuned_bottles_on_site_v3 (1).pt")
print("YOLO model loaded successfully!")

# Load classifier checkpoint
checkpoint_path = r"C:\Users\purva\Documents\Github\gass_GASSY\classifier-setup\src\classifier\models\bottle_classifier_fold_2.pth"
state_dict = torch.load(checkpoint_path, map_location=device)


num_classes = 2
classifier_names = ['not ok', 'ok']


classifier = models.efficientnet_b0(weights=None)
in_features = classifier.classifier[1].in_features


classifier.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features, num_classes)
)


classifier.load_state_dict(state_dict, strict=False)
classifier.to(device)
classifier.eval()

# Transforms for classifier
classifier_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Function to get consistent random colors
def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))


def classify_gas_bottle(frame, x1, y1, x2, y2):
    roi = frame[y1:y2, x1:x2]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(roi_rgb)
    input_tensor = classifier_transforms(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = classifier(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = torch.max(probabilities).item()
    return predicted_class, confidence

# Video capture
video_path = r"C:\Users\purva\Documents\Github\gass_Gassy\videos\14_55_front_cropped.mp4"
videoCap = cv2.VideoCapture(video_path)

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

                # Classify detected gas bottle
                classifier_class, classifier_conf = classify_gas_bottle(frame, x1, y1, x2, y2)
                classifier_label = classifier_names[classifier_class]

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                detection_text = f"{class_name} {conf:.2f}"
                classification_text = f"{classifier_label} {classifier_conf:.2f}"
                cv2.putText(frame, detection_text, (x1, max(y1 - 30, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
                cv2.putText(frame, classification_text, (x1, max(y1 - 10, 40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Tracking gas bottles', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCap.release()
cv2.destroyAllWindows()

# Free GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
