import cv2
import random
from ultralytics import YOLO
import numpy as np
from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn

yolo = YOLO("models/yolo11x_finetuned_bottles_on_site_v3.pt")

# Load the classifier model
classifier_name = 'efficientnet-b0'
file_name = 'models/classifier_efficientNet.pth'
out_dim = 1

classifier = EfficientNet.from_name(classifier_name)
classifier.load_state_dict(torch.load(file_name), strict=False)

classifier._fc = nn.Linear(classifier._fc.in_features, out_dim)

# Define class label mapping
class_labels = {
    0: "ok",
    1: "notprimagaz",
    2: "damaged",
    3: "dirty"
    # Add more labels as needed
}

def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

video_path = "videos/14_55_front_cropped.mp4"
videoCap = cv2.VideoCapture(video_path)

frame_count = 0

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
                
                # Crop the detected object for classification
                cropped_img = frame[y1:y2, x1:x2]
                if cropped_img.size > 0:  # Ensure the cropped image is valid
                    resized_img = cv2.resize(cropped_img, (224, 224))  # Resize to classifier input size
                    normalized_img = resized_img / 255.0  # Normalize pixel values
                    input_img = np.expand_dims(normalized_img, axis=0)  # Add batch dimension
                    
                    # Convert the input to a PyTorch tensor
                    input_tensor = torch.tensor(input_img, dtype=torch.float32).permute(0, 3, 1, 2)  # Change shape to (batch_size, channels, height, width)
                    
                    # Ensure the model is in evaluation mode
                    classifier.eval()
                    
                    # Perform inference
                    with torch.no_grad():  # Disable gradient computation for inference
                        class_probs = classifier(input_tensor).numpy()
                    
                    class_label = np.argmax(class_probs)
                    class_conf = class_probs[0][class_label]
                    
                    # Get the string label for the predicted class
                    label_text = class_labels.get(class_label, "unknown")  # Default to "unknown" if label is not found
                    
                    # Add classification result to the box caption
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                    cv2.putText(frame, f"{class_name} {conf:.2f} | {label_text} ({class_conf:.2f})",
                                (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, colour, 2)
    
    # Display the frame
    cv2.imshow('Tracking gas bottles', frame)
    
    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_count += 1

videoCap.release()