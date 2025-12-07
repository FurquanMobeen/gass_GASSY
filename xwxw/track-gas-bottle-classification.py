import cv2
import random
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image
import torchvision.models as models

# Load YOLO model for detection
yolo = YOLO("models/yolo11x_finetuned_bottles_on_site_v3.pt")

# Load ConvNext classifier
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the classifier checkpoint
checkpoint = torch.load("models/trained_convnext_classifier.pth", map_location=device, weights_only=False)

# Create a simple wrapper model that matches the saved structure
class ClassifierModel(torch.nn.Module):
    def __init__(self, state_dict, num_classes):
        super().__init__()
        # Create ConvNeXt backbone 
        self.backbone = models.convnext_base(weights=None)
        # Replace the classifier head to match the saved model structure
        self.backbone.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.LayerNorm(1024, eps=1e-06, elementwise_affine=True),  # Layer 2
            torch.nn.Linear(1024, num_classes, bias=True)  # Layer 3
        )
        
    def forward(self, x):
        return self.backbone(x)

# Create model and load weights
num_classes = checkpoint['num_classes']
classifier = ClassifierModel(checkpoint['model_state_dict'], num_classes)
classifier.load_state_dict(checkpoint['model_state_dict'])
classifier.to(device)
classifier.eval()

# Define transforms for the classifier (adjust these based on your training setup)
classifier_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

def classify_gas_bottle(frame, x1, y1, x2, y2):
    
    # Extract the region of interest
    roi = frame[y1:y2, x1:x2]
    
    # Convert BGR to RGB
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(roi_rgb)
    
    # Apply transforms
    input_tensor = classifier_transforms(pil_image).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = classifier(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = torch.max(probabilities).item()
    
    return predicted_class, confidence

video_path = "videos/input/14_43_front_cropped.mp4"
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
                
                # Classify the detected gas bottle
                classifier_class, classifier_conf = classify_gas_bottle(frame, x1, y1, x2, y2)
                # Use the class names from the trained model
                classifier_names = checkpoint['class_names']
                classifier_label = classifier_names[classifier_class] if classifier_class < len(classifier_names) else f"Class_{classifier_class}"
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                
                # Display both detection and classification results
                detection_text = f"{class_name} {conf:.2f}"
                classification_text = f"{classifier_label} {classifier_conf:.2f}"
                
                cv2.putText(frame, detection_text,
                            (x1, max(y1 - 30, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, colour, 2)
                cv2.putText(frame, classification_text,
                            (x1, max(y1 - 10, 40)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)  # Green for classification
    
    # Display the frame
    cv2.imshow('Tracking gas bottles', frame)
    
    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_count += 1

videoCap.release()
cv2.destroyAllWindows()

# Clean up GPU memory if using CUDA
if torch.cuda.is_available():
    torch.cuda.empty_cache()