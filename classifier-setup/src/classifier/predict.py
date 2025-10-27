import torch
from torchvision import models, transforms
from PIL import Image
import os

def predict(image_path):
    model_dir = "models/bottle_classifier.pth"
    data_dir = "data"
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transforms(image).unsqueeze(0)
    
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    
    class_names = sorted(os.listdir(data_dir))
    model.fc = torch.nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(model_dir, map_location='cpu'))
    model.eval()
    
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        predicted_label = class_names[preds.item()]
        
    print(f"Predicted label: {predicted_label}")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bottle classifier prediction")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    args = parser.parse_args()
    
    predict(args.image)