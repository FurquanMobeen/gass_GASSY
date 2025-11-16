import os
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
VAL_DIR = r"C:\Users\purva\Documents\Github\gass_Gassy\classifier-setup\data\val"
CHECKPOINT_PATH = r"C:\Users\purva\Documents\Github\gass_GASSY\classifier-setup\src\classifier\models\bottle_classifier_fold_5.pth"
DISPLAY_COUNT = 6  # Number of images to display

# --- DEVICE ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- CLASS LABELS ---
classifier_names = ['nok', 'ok']
num_classes = len(classifier_names)

# --- LOAD CLASSIFIER ---
classifier = models.efficientnet_b0(weights=None)
in_features = classifier.classifier[1].in_features
classifier.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features, num_classes)
)

state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
classifier.load_state_dict(state_dict, strict=True)
classifier.to(device)
classifier.eval()

# --- TRANSFORMS ---
classifier_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- DEBUG & PREDICTION FUNCTION ---
def debug_predict(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = classifier_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = classifier(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()

    print(f"\n--- DEBUG {os.path.basename(image_path)} ---")
    print("Logits:", outputs.cpu().numpy())
    print("Softmax probs [nok, ok]:", probs.cpu().numpy())
    print("Predicted index:", pred_idx)
    print("Predicted class:", classifier_names[pred_idx])
    print("Confidence:", confidence)

    return classifier_names[pred_idx], confidence

# --- GET ALL IMAGES IN VAL FOLDER ---
def get_val_images(val_dir):
    images = [os.path.join(val_dir, f) for f in os.listdir(val_dir)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return images

# --- DISPLAY RESULTS (randomized subset) ---
def show_predictions(images, preds, confs, display_count=6):
    # Randomize images
    combined = list(zip(images, preds, confs))
    random.shuffle(combined)
    selected = combined[:display_count]

    plt.figure(figsize=(14, 6))
    for i, (img_path, pred, conf) in enumerate(selected, 1):
        img = Image.open(img_path).convert('RGB')
        plt.subplot(2, (display_count + 1) // 2, i)
        plt.imshow(img)
        plt.title(f"{pred}\n({conf:.2f})", color='red' if pred == 'nok' else 'green')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# --- MAIN ---
if __name__ == "__main__":
    val_images = get_val_images(VAL_DIR)
    print(f"Total images in val folder: {len(val_images)}")

    preds, confs = [], []

    for img_path in val_images:
        pred, conf = debug_predict(img_path)
        preds.append(pred)
        confs.append(conf)

    # --- Show a randomized subset of predictions ---
    show_predictions(val_images, preds, confs, display_count=DISPLAY_COUNT)
