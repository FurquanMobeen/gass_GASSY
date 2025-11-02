import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the Keras model
model_path = "models/classifier_augmented.keras"
classifier = load_model(model_path)

# Define the directory containing the images
image_dir = "images/classification/examples/ok"

# Define class labels
class_labels = {
    0: "ok",
    1: "notprimagaz",
    2: "damaged",
    3: "dirty"
}

# Initialize counters for predictions
class_counts = {label: 0 for label in class_labels.values()}

# Process each image in the directory
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load image {image_path}")
        continue

    # Preprocess the image
    resized_img = cv2.resize(image, (224, 224))  # Resize to model input size
    normalized_img = resized_img / 255.0  # Normalize pixel values
    input_img = np.expand_dims(normalized_img, axis=0)  # Add batch dimension

    # Perform prediction
    class_probs = classifier.predict(input_img)
    class_label = np.argmax(class_probs)
    class_name = class_labels[class_label]

    # Increment the count for the predicted class
    class_counts[class_name] += 1

    print(f"Image: {image_name}, Predicted: {class_name}, Confidence: {class_probs[0][class_label]:.2f}")

# Plot the overall results
plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Classification Results")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()