import cv2
import random
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

yolo = YOLO("models/yolo11x_finetuned_bottles_on_site_v3.pt")

# Load the classifier model
classifier = tf.keras.models.load_model('models/classifier_augmented.keras')

# Display all class names of the classifier model
classifier_class_names = ['OK', 'damaged', 'dirty', 'not PRIMAGAZ']

def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

# Load the image
image_path = "images/ok.JPG"
frame = cv2.imread(image_path)

if frame is None:
    print(f"Error: Unable to load image at {image_path}")
else:
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
                    
                    # Perform inference using the Keras model
                    classifier_prediction = classifier.predict(input_img)
                    classifier_score = tf.nn.softmax(classifier_prediction)
                    class_conf = float(tf.reduce_max(classifier_score))  # Get the maximum confidence score
                    
                    # Add classification result to the box caption
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                    cv2.putText(frame, f"{class_name} {conf:.2f} | {classifier_class_names[np.argmax(classifier_score)]} ({class_conf:.2f})",
                                (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, colour, 2)
    
    # Display the image
    cv2.imshow('Tracking gas bottles', frame)
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()