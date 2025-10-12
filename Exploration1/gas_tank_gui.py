import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import os

class GasTankDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gas Tank Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.model = None
        self.current_image = None
        self.original_image = None
        
        # Load model
        self.load_model()
        
        # Create GUI
        self.create_widgets()
        
    def load_model(self):
        """Load the YOLOv8 model"""
        try:
            # Try to load the trained model first
            model_path = "runs/detect/gas_tank_model_m/weights/best.pt"
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                print("✅ Loaded custom trained YOLOv8m model")
            else:
                # Fallback to pretrained model
                self.model = YOLO("yolov8m.pt")
                print("⚠️ Using pretrained YOLOv8m model (custom model not found)")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            
    def create_widgets(self):
        """Create the GUI widgets"""
        # Title
        title_label = tk.Label(
            self.root, 
            text="⛽ Gas Tank Detection System", 
            font=('Arial', 20, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=10)
        
        # Control frame
        control_frame = tk.Frame(self.root, bg='#f0f0f0')
        control_frame.pack(pady=10)
        
        # Upload button
        upload_btn = tk.Button(
            control_frame,
            text="📁 Upload Image",
            command=self.upload_image,
            font=('Arial', 12),
            bg='#3498db',
            fg='white',
            padx=20,
            pady=5
        )
        upload_btn.pack(side=tk.LEFT, padx=5)
        
        # Predict button
        self.predict_btn = tk.Button(
            control_frame,
            text="🔍 Detect Gas Tank",
            command=self.predict_image,
            font=('Arial', 12),
            bg='#27ae60',
            fg='white',
            padx=20,
            pady=5,
            state='disabled'
        )
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        # Reset button
        reset_btn = tk.Button(
            control_frame,
            text="🔄 Reset",
            command=self.reset_image,
            font=('Arial', 12),
            bg='#e74c3c',
            fg='white',
            padx=20,
            pady=5
        )
        reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Confidence threshold
        threshold_frame = tk.Frame(self.root, bg='#f0f0f0')
        threshold_frame.pack(pady=5)
        
        tk.Label(threshold_frame, text="Confidence Threshold:", bg='#f0f0f0').pack(side=tk.LEFT)
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = tk.Scale(
            threshold_frame,
            from_=0.1,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.confidence_var,
            bg='#f0f0f0'
        )
        confidence_scale.pack(side=tk.LEFT, padx=10)
        
        # Image frame
        image_frame = tk.Frame(self.root, bg='#f0f0f0')
        image_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Image display
        self.image_label = tk.Label(
            image_frame,
            text="No image loaded",
            bg='white',
            relief=tk.SUNKEN,
            bd=2
        )
        self.image_label.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        # Results frame
        results_frame = tk.Frame(self.root, bg='#f0f0f0')
        results_frame.pack(pady=10, fill=tk.X)
        
        self.results_text = tk.Text(
            results_frame,
            height=6,
            bg='white',
            font=('Arial', 10),
            wrap=tk.WORD
        )
        self.results_text.pack(pady=5, padx=20, fill=tk.X)
        
        # Scrollbar for results
        scrollbar = tk.Scrollbar(self.results_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_text.yview)
        
    def upload_image(self):
        """Upload and display an image"""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load and display image
                self.original_image = Image.open(file_path)
                self.current_image = self.original_image.copy()
                self.display_image(self.current_image)
                
                # Enable predict button
                self.predict_btn.config(state='normal')
                
                # Clear results
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, f"Image loaded: {os.path.basename(file_path)}\n")
                self.results_text.insert(tk.END, f"Size: {self.original_image.size[0]} x {self.original_image.size[1]} pixels\n\n")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_image(self, image):
        """Display image in the GUI"""
        # Resize image to fit the display area
        display_size = (600, 400)
        image_resized = image.copy()
        image_resized.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(image_resized)
        
        # Update label
        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo  # Keep a reference
    
    def predict_image(self):
        """Make prediction on the current image"""
        if self.current_image is None or self.model is None:
            messagebox.showwarning("Warning", "Please load an image and ensure model is loaded")
            return
            
        try:
            # Convert PIL image to numpy array
            img_array = np.array(self.current_image)
            
            # Run prediction
            confidence_threshold = self.confidence_var.get()
            results = self.model(img_array, conf=confidence_threshold)
            
            # Analyze results
            analysis = self.analyze_results(results)
            
            # Update results text
            self.results_text.delete(1.0, tk.END)
            
            if analysis['gas_tank_detected']:
                self.results_text.insert(tk.END, "🎉 GAS TANK DETECTED!\n", "success")
                self.results_text.insert(tk.END, f"Confidence: {analysis['max_confidence']:.2%}\n")
                self.results_text.insert(tk.END, f"Gas tanks found: {analysis['gas_tank_count']}\n")
                self.results_text.insert(tk.END, f"Bubbles found: {analysis['bubble_count']}\n\n")
            else:
                self.results_text.insert(tk.END, "❌ NO GAS TANK DETECTED\n", "failure")
                self.results_text.insert(tk.END, f"Total detections: {analysis['detection_count']}\n")
                self.results_text.insert(tk.END, f"Bubbles found: {analysis['bubble_count']}\n\n")
            
            # Show detailed results
            self.results_text.insert(tk.END, "Detailed Detection Results:\n")
            for i, result in enumerate(results):
                boxes = result.boxes
                if boxes is not None:
                    for j, box in enumerate(boxes):
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = box.conf[0].cpu().numpy()
                        class_name = result.names[class_id]
                        coords = box.xyxy[0].cpu().numpy()
                        
                        self.results_text.insert(tk.END, f"Detection {j+1}: {class_name} (Confidence: {confidence:.3f})\n")
                        self.results_text.insert(tk.END, f"Coordinates: ({coords[0]:.0f}, {coords[1]:.0f}, {coords[2]:.0f}, {coords[3]:.0f})\n")
                else:
                    self.results_text.insert(tk.END, "No detections found\n")
            
            # Draw predictions on image
            annotated_image = self.draw_predictions(self.original_image, results)
            self.current_image = annotated_image
            self.display_image(annotated_image)
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def draw_predictions(self, image, results):
        """Draw bounding boxes and labels on the image"""
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = result.names[class_id]
                    
                    # Choose color based on class
                    if class_name == 'gas-tank':
                        color = (0, 255, 0)  # Green for gas tank
                    else:
                        color = (255, 0, 0)  # Red for bubble
                    
                    # Draw bounding box
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw label
                    label = f"{class_name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(img_cv, (x1, y1 - label_size[1] - 10), 
                                 (x1 + label_size[0], y1), color, -1)
                    cv2.putText(img_cv, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Convert back to RGB
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
    
    def analyze_results(self, results):
        """Analyze the prediction results"""
        gas_tank_detected = False
        max_confidence = 0
        detection_count = 0
        bubble_count = 0
        gas_tank_count = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = box.conf[0].cpu().numpy()
                    class_name = result.names[class_id]
                    
                    detection_count += 1
                    
                    if class_name == 'gas-tank':
                        gas_tank_detected = True
                        gas_tank_count += 1
                        max_confidence = max(max_confidence, confidence)
                    elif class_name == 'bubble':
                        bubble_count += 1
        
        return {
            'gas_tank_detected': gas_tank_detected,
            'max_confidence': max_confidence,
            'detection_count': detection_count,
            'gas_tank_count': gas_tank_count,
            'bubble_count': bubble_count
        }
    
    def reset_image(self):
        """Reset to original image"""
        if self.original_image:
            self.current_image = self.original_image.copy()
            self.display_image(self.current_image)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Image reset to original\n")

def main():
    root = tk.Tk()
    app = GasTankDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()