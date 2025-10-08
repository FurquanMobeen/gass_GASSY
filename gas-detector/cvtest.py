from ultralytics import YOLO
import cv2


# download yolo-model

model = YOLO("yolov8n.pt")

# use this mp4 file as a reference

cap = cv2.VideoCapture("14_55_back_right_cropped.mp4")
if not cap.isOpened():
    print("Error in opening video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break # when video ends
    
    results = model(frame, stream=True)
    for r in results:
        annotated_frame = r.plot()
        
        
    cv2.imshow("Gassbottle", annotated_frame)
    
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
cap.release()
cv2.destroyAllWindows()