import cv2
import numpy as np
import os

sift = cv2.xfeatures2d.SIFT_create()
img = cv2.imread("images/template/template.jpg", cv2.IMREAD_GRAYSCALE)
kp_image, desc_image = sift.detectAndCompute(img, None)

video = cv2.VideoCapture("videos/14_55_top_cropped.mp4")
vo_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
vo_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

#Initializing the matching algorithm
index_params = dict(algorithm=0, trees=5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

def generate_labels(dst, count): # Get dst and count from the template matching function
    x_top = np.int32(dst[0][0][0])
    y_top = np.int32(dst[0][0][1])
    x_bottom = np.int32(dst[1][0][0])
    y_bottom = np.int32(dst[1][0][1])

    if(x_top > 0 and y_top > 0 and x_bottom > 0 and y_bottom > 0): #We only want non-zero co-ordinate values
        name = str("Image"+str(count)+".txt")
        with open(name, "w+") as f:
            f.write(str(x_top)+" "+str(y_top)+" "+str(x_bottom)+" "+str(y_bottom)+"\n")
        f.close()

output_dir = "images/label-test"
os.makedirs(output_dir, exist_ok=True)

count = 0
while True:
    _,frame = video.read() #Parsing the video frame by frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_grayframe, desc_grayframe = sift.detectAndCompute(gray, None)
    matches = flann.knnMatch(desc_image, desc_grayframe, k=2)

    good = [] #List that stores all the matching points
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if (len(good)>7): # Threshold for number of matched features
        query_pts = np.float32([kp_image[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
   
        # Perspective transform
        h, w = img.shape
        pts = np.float32([[0, 0], [w, h]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        
        #Drawing a rectangle over the matched area. This is just for your reference.
        #We don't want a rectangle drawn over the template area when feeding the image into the CNN
        #Please comment out Line 32 when generating the dataset
        cv2.rectangle(frame, (np.int32(dst[0][0][0]),np.int32(dst[0][0][1])), (np.int32(dst[1][0][0]),  np.int32(dst[1][0][1])), (0,255, 255), 3)
        cv2.imwrite(os.path.join(output_dir, f"Image{count}.jpg"), frame)
        
        generate_labels(dst, count) #Fucntion for generating labels, given below
        count += 1
    
    key = cv2.waitKey(27)
    if key & 0xFF == ord('q'): #press q to terminate the process
        break