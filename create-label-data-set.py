import cv2
import numpy as np
import os
import glob

sift = cv2.xfeatures2d.SIFT_create()

# Use the first frame as template image
template_path = "frames/images/front/frame_0000.jpg"
img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Could not read template image from {template_path}")
    exit(1)

kp_image, desc_image = sift.detectAndCompute(img, None)

# Get all frame images from the directory
frame_pattern = "frames/images/front/frame_*.jpg"
frame_files = sorted(glob.glob(frame_pattern))

if not frame_files:
    print(f"Error: No frame files found in pattern {frame_pattern}")
    exit(1)

print(f"Found {len(frame_files)} frames to process")
vo_h, vo_w = img.shape  # Get dimensions from template image

#Initializing the matching algorithm
index_params = dict(algorithm=0, trees=5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

def generate_labels(dst, count): # Get dst and count from the template matching function
    x_top = np.int32(dst[0][0][0])
    y_top = np.int32(dst[0][0][1])
    x_bottom = np.int32(dst[1][0][0])
    y_bottom = np.int32(dst[1][0][1])

    if(x_top >= 0 and y_top >= 0 and x_bottom > 0 and y_bottom > 0): #We only want non-negative co-ordinate values
        name = str("Image"+str(count)+".txt")
        with open(name, "w+") as f:
            f.write(str(x_top)+" "+str(y_top)+" "+str(x_bottom)+" "+str(y_bottom)+"\n")
        f.close()

output_dir = "images/label-test"
os.makedirs(output_dir, exist_ok=True)

count = 0
for frame_file in frame_files:
    # Read each frame image
    frame = cv2.imread(frame_file)
    if frame is None:
        print(f"Warning: Could not read frame {frame_file}")
        continue
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_grayframe, desc_grayframe = sift.detectAndCompute(gray, None)
    
    if desc_grayframe is None:
        print(f"Warning: No features found in {frame_file}")
        continue
        
    matches = flann.knnMatch(desc_image, desc_grayframe, k=2)

    good = [] #List that stores all the matching points
    if matches is not None and len(matches) > 0:
        for match_pair in matches:
            if len(match_pair) == 2:  # Ensure we have 2 matches for distance ratio test
                m, n = match_pair
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
        
        generate_labels(dst, count) #Function for generating labels, given below
        count += 1
        print(f"Processed frame {count}: {frame_file}")
    else:
        print(f"Insufficient good matches ({len(good)}) for {frame_file}")

print(f"Processing complete. Processed {count} frames with valid matches out of {len(frame_files)} total frames.")