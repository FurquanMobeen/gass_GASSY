import cv2
import os

def extract_frames_stepped(video_path, output_folder, step=10):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    
    # Get the total number of frames (just for the progress message)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing '{video_name}' ({total_frames} total frames). Saving every {step}th frame...")
    
    frame_count = 0
    saved_count = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        # Only save if the current frame_count is divisible by the step
        # e.g., 0, 10, 20, 30...
        if frame_count % step == 0:
            # We name the file using the ACTUAL frame number so you can map it back to the video
            # Example: Video_00010.jpg, Video_00020.jpg
            output_filename = f"{video_name}_{frame_count:05d}.jpg"
            output_path = os.path.join(output_folder, output_filename)
            
            cv2.imwrite(output_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Done! Saved {saved_count} images to '{output_folder}'")

# --- Usage ---
if __name__ == "__main__":
    # Change the 'step' value below to skip more or fewer frames
    extract_frames_stepped("gass_GASSY/videos/14_55_top_cropped.mp4", "extracted_frames_top_cropped", step=20)