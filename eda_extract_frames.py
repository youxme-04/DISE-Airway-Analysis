import cv2
import os

video_dir = r"video DISE patient 001-100"
output_dir = r"samples"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

labeled_videos = {
    "pt003.mp4": "AP",
    "pt011.mp4": "AP",
    "pt016.mp4": "AP",
    "pt027.mp4": "Concentric",
    "pt028.mp4": "AP_Lateral"
}

for video_file, label in labeled_videos.items():
    video_path = os.path.join(video_dir, video_file)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video {video_file}")
        continue
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Get a frame exactly from the middle of the video
    middle_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    
    ret, frame = cap.read()
    if ret:
        output_file = os.path.join(output_dir, f"{video_file.replace('.mp4', '')}_{label}.jpg")
        cv2.imwrite(output_file, frame)
        print(f"Saved {output_file} (Original Size: {frame.shape}, Total Frames: {total_frames}, FPS: {fps})")
    else:
        print(f"Could not read frame from {video_file}")
        
    cap.release()

print("Extraction complete.")
