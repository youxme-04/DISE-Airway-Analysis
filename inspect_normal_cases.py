import os
import cv2
import numpy as np

def analyze_intensities(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 1. ROI detection (from test_new_classifier)
    frame_indices = [int(total_frames * p) for p in [0.05, 0.15, 0.3, 0.5, 0.7, 0.85]]
    valid_boxes = []
    for f_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_c = max(contours, key=cv2.contourArea)
            c_area = cv2.contourArea(largest_c)
            if c_area > (frame.shape[0] * frame.shape[1] * 0.05):
                bx, by, bw, bh = cv2.boundingRect(largest_c)
                aspect = bw / bh if bh > 0 else 0
                if 0.75 <= aspect <= 1.3:
                    valid_boxes.append((bx, by, bw, bh))
                    
    if valid_boxes:
        x = int(np.median([box[0] for box in valid_boxes]))
        y = int(np.median([box[1] for box in valid_boxes]))
        w = int(np.median([box[2] for box in valid_boxes]))
        h = int(np.median([box[3] for box in valid_boxes]))
    else:
        x, y, w, h = 0, 0, 640, 480
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_brightness = []
    frame_areas = []
    
    mask_circle = np.zeros((224, 224), dtype=np.uint8)
    cv2.circle(mask_circle, (112, 112), 108, 255, -1)
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        
        crop_f = frame[y:y+h, x:x+w]
        crop_f_resized = cv2.resize(crop_f, (224, 224))
        
        # Calculate mean brightness of the crop frame
        gray = cv2.cvtColor(crop_f_resized, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray[mask_circle > 0])
        frame_brightness.append(mean_val)
        
    cap.release()
    return frame_brightness

def main():
    video_dir = r"video DISE patient 001-100"
    pt16_path = os.path.join(video_dir, "pt016.mp4")
    if os.path.exists(pt16_path):
        b_vals = analyze_intensities(pt16_path)
        print("Pt16 Frame Brightness Statistics:")
        print(f"Total Frames: {len(b_vals)}")
        print(f"Min Brightness: {min(b_vals):.1f}")
        print(f"Max Brightness: {max(b_vals):.1f}")
        print(f"Mean Brightness: {np.mean(b_vals):.1f}")
        print(f"Std Brightness: {np.std(b_vals):.1f}")
        
        # Print first few low brightness frames
        low_idx = [idx for idx, b in enumerate(b_vals) if b < 25]
        print(f"Number of frames with brightness < 25: {len(low_idx)}")
        if low_idx:
            print(f"Low brightness frame indices: {low_idx[:20]}...")
            
if __name__ == '__main__':
    main()
