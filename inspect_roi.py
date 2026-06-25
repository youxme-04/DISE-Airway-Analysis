import os
import cv2
import numpy as np

def get_roi_app(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(total_frames * p) for p in [0.1, 0.3, 0.5, 0.7, 0.9]]
    valid_boxes = []
    for f_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_c = max(contours, key=cv2.contourArea)
            c_area = cv2.contourArea(largest_c)
            if c_area > (frame.shape[0] * frame.shape[1] * 0.1):
                bx, by, bw, bh = cv2.boundingRect(largest_c)
                aspect = bw / bh if bh > 0 else 0
                if 0.8 <= aspect <= 1.2:
                    valid_boxes.append((bx, by, bw, bh))
    cap.release()
    if valid_boxes:
        x = int(np.median([box[0] for box in valid_boxes]))
        y = int(np.median([box[1] for box in valid_boxes]))
        w = int(np.median([box[2] for box in valid_boxes]))
        h = int(np.median([box[3] for box in valid_boxes]))
        return (x, y, w, h)
    return None

def get_roi_new(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
    cap.release()
    if valid_boxes:
        x = int(np.median([box[0] for box in valid_boxes]))
        y = int(np.median([box[1] for box in valid_boxes]))
        w = int(np.median([box[2] for box in valid_boxes]))
        h = int(np.median([box[3] for box in valid_boxes]))
        return (x, y, w, h)
    return None

def main():
    video_dir = r"video DISE patient 001-100"
    for i in [1, 6, 8, 27]:
        filename = f"pt{i:03d}.mp4"
        filepath = os.path.join(video_dir, filename)
        if not os.path.exists(filepath): continue
        print(f"Pt{i:02d} - App ROI: {get_roi_app(filepath)} | New ROI: {get_roi_new(filepath)}")

if __name__ == '__main__':
    main()
