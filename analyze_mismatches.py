import os
import cv2
import numpy as np

ground_truth_db = {
    1: {"class": "AP", "degree": 2, "reduction": 88.5},
    2: {"class": "AP", "degree": 2, "reduction": 91.2},
    3: {"class": "AP", "degree": 2, "reduction": 95.0},
    4: {"class": "AP", "degree": 2, "reduction": 89.0},
    5: {"class": "AP", "degree": 2, "reduction": 96.5},
    6: {"class": "AP", "degree": 2, "reduction": 98.2},
    7: {"class": "AP", "degree": 2, "reduction": 92.0},
    8: {"class": "AP", "degree": 1, "reduction": 64.0},
    9: {"class": "AP", "degree": 2, "reduction": 87.5},
    10: {"class": "AP", "degree": 1, "reduction": 68.0},
    11: {"class": "AP", "degree": 2, "reduction": 93.0},
    12: {"class": "AP", "degree": 1, "reduction": 62.0},
    13: {"class": "Normal", "degree": 0, "reduction": 12.0},
    14: {"class": "Concentric", "degree": 2, "reduction": 94.5},
    15: {"class": "AP", "degree": 1, "reduction": 65.5},
    16: {"class": "Normal", "degree": 0, "reduction": 14.0},
    17: {"class": "AP", "degree": 2, "reduction": 89.5},
    18: {"class": "AP", "degree": 1, "reduction": 63.5},
    19: {"class": "AP", "degree": 2, "reduction": 92.5},
    20: {"class": "AP", "degree": 2, "reduction": 94.0},
    21: {"class": "AP", "degree": 1, "reduction": 61.5},
    22: {"class": "AP", "degree": 2, "reduction": 90.5},
    23: {"class": "AP", "degree": 1, "reduction": 67.2},
    24: {"class": "AP", "degree": 2, "reduction": 93.8},
    25: {"class": "AP", "degree": 1, "reduction": 62.5},
    26: {"class": "Concentric", "degree": 1, "reduction": 68.5},
    27: {"class": "Concentric", "degree": 2, "reduction": 96.0},
    28: {"class": "AP", "degree": 2, "reduction": 88.0},
    29: {"class": "AP", "degree": 2, "reduction": 91.0},
    30: {"class": "AP", "degree": 2, "reduction": 92.2}
}

def resample_contour(contour, num_points=32):
    pts = contour.reshape(-1, 2).astype(np.float32)
    diffs = np.diff(pts, axis=0, append=[pts[0]])
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    total_dist = cum_dists[-1]
    if total_dist == 0:
        return [[112.0, 112.0]] * num_points
    target_dists = np.linspace(0, total_dist, num_points, endpoint=False)
    pts_closed = np.vstack([pts, pts[0]])
    resampled_x = np.interp(target_dists, cum_dists, pts_closed[:, 0])
    resampled_y = np.interp(target_dists, cum_dists, pts_closed[:, 1])
    return np.column_stack((resampled_x, resampled_y)).tolist()

def extract_robust_contour(crop_f_resized, mask_circle, prev_contour=None, prev_area=None):
    gray_seq = cv2.cvtColor(crop_f_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_seq, (5, 5), 0)
    blurred[mask_circle == 0] = 255
    
    thresholds = range(15, 61, 5)
    best_c = None
    best_area = 0.0
    best_score = -1e9
    
    prev_centroid = None
    if prev_contour is not None and len(prev_contour) > 0:
        prev_centroid = np.mean(prev_contour, axis=0)
        
    for th in thresholds:
        _, thresh_seq = cv2.threshold(blurred, th, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        thresh_seq = cv2.morphologyEx(thresh_seq, cv2.MORPH_CLOSE, kernel)
        contours_seq, _ = cv2.findContours(thresh_seq, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours_seq:
            c_area = cv2.contourArea(c)
            if c_area < 150:
                continue
            local_mask = np.zeros_like(thresh_seq)
            cv2.drawContours(local_mask, [c], -1, 255, -1)
            c_filled, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not c_filled:
                continue
            c = c_filled[0]
            c_area = cv2.contourArea(c)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx_pt = M["m10"] / M["m00"]
                cy_pt = M["m01"] / M["m00"]
            else:
                cx_pt, cy_pt = 112.0, 112.0
            dist_from_center = np.sqrt((cx_pt - 112.0)**2 + (cy_pt - 112.0)**2)
            if dist_from_center > 90:
                continue
            solidity = 1.0
            if len(c) >= 5:
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = c_area / hull_area
            if solidity < 0.6:
                continue
            score = c_area - 0.5 * (dist_from_center ** 2) + 200.0 * solidity
            if prev_centroid is not None and prev_area is not None:
                dist_from_prev = np.sqrt((cx_pt - prev_centroid[0])**2 + (cy_pt - prev_centroid[1])**2)
                area_change = abs(c_area - prev_area)
                score -= 8.0 * dist_from_prev
                if area_change > 1000.0:
                    score -= 5000.0
                if dist_from_prev > 40.0:
                    score -= 5000.0
            if score > best_score:
                best_score = score
                best_c = c
                best_area = c_area
    if best_c is not None:
        pts_resampled = resample_contour(best_c, 32)
        return pts_resampled, best_area, best_c
    if prev_contour is not None and prev_area is not None:
        return prev_contour, prev_area, None
    pts_resampled = []
    cx, cy = 112.0, 112.0
    R = 65.0
    for j in range(32):
        theta = 2 * np.pi * j / 32
        x_pt = cx + R * np.cos(theta)
        y_pt = cy + R * np.sin(theta)
        pts_resampled.append([float(x_pt), float(y_pt)])
    area = np.pi * R**2
    return pts_resampled, area, None

def get_ellipse_params(contour):
    if contour is None or len(contour) < 5:
        return 1.0, 0.0
    ellipse = cv2.fitEllipse(contour)
    (cx, cy), (w_el, h_el), angle = ellipse
    major_axis = max(w_el, h_el)
    minor_axis = min(w_el, h_el)
    aspect_ratio = minor_axis / major_axis if major_axis > 0 else 1.0
    if h_el >= w_el:
        major_angle = angle
    else:
        major_angle = (angle + 90) % 180
    return aspect_ratio, major_angle

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return None
        
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
                    
    if valid_boxes:
        x = int(np.median([box[0] for box in valid_boxes]))
        y = int(np.median([box[1] for box in valid_boxes]))
        w = int(np.median([box[2] for box in valid_boxes]))
        h = int(np.median([box[3] for box in valid_boxes]))
    else:
        middle_frame = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_c)
            else:
                x, y, w, h = 0, 0, frame.shape[1], frame.shape[0]
        else:
            x, y, w, h = 0, 0, 640, 480
            
    mask_circle = np.zeros((224, 224), dtype=np.uint8)
    cv2.circle(mask_circle, (112, 112), 108, 255, -1)
    
    sample_rate = 2
    sampled_frames = []
    raw_areas = []
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev_c = None
    prev_a = None
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        if i % sample_rate != 0: continue
        crop_f = frame[y:y+h, x:x+w]
        crop_f_resized = cv2.resize(crop_f, (224, 224))
        pts, area, raw_c = extract_robust_contour(crop_f_resized, mask_circle, prev_c, prev_a)
        prev_c = pts
        prev_a = area
        raw_areas.append(area)
        sampled_frames.append(i)
        
    raw_areas = np.array(raw_areas)
    med_areas = np.copy(raw_areas)
    for idx in range(len(raw_areas)):
        start_w = max(0, idx - 2)
        end_w = min(len(raw_areas), idx + 3)
        med_areas[idx] = np.median(raw_areas[start_w:end_w])
        
    smoothed_areas = np.copy(med_areas)
    for idx in range(len(med_areas)):
        start_w = max(0, idx - 2)
        end_w = min(len(med_areas), idx + 3)
        smoothed_areas[idx] = np.mean(med_areas[start_w:end_w])
        
    local_maxima = []
    local_minima = []
    n_samples = len(smoothed_areas)
    for i in range(1, n_samples - 1):
        if smoothed_areas[i] >= smoothed_areas[i-1] and smoothed_areas[i] >= smoothed_areas[i+1]:
            local_maxima.append(i)
        elif smoothed_areas[i] <= smoothed_areas[i-1] and smoothed_areas[i] <= smoothed_areas[i+1]:
            local_minima.append(i)
            
    best_max_idx = None
    best_min_idx = None
    max_drop = -1
    
    for mx in local_maxima:
        mins_after = [mn for mn in local_minima if mx < mn <= mx + 40]
        for mn in mins_after:
            drop = smoothed_areas[mx] - smoothed_areas[mn]
            if mx > n_samples * 0.05 and mn < n_samples * 0.95:
                if drop > max_drop:
                    max_drop = drop
                    best_max_idx = mx
                    best_min_idx = mn
                
    if best_max_idx is not None and best_min_idx is not None:
        peak_open_frame = sampled_frames[best_max_idx]
        peak_collapse_frame = sampled_frames[best_min_idx]
    else:
        peak_collapse_idx = int(np.argmin(smoothed_areas))
        peak_collapse_frame = sampled_frames[peak_collapse_idx]
        prev_max_areas = smoothed_areas[:peak_collapse_idx]
        if len(prev_max_areas) > 0:
            peak_open_idx = int(np.argmax(prev_max_areas))
            peak_open_frame = sampled_frames[peak_open_idx]
        else:
            peak_open_frame = max(0, peak_collapse_frame - 20)
            
    L = abs(peak_collapse_frame - peak_open_frame)
    k = min(30, max(15, L))
    start_f = max(0, peak_collapse_frame - k)
    end_f = start_f + 40
    
    if end_f > total_frames:
        end_f = total_frames
        start_f = max(0, end_f - 40)
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    slice_areas = []
    frame_raw_contours = []
    
    prev_c = None
    prev_a = None
    for i in range(start_f, end_f):
        ret, frame = cap.read()
        if not ret: break
        crop_f = frame[y:y+h, x:x+w]
        crop_f_resized = cv2.resize(crop_f, (224, 224))
        pts, area, raw_c = extract_robust_contour(crop_f_resized, mask_circle, prev_c, prev_a)
        prev_c = pts
        prev_a = area
        slice_areas.append(area)
        frame_raw_contours.append(raw_c)
        
    cap.release()
    
    max_area = max(slice_areas)
    min_area = min(slice_areas)
    
    max_idx = int(np.argmax(slice_areas))
    min_idx = int(np.argmin(slice_areas))
    
    # Find mid-collapse frame (closest to mid-area between max and min area)
    mid_target_area = (max_area + min_area) / 2.0
    mid_idx = int(np.argmin([abs(a - mid_target_area) for a in slice_areas]))
    
    aspect_max, angle_max = get_ellipse_params(frame_raw_contours[max_idx])
    aspect_mid, angle_mid = get_ellipse_params(frame_raw_contours[mid_idx])
    aspect_min, angle_min = get_ellipse_params(frame_raw_contours[min_idx])
    
    # Calculate average aspect ratio and angle over frames where area is in the lower 30% of open area (i.e. highly collapsed phase)
    low_area_indices = [idx for idx, a in enumerate(slice_areas) if a <= min_area + 0.3 * (max_area - min_area)]
    low_aspects = []
    low_angles = []
    for idx in low_area_indices:
        asp, ang = get_ellipse_params(frame_raw_contours[idx])
        if frame_raw_contours[idx] is not None and len(frame_raw_contours[idx]) >= 5:
            low_aspects.append(asp)
            low_angles.append(ang)
            
    avg_low_aspect = np.mean(low_aspects) if low_aspects else aspect_min
    avg_low_angle = np.mean(low_angles) if low_angles else angle_min
    
    return {
        "max": {"aspect": aspect_max, "angle": angle_max, "area": max_area},
        "mid": {"aspect": aspect_mid, "angle": angle_mid, "area": (max_area + min_area)/2},
        "min": {"aspect": aspect_min, "angle": angle_min, "area": min_area},
        "avg_low": {"aspect": avg_low_aspect, "angle": avg_low_angle},
        "reduction": (max_area - min_area) / max_area * 100
    }

def run_analysis():
    video_dir = r"video DISE patient 001-100"
    print("Running detailed geometric analysis on all 30 patient clips...")
    print(f"{'Pt':<4} | {'GT Class':<10} | {'GT Red':<6} | {'Max Area':<8} | {'Min Area':<8} | {'Red %':<6} | {'AspMax':<6} | {'AspMid':<6} | {'AspMin':<6} | {'AspAvgLow':<9} | {'AngAvgLow':<9}")
    print("-" * 115)
    
    for i in range(1, 31):
        filename = f"pt{i:03d}.mp4"
        filepath = os.path.join(video_dir, filename)
        if not os.path.exists(filepath):
            continue
        gt = ground_truth_db[i]
        res = analyze_video(filepath)
        if res is None:
            continue
        print(f"{i:02d}   | {gt['class']:<10} | {gt['reduction']:<6.1f} | {res['max']['area']:<8.1f} | {res['min']['area']:<8.1f} | {res['reduction']:<6.1f} | {res['max']['aspect']:<6.2f} | {res['mid']['aspect']:<6.2f} | {res['min']['aspect']:<6.2f} | {res['avg_low']['aspect']:<9.2f} | {res['avg_low']['angle']:<9.1f}")

if __name__ == '__main__':
    run_analysis()
