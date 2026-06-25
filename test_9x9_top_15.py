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

agreement_cases = [4, 6, 13, 14, 20, 21, 24]

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
    
    thresholds = range(15, 51, 5)
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
            if c_area < 100:
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
            if dist_from_center > 95:
                continue
                
            solidity = 1.0
            if len(c) >= 5:
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = c_area / hull_area
                    
            if solidity < 0.55:
                continue
                
            score = c_area - 0.4 * (dist_from_center ** 2) + 300.0 * solidity
            
            if prev_centroid is not None and prev_area is not None:
                dist_from_prev = np.sqrt((cx_pt - prev_centroid[0])**2 + (cy_pt - prev_centroid[1])**2)
                area_change = abs(c_area - prev_area)
                
                score -= 6.0 * dist_from_prev
                if area_change > 1200.0:
                    score -= 3000.0
                if dist_from_prev > 45.0:
                    score -= 3000.0
                    
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

def evaluate_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return None
        
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
        mn = min(w, h)
        x, y, w, h = (w - mn)//2, (h - mn)//2, mn, mn
        
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
        pts, area, _ = extract_robust_contour(crop_f_resized, mask_circle, prev_c, prev_a)
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
    
    max_lumen_area = max(slice_areas) if slice_areas else 1.0
    min_lumen_area = min(slice_areas) if slice_areas else 0.0
    
    sorted_areas = sorted(slice_areas)
    # 15% averaging window
    n_collapsed = max(1, int(len(sorted_areas) * 0.15))
    avg_low_area = np.mean(sorted_areas[:n_collapsed])
    reduction_percent = (max_lumen_area - avg_low_area) / max_lumen_area * 100
    
    max_idx = int(np.argmax(slice_areas))
    open_contour = frame_raw_contours[max_idx]
    
    low_area_indices = [idx for idx, a in enumerate(slice_areas) if a <= min_lumen_area + 0.3 * (max_lumen_area - min_lumen_area)]
    aspect_ratios = []
    angles = []
    contours_to_use = []
    for idx in low_area_indices:
        c = frame_raw_contours[idx]
        if c is not None and len(c) >= 5:
            contours_to_use.append(c)
            ellipse = cv2.fitEllipse(c)
            (cx, cy), (w_el, h_el), angle = ellipse
            major = max(w_el, h_el)
            minor = min(w_el, h_el)
            asp = minor / major if major > 0 else 1.0
            ang = angle if h_el >= w_el else (angle + 90) % 180
            aspect_ratios.append(asp)
            angles.append(ang)
            
    avg_aspect = np.mean(aspect_ratios) if aspect_ratios else 1.0
    avg_angle = np.mean(angles) if angles else 0.0
    
    if open_contour is not None and len(open_contour) >= 5:
        open_ellipse = cv2.fitEllipse(open_contour)
        (open_cx, open_cy), (open_w, open_h), open_angle = open_ellipse
        baseline_angle = open_angle if open_h >= open_w else (open_angle + 90) % 180
    else:
        open_cx, open_cy = 112.0, 112.0
        baseline_angle = 90.0
        
    def get_rotated_bounding_box(contour, cx, cy, angle_deg):
        if contour is None or len(contour) == 0:
            return 1.0, 1.0
        pts = contour.reshape(-1, 2).astype(np.float32)
        angle_rad = -np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        pts_centered = pts - np.array([cx, cy])
        pts_rotated = np.dot(pts_centered, R.T)
        xs = pts_rotated[:, 0]
        ys = pts_rotated[:, 1]
        return max(1.0, np.max(xs) - np.min(xs)), max(1.0, np.max(ys) - np.min(ys))
        
    w_open, h_open = get_rotated_bounding_box(open_contour, open_cx, open_cy, baseline_angle)
    
    w_coll_list, h_coll_list = [], []
    for c in contours_to_use:
        wc, hc = get_rotated_bounding_box(c, open_cx, open_cy, baseline_angle)
        w_coll_list.append(wc)
        h_coll_list.append(hc)
        
    w_collapsed = np.mean(w_coll_list) if w_coll_list else w_open
    h_collapsed = np.mean(h_coll_list) if h_coll_list else h_open
    
    red_major = 1.0 - (w_collapsed / w_open)
    red_minor = 1.0 - (h_collapsed / h_open)
    
    prediction_class = 'Concentric'
    if reduction_percent <= 50:
        prediction_class = 'Normal'
        degree = 0
    else:
        if reduction_percent > 75:
            degree = 2
        else:
            degree = 1
            
        if red_major > 0.45 and red_minor > 0.45 and avg_aspect >= 0.35:
            prediction_class = 'Concentric'
        else:
            if 45.0 <= avg_angle <= 135.0:
                prediction_class = 'AP'
            else:
                prediction_class = 'Lateral'
                
    return prediction_class, degree, reduction_percent

def evaluate_all():
    video_dir = r"video DISE patient 001-100"
    print("Evaluating 9x9 Morphological Closing WITHOUT recovery (15% Window)...\n")
    agreed_matches = 0
    overall_matches = 0
    
    for i in range(1, 31):
        filename = f"pt{i:03d}.mp4"
        filepath = os.path.join(video_dir, filename)
        if not os.path.exists(filepath): continue
        
        gt = ground_truth_db[i]
        res = evaluate_video(filepath)
        if res is None: continue
        
        pred_class, pred_deg, pred_red = res
        match = (gt["class"] == pred_class) and (gt["degree"] == pred_deg)
        
        if match:
            overall_matches += 1
            if i in agreement_cases:
                agreed_matches += 1
                
        status = "MATCH" if match else "MISMATCH"
        print(f"Pt{i:02d} | GT: {gt['class']} Deg {gt['degree']} | Pred: {pred_class} Deg {pred_deg} ({pred_red:.1f}%) | {status}")
        
    print(f"\nSummary (9x9, No Recovery, 15% Window):")
    print(f"Agreement Cases: {agreed_matches}/7 ({agreed_matches/7*100:.1f}%)")
    print(f"Overall Cases: {overall_matches}/30 ({overall_matches/30*100:.1f}%)")

if __name__ == '__main__':
    evaluate_all()
