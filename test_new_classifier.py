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
    
    # Limit threshold search range to prevent segmenting dark tissues and outer circular boundary
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
            if c_area < 100:  # Slightly lower area limit to catch tiny closed lumens
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
                
            # Base score: favor larger area, centrality, and solidity
            score = c_area - 0.4 * (dist_from_center ** 2) + 300.0 * solidity
            
            # Tracking continuity constraints
            if prev_centroid is not None and prev_area is not None:
                dist_from_prev = np.sqrt((cx_pt - prev_centroid[0])**2 + (cy_pt - prev_centroid[1])**2)
                area_change = abs(c_area - prev_area)
                
                # Soften penalty if we are recovering from a small area trap
                # i.e., if prev_area was small and new contour is large and solid, allow it!
                is_recovery = (prev_area < 1200) and (c_area > 1500) and (solidity > 0.8)
                
                if not is_recovery:
                    score -= 6.0 * dist_from_prev
                    if area_change > 1200.0:
                        score -= 3000.0
                    if dist_from_prev > 45.0:
                        score -= 3000.0
                else:
                    # Give a boost for recovery to correct contour
                    score += 500.0
                    
            if score > best_score:
                best_score = score
                best_c = c
                best_area = c_area
                
    if best_c is not None:
        pts_resampled = resample_contour(best_c, 32)
        return pts_resampled, best_area, best_c
        
    if prev_contour is not None and prev_area is not None:
        return prev_contour, prev_area, None
        
    # Default circular backup contour
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

def analyze_and_classify(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return None
        
    # ROI Detection: Scan frames to find the actual endoscopy circular frame
    frame_indices = [int(total_frames * p) for p in [0.05, 0.15, 0.3, 0.5, 0.7, 0.85]]
    valid_boxes = []
    for f_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret: continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # We use Otsu's thresholding to find the bright circle of the endoscopy
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
        # Fallback to center crop
        middle_frame = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        if ret:
            x, y, w, h = 0, 0, frame.shape[1], frame.shape[0]
            # Try to crop square from center
            mn = min(w, h)
            x = (w - mn) // 2
            y = (h - mn) // 2
            w = mn
            h = mn
        else:
            x, y, w, h = 0, 0, 640, 480
            
    mask_circle = np.zeros((224, 224), dtype=np.uint8)
    cv2.circle(mask_circle, (112, 112), 108, 255, -1)
    
    sample_rate = 2
    sampled_frames = []
    raw_areas = []
    raw_contours = []
    
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
        raw_contours.append(raw_c)
        sampled_frames.append(i)
        
    # Smooth the area signal using Median & Moving Average
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
        
    # Find breathing cycles (local max/min)
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
    reduction_percent = ((max_lumen_area - min_lumen_area) / max_lumen_area * 100) if max_lumen_area > 0 else 0.0
    
    # Classification Logic
    max_idx = int(np.argmax(slice_areas))
    min_idx = int(np.argmin(slice_areas))
    
    open_contour = frame_raw_contours[max_idx]
    collapsed_contour = frame_raw_contours[min_idx]
    
    # Find the average contour properties in the lowest 30% area phase to reduce noise
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
    
    # Baseline anatomy rotation & projection
    if open_contour is not None and len(open_contour) >= 5:
        open_ellipse = cv2.fitEllipse(open_contour)
        (open_cx, open_cy), (open_w, open_h), open_angle = open_ellipse
        baseline_angle = open_angle if open_h >= open_w else (open_angle + 90) % 180
    else:
        open_cx, open_cy = 112.0, 112.0
        baseline_angle = 90.0 # Default horizontal baseline
        
    # Align and project open contour
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
        width = np.max(xs) - np.min(xs)
        height = np.max(ys) - np.min(ys)
        return max(1.0, width), max(1.0, height)
        
    w_open, h_open = get_rotated_bounding_box(open_contour, open_cx, open_cy, baseline_angle)
    
    # Calculate average rotated box in collapsed phase
    w_coll_list = []
    h_coll_list = []
    for c in contours_to_use:
        wc, hc = get_rotated_bounding_box(c, open_cx, open_cy, baseline_angle)
        w_coll_list.append(wc)
        h_coll_list.append(hc)
        
    w_collapsed = np.mean(w_coll_list) if w_coll_list else w_open
    h_collapsed = np.mean(h_coll_list) if h_coll_list else h_open
    
    # Reductions along baseline major and minor axes
    red_major = 1.0 - (w_collapsed / w_open)
    red_minor = 1.0 - (h_collapsed / h_open)
    
    # Classification Rule
    # If both axes reduce significantly, or aspect ratio is high, it is Concentric.
    # Concentric collapse usually has red_major and red_minor both high, and aspect ratio relatively high.
    # AP collapse: vertical collapse is dominant (height reduces much more than width).
    # Lateral collapse: horizontal collapse is dominant (width reduces much more than height).
    
    # Let's check which class fits best
    prediction_class = 'Concentric'
    
    if reduction_percent <= 50:
        prediction_class = 'Normal'
        degree = 0
    else:
        # Determine degree of collapse
        if reduction_percent > 75:
            degree = 2
        else:
            degree = 1
            
        # Classify pattern
        # If it collapses from all sides (Concentric)
        # Clinically, Concentric collapse reduces both dimensions.
        # Let's look at the ratio of reduction:
        if red_major > 0.45 and red_minor > 0.45 and avg_aspect >= 0.38:
            prediction_class = 'Concentric'
        else:
            # Squeezed along one dimension
            # AP collapse: squeezes vertically. If the baseline angle is horizontal (e.g. 60 to 120 deg),
            # then minor axis is vertical. So vertical dimension H reduces.
            # Let's check the angle of the collapsed ellipse:
            # If avg_angle is between 45 and 135 degrees, the major axis of collapsed ellipse is horizontal,
            # which is AP collapse.
            if 45.0 <= avg_angle <= 135.0:
                prediction_class = 'AP'
            else:
                prediction_class = 'Lateral'
                
    return {
        "prediction_class": prediction_class,
        "degree": degree,
        "reduction": reduction_percent,
        "roi": (x, y, w, h),
        "max_area": max_lumen_area,
        "min_area": min_lumen_area,
        "red_major": red_major,
        "red_minor": red_minor,
        "avg_aspect": avg_aspect,
        "avg_angle": avg_angle
    }

def evaluate_all():
    video_dir = r"video DISE patient 001-100"
    print("Evaluating NEW tracking & classification algorithms...\n")
    print(f"{'Patient':<8} | {'GT Class':<10} | {'Pred Class':<10} | {'GT Deg':<6} | {'Pred Deg':<8} | {'GT Red':<6} | {'Pred Red':<8} | {'ROI BBox':<16} | {'RedMaj':<6} | {'RedMin':<6} | {'Status'}")
    print("-" * 125)
    
    agreed_matches = 0
    test_matches = 0
    total_agreed = len(agreement_cases)
    total_test = 30 - total_agreed
    
    for i in range(1, 31):
        filename = f"pt{i:03d}.mp4"
        filepath = os.path.join(video_dir, filename)
        if not os.path.exists(filepath):
            continue
            
        gt = ground_truth_db[i]
        pred = analyze_and_classify(filepath)
        
        if pred is None:
            continue
            
        class_match = (gt["class"] == pred["prediction_class"])
        deg_match = (gt["degree"] == pred["degree"])
        match = class_match and deg_match
        
        status = "MATCH" if match else "MISMATCH"
        if not class_match and deg_match:
            status += " (Class mismatch)"
        elif class_match and not deg_match:
            status += " (Degree mismatch)"
        elif not class_match and not deg_match:
            status += " (Both mismatch)"
            
        is_agreement = i in agreement_cases
        case_type = "Agreed" if is_agreement else "Test"
        
        if is_agreement:
            if match:
                agreed_matches += 1
        else:
            if match:
                test_matches += 1
                
        roi_str = f"({pred['roi'][0]},{pred['roi'][1]},{pred['roi'][2]},{pred['roi'][3]})"
        print(f"Pt{i:02d} ({case_type:<6}) | {gt['class']:<10} | {pred['prediction_class']:<10} | {gt['degree']:<6} | {pred['degree']:<8} | {gt['reduction']:<6.1f} | {pred['reduction']:<8.1f} | {roi_str:<16} | {pred['red_major']:<6.2f} | {pred['red_minor']:<6.2f} | {status}")
        
    print("\nSummary statistics:")
    print(f"Agreement Cases Accuracy: {agreed_matches}/{total_agreed} ({agreed_matches/total_agreed*100:.1f}%)")
    print(f"Test Cases Accuracy: {test_matches}/{total_test} ({test_matches/total_test*100:.1f}%)")
    print(f"Overall Accuracy: {agreed_matches + test_matches}/30 ({(agreed_matches + test_matches)/30*100:.1f}%)")

if __name__ == '__main__':
    evaluate_all()
