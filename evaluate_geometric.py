import os
import cv2
import numpy as np

# Ground truth database
ground_truth_db = {
    1: {"class": "AP", "degree": 2, "reduction": 88.5, "aspect": 0.25, "angle": 90.0},
    2: {"class": "AP", "degree": 2, "reduction": 91.2, "aspect": 0.22, "angle": 90.0},
    3: {"class": "AP", "degree": 2, "reduction": 95.0, "aspect": 0.18, "angle": 90.0},
    4: {"class": "AP", "degree": 2, "reduction": 89.0, "aspect": 0.28, "angle": 90.0},
    5: {"class": "AP", "degree": 2, "reduction": 96.5, "aspect": 0.15, "angle": 90.0},
    6: {"class": "AP", "degree": 2, "reduction": 98.2, "aspect": 0.12, "angle": 90.0},
    7: {"class": "AP", "degree": 2, "reduction": 92.0, "aspect": 0.20, "angle": 90.0},
    8: {"class": "AP", "degree": 1, "reduction": 64.0, "aspect": 0.45, "angle": 90.0},
    9: {"class": "AP", "degree": 2, "reduction": 87.5, "aspect": 0.24, "angle": 90.0},
    10: {"class": "AP", "degree": 1, "reduction": 68.0, "aspect": 0.40, "angle": 90.0},
    11: {"class": "AP", "degree": 2, "reduction": 93.0, "aspect": 0.19, "angle": 90.0},
    12: {"class": "AP", "degree": 1, "reduction": 62.0, "aspect": 0.48, "angle": 90.0},
    13: {"class": "Normal", "degree": 0, "reduction": 12.0, "aspect": 0.85, "angle": 45.0},
    14: {"class": "Concentric", "degree": 2, "reduction": 94.5, "aspect": 0.72, "angle": 45.0},
    15: {"class": "AP", "degree": 1, "reduction": 65.5, "aspect": 0.42, "angle": 90.0},
    16: {"class": "Normal", "degree": 0, "reduction": 14.0, "aspect": 0.82, "angle": 45.0},
    17: {"class": "AP", "degree": 2, "reduction": 89.5, "aspect": 0.22, "angle": 90.0},
    18: {"class": "AP", "degree": 1, "reduction": 63.5, "aspect": 0.46, "angle": 90.0},
    19: {"class": "AP", "degree": 2, "reduction": 92.5, "aspect": 0.21, "angle": 90.0},
    20: {"class": "AP", "degree": 2, "reduction": 94.0, "aspect": 0.18, "angle": 90.0},
    21: {"class": "AP", "degree": 1, "reduction": 61.5, "aspect": 0.49, "angle": 90.0},
    22: {"class": "AP", "degree": 2, "reduction": 90.5, "aspect": 0.23, "angle": 90.0},
    23: {"class": "AP", "degree": 1, "reduction": 67.2, "aspect": 0.43, "angle": 90.0},
    24: {"class": "AP", "degree": 2, "reduction": 93.8, "aspect": 0.17, "angle": 90.0},
    25: {"class": "AP", "degree": 1, "reduction": 62.5, "aspect": 0.47, "angle": 90.0},
    26: {"class": "Concentric", "degree": 1, "reduction": 68.5, "aspect": 0.75, "angle": 45.0},
    27: {"class": "Concentric", "degree": 2, "reduction": 96.0, "aspect": 0.70, "angle": 45.0},
    28: {"class": "AP", "degree": 2, "reduction": 88.0, "aspect": 0.26, "angle": 90.0},
    29: {"class": "AP", "degree": 2, "reduction": 91.0, "aspect": 0.20, "angle": 90.0},
    30: {"class": "AP", "degree": 2, "reduction": 92.2, "aspect": 0.19, "angle": 90.0}
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
                is_recovery = (prev_area < 1200) and (c_area > 1500) and (solidity > 0.8)
                if not is_recovery:
                    score -= 6.0 * dist_from_prev
                    if area_change > 1200.0:
                        score -= 3000.0
                    if dist_from_prev > 45.0:
                        score -= 3000.0
                else:
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
    pts_resampled = []
    cx, cy = 112.0, 112.0
    R = 6.0
    for j in range(32):
        theta = 2 * np.pi * j / 32
        x_pt = cx + R * np.cos(theta)
        y_pt = cy + R * np.sin(theta)
        pts_resampled.append([float(x_pt), float(y_pt)])
    area = np.pi * R**2
    return pts_resampled, area, None

def evaluate_proposed_logic():
    video_dir = r"video DISE patient 001-100"
    matches = 0
    agreed_matches = 0
    total_agreed = len(agreement_cases)
    
    print("Evaluating with consecutive failures check and adaptive centering...\n")
    print(f"{'Patient':<8} | {'GT Class':<10} | {'Pred Class':<10} | {'GT Deg':<6} | {'Pred Deg':<8} | {'GT Red':<6} | {'Pred Red':<8} | {'Aspect':<6} | {'Status'}")
    print("-" * 100)
    
    for i in range(1, 31):
        filename = f"pt{i:03d}.mp4"
        filepath = os.path.join(video_dir, filename)
        if not os.path.exists(filepath):
            continue
            
        cap = cv2.VideoCapture(filepath)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            continue
            
        # ROI detection
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
            
        mask_circle = np.zeros((224, 224), dtype=np.uint8)
        cv2.circle(mask_circle, (112, 112), 108, 255, -1)
        
        sample_rate = 2
        sampled_frames = []
        raw_areas = []
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        prev_c = None
        prev_a = None
        consecutive_failures = 0
        for k in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            if k % sample_rate != 0: continue
            crop_f = frame[y:y+h, x:x+w]
            crop_f_resized = cv2.resize(crop_f, (224, 224))
            
            pts, area, raw_c = extract_robust_contour(crop_f_resized, mask_circle, prev_c, prev_a)
            if raw_c is None:
                consecutive_failures += 1
            else:
                consecutive_failures = 0
                
            if consecutive_failures > 3:
                prev_c = None
                prev_a = None
            else:
                prev_c = pts
                prev_a = area
                
            raw_areas.append(area)
            sampled_frames.append(k)
            
        cap.release()
        
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
        for k in range(1, n_samples - 1):
            if smoothed_areas[k] >= smoothed_areas[k-1] and smoothed_areas[k] >= smoothed_areas[k+1]:
                local_maxima.append(k)
            elif smoothed_areas[k] <= smoothed_areas[k-1] and smoothed_areas[k] <= smoothed_areas[k+1]:
                local_minima.append(k)
                
        # Chronological first transition logic with start_pct=0.22 and max_area=3500
        best_max_idx = None
        best_min_idx = None
        found = False
        
        for mx in sorted(local_maxima):
            if found:
                break
            if mx < n_samples * 0.22:
                continue
            if smoothed_areas[mx] > 3500:
                continue
                
            mins_after = [mn for mn in local_minima if mx < mn <= mx + 40]
            for mn in sorted(mins_after):
                drop = smoothed_areas[mx] - smoothed_areas[mn]
                if mn < n_samples * 0.90:
                    if drop > 600:
                        best_max_idx = mx
                        best_min_idx = mn
                        found = True
                        break
                        
        if not found:
            max_drop = -1
            for mx in local_maxima:
                if mx < n_samples * 0.22 or smoothed_areas[mx] > 3500:
                    continue
                mins_after = [mn for mn in local_minima if mx < mn <= mx + 40]
                for mn in mins_after:
                    drop = smoothed_areas[mx] - smoothed_areas[mn]
                    if mn < n_samples * 0.90:
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
                
        # Improved start frame logic: if transition is long (> 30 frames), start at peak_open - 5
        L_trans = peak_collapse_frame - peak_open_frame
        if L_trans > 30:
            start_f = max(0, peak_open_frame - 5)
        else:
            transition_center = (peak_open_frame + peak_collapse_frame) // 2
            start_f = max(0, transition_center - 20)
        end_f = start_f + 40
        if end_f > total_frames:
            end_f = total_frames
            start_f = max(0, end_f - 40)
            
        # Run classification
        cap = cv2.VideoCapture(filepath)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        slice_areas = []
        frame_raw_contours = []
        prev_c = None
        prev_a = None
        consecutive_failures = 0
        for k in range(start_f, end_f):
            ret, frame = cap.read()
            if not ret: break
            crop_f = frame[y:y+h, x:x+w]
            crop_f_resized = cv2.resize(crop_f, (224, 224))
            pts, area, raw_c = extract_robust_contour(crop_f_resized, mask_circle, prev_c, prev_a)
            if raw_c is None:
                consecutive_failures += 1
            else:
                consecutive_failures = 0
                
            if consecutive_failures > 3:
                prev_c = None
                prev_a = None
            else:
                prev_c = pts
                prev_a = area
            slice_areas.append(area)
            frame_raw_contours.append(raw_c)
        cap.release()
        
        max_lumen_area = max(slice_areas) if slice_areas else 1.0
        min_lumen_area = min(slice_areas) if slice_areas else 0.0
        reduction_percent = ((max_lumen_area - min_lumen_area) / max_lumen_area * 100) if max_lumen_area > 0 else 0.0
        
        classifiable_contour = None
        sorted_indices = np.argsort(slice_areas)
        for idx_c in sorted_indices:
            c = frame_raw_contours[idx_c]
            if c is not None and len(c) >= 5:
                classifiable_contour = c
                break
                
        prediction_class = 'Concentric'
        major_angle = 0.0
        aspect_ratio = 1.0
        
        if classifiable_contour is not None:
            ellipse = cv2.fitEllipse(classifiable_contour)
            (cx, cy), (w_el, h_el), angle = ellipse
            major_axis = max(w_el, h_el)
            minor_axis = min(w_el, h_el)
            aspect_ratio = minor_axis / major_axis if major_axis > 0 else 1.0
            if h_el >= w_el:
                major_angle = angle
            else:
                major_angle = (angle + 90) % 180
                
            if aspect_ratio >= 0.62:
                prediction_class = 'Concentric'
            else:
                score_ap = (1.0 - aspect_ratio) * (1.0 - abs(major_angle - 90.0) / 90.0)
                score_lateral = (1.0 - aspect_ratio) * (1.0 - min(major_angle, 180.0 - major_angle) / 90.0)
                if score_ap >= score_lateral:
                    prediction_class = 'AP'
                else:
                    prediction_class = 'Lateral'
        else:
            prediction_class = 'Concentric'
            
        if reduction_percent > 75:
            degree = 2
        elif reduction_percent > 50:
            degree = 1
        else:
            degree = 0
            prediction_class = 'Normal'
            
        gt = ground_truth_db[i]
        class_match = (gt["class"] == prediction_class)
        deg_match = (gt["degree"] == degree)
        is_match = class_match and deg_match
        if is_match:
            matches += 1
            if i in agreement_cases:
                agreed_matches += 1
                
        status = "MATCH" if is_match else "MISMATCH"
        if i == 2:
            print(f"**Pt02 Selected Range**: {start_f} - {end_f} ({start_f/30.0:.2f}s - {end_f/30.0:.2f}s)")
        print(f"Pt{i:02d} | {gt['class']:<10} | {prediction_class:<10} | {gt['degree']:<6} | {degree:<8} | {gt['reduction']:<6.1f} | {reduction_percent:<8.1f} | {aspect_ratio:<6.2f} | {status}")
        
    print(f"\nOverall Accuracy: {matches}/30 ({matches/30*100:.1f}%), Agreed Cases: {agreed_matches}/7")

evaluate_proposed_logic()
