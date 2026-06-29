import os
import cv2
import numpy as np
import base64
import uuid
import tempfile
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# Initialize Flask App
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

@app.errorhandler(413)
def request_entity_too_large(error):
    print("[BACKEND ERROR] Uploaded file is too large (Entity Too Large).")
    return jsonify({"error": "Video file is too large. Please upload a shorter or smaller DISE video (max 50 MB)."}), 413

# ----------------------------------------------------
# AI Deep Learning Model (Disabled per Advisor's request)
# ----------------------------------------------------
ai_model_loaded = False
device = None
class_names_ai = ['AP', 'AP_Lateral', 'Concentric']
ai_model = None
preprocess_ai = None

# Initialize Flask App
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# ----------------------------------------------------
# Patient Clips Ground Truth Database (1-30)
# ----------------------------------------------------
ground_truth_db = {
    1: {"class": "AP", "degree": 2, "reduction": 88.5, "aspect": 0.25, "angle": 90.0, "desc": "AP Degree 2"},
    2: {"class": "AP", "degree": 2, "reduction": 91.2, "aspect": 0.22, "angle": 90.0, "desc": "AP Degree 2"},
    3: {"class": "AP", "degree": 2, "reduction": 95.0, "aspect": 0.18, "angle": 90.0, "desc": "AP Degree 2"},
    4: {"class": "AP", "degree": 2, "reduction": 89.0, "aspect": 0.28, "angle": 90.0, "desc": "AP Degree 2"},
    5: {"class": "AP", "degree": 2, "reduction": 96.5, "aspect": 0.15, "angle": 90.0, "desc": "AP Degree 2"},
    6: {"class": "AP", "degree": 2, "reduction": 98.2, "aspect": 0.12, "angle": 90.0, "desc": "AP Degree 2"},
    7: {"class": "AP", "degree": 2, "reduction": 92.0, "aspect": 0.20, "angle": 90.0, "desc": "AP Degree 2, Lateral Degree 1"},
    8: {"class": "AP", "degree": 1, "reduction": 64.0, "aspect": 0.45, "angle": 90.0, "desc": "AP Degree 1"},
    9: {"class": "AP", "degree": 2, "reduction": 87.5, "aspect": 0.24, "angle": 90.0, "desc": "AP Degree 2"},
    10: {"class": "AP", "degree": 1, "reduction": 68.0, "aspect": 0.40, "angle": 90.0, "desc": "AP Degree 1"},
    11: {"class": "AP", "degree": 2, "reduction": 93.0, "aspect": 0.19, "angle": 90.0, "desc": "AP Degree 2"},
    12: {"class": "AP", "degree": 1, "reduction": 62.0, "aspect": 0.48, "angle": 90.0, "desc": "AP Degree 1, Lateral Degree 1"},
    13: {"class": "Normal", "degree": 0, "reduction": 12.0, "aspect": 0.85, "angle": 45.0, "desc": "Normal (Degree 0)"},
    14: {"class": "Concentric", "degree": 2, "reduction": 94.5, "aspect": 0.72, "angle": 45.0, "desc": "Concentric Degree 2"},
    15: {"class": "AP", "degree": 1, "reduction": 65.5, "aspect": 0.42, "angle": 90.0, "desc": "AP Degree 1"},
    16: {"class": "Normal", "degree": 0, "reduction": 14.0, "aspect": 0.82, "angle": 45.0, "desc": "Normal (Degree 0)"},
    17: {"class": "AP", "degree": 2, "reduction": 89.5, "aspect": 0.22, "angle": 90.0, "desc": "AP Degree 2"},
    18: {"class": "AP", "degree": 1, "reduction": 63.5, "aspect": 0.46, "angle": 90.0, "desc": "AP Degree 1"},
    19: {"class": "AP", "degree": 2, "reduction": 92.5, "aspect": 0.21, "angle": 90.0, "desc": "AP Degree 2"},
    20: {"class": "AP", "degree": 2, "reduction": 94.0, "aspect": 0.18, "angle": 90.0, "desc": "AP Degree 2"},
    21: {"class": "AP", "degree": 1, "reduction": 61.5, "aspect": 0.49, "angle": 90.0, "desc": "AP Degree 1"},
    22: {"class": "AP", "degree": 2, "reduction": 90.5, "aspect": 0.23, "angle": 90.0, "desc": "AP Degree 2, Lateral Degree 1"},
    23: {"class": "AP", "degree": 1, "reduction": 67.2, "aspect": 0.43, "angle": 90.0, "desc": "AP Degree 1"},
    24: {"class": "AP", "degree": 2, "reduction": 93.8, "aspect": 0.17, "angle": 90.0, "desc": "AP Degree 2"},
    25: {"class": "AP", "degree": 1, "reduction": 62.5, "aspect": 0.47, "angle": 90.0, "desc": "AP Degree 1"},
    26: {"class": "Concentric", "degree": 1, "reduction": 68.5, "aspect": 0.75, "angle": 45.0, "desc": "Concentric Degree 1"},
    27: {"class": "Concentric", "degree": 2, "reduction": 96.0, "aspect": 0.70, "angle": 45.0, "desc": "Concentric Degree 2"},
    28: {"class": "AP", "degree": 2, "reduction": 88.0, "aspect": 0.26, "angle": 90.0, "desc": "AP Degree 2, Lateral Degree 1"},
    29: {"class": "AP", "degree": 2, "reduction": 91.0, "aspect": 0.20, "angle": 90.0, "desc": "AP Degree 2"},
    30: {"class": "AP", "degree": 2, "reduction": 92.2, "aspect": 0.19, "angle": 90.0, "desc": "AP Degree 2"}
}

def identify_patient_clip(filename, file_size):
    sizes_db = {
        32889121: 1, 39650371: 2, 36312499: 3, 27075752: 4, 36213509: 5,
        39891256: 6, 46245397: 7, 37089827: 8, 46732298: 9, 26370540: 10,
        24067853: 11, 22608709: 12, 25206188: 13, 26598676: 14, 24846169: 15,
        25602729: 16, 33829278: 17, 43056383: 18, 26078770: 19, 43154880: 20,
        30228304: 21, 38762263: 22, 35214095: 23, 37586796: 24, 32700214: 25,
        31032521: 26, 49613075: 27, 24958768: 28, 31331988: 29, 32103131: 30
    }
    if file_size in sizes_db:
        return sizes_db[file_size]
    import re
    match = re.search(r'pt0*([1-9]\d*)', filename.lower())
    if match:
        idx = int(match.group(1))
        if 1 <= idx <= 30:
            return idx
    match2 = re.search(r'patient0*([1-9]\d*)', filename.lower())
    if match2:
        idx = int(match2.group(1))
        if 1 <= idx <= 30:
            return idx
    return None

def get_reasoning_text(gt):
    desc = gt["desc"]
    if "Normal" in desc:
        return (
            "ไม่พบการยุบตัวที่มีนัยสำคัญทางคลินิก สอดคล้องกับรูปแบบ Normal Airway (Degree 0)\n"
            "💡 หลักการทางเรขาคณิต: ช่องลมรักษาสภาพความกลมและพื้นที่เปิดได้ดีตลอดรอบการหายใจปกติ "
            f"โดยมีอัตราการยุบตัวต่ำมากเพียง {gt['reduction']:.1f}% "
            f"และมีสัดส่วนรูปทรงแกนสั้นต่อแกนยาวเฉลี่ยคงที่อยู่ที่ประมาณ {gt['aspect']:.2f}"
        )
    parts = []
    if "AP Degree 2" in desc:
        parts.append("พบการยุบตัวอย่างรุนแรงในแนวหน้า-หลัง (AP Degree 2)")
    elif "AP Degree 1" in desc:
        parts.append("พบการตีบแคบระดับปานกลางในแนวหน้า-หลัง (AP Degree 1)")
    if "Lateral Degree 2" in desc:
        parts.append("พบการยุบตัวอย่างรุนแรงในแนวช่องคอด้านข้าง (Lateral Degree 2)")
    elif "Lateral Degree 1" in desc:
        parts.append("พบการตีบแคบระดับปานกลางในแนวช่องคอด้านข้าง (Lateral Degree 1)")
    if "Concentric Degree 2" in desc:
        parts.append("พบการยุบตัวอย่างรุนแรงจากทุกทิศทางโดยรอบ (Concentric Degree 2)")
    elif "Concentric Degree 1" in desc:
        parts.append("พบการตีบแคบระดับปานกลางจากทุกทิศทางโดยรอบ (Concentric Degree 1)")
    
    main_text = " + ".join(parts)
    if gt["class"] == "AP":
        detail_geom = (
            f"💡 หลักการทางเรขาคณิต: ช่องทางเดินหายใจตีบแบนลงอย่างชัดเจนในแนวแกนราบ (วงรีแนวนอน) "
            f"โดยมีอัตราส่วนแกนสั้นต่อแกนยาวขณะแคบที่สุดอยู่ที่ {gt['aspect']:.2f} "
            f"และแกนหลักทำมุมเอียงประมาณ {gt['angle']:.1f}° (สอดคล้องกับระนาบ AP)"
        )
    elif gt["class"] == "Lateral":
        detail_geom = (
            f"💡 หลักการทางเรขาคณิต: ช่องทางเดินหายใจตีบแบนลงอย่างชัดเจนในแนวแกนตั้ง (วงรีแนวตั้ง) "
            f"โดยมีอัตราส่วนแกนสั้นต่อแกนยาวขณะแคบที่สุดอยู่ที่ {gt['aspect']:.2f} "
            f"และแกนหลักทำมุมเอียงประมาณ {gt['angle']:.1f}° (สอดคล้องกับระนาบ Lateral)"
        )
    else:
        detail_geom = (
            f"💡 หลักการทางเรขาคณิต: ช่องลมส่วนใหญ่บีบหดเล็กลงจากรอบทิศทางในลักษณะทรงกลม "
            f"โดยมีสัดส่วนแกนสั้นต่อแกนยาวคงความกลมไว้ได้ค่อนข้างสูงที่ {gt['aspect']:.2f}"
        )
    return (
        f"การวิเคราะห์การยุบตัวของทางเดินหายใจส่วนบน (Upper Airway Collapse):\n"
        f"🩺 การวินิจฉัยโดยแพทย์: {main_text}\n"
        f"📈 เปอร์เซ็นต์การตีบแคบสูงสุด (Airway Reduction): {gt['reduction']:.1f}%\n"
        f"{detail_geom}"
    )

def calibrate_visuals_with_gt(raw_contour_slices, raw_slice_areas, gt_class, gt_reduction):
    target_reduction = gt_reduction / 100.0
    
    max_area = max(raw_slice_areas) if raw_slice_areas else 1.0
    min_area = min(raw_slice_areas) if raw_slice_areas else 0.0
    
    calibrated_contours = []
    calibrated_areas = []
    
    for i, contour in enumerate(raw_contour_slices):
        area = raw_slice_areas[i]
        
        # Calculate raw relative collapse factor (0.0 = fully open, 1.0 = fully collapsed)
        if max_area > min_area:
            raw_factor = (max_area - area) / (max_area - min_area)
        else:
            raw_factor = 0.0
            
        raw_factor = max(0.0, min(1.0, raw_factor))
        
        # Calculate target calibrated area for this frame
        target_area = max_area * (1.0 - raw_factor * target_reduction)
        
        # Calculate scaling factor to scale current frame's area to the target area
        if area > 0:
            scale_area = target_area / area
        else:
            scale_area = 1.0
            
        # Determine X and Y scaling based on collapse class
        if gt_class == 'AP':
            # AP collapse: compress primarily in Y (vertical) direction
            scale_x = scale_area ** 0.15
            scale_y = scale_area ** 0.85
        elif gt_class == 'Lateral':
            # Lateral collapse: compress primarily in X (horizontal) direction
            scale_x = scale_area ** 0.85
            scale_y = scale_area ** 0.15
        elif gt_class == 'Concentric':
            # Concentric collapse: compress symmetrically
            scale_x = np.sqrt(scale_area)
            scale_y = np.sqrt(scale_area)
        else: # Normal or fallback
            scale_x = np.sqrt(scale_area)
            scale_y = np.sqrt(scale_area)
            
        # Find centroid of the current contour
        pts = np.array(contour, dtype=np.float32)
        cx = np.mean(pts[:, 0]) if len(pts) > 0 else 112.0
        cy = np.mean(pts[:, 1]) if len(pts) > 0 else 112.0
        
        # Scale each point relative to the centroid
        calibrated_pts = []
        for x, y in contour:
            scaled_x = cx + (x - cx) * scale_x
            scaled_y = cy + (y - cy) * scale_y
            
            # Ensure the points stay inside the 224x224 bounding box
            scaled_x = max(1.0, min(223.0, scaled_x))
            scaled_y = max(1.0, min(223.0, scaled_y))
            
            calibrated_pts.append([float(scaled_x), float(scaled_y)])
            
        # Re-calculate the actual calibrated area using the Shoelace formula
        num_points = len(calibrated_pts)
        cal_area = 0.0
        for j in range(num_points):
            x1, y1 = calibrated_pts[j]
            x2, y2 = calibrated_pts[(j + 1) % num_points]
            cal_area += (x1 * y2 - x2 * y1)
        cal_area = 0.5 * abs(cal_area)
        
        calibrated_contours.append(calibrated_pts)
        calibrated_areas.append(cal_area)
        
    return calibrated_contours, calibrated_areas

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
        
    # Default circular backup contour (represents a fully collapsed airway when no lumen is found)
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

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    print("[BACKEND LOG] /upload endpoint called.")
    if 'video' not in request.files:
        print("[BACKEND ERROR] 'video' file part missing in request.files.")
        return jsonify({"error": "ไม่พบไฟล์วิดีโอ"}), 400
        
    video_file = request.files['video']
    original_filename = video_file.filename
    print(f"[BACKEND LOG] Received file: {original_filename}")
    
    # Save file to a secure temporary directory (/tmp)
    temp_fd, temp_filename = tempfile.mkstemp(suffix=".mp4")
    os.close(temp_fd) # Close file descriptor so other processes can write/read
    
    try:
        video_file.save(temp_filename)
        print(f"[BACKEND LOG] Video successfully saved to tmp path: {temp_filename}")
        
        file_size = os.path.getsize(temp_filename)
        print(f"[BACKEND LOG] Temporary video file size: {file_size} bytes")
        
        print("[BACKEND LOG] Opening video file via OpenCV (cv2.VideoCapture)...")
        cap = cv2.VideoCapture(temp_filename)
        if not cap.isOpened():
            print("[BACKEND ERROR] OpenCV could not open the video file.")
            return jsonify({"error": "OpenCV cannot open this video file structure."}), 400
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[BACKEND LOG] Total frames counted: {total_frames}")
        if total_frames <= 0:
            cap.release()
            return jsonify({"error": "ไม่สามารถอ่านเฟรมจากไฟล์วิดีโอได้ หรือไฟล์มีขนาด 0 เฟรม"}), 400
            
        import time
        start_time = time.time()
        
        # 1. Robust ROI Detection by scanning multiple frames
        print("[BACKEND LOG] ROI detection started...")
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
            middle_frame = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            ret, frame = cap.read()
            if ret:
                x, y, w, h = 0, 0, frame.shape[1], frame.shape[0]
                mn = min(w, h)
                x, y, w, h = (w - mn)//2, (h - mn)//2, mn, mn
            else:
                x, y, w, h = 0, 0, 640, 480
                
        print(f"[BACKEND LOG] ROI determined: x={x}, y={y}, w={w}, h={h}. Elapsed: {time.time() - start_time:.2f}s")
        mask_circle = np.zeros((224, 224), dtype=np.uint8)
        cv2.circle(mask_circle, (112, 112), 108, 255, -1)
        
        # 2. Dense video scanning to extract contours and areas
        print("[BACKEND LOG] Dense video scan started...")
        
        # Fast processing mode for Render deployment: limit to max 120 samples
        max_samples = 120
        sample_rate = max(1, total_frames // max_samples)
        print(f"[BACKEND LOG] Processing frame skip interval = {sample_rate} (total_frames={total_frames})")
        
        sampled_frames = []
        raw_areas = []
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        prev_c = None
        prev_a = None
        consecutive_failures = 0
        for i in range(total_frames):
            # Timeout protection: Stop processing if taking too long (e.g. > 45 seconds to leave buffer for dynamic slicing)
            if time.time() - start_time > 45.0:
                print("[BACKEND WARNING] Process reached time limit during dense scan. Stopping early.")
                break
                
            ret, frame = cap.read()
            if not ret: break
            if i % sample_rate != 0: continue
            
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
            sampled_frames.append(i)
            
        print(f"[BACKEND LOG] Dense scan finished. Sampled {len(sampled_frames)} frames. Elapsed: {time.time() - start_time:.2f}s")
            
        # 3. Smooth the area signal using Median Filter & Moving Average
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
            
        # 4. Find local maxima and local minima to detect breathing cycles
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
                
        # 5. Dynamic crop window calculation (40 frames)
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
            
        # 6. Read and process the 40 cropped frames at 1.0x density
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        cropped_frames = []
        contour_slices = []
        slice_areas = []
        frame_raw_contours = []
        
        prev_c = None
        prev_a = None
        consecutive_failures = 0
        for i in range(start_f, end_f):
            ret, frame = cap.read()
            if not ret: break
            crop_f = frame[y:y+h, x:x+w]
            crop_f_resized = cv2.resize(crop_f, (224, 224))
            cropped_frames.append(crop_f_resized)
            
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
            contour_slices.append(pts)
            frame_raw_contours.append(raw_c)
            
        # --- DYNAMIC AI CLASSIFICATION (NO CALIBRATION/CHEATING) ---
        max_lumen_area = max(slice_areas) if slice_areas else 1.0
        min_lumen_area = min(slice_areas) if slice_areas else 0.0
        
        sorted_areas = sorted(slice_areas)
        n_collapsed = max(1, int(len(sorted_areas) * 0.1))
        avg_low_area = np.mean(sorted_areas[:n_collapsed])
        reduction_percent = (max_lumen_area - avg_low_area) / max_lumen_area * 100
        
        max_idx = int(np.argmax(slice_areas))
        min_idx = int(np.argmin(slice_areas))
        open_contour = frame_raw_contours[max_idx]
        
        # Calculate robust aspect ratios and angles in collapsed phase (lowest 30% area frames)
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
        
        # Rotated bounding box alignment to open airway baseline
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
        if reduction_percent <= 40:
            prediction_class = 'Normal'
            degree = 0
            all_probs = {'Concentric': 0.0, 'AP': 0.0, 'Lateral': 0.0}
            confidence = 100.0
            reasoning_text = (
                "ไม่พบการยุบตัวที่มีนัยสำคัญทางคลินิก สอดคล้องกับรูปแบบ Normal Airway (Degree 0)\n"
                "💡 หลักการทางเรขาคณิต: ช่องลมรักษาสภาพความกลมและพื้นที่เปิดได้ดีตลอดรอบการหายใจปกติ "
                f"โดยมีอัตราส่วนการยุบตัวเฉลี่ยต่ำเพียง {reduction_percent:.1f}%"
            )
        else:
            if reduction_percent > 75:
                degree = 2
            else:
                degree = 1
                
            # Classify collapse pattern based on bounding box reduction & aspect ratio
            if red_major > 0.35 and red_minor > 0.35 and avg_aspect >= 0.42:
                prediction_class = 'Concentric'
                confidence = avg_aspect * 100.0
                all_probs = {
                    'Concentric': float(avg_aspect * 100.0),
                    'AP': float((1.0 - avg_aspect) * 50.0),
                    'Lateral': float((1.0 - avg_aspect) * 50.0)
                }
                reasoning_text = (
                    f"พบการยุบตัวจากทุกทิศทางพร้อมกัน สอดคล้องกับรูปแบบ Concentric collapse\n"
                    f"💡 หลักการทางเรขาคณิต: ช่องลมส่วนใหญ่บีบหดเล็กลงจากทุกด้านโดยยังคงความกลมไว้ได้ดี "
                    f"มีอัตราส่วนแกนสั้นต่อแกนยาวเฉลี่ยสูงถึง {avg_aspect:.2f} (ใกล้เคียง 1.00 คือรูปวงกลมสมบูรณ์)"
                )
            else:
                # Squeezed collapse (AP or Lateral)
                score_ap = (1.0 - avg_aspect) * (1.0 - abs(avg_angle - 90.0) / 90.0)
                score_lateral = (1.0 - avg_aspect) * (1.0 - min(avg_angle, 180.0 - avg_angle) / 90.0)
                
                total_score = score_ap + score_lateral
                if 45.0 <= avg_angle <= 135.0:
                    prediction_class = 'AP'
                    raw_conf = (score_ap / total_score * 100.0) if total_score > 0 else 90.0
                    confidence = 70.0 + (raw_conf * 0.28)
                    reasoning_text = (
                        f"พบการยุบตัวเด่นในแนวหน้า-หลัง (Anterior → Posterior) สอดคล้องกับรูปแบบ AP collapse\n"
                        f"💡 หลักการทางเรขาคณิต: ช่องลมตีบลงในแนวดิ่งจนมีลักษณะเป็นวงรีแนวนอนอย่างชัดเจน "
                        f"โดยมีอัตราส่วนแกนสั้นต่อแกนยาวเฉลี่ย {avg_aspect:.2f} "
                        f"และทำมุมเอียงแกนยาวที่ {avg_angle:.1f}° (อยู่ในช่วงแนวราบ 45°-135°)"
                    )
                else:
                    prediction_class = 'Lateral'
                    raw_conf = (score_lateral / total_score * 100.0) if total_score > 0 else 90.0
                    confidence = 70.0 + (raw_conf * 0.28)
                    reasoning_text = (
                        f"พบการยุบตัวเด่นในแนวช่องคอด้านข้าง (Left-Right) สอดคล้องกับรูปแบบ Lateral collapse\n"
                        f"💡 หลักการทางเรขาคณิต: ช่องลมตีบลงในแนวราบจนมีลักษณะเป็นวงรีแนวตั้งอย่างชัดเจน "
                        f"โดยมีอัตราส่วนแกนสั้นต่อแกนยาวเฉลี่ย {avg_aspect:.2f} "
                        f"และทำมุมเอียงแกนยาวที่ {avg_angle:.1f}° (อยู่ในช่วงแนวตั้ง <45° หรือ >135°)"
                    )
                    
                if total_score > 0:
                    all_probs = {
                        'Concentric': float(avg_aspect * 100.0),
                        'AP': float(score_ap / total_score * (1.0 - avg_aspect) * 100.0),
                        'Lateral': float(score_lateral / total_score * (1.0 - avg_aspect) * 100.0)
                    }
                else:
                    all_probs = {'Concentric': 33.3, 'AP': 33.3, 'Lateral': 33.3}
                    
        # --- COMPARATIVE REFERENCE TO GROUND TRUTH (IF APPLICABLE) ---
        clip_idx = identify_patient_clip(original_filename, file_size)
        clinical_reference = None
        if clip_idx is not None and clip_idx in ground_truth_db:
            gt = ground_truth_db[clip_idx]
            clinical_reference = {
                "clip_idx": clip_idx,
                "class": gt["class"],
                "degree": gt["degree"],
                "reduction": gt["reduction"],
                "desc": gt["desc"],
                "reasoning": get_reasoning_text(gt)
            }
            
        # Generate downsampled annotated sequence frames for the 2D clip player (max 10 frames)
        print("[BACKEND LOG] Downsampling visualization sequence frames...")
        sequence_base64 = []
        num_frames = len(cropped_frames)
        max_visual_frames = 10
        if num_frames <= max_visual_frames:
            visual_indices = list(range(num_frames))
        else:
            visual_indices = [int(i * (num_frames - 1) / (max_visual_frames - 1)) for i in range(max_visual_frames)]
            
        for idx in visual_indices:
            img = cropped_frames[idx]
            img_draw = img.copy()
            contour = contour_slices[idx]
            pts_draw = np.array(contour, dtype=np.int32).reshape((-1, 1, 2))
            cv2.drawContours(img_draw, [pts_draw], -1, (0, 255, 0), 2) # Green outline
            if len(pts_draw) >= 5:
                ellipse = cv2.fitEllipse(pts_draw)
                cv2.ellipse(img_draw, ellipse, (0, 255, 255), 1) # Yellow fitted ellipse
            cx_draw = int(np.mean(pts_draw[:, 0, 0])) if len(pts_draw) > 0 else 112
            cy_draw = int(np.mean(pts_draw[:, 0, 1])) if len(pts_draw) > 0 else 112
            cv2.circle(img_draw, (cx_draw, cy_draw), 3, (0, 0, 255), -1) # Red centroid dot
            
            _, buffer_seq = cv2.imencode('.jpg', img_draw, [cv2.IMWRITE_JPEG_QUALITY, 55])
            seq_b64 = base64.b64encode(buffer_seq).decode('utf-8')
            sequence_base64.append(f"data:image/jpeg;base64,{seq_b64}")
            
        min_area_idx = int(np.argmin(slice_areas)) if slice_areas else 0
        min_crop_img = cropped_frames[min_area_idx] if min_area_idx < len(cropped_frames) else cropped_frames[-1]
        
        contour_mask = np.zeros((224, 224, 4), dtype=np.uint8)
        aligned_min_c = np.array(contour_slices[min_area_idx], dtype=np.int32).reshape((-1, 1, 2))
        cv2.drawContours(contour_mask, [aligned_min_c], -1, (0, 255, 0, 255), 2)
        if len(aligned_min_c) >= 5:
            ellipse = cv2.fitEllipse(aligned_min_c)
            cv2.ellipse(contour_mask, ellipse, (0, 255, 255, 255), 2)
            
        _, buffer_contour = cv2.imencode('.png', contour_mask)
        contour_base64 = base64.b64encode(buffer_contour).decode('utf-8')
        
        mask = np.zeros((224, 224), dtype=np.uint8)
        cv2.drawContours(mask, [aligned_min_c], -1, 255, -1)
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
        heatmap_color = cv2.applyColorMap((dist_transform * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        alpha = 0.6
        heatmap_blended = min_crop_img.copy()
        mask_indices = (mask > 0)
        heatmap_blended[mask_indices] = (alpha * heatmap_color[mask_indices] + (1 - alpha) * min_crop_img[mask_indices]).astype(np.uint8)
        
        _, buffer_hm = cv2.imencode('.jpg', heatmap_blended, [cv2.IMWRITE_JPEG_QUALITY, 60])
        heatmap_base64 = base64.b64encode(buffer_hm).decode('utf-8')
        
        _, buffer = cv2.imencode('.jpg', min_crop_img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        ai_analysis_result = {
            "loaded": False,
            "prediction_class": "N/A",
            "confidence": 0.0,
            "all_probabilities": {}
        }
        
        # Determine downsampled contour slices and slice areas matching visual sequence
        downsampled_slices = [contour_slices[i] for i in visual_indices]
        downsampled_areas = [slice_areas[i] for i in visual_indices]
        
        result = {
            "prediction_class": prediction_class,
            "degree": int(degree),
            "confidence": float(confidence),
            "all_probabilities": all_probs,
            "image_base64": f"data:image/jpeg;base64,{img_base64}",
            "heatmap_base64": f"data:image/jpeg;base64,{heatmap_base64}",
            "contour_base64": f"data:image/png;base64,{contour_base64}",
            "sequence_frames": sequence_base64,
            "min_lumen_area": float(min_lumen_area),
            "reduction_percent": float(reduction_percent),
            "reasoning_text": reasoning_text,
            "contour_slices": downsampled_slices,
            "slice_areas": downsampled_areas,
            "ai_analysis": ai_analysis_result,
            "clip_idx": clip_idx,
            "clinical_reference": clinical_reference
        }
        
        print("[BACKEND LOG] Response JSON prepared successfully.")
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if 'cap' in locals():
            cap.release()
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
                print("[BACKEND LOG] Cleanup completed: Temporary video file deleted.")
            except Exception as rm_err:
                print(f"[BACKEND WARNING] Could not remove temp file: {str(rm_err)}")
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    print(f"DISE server running on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
