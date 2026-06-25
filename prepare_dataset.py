import cv2
import os

video_dir = r"video DISE patient 001-100"
dataset_dir = r"dataset\train"

# ข้อมูลเฉลยชุดแรก 5 คลิป
labeled_videos = {
    "pt003.mp4": "AP",
    "pt011.mp4": "AP",
    "pt016.mp4": "AP",
    "pt027.mp4": "Concentric",
    "pt028.mp4": "AP_Lateral"
}

# สร้างโฟลเดอร์สำหรับเก็บภาพแยกตามคลาส
for label in ['AP', 'Lateral', 'AP_Lateral', 'Concentric']:
    folder_path = os.path.join(dataset_dir, label)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

print("กำลังเริ่มสกัดเฟรมภาพจากวิดีโอเข้า Dataset...")

for video_file, label in labeled_videos.items():
    video_path = os.path.join(video_dir, video_file)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"อ่านวิดีโอ {video_file} ไม่ได้")
        continue

    # บันทึกทุกๆ 30 เฟรม (ประมาณทุกๆ 1 วินาที เพื่อไม่ให้ภาพซ้ำซ้อนกันเกินไป)
    frame_interval = 30 
    count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % frame_interval == 0:
            # ใช้ลอจิกตัดขอบอัตโนมัติที่เราออกแบบไว้
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # ครอปภาพ
                cropped_img = frame[y:y+h, x:x+w]
                
                # ย่อภาพลงเล็กน้อยให้เหมาะกับ PyTorch (ไม่ต้องใหญ่มากจะได้เทรนไวๆ)
                cropped_img = cv2.resize(cropped_img, (224, 224))
                
                # เซฟลงโฟลเดอร์ประเภทโรคของมัน
                img_name = f"{video_file.replace('.mp4','')}_frame{count}.jpg"
                save_path = os.path.join(dataset_dir, label, img_name)
                cv2.imwrite(save_path, cropped_img)
                saved_count += 1
                
        count += 1
        
    cap.release()
    print(f"วิดีโอ {video_file}: สกัดได้ {saved_count} ภาพ => นำไปเก็บที่ {label}")

print("-----------------------------------------")
print("สร้าง Dataset สำเร็จ 100% พร้อมป้อนให้ PyTorch แล้ว!")
