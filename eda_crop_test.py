import cv2
import os
import numpy as np

input_dir = r"samples"
output_dir = r"samples_cropped"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ดึงรายชื่อไฟล์รูปภาพทั้งหมดในโฟลเดอร์ samples
for img_name in os.listdir(input_dir):
    if not img_name.endswith(('.jpg', '.png')):
        continue
        
    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"อ่านภาพ {img_name} ไม่ได้")
        continue

    # 1. แปลงภาพเป็นภาพขาวดำ (Grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. ปรับให้ส่วนที่มืด (ขอบดำ) กลายเป็นสีดำสนิท (0) และส่วนที่เป็นเนื้อคนกลายเป็นสีขาว (255)
    # ใช้ Threshold ที่ 15 เพื่อกรองก้อนเมกเซลล์สีดำๆ และ Noise ของวิดีโอ
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    
    # 3. ค้นหาก้อนรูปทรงทั้งหมดในภาพ (หา Contours)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"ไม่พบพื้นที่ในภาพ {img_name}")
        continue
        
    # 4. เลือกก้อนรูปทรงที่ "ใหญ่ที่สุด" ซึ่งก็คือกรอบสี่เหลี่ยมของกล้องส่อง
    # สาเหตุที่ต้องเลือกอันใหญ่สุด เพื่อตัดตัวอักษร PENTAX หรือเวลาตรงมุมจอทิ้งไป
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 5. หาขอบเขต (Bounding Box) ของก้อนที่ใหญ่ที่สุด
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # 6. ตัดภาพ (Cropping) ด้วยพิกัดที่หามาได้
    cropped_img = img[y:y+h, x:x+w]
    
    # เซฟภาพที่โดนตัดขอบแล้ว
    output_path = os.path.join(output_dir, img_name)
    cv2.imwrite(output_path, cropped_img)
    print(f"ตัดขอบสำเร็จ: {img_name} (พิกัด: {x},{y} ขนาด {w}x{h})")

print("ทำการ Crop ตัดขอบเสร็จสิ้น 100%")
