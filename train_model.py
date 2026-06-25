import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time

# 1. ตั้งค่าพื้นฐาน
data_dir = r"dataset\train"  # โฟลเดอร์รูปที่เราสกัดมาเก็บไว้
batch_size = 16
epochs = 5      # ทดสอบลูประบบสัก 5 รอบก็พอสำหรับ Proof of Concept

# ตรวจสอบว่าคอมเรามีการ์ดจอแรงๆ ไหม (ถ้าไม่มีระบบจะใช้ CPU ธรรมดารัน)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"กำลังใช้หน่วยประมวลผล: {device}")

# 2. ปรับแต่งรูปภาพ (Data Augmentation & Normalization)
# โมเดลต้องการภาพขนาด 224x224 และค่าสีแบบปกติ
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # กลับซ้ายขวาบ้างสุ่มๆ ให้ AI เก่งขึ้น
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# (ข้ามการลบโฟลเดอร์อัตโนมัติ เพราะ OneDrive ล็อกไฟล์ไว้)

# 3. โหลดชุดข้อมูล (Dataset)
dataset = datasets.ImageFolder(data_dir, data_transforms)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
class_names = dataset.classes
print(f"พบประเภทการยุบตัว: {class_names}")
print(f"จำนวนภาพที่พร้อมให้ AI เรียนรู้ทั้งหมด: {len(dataset)} รูปภาพ")

# 4. เรียกสมอง AI อัจฉริยะ (Transfer Learning ด้วย MobileNetV2)
print("กำลังนิมนต์โมเดล MobileNetV2 มาเป็นสมอง...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# เปลี่ยนสมองส่วนปลาย (Classifier) จากเดิมมันทายหมา-แมวรูป 1,000 ชนิด ให้มาทายแค่ 4 ชนิดของเราแทน
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(class_names))

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. กระบวนการฝึกสอน (Training Loop)
print("-" * 30)
print("เริ่มการฝึกสอน AI (Training)...")
start_time = time.time()

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # ล้างสมอง
        optimizer.zero_grad()
        # คิดคำตอบ
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        # หาว่าผิดไปเท่าไหร่
        loss = criterion(outputs, labels)
        
        # ปรับจูนสมองใหม่ให้ฉลาดขึ้น
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataset)
    epoch_acc = corrects.double() / len(dataset)

    print(f"รอบที่ (Epoch) {epoch+1}/{epochs} | ค่าความผิดพลาด (Loss): {epoch_loss:.4f} | ความแม่นยำ (Acc): {epoch_acc*100:.2f}%")

time_elapsed = time.time() - start_time
print(f"การสอน AI สำเร็จ! ใช้เวลาไป {time_elapsed // 60:.0f} นาที {time_elapsed % 60:.0f} วินาที")

# 6. เซฟโมเดล
save_path = "dise_model_poc.pth"
torch.save(model.state_dict(), save_path)
print(f"เซฟความจำ AI เสร็จสมบูรณ์ บันทึกไว้ที่ไฟล์: {save_path}")
print("สามารถเอาไฟล์นี้ไปเสียบใส่โปรแกรมเว็บไซต์เพื่อใช้งานจริงได้เลยครับ!")
