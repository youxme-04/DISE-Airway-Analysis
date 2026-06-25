import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

# 1. โหลดสมอง AI ที่เราเทรนเสร็จแล้ว (dise_model_poc.pth)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = ['AP', 'AP_Lateral', 'Concentric'] # คลาสที่ AI รู้จักตอนสอน (เรียงตามโฟลเดอร์)

print("กำลังโหลดสมอง AI...")
model = models.mobilenet_v2()
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load("dise_model_poc.pth", map_location=device))
model.eval()
model.to(device)

# 2. ฟังก์ชันเตรียมรูปภาพให้เป็นฟอร์แมตที่ AI คุ้นเคย
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. ลองทดสอบกับรูปภาพของ pt027 (ที่เป็น Concentric)
# หา path ของรูปตัวอย่าง pt027 ที่ถูกสกัดมา
pt027_folder = r"dataset\train\Concentric"
images = [f for f in os.listdir(pt027_folder) if f.startswith("pt027")]

if not images:
    print("ไม่พบรูปไฟล์ pt027 ในโฟลเดอร์ Concentric ครับ")
else:
    test_image_path = os.path.join(pt027_folder, images[0]) # เอารูปแรกมาเทส
    img = Image.open(test_image_path).convert('RGB')
    
    # ป้อนเข้า AI
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
    print("-" * 30)
    print(f"กำลังวิเคราะห์ไฟล์: {test_image_path}")
    print("ผลวินิจฉัยจาก AI (จากสมองที่เพิ่งเทรนเสร็จ):")
    
    # แสดงเปอร์เซ็นต์ของทุกโรค
    for i, prob in enumerate(probabilities):
        print(f"{class_names[i]}: {prob.item()*100:.2f}%")
        
    # ทายผลอันที่สูงที่สุด
    winner_prob, winner_idx = torch.max(probabilities, 0)
    print("--------------------------------")
    print(f"-> AI ฟันธงว่าเป็น: {class_names[winner_idx.item()]} (มั่นใจ {winner_prob.item()*100:.2f}%)")
