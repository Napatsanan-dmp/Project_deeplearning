import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms

# ==========================
# 1️⃣  สร้างโมเดล SimpleCNN
# ==========================
class SimpleCNN_with_BatchNorm(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleCNN_with_BatchNorm, self).__init__()
        # Input: (Batch, 1, 48, 48)

        # Block 1: Conv -> BatchNorm -> ReLU -> Pool
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # <--- เพิ่ม
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # -> (Batch, 32, 24, 24)

        # Block 2: Conv -> BatchNorm -> ReLU -> Pool
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) # <--- เพิ่ม
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # -> (Batch, 64, 12, 12)

        # Block 3: Conv -> BatchNorm -> ReLU -> Pool
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) # <--- เพิ่ม
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # -> (Batch, 128, 6, 6)

        # Flatten -> 128 * 6 * 6 = 4608
        
        # Block 4: Fully Connected
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.bn_fc1 = nn.BatchNorm1d(512) # <--- เพิ่ม (ใช้ BatchNorm1d สำหรับ FC)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes) # 7 classes

    def forward(self, x):
        # ลำดับที่นิยมคือ Conv -> BN -> ReLU
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(-1, 128 * 6 * 6)

        # ลำดับที่นิยมคือ FC -> BN -> ReLU
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x) # (ที่ Output layer สุดท้ายไม่ต้องมี ReLU/BatchNorm)

        return x

# ==========================
# 2️⃣ โหลดโมเดลที่เทรนแล้ว
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN_with_BatchNorm(num_classes=7)
model.load_state_dict(torch.load('fer2013_cnn_model_with_evaluate.pth', map_location=device))
model.to(device)
model.eval()

# ==========================
# 3️⃣ Transform สำหรับ Inference
# ==========================
inference_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ==========================
# 4️⃣ โหลด Face Detector
# ==========================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ==========================
# 5️⃣ เปิดเว็บแคม
# ==========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y+h, x:x+w]
        face_tensor = inference_transform(roi_gray)
        face_tensor = face_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(face_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            emotion = EMOTIONS[predicted_idx.item()]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    cv2.imshow('Real-Time Facial Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
