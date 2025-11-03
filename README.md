# Facial Emotion Recognition using Convolutional Neural Network (CNN)
**ระบบจำแนกอารมณ์จากภาพใบหน้าโดยใช้โครงข่ายประสาทเทียมแบบคอนโวลูชัน (CNN)**
โปรเจกต์นี้เป็นส่วนหนึ่งของรายวิชา **Deep Learning**  
โดยมีวัตถุประสงค์เพื่อพัฒนาโมเดลที่สามารถจำแนกอารมณ์ของมนุษย์จากภาพใบหน้า  
ด้วยการใช้ชุดข้อมูล **FER2013** และสถาปัตยกรรม **Convolutional Neural Network (CNN)** บน **PyTorch Framework**  
รวมถึงเชื่อมต่อกับกล้องเว็บแคมเพื่อให้ระบบสามารถตรวจจับอารมณ์ได้แบบเรียลไทม์

---

## จัดทำโดย

| ลำดับ | ชื่อ - นามสกุล | รหัสนิสิต |
|--------|------------------|-------------|
| 1 | นางสาวนภัสนันท์ ดามะพร | 6610502102 |
| 2 | นายสรัล สังข์วร | 6610505594 |

---

##  Dataset

ในโครงงานนี้ใช้ชุดข้อมูลเดียวกับที่ปรากฏในบทความ  
**[Facial Emotion Recognition by Gaurav Sharma (Kaggle Notebook)](https://www.kaggle.com/code/gauravsharma99/facial-emotion-recognition)**  
ซึ่งเป็นการนำชุดข้อมูล **FER2013 (Facial Expression Recognition 2013)** จาก Kaggle มาใช้งาน  

**รายละเอียดของ Dataset:**
- จำนวนภาพทั้งหมด: 35,887 ภาพ (48×48 pixels, grayscale)
- จำแนกอารมณ์ได้ 7 ประเภท ได้แก่  Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- ข้อมูลถูกเก็บอยู่ในไฟล์ `fer2013.csv` ซึ่งภายในบรรจุข้อมูลพิกเซลของภาพและ label อารมณ์


>  ในโครงงานนี้ ข้อมูล `fer2013.csv` ถูกนำมา Preprocess ใหม่  และแปลงเป็น Tensor  
> โดยใช้ PyTorch เพื่อสร้าง DataLoader สำหรับการฝึกโมเดล CNN และประเมินผล

---

## Project Files Description

| ชื่อไฟล์ | รายละเอียด |
|-----------|-------------|
|  **README.md** | ไฟล์อธิบายโปรเจกต์ (ไฟล์นี้) |
|  **Facial_Emotion_Recognition.ipynb** | Notebook หลัก ใช้สำหรับเทรนโมเดล CNN บน Google Colab พร้อมขั้นตอน Preprocess และ Evaluate |
|  **fer2013_cnn_model_with_evaluate.pth** | โมเดลที่ผ่านการเทรนเรียบร้อยแล้ว (น้ำหนักของ CNN ที่ดีที่สุด) |
|  **haarcascade_frontalface_default.xml** | โมเดลตรวจจับใบหน้าจาก OpenCV (Haar Cascade Classifier) |
|  **test_optimized.py** | สคริปต์หลักสำหรับรันโมเดลจริง ตรวจจับใบหน้าแบบ Real-Time ผ่านกล้องเว็บแคม |

---

## การติดตั้งที่จำเป็น (Requirements)

ก่อนรันโปรเจกต์ ให้ติดตั้งไลบรารีดังนี้:
```bash
pip install torch torchvision
pip install opencv-python
pip install matplotlib seaborn scikit-learn
```

