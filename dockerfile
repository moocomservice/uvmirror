FROM python:3.10-slim

WORKDIR /app

# ติดตั้ง dependencies สำหรับ OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# ติดตั้ง NumPy เวอร์ชันที่เข้ากันได้ก่อน
RUN pip install --no-cache-dir numpy==1.24.3

# ติดตั้ง dependencies หลักที่ต้องเข้ากันกับ NumPy
RUN pip install --no-cache-dir torch==1.11.0 torchvision==0.12.0
RUN pip install --no-cache-dir opencv-python-headless==4.5.5.64
RUN pip install --no-cache-dir Pillow==9.0.1

# ติดตั้ง Gradio
RUN pip install --no-cache-dir gradio>=3.50.2

# คัดลอกโค้ดแอพพลิเคชัน
COPY . .

# สร้างโฟลเดอร์สำหรับเก็บไฟล์ชั่วคราว
RUN mkdir -p output
RUN mkdir -p models

# ดาวน์โหลดโมเดล YOLOv5
RUN if [ ! -f models/yolov5s.pt ]; then \
    wget -q https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt -O models/yolov5s.pt; \
    fi

# ตั้งค่าพอร์ต
EXPOSE 7860

# รัน Gradio app
CMD ["python", "app_gradio.py"]