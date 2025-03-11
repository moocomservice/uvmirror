FROM python:3.9-slim

WORKDIR /app

# ติดตั้ง dependencies ที่จำเป็น
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# สร้างโฟลเดอร์ที่จำเป็น
RUN mkdir -p models output

# คัดลอก requirements และติดตั้ง dependencies ก่อน
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip wheel setuptools
RUN pip install --no-cache-dir -r requirements.txt

# ดาวน์โหลดโมเดล YOLOv5 ไว้ล่วงหน้า
RUN mkdir -p /app/models
RUN wget -q https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5n.pt -O /app/models/yolov5s.pt

# คัดลอกโค้ดแอปพลิเคชัน
COPY . .

# เปิดพอร์ตที่ Render ต้องการ
ENV PORT=10000
EXPOSE 10000

# คำสั่งเริ่มต้นแอป
CMD gunicorn --bind 0.0.0.0:$PORT --timeout 600 app:app
