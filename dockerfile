FROM python:3.9-slim

WORKDIR /app

# ติดตั้ง dependencies ที่จำเป็นสำหรับ OpenCV และ NumPy
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# คัดลอก requirements.txt
COPY requirements.txt .

# ติดตั้ง pre-built wheels ของ NumPy และ pandas
RUN pip install --no-cache-dir wheel
RUN pip install --no-cache-dir numpy==1.22.4 pandas==1.4.2

# ติดตั้ง dependencies ที่เหลือ
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกไฟล์โปรเจ็กต์ทั้งหมด
COPY . .

# สร้างโฟลเดอร์ที่จำเป็น
RUN mkdir -p models output

# เปิดพอร์ต
EXPOSE 10000

# คำสั่งเริ่มต้นแอป
CMD gunicorn --bind=0.0.0.0:10000 app:app
