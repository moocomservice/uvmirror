from flask import Flask, request, jsonify, send_file, send_from_directory
import torch
import os
import cv2
import numpy as np
from PIL import Image
import time
import io
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # เพิ่ม CORS เพื่อให้สามารถเรียกใช้ API จากโดเมนอื่นได้

# ตั้งค่าตำแหน่งของโมเดล
execution_path = os.getcwd()
model_path = os.path.join(execution_path, "models/yolov5s.pt")

# สร้างโฟลเดอร์ output และ static ถ้ายังไม่มี
os.makedirs(os.path.join(execution_path, "output"), exist_ok=True)
os.makedirs(os.path.join(execution_path, "static"), exist_ok=True)

# โหลดโมเดล YOLOv5
print("กำลังโหลดโมเดล YOLOv5...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.conf = 0.25
model.iou = 0.45
print("โหลดโมเดลเรียบร้อยแล้ว")

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# เส้นทางสำหรับดูไฟล์ static อื่นๆ (CSS, JS, รูปภาพ)
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

# เส้นทางสำหรับดูไฟล์ผลลัพธ์
@app.route('/output/<filename>')
def get_output(filename):
    return send_file(os.path.join(execution_path, "output", filename))

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_file = request.files['image']
    input_path = os.path.join(execution_path, "output", "input.jpg")
    output_path = os.path.join(execution_path, "output", "output.jpg")
    cream_mask_path = os.path.join(execution_path, "output", "cream_mask.jpg")
    
    # เช็คว่าเป็นการตรวจจับแบบเรียลไทม์หรือไม่
    is_realtime = 'realtime' in request.args and request.args['realtime'] == 'true'
    
    # บันทึกภาพที่อัปโหลด
    image_file.save(input_path)
    
    # อ่านภาพและตรวจจับวัตถุ
    img = cv2.imread(input_path)
    if img is None:
        return jsonify({"error": "Could not read uploaded image"}), 400
    
    # ลดขนาดภาพถ้าเป็นการทำงานแบบเรียลไทม์เพื่อความเร็ว
    if is_realtime and max(img.shape[0], img.shape[1]) > 640:
        scale_factor = 640 / max(img.shape[0], img.shape[1])
        new_width = int(img.shape[1] * scale_factor)
        new_height = int(img.shape[0] * scale_factor)
        img = cv2.resize(img, (new_width, new_height))
    
    # สร้างภาพผลลัพธ์
    output_img = img.copy()
    
    # เริ่มต้นตัวแปรสำหรับการตรวจจับ
    cream_positions = 0
    cream_detected = False
    cream_areas = []
    
    # YOLO สำหรับตรวจจับใบหน้า
    start_time = time.time()
    results = model(img)
    detections = results.pandas().xyxy[0]
    yolo_time = time.time() - start_time
    print(f"YOLO ใช้เวลา: {yolo_time:.2f} วินาที")
    
    # หาใบหน้า (ใช้ทั้ง person และ face class จาก YOLO)
    face_detections = detections[(detections['name'] == 'person') | (detections['name'] == 'face')]
    
    # เตรียมรายการสำหรับเก็บใบหน้า
    faces = []
    
    # ถ้าไม่พบใบหน้าด้วย YOLO ให้ใช้วิธีตรวจจับใบหน้าด้วย OpenCV แทน
    if len(face_detections) == 0:
        # ใช้ตัวตรวจจับใบหน้าของ OpenCV
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_cv = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # สร้างรายการใบหน้าจากผลลัพธ์ของ OpenCV
        for (x, y, w, h) in faces_cv:
            faces.append({
                "box": {
                    "x1": int(x),
                    "y1": int(y),
                    "x2": int(x + w),
                    "y2": int(y + h)
                }
            })
    else:
        # สร้างรายการใบหน้าจากผลลัพธ์ YOLO
        for _, row in face_detections.iterrows():
            faces.append({
                "box": {
                    "x1": int(row['xmin']),
                    "y1": int(row['ymin']),
                    "x2": int(row['xmax']),
                    "y2": int(row['ymax'])
                }
            })
    
    # ถ้ายังไม่พบใบหน้า ให้ใช้ภาพทั้งหมดเป็นพื้นที่ตรวจจับ
    if len(faces) == 0:
        faces = [{
            "box": {
                "x1": 0,
                "y1": 0,
                "x2": img.shape[1],
                "y2": img.shape[0]
            }
        }]
    
    # แปลงภาพเป็น HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # คำนวณค่าความสว่างเฉลี่ยของใบหน้า
    face_brightness = []
    for face in faces:
        x1, y1 = face["box"]["x1"], face["box"]["y1"]
        x2, y2 = face["box"]["x2"], face["box"]["y2"]
        
        face_img = img[y1:y2, x1:x2]
        if face_img.size == 0:
            continue
            
        face_hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
        avg_brightness = np.mean(face_hsv[:,:,2])
        face_brightness.append(avg_brightness)
    
    # ถ้ามีข้อมูลความสว่างของใบหน้า คำนวณค่าเฉลี่ย
    if face_brightness:
        avg_face_brightness = np.mean(face_brightness)
        min_cream_brightness = min(255, avg_face_brightness + 30)
    else:
        min_cream_brightness = 210
    
    if not is_realtime:
        print(f"ความสว่างเฉลี่ยของใบหน้า: {avg_face_brightness if 'avg_face_brightness' in locals() else 'ไม่สามารถคำนวณได้'}")
        print(f"ความสว่างขั้นต่ำสำหรับการตรวจจับครีม: {min_cream_brightness}")
    
    # กำหนดช่วงสีของครีม
    min_cream_brightness_int = int(min_cream_brightness)
    min_cream_brightness_int = max(0, min(255, min_cream_brightness_int))
    min_cream_brightness_int = min(min_cream_brightness_int - 10, 255)
    
    lower_white = np.array([0, 0, min_cream_brightness_int], dtype=np.uint8)
    upper_white = np.array([180, 50, 255], dtype=np.uint8)
    
    # สร้างมาสก์สำหรับสีขาวในภาพทั้งหมด
    white_mask_global = cv2.inRange(hsv_img, lower_white, upper_white)
    
    # เพิ่มการ dilate ให้กับมาสก์สีขาว
    kernel = np.ones((5,5), np.uint8)
    white_mask_dilated = cv2.dilate(white_mask_global, kernel, iterations=3)
    
    # สร้างมาสก์ความแตกต่างของความสว่าง
    gradient_mask = cv2.morphologyEx(hsv_img[:,:,2], cv2.MORPH_GRADIENT, kernel)
    _, gradient_threshold = cv2.threshold(gradient_mask, 15, 255, cv2.THRESH_BINARY)
    gradient_dilated = cv2.dilate(gradient_threshold, kernel, iterations=2)
    
    # รวมมาสก์
    cream_mask = cv2.bitwise_and(white_mask_global, gradient_dilated)
    
    # กรองเสียงรบกวน
    cream_mask = cv2.morphologyEx(cream_mask, cv2.MORPH_OPEN, kernel)
    cream_mask = cv2.morphologyEx(cream_mask, cv2.MORPH_CLOSE, kernel)
    
    # หา contours ของครีม
    contours, _ = cv2.findContours(cream_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # ตั้งค่าขนาดพื้นที่ขั้นต่ำของครีม
    min_cream_area = 500 if is_realtime else 500
    
    # กรองเฉพาะพื้นที่ครีมที่มีขนาดใหญ่พอและพิจารณารูปร่าง
    valid_cream_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_cream_area:
            continue
            
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
            
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if is_realtime:
            if 0.05 < circularity < 0.9:
                valid_cream_contours.append(contour)
                cream_positions += 1
        else:
            if 0.1 < circularity < 0.8:
                valid_cream_contours.append(contour)
                cream_positions += 1
    
    # สร้างมาสก์สุดท้ายสำหรับแสดงผล
    final_mask = np.zeros_like(img[:,:,0])
    
    # ตรวจสอบว่าพบครีมหรือไม่
    if len(valid_cream_contours) > 0:
        cream_detected = True
        
        # คำนวณขนาดเฉลี่ยของพื้นที่ครีม
        avg_area = sum([cv2.contourArea(c) for c in valid_cream_contours]) / len(valid_cream_contours)
        min_valid_area = max(min_cream_area, avg_area * 0.2)
        
        # วาดลงบนมาสก์สุดท้าย
        displayed_contours = []
        for contour in valid_cream_contours:
            area = cv2.contourArea(contour)
            if area >= min_valid_area:
                displayed_contours.append(contour)
                cv2.drawContours(final_mask, [contour], -1, 255, -1)
                
                # วาดลงบนรูปต้นฉบับเฉพาะในโหมดปกติ
                if not is_realtime:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(output_img, f"Cream: {int(area)} px", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # ปรับจำนวนตำแหน่งครีมที่พบ
        cream_positions = len(displayed_contours)
    
    # เตรียมภาพผลลัพธ์สุดท้าย
    # ปรับแต่งภาพให้เป็น grayscale และเน้นรายละเอียดใบหน้า
    gray_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
    
    # ปรับความคมชัดเพื่อเน้นรายละเอียดผิว
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(gray_img)
    
    # เพิ่มความคมชัดด้วย unsharp masking
    gaussian_blur = cv2.GaussianBlur(enhanced_img, (0, 0), 3.0)
    enhanced_img = cv2.addWeighted(enhanced_img, 1.5, gaussian_blur, -0.5, 0)
    
    # เพิ่ม contrast
    enhanced_img = cv2.convertScaleAbs(enhanced_img, alpha=1.1, beta=5)
    
    # แปลงกลับเป็นภาพสี
    output_color = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
    
    # ถ้ามีครีม วาดลงบนภาพที่ปรับแต่งแล้ว
    if cream_detected:
        # คำนวณขนาดเฉลี่ยของพื้นที่ครีมอีกครั้ง
        if len(valid_cream_contours) > 0:
            avg_area = sum([cv2.contourArea(c) for c in valid_cream_contours]) / len(valid_cream_contours)
            min_valid_area = max(500, avg_area * 0.2)
        
        # สร้างมาสก์ว่างสำหรับการวาดพื้นที่ครีม
        cream_overlay = np.zeros_like(output_color)
        
        # กรองเฉพาะพื้นที่ที่มีขนาดใหญ่พอ "ก่อน" แล้วค่อยขยาย contour
        displayed_contours = []
        dilated_contours_list = []
        
        for contour in valid_cream_contours:
            area = cv2.contourArea(contour)
            if area >= min_valid_area:
                displayed_contours.append(contour)
                
                # ขยาย contour ด้วยการ dilate
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                
                contour_mask = np.zeros_like(cream_mask)
                cv2.drawContours(contour_mask, [approx_contour], -1, 255, -1)
                dilated_contour_mask = cv2.dilate(contour_mask, np.ones((7,7), np.uint8), iterations=2)
                
                dilated_contours, _ = cv2.findContours(dilated_contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(dilated_contours) > 0:
                    dilated_contours_list.append(dilated_contours[0])
                    cv2.drawContours(cream_overlay, dilated_contours, -1, (255, 255, 255), -1)
        
        # เบลอขอบของครีมด้วย Gaussian blur
        blurred_overlay = cv2.GaussianBlur(cream_overlay, (21, 21), 0)
        
        # สร้างมาสก์ alpha เพื่อการผสมแบบ smooth
        if len(blurred_overlay.shape) == 3:
            alpha_mask = cv2.cvtColor(blurred_overlay, cv2.COLOR_BGR2GRAY)
        else:
            alpha_mask = blurred_overlay.copy()
        
        # ปรับให้เป็นค่าระหว่าง 0-1 สำหรับการผสม
        alpha_mask = alpha_mask.astype(float) / 255.0
        
        # สร้างภาพ overlay สีดำสำหรับครีม
        black_overlay = np.zeros_like(output_color)
        
        # ผสมภาพโดยใช้ alpha blending
        for c in range(3):
            output_color[:,:,c] = output_color[:,:,c] * (1 - alpha_mask) + black_overlay[:,:,c] * alpha_mask
        
        # วาด bounding boxes เฉพาะในโหมดไม่ใช่ realtime
        if not is_realtime:
            for i, contour in enumerate(displayed_contours):
                if i < len(dilated_contours_list):
                    original_area = cv2.contourArea(contour)
                    
                    dilated_contour = dilated_contours_list[i]
                    dilated_x, dilated_y, dilated_w, dilated_h = cv2.boundingRect(dilated_contour)
                    
                    # วาดกรอบสี่เหลี่ยมสีแดงใช้ขนาดของ contour ที่ขยายแล้ว
                    cv2.rectangle(output_color, (dilated_x, dilated_y), (dilated_x + dilated_w, dilated_y + dilated_h), (0, 0, 255), 2)
                    
                    # เขียนข้อความระบุว่าเป็นครีมพร้อมขนาด
                    area_text = f"Cream: {int(original_area)} px"
                    cv2.putText(output_color, area_text, (dilated_x, dilated_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # บันทึกภาพผลลัพธ์
    cv2.imwrite(output_path, output_color)
    
    # บันทึกมาสก์
    if cream_detected:
        cv2.imwrite(cream_mask_path, final_mask)
    
    # เพิ่มข้อมูลเวลาการประมวลผล
    end_time = time.time()
    process_time = end_time - start_time
    
    return jsonify({
        "detections": faces,
        "cream_detected": cream_detected,
        "cream_positions": cream_positions,
        "output_image": f"/output/output.jpg?t={int(time.time())}",
        "cream_mask": f"/output/cream_mask.jpg?t={int(time.time())}" if cream_detected else None,
        "process_time": process_time
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)