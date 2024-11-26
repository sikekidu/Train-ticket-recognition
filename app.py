from flask import Flask, request, jsonify, send_file, render_template
import cv2
import numpy as np
import os
from pyzbar.pyzbar import decode
import pytesseract
from flask_cors import CORS  
import shutil

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def detect_blue_bottom(image):
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义蓝色的HSV范围
    lower_blue = np.array([95, 150, 120])
    upper_blue = np.array([105, 220, 190])

    # 创建掩码
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 找到蓝色区域的轮尾
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤掉小的轮尾
    blue_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 2000]  # 调整面积阀值

    # 合并靠得太近的轮尾
    merged_contours = []
    for i, cnt1 in enumerate(blue_contours):
        x1, y1, w1, h1 = cv2.boundingRect(cnt1)
        merged = False
        for j, cnt2 in enumerate(blue_contours):
            if i != j:
                x2, y2, w2, h2 = cv2.boundingRect(cnt2)
                if abs(x1 - x2) < 40 and abs(y1 - y2) < 40:  # 调整合并阀值
                    x = min(x1, x2)
                    y = min(y1, y2)
                    w = max(x1 + w1, x2 + w2) - x
                    h = max(y1 + h1, y2 + h2) - y
                    merged_contours.append(np.array([[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]], dtype=np.int32))
                    merged = True
                    break
        if not merged:
            merged_contours.append(cnt1)

    print(f"Found {len(merged_contours)} blue bottoms after merging")  # 调试日志

    return merged_contours

def crop_tickets(image, blue_contours):
    tickets = []
    ticket_aspect_ratio = 85 / 54  # 火车票的长宽比
    for cnt in blue_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ticket_width = w
        ticket_height = int(ticket_width / ticket_aspect_ratio)

        # 计算火车票的顶部位置
        ticket_top = y - ticket_height

        # 确保裁切区域在图像范围内
        if ticket_top < 0:
            ticket_top = 0

        # 扩展裁切区域以确保覆盖整张火车票
        ticket_top = max(0, ticket_top - 60)  # 增加边距
        ticket_bottom = min(image.shape[0], y + 60)  # 增加边距
        ticket_left = max(0, x - 60)  # 增加边距
        ticket_right = min(image.shape[1], x + ticket_width + 60)  # 增加边距

        ticket = image[ticket_top:ticket_bottom, ticket_left:ticket_right]
        tickets.append(ticket)
        print(f"Cropped ticket at ({ticket_left}, {ticket_top}) with size ({ticket_right - ticket_left}, {ticket_bottom - ticket_top})")  # 调试日志

    return tickets

def process_image_with_blue_bottom(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 检测蓝色底边
    blue_contours = detect_blue_bottom(image)

    # 裁切火车票
    tickets = crop_tickets(image, blue_contours)

    return tickets

def decode_qr_codes(tickets):
    qr_codes = []
    for ticket in tickets:
        decoded_objects = decode(ticket)
        for obj in decoded_objects:
            qr_codes.append(obj.data.decode('utf-8'))
    return qr_codes

def extract_ticket_info(ticket):
    # 使用pytesseract进行OCR处理
    text = pytesseract.image_to_string(ticket, lang='chi_sim')
    # 提取名字和时间信息
    # 这里假设名字和时间信息在文本的前几行
    lines = text.split('\n')
    name = None
    time = None
    for line in lines:
        if '乘客' in line:
            name = line.split('：')[1].strip()
        if '日期' in line:
            time = line.split('：')[1].strip()
    return name, time

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        print(f"Saved uploaded file to {file_path}")  # 调试日志

        # 处理图像
        tickets = process_image_with_blue_bottom(file_path)

        print(f"Processed image and found {len(tickets)} tickets")  # 调试日志

        # 解码二维码
        qr_codes = decode_qr_codes(tickets)

        print(f"Decoded QR codes: {qr_codes}")  # 调试日志

        # 保存分割后的火车票
        output_folder = 'output'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        ticket_paths = []
        for i, ticket in enumerate(tickets):
            name, time = extract_ticket_info(ticket)
            if name and time:
                ticket_filename = f'{name}_{time}.jpg'
            else:
                ticket_filename = f'ticket_{i}.jpg'
            ticket_path = os.path.join(output_folder, ticket_filename)
            cv2.imwrite(ticket_path, ticket)
            ticket_paths.append(ticket_path)

        # 清理上传和中间文件夹
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
        if os.path.exists('intermediate'):
            shutil.rmtree('intermediate')

        return jsonify({'message': 'File uploaded and processed successfully', 'ticket_paths': ticket_paths, 'qr_codes': qr_codes})

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    file_path = os.path.join('output', filename)
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)
