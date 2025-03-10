import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from flask import Flask, request, jsonify, render_template
from PIL import Image
import os

app = Flask(__name__, template_folder="templates")  # Đảm bảo Flask tìm thấy index.html

# Danh sách nhãn cử chỉ tay
CLASS_LABELS = [
    "01_palm", "02_l", "03_fist", "04_fist_moved", "05_thumb",
    "06_index", "07_ok", "08_palm_moved", "09_c", "10_down"
]

MODEL_PATH = "best_model.pth"

# Load mô hình
def load_model():
    model = models.resnet50()
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, len(CLASS_LABELS))
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Tiền xử lý ảnh
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(image).unsqueeze(0)  

# Hiển thị trang web
@app.route('/')
def index():
    return render_template('index.html')

# API dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']
    try:
        image = Image.open(file).convert('RGB')  # Mở file ảnh
        image = transform_image(image)  # Chuyển đổi ảnh

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            label = CLASS_LABELS[predicted.item()]

        return jsonify({"result": label})  # Đổi key từ "gesture" thành "result"
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
