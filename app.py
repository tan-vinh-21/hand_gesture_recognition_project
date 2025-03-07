import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def check_environment():
    """Kiểm tra môi trường làm việc và files"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print("\n=== THÔNG TIN MÔI TRƯỜNG ===")
    print(f"Thư mục hiện tại: {current_dir}")
    print("\nDanh sách files trong thư mục:")
    for file in os.listdir(current_dir):
        file_path = os.path.join(current_dir, file)
        file_size = os.path.getsize(file_path) if os.path.isfile(file_path) else "N/A"
        print(f"- {file} (Size: {file_size} bytes)")
    print("===========================\n")
    return current_dir

def predict_image(model_path, image_path, target_size=(224, 224), class_names=None):
    """
    Hàm dự đoán ảnh sử dụng model đã train
    
    Parameters:
    model_path: đường dẫn đến file model .h5
    image_path: đường dẫn đến file ảnh cần dự đoán
    target_size: kích thước ảnh đầu vào model yêu cầu, mặc định là (224, 224)
    class_names: list tên các class (tùy chọn)
    """
    try:
        # Kiểm tra file model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy file model tại: {model_path}")
        print(f"File model tồn tại: {model_path}")
        print(f"Kích thước file model: {os.path.getsize(model_path)} bytes")
            
        # Kiểm tra file ảnh
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Không tìm thấy file ảnh tại: {image_path}")
        print(f"File ảnh tồn tại: {image_path}")
        print(f"Kích thước file ảnh: {os.path.getsize(image_path)} bytes")
        
        # Load model
        print(f"\nĐang load model...")
        model = keras.models.load_model(model_path)
        print("Load model thành công!")
        
        # Đọc và tiền xử lý ảnh
        print(f"Đang đọc ảnh...")
        original_img = cv2.imread(image_path)
        if original_img is None:
            # Thử đọc với các phương thức khác
            try:
                original_img = plt.imread(image_path)
                print("Đọc ảnh thành công bằng plt.imread")
            except:
                raise ValueError(f"Không thể đọc file ảnh bằng cả cv2.imread và plt.imread")
            
        print(f"Đọc ảnh thành công! Kích thước ảnh: {original_img.shape}")
            
        # Giữ ảnh gốc để hiển thị
        if len(original_img.shape) == 2:  # Grayscale image
            img_display = np.stack([original_img]*3, axis=-1)
        else:
            img_display = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Xử lý ảnh cho prediction
        img = cv2.resize(img_display, target_size)
        img = img / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Thêm batch dimension
        
        # Thực hiện dự đoán
        print("Đang thực hiện dự đoán...")
        prediction = model.predict(img, verbose=0)
        
        # Hiển thị kết quả
        plt.figure(figsize=(10, 5))
        
        # Hiển thị ảnh
        plt.subplot(1, 2, 1)
        plt.imshow(img_display)
        plt.title('Ảnh đầu vào')
        plt.axis('off')
        
        # Hiển thị kết quả dự đoán
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class] * 100
        
        plt.subplot(1, 2, 2)
        plt.bar(range(len(prediction[0])), prediction[0])
        plt.title('Kết quả dự đoán')
        if class_names:
            plt.xticks(range(len(prediction[0])), class_names, rotation=45)
        
        # In kết quả
        print("\nKết quả dự đoán:")
        print(f"Class: {class_names[predicted_class] if class_names else predicted_class}")
        print(f"Độ tin cậy: {confidence:.2f}%")
        
        plt.tight_layout()
        plt.show()
        
        return prediction, original_img
        
    except Exception as e:
        print(f"\nLỗi chi tiết: {str(e)}")
        print(f"Loại lỗi: {type(e).__name__}")
        return None, None

if __name__ == "__main__":
    # Kiểm tra môi trường
    current_dir = check_environment()
    
    # Đường dẫn files
    model_path = os.path.join(current_dir, "best_model.h5")
    image_path = os.path.join(current_dir, "1.png")
    
    # Thêm tên các class nếu có
    class_names = ['Palm', 'l', 'Fist', 'Fist_moved', 'Thumb', 'Index', 'Ok', 'Palm_moved', 'C', 'Down']
    
    # Thực hiện dự đoán
    prediction, img = predict_image(model_path, image_path, class_names=class_names)