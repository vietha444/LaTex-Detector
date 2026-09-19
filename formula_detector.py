import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any

# Placeholder cho việc import YOLO. Nếu chạy thực tế cần cài ultralytics
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

class FormulaDetector:
    def __init__(self, model_path: str = 'yolov8n.pt'):
        """
        Khởi tạo mô hình phát hiện công thức.
        Nên sử dụng mô hình YOLOv8 đã được fine-tune cho việc phát hiện công thức.
        (ví dụ: yolov8_formula.pt)
        """
        self.model_path = model_path
        if HAS_YOLO:
            try:
                self.model = YOLO(self.model_path)
                print(f"Đã load model YOLO từ: {self.model_path}")
            except Exception as e:
                print(f"Không thể load model YOLO. Đang chạy ở chế độ giả lập. Chi tiết lỗi: {e}")
                self.model = None
        else:
            print("Cảnh báo: Chưa cài đặt ultralytics. Đang chạy ở chế độ giả lập (Mock).")
            self.model = None

    def detect_and_crop(self, image: Image.Image, page_idx: int, output_dir: str = None) -> List[Dict[str, Any]]:
        """
        Nhận diện công thức trên ảnh, cắt (crop) chúng và trả về danh sách dữ liệu.
        """
        # Chuyển đổi ảnh từ PIL sang dạng OpenCV (numpy array BGR)
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results_data = []

        if self.model:
            # Chạy YOLOv8 inference
            results = self.model(image)
            boxes = results[0].boxes
            
            for i, box in enumerate(boxes):
                # Lấy tọa độ bounding box
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                
                # Đảm bảo tọa độ hợp lệ
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    continue # Bỏ qua box bị lỗi diện tích = 0
                    
                confidence = float(box.conf[0])
                cls = int(box.cls[0]) # 0 có thể là 'inline', 1 có thể là 'display'
                
                # Cắt ảnh
                cropped_img = image.crop((x1, y1, x2, y2))
                
                # Kiểm tra lại kích thước sau cắt
                if cropped_img.width <= 0 or cropped_img.height <= 0:
                    continue
                    
                crop_info = {
                    "page": page_idx + 1,
                    "box_idx": i + 1,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "class_id": cls,
                    "image": cropped_img
                }
                results_data.append(crop_info)
        else:
            # CHẾ ĐỘ GIẢ LẬP (MOCK) KHI CHƯA CÓ MODEL THẬT
            print("Đang giả lập việc nhận diện...")
            h, w = cv_image.shape[:2]
            
            # Đảm bảo box giả định luôn có diện tích > 0
            x1, y1 = int(w*0.25), int(h*0.25)
            x2, y2 = int(w*0.75), int(h*0.33)
            
            if x2 <= x1: x2 = x1 + 10
            if y2 <= y1: y2 = y1 + 10
            
            cropped_img = image.crop((x1, y1, x2, y2))
            
            if cropped_img.width > 0 and cropped_img.height > 0:
                results_data.append({
                    "page": page_idx + 1,
                    "box_idx": 1,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": 0.99,
                    "class_id": 0,
                    "image": cropped_img
                })

        # Lưu ảnh đã cắt nếu được yêu cầu
        if output_dir and results_data:
            os.makedirs(output_dir, exist_ok=True)
            for item in results_data:
                filename = f"page_{item['page']}_formula_{item['box_idx']}.png"
                filepath = os.path.join(output_dir, filename)
                item['image'].save(filepath)
                item['filepath'] = filepath

        return results_data
