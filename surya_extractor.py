from PIL import Image, ImageDraw
from typing import List, Dict, Any, Tuple
import numpy as np

class SuryaExtractor:
    def __init__(self):
        print("Đang khởi tạo Surya Layout Predictor và Pix2Tex...")
        self.is_mock = False
        
        # Tải mô hình Layout từ Surya
        try:
            from surya.inference import SuryaInferenceManager
            from surya.layout import LayoutPredictor
            print("Khởi tạo Surya Inference Manager...")
            # Surya sẽ tự động chọn GPU/CPU tùy hệ thống
            self.layout_predictor = LayoutPredictor(SuryaInferenceManager())
            print("Đã tải xong mô hình Surya Layout!")
        except ImportError:
            print("Lỗi: Chưa cài đặt thư viện surya-ocr (pip install surya-ocr)")
            self.layout_predictor = None
            self.is_mock = True
            
        # Tải mô hình OCR Toán học từ Pix2Tex (LaTeX-OCR gốc, rất nhẹ)
        try:
            from pix2tex.cli import LatexOCR
            print("Khởi tạo Pix2Tex (LaTeX-OCR)...")
            self.latex_ocr = LatexOCR()
            print("Đã tải xong Pix2Tex!")
        except ImportError:
            print("Lỗi: Chưa cài đặt pix2tex")
            self.latex_ocr = None

    def process_page(self, image: Image.Image, page_idx: int, extract_all: bool = False) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        results_data = []
        image = image.convert('RGB')
        
        annotated_img = image.copy()
        draw = ImageDraw.Draw(annotated_img)
        
        if self.is_mock or not self.layout_predictor:
            return annotated_img, results_data

        try:
            # Nhận diện bố cục toàn trang bằng Surya
            layout_predictions = self.layout_predictor([image])
            page_prediction = layout_predictions[0]
        except Exception as e:
            print(f"Lỗi khi chạy Surya Layout: {e}")
            return annotated_img, results_data
            
        box_idx = 1
        
        # page_prediction.bboxes chứa danh sách các bounding box
        for b in page_prediction.bboxes:
            block_type = b.label.lower()  # Surya nhãn: "Equation", "Text", "Table", "Figure", v.v.
            
            # Surya sử dụng bbox dạng [x0, y0, x1, y1]
            try:
                x1, y1, x2, y2 = b.bbox
            except:
                continue
                
            is_formula = (block_type == "equation")
            
            # Lọc theo yêu cầu
            if is_formula or extract_all:
                if is_formula:
                    color = "red"
                    label = f"Math (Surya)"
                else:
                    color = "blue"
                    label = f"{block_type.capitalize()}"
                    
                # Vẽ khung Bounding Box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                draw.rectangle([x1, max(0, y1 - 15), x1 + 100, max(0, y1)], fill=color)
                draw.text((x1 + 2, max(0, y1 - 15)), label, fill="white")
                
                # Mở rộng nhẹ vùng crop (padding 5 pixels) để OCR nhận diện tốt hơn
                cx1, cy1 = max(0, x1 - 5), max(0, y1 - 5)
                cx2, cy2 = min(image.width, x2 + 5), min(image.height, y2 + 5)
                
                if cx2 - cx1 > 0 and cy2 - cy1 > 0:
                    cropped_img = image.crop((cx1, cy1, cx2, cy2))
                    
                    # Chạy OCR
                    latex_str = ""
                    if self.latex_ocr:
                        try:
                            # Pix2Tex chuyên dịch ảnh công thức đơn lẻ
                            latex_str = self.latex_ocr(cropped_img)
                        except Exception as e:
                            latex_str = f"Lỗi OCR: {e}"
                            
                    results_data.append({
                        "page": page_idx + 1,
                        "box_idx": box_idx,
                        "bbox": [x1, y1, x2, y2],
                        "type": block_type,
                        "latex": latex_str,
                        "image": cropped_img
                    })
                    box_idx += 1

        return annotated_img, results_data
