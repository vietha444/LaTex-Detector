from PIL import Image
from typing import Dict, Any, List

try:
    from pix2tex.cli import LatexOCR
    HAS_PIX2TEX = True
except ImportError:
    HAS_PIX2TEX = False

class LatexOCRModule:
    def __init__(self):
        """
        Khởi tạo mô hình Pix2Tex (LaTeX-OCR).
        Trong lần chạy đầu tiên, model checkpoints sẽ được tải xuống tự động.
        """
        if HAS_PIX2TEX:
            print("Đang khởi tạo mô hình Pix2Tex...")
            self.model = LatexOCR()
            print("Khởi tạo Pix2Tex thành công.")
        else:
            print("Cảnh báo: Chưa cài đặt pix2tex. Đang chạy ở chế độ giả lập (Mock).")
            self.model = None

    def recognize(self, image: Image.Image) -> str:
        """
        Dịch 1 ảnh chứa công thức thành chuỗi LaTeX.
        """
        # Kiểm tra kích thước ảnh hợp lệ
        if image.width <= 0 or image.height <= 0:
            return "Lỗi: Ảnh bị cắt quá nhỏ hoặc kích thước không hợp lệ"
            
        if self.model:
            # Predict với ảnh truyền vào
            try:
                prediction = self.model(image)
                return prediction
            except Exception as e:
                return f"Lỗi khi nhận dạng ảnh: {e}"
        else:
            # MOCK response
            return "\\int_{a}^{b} x^2 dx"

    def process_batch(self, formula_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Xử lý danh sách các ảnh công thức đã cắt.
        Cập nhật thêm trường 'latex' vào mỗi item.
        """
        for item in formula_items:
            img = item.get('image')
            if img:
                latex_str = self.recognize(img)
                item['latex'] = latex_str
                print(f"Nhận diện xong [Trang {item['page']} - BBox {item['box_idx']}]: {latex_str}")
        return formula_items
