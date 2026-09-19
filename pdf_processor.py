import os
from pdf2image import convert_from_path
from PIL import Image
from typing import List

class PDFProcessor:
    def __init__(self, dpi: int = 300, poppler_path: str = None):
        """
        Khởi tạo PDFProcessor.
        :param dpi: Độ phân giải của ảnh trích xuất.
        :param poppler_path: Đường dẫn tới bin của thư mục poppler (nếu dùng Windows và chưa cấu hình PATH).
        """
        self.dpi = dpi
        self.poppler_path = poppler_path

    def process(self, pdf_path: str, output_dir: str = None) -> List[Image.Image]:
        """
        Chuyển đổi file PDF thành danh sách các hình ảnh (Pillow Images).
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Không tìm thấy file PDF tại: {pdf_path}")
            
        print(f"Đang xử lý PDF: {pdf_path} với DPI = {self.dpi}...")
        
        try:
            if self.poppler_path:
                images = convert_from_path(pdf_path, dpi=self.dpi, poppler_path=self.poppler_path)
            else:
                images = convert_from_path(pdf_path, dpi=self.dpi)
        except Exception as e:
            print("Lỗi khi chuyển đổi PDF. Bạn đã cài đặt 'poppler' chưa?")
            raise e

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for i, image in enumerate(images):
                img_path = os.path.join(output_dir, f"page_{i + 1}.png")
                image.save(img_path, "PNG")
                print(f"Đã lưu trang {i + 1} thành: {img_path}")
                
        return images
