from PIL import Image, ImageDraw
from typing import List, Dict, Any, Tuple
import numpy as np

class SmartExtractor:
    def __init__(self):
        print("Đang khởi tạo Pix2Text Engine...")
        try:
            from pix2text import Pix2Text
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # Cố gắng sử dụng CUDA nếu có, BẮT BUỘC dùng backend pytorch để tránh lỗi ONNX cublas
            self.p2t = Pix2Text.from_config(
                device=device, 
                mfd_config={'backend': 'pytorch'},
                layout_config={'backend': 'pytorch'}
            )
            self.is_mock = False
        except ImportError:
            print("Lỗi Import Pix2Text!")
            self.p2t = None
            self.is_mock = True

    def process_page(self, image: Image.Image, page_idx: int, extract_all: bool = False) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        results_data = []
        image = image.convert('RGB')
        
        # Tạo một bản sao để vẽ Bounding Box lên toàn bộ trang
        annotated_img = image.copy()
        draw = ImageDraw.Draw(annotated_img)
        
        if self.is_mock:
            return annotated_img, results_data

        try:
            # Nhận diện toàn trang
            out = self.p2t.recognize_page(image)
        except Exception as e:
            print(f"Lỗi recognize_page: {e}")
            return annotated_img, results_data
        
        # Trích xuất danh sách các khối (blocks)
        blocks = getattr(out, 'elements', out) if not isinstance(out, dict) else out.get('elements', out)
        if not isinstance(blocks, (list, tuple)):
            try:
                blocks = list(blocks)
            except:
                blocks = []
                
        box_idx = 1
        for block in blocks:
            # Ép kiểu an toàn về dictionary
            if hasattr(block, 'model_dump'):
                b_dict = block.model_dump()
            elif hasattr(block, 'dict'):
                b_dict = block.dict()
            elif isinstance(block, dict):
                b_dict = block
            else:
                b_dict = block.__dict__ if hasattr(block, '__dict__') else {}
                
            # Lấy loại khối (type)
            raw_type = b_dict.get('type', 'unknown')
            block_type = str(raw_type).lower()
            if '.' in block_type: # VD: ElementType.ISOLATED -> isolated
                block_type = block_type.split('.')[-1]
                
            text_content = b_dict.get('text', '')
            box = b_dict.get('position', b_dict.get('box', []))
            
            if box is None or len(box) == 0:
                continue
                
            # Tính toán tọa độ
            try:
                box_arr = np.array(box)
                x1, x2 = float(np.min(box_arr[:, 0])), float(np.max(box_arr[:, 0]))
                y1, y2 = float(np.min(box_arr[:, 1])), float(np.max(box_arr[:, 1]))
            except Exception:
                continue
                
            # Điều kiện là công thức toán học
            is_formula = any(kw in block_type for kw in ['isolated', 'formula', 'equation', 'math', 'embedding'])
            has_math_symbols = ('$' in text_content) or ('\\' in text_content)
            
            # Đóng khung trên ảnh toàn cảnh
            if is_formula or has_math_symbols:
                color = "red" # Đỏ cho công thức
                label = f"Math ({block_type})"
            else:
                color = "blue" # Xanh cho text thường
                label = f"Text ({block_type})"
                
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            # Vẽ background cho chữ dễ nhìn
            draw.rectangle([x1, max(0, y1 - 15), x1 + 80, max(0, y1)], fill=color)
            draw.text((x1 + 2, max(0, y1 - 15)), label, fill="white")
            
            # Trích xuất nếu thỏa mãn điều kiện
            if is_formula or has_math_symbols or extract_all:
                cx1, cy1 = max(0, x1 - 3), max(0, y1 - 3)
                cx2, cy2 = min(image.width, x2 + 3), min(image.height, y2 + 3)
                
                if cx2 - cx1 > 0 and cy2 - cy1 > 0:
                    cropped_img = image.crop((cx1, cy1, cx2, cy2))
                    results_data.append({
                        "page": page_idx + 1,
                        "box_idx": box_idx,
                        "bbox": [x1, y1, x2, y2],
                        "type": block_type,
                        "latex": text_content,
                        "image": cropped_img
                    })
                    box_idx += 1

        return annotated_img, results_data
