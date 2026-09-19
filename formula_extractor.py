"""
Formula Extractor V3 - Sử dụng layout_parser.parse() nội bộ của Pix2Text 
để lấy bounding box chính xác mà không bị mất dữ liệu (position=None).
"""
from PIL import Image, ImageDraw
from typing import List, Dict, Any, Tuple
import numpy as np

class FormulaExtractor:
    def __init__(self):
        print("Đang khởi tạo Engine V3...")
        self.is_mock = False
        
        try:
            from pix2text import Pix2Text
            import torch
            
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Device: {self.device}")
            
            # Khởi tạo toàn bộ pipeline nhưng ép backend PyTorch để né lỗi ONNX nếu có
            self.p2t = Pix2Text.from_config(
                device=self.device, 
                mfd_config={'backend': 'pytorch'},
                layout_config={'backend': 'pytorch'}
            )
            print("Đã tải xong Pix2Text V3!")
            
            self.layout_parser = self.p2t.layout_parser
            
        except Exception as e:
            print(f"Lỗi khởi tạo Pix2Text: {e}")
            self.is_mock = True

    def process_page(
        self, image: Image.Image, page_idx: int, 
        conf_threshold: float = 0.25, extract_all: bool = False
    ) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        
        image = image.convert('RGB')
        annotated_img = image.copy()
        draw = ImageDraw.Draw(annotated_img)
        results_data = []
        
        if self.is_mock:
            return annotated_img, results_data

        try:
            # 1. Gọi trực tiếp bộ parse để LẤY BOUNDING BOX (tránh lỗi position=None)
            # Signature: parse(img, table_as_image=False, imgsz=1024, conf=0.2, ...)
            # Trả về tuple: (danh_sách_blocks, ...)
            parse_res = self.layout_parser.parse(image, False, conf=conf_threshold)
            blocks = parse_res[0] if isinstance(parse_res, tuple) else parse_res
        except Exception as e:
            print(f"Lỗi phân tích layout: {e}")
            import traceback; traceback.print_exc()
            return annotated_img, results_data

        box_idx = 1

        for block in blocks:
            # block thường có cấu trúc dictionary: {'type': '...', 'position': array([[x,y],...])}
            b_type = str(block.get('type', 'unknown')).lower()
            if '.' in b_type:
                b_type = b_type.split('.')[-1]
                
            pos = block.get('position', None)
            if pos is None:
                continue
                
            # pos thường là list/array 4 điểm: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            try:
                pos_arr = np.array(pos)
                x1, x2 = float(np.min(pos_arr[:, 0])), float(np.max(pos_arr[:, 0]))
                y1, y2 = float(np.min(pos_arr[:, 1])), float(np.max(pos_arr[:, 1]))
            except:
                continue

            # Xác định xem có phải công thức không
            is_formula = any(kw in b_type for kw in ['isolated', 'formula', 'equation', 'math', 'embedding'])
            
            if is_formula:
                color = "red"
                label = f"Math ({b_type})"
            else:
                color = "#4488ff"
                label = f"Text ({b_type})"

            # Vẽ bounding box lên ảnh toàn cảnh
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.rectangle([x1, max(0, y1 - 18), x1 + len(label)*6 + 10, max(0, y1)], fill=color)
            draw.text((x1 + 2, max(0, y1 - 16)), label, fill="white")

            # Chỉ crop phần công thức (hoặc tất cả nếu được yêu cầu)
            if is_formula or extract_all:
                pad = 3
                cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
                cx2, cy2 = min(image.width, x2 + pad), min(image.height, y2 + pad)
                
                if cx2 - cx1 > 5 and cy2 - cy1 > 5:
                    cropped_img = image.crop((cx1, cy1, cx2, cy2))
                    
                    # 2. Dịch LaTeX trực tiếp trên ảnh cắt
                    latex_str = ""
                    try:
                        # gọi recognize cho 1 ảnh (OCR)
                        rec_res = self.p2t.recognize(cropped_img)
                        if isinstance(rec_res, dict):
                            latex_str = rec_res.get('text', str(rec_res))
                        else:
                            latex_str = str(rec_res)
                    except Exception as e:
                        latex_str = f"Lỗi OCR: {e}"

                    results_data.append({
                        "page": page_idx + 1,
                        "box_idx": box_idx,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(block.get('score', 1.0)),
                        "type": b_type,
                        "latex": latex_str.strip(),
                        "image": cropped_img
                    })
                    box_idx += 1

        return annotated_img, results_data
