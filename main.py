import os
import json
import argparse
from pdf_processor import PDFProcessor
from formula_detector import FormulaDetector
from latex_ocr_module import LatexOCRModule

def main():
    parser = argparse.ArgumentParser(description="PDF Formula to LaTeX OCR Pipeline")
    parser.add_argument("-i", "--input", type=str, required=True, help="Đường dẫn đến file PDF")
    parser.add_argument("-o", "--output", type=str, default="output", help="Thư mục xuất kết quả")
    parser.add_argument("--poppler", type=str, default=None, help="Đường dẫn đến poppler/bin (cho Windows)")
    parser.add_argument("--yolo", type=str, default="yolov8n.pt", help="Đường dẫn đến file weights YOLOv8")
    args = parser.parse_args()

    input_pdf = args.input
    output_dir = args.output
    
    # Tạo cấu trúc thư mục đầu ra
    pages_dir = os.path.join(output_dir, "pages")
    crops_dir = os.path.join(output_dir, "crops")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)

    print(f"=== Bắt đầu quy trình xử lý cho: {input_pdf} ===")

    # 1. Chuyển PDF thành hình ảnh
    processor = PDFProcessor(dpi=300, poppler_path=args.poppler)
    try:
        pages = processor.process(input_pdf, output_dir=pages_dir)
    except Exception as e:
        print(f"Lỗi khi xử lý PDF: {e}")
        return

    # 2. Khởi tạo các mô hình AI
    detector = FormulaDetector(model_path=args.yolo)
    latex_ocr = LatexOCRModule()

    # Lưu kết quả tổng hợp
    final_results = []

    # 3. Chạy Pipeline trên từng trang
    for page_idx, page_img in enumerate(pages):
        print(f"\n--- Xử lý trang {page_idx + 1}/{len(pages)} ---")
        
        # Bước 2.1: Phát hiện và cắt công thức
        formula_items = detector.detect_and_crop(
            image=page_img, 
            page_idx=page_idx, 
            output_dir=crops_dir
        )
        print(f"Đã phát hiện {len(formula_items)} công thức trên trang {page_idx + 1}.")

        # Bước 2.2: Chuyển đổi ảnh sang LaTeX
        formula_items = latex_ocr.process_batch(formula_items)
        
        # Xóa trường 'image' (Pillow object) trước khi lưu JSON
        for item in formula_items:
            if 'image' in item:
                del item['image']
        
        final_results.extend(formula_items)

    # 4. Xuất báo cáo (JSON)
    report_path = os.path.join(output_dir, "formulas_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
        
    print(f"\n=== Hoàn tất! Báo cáo được lưu tại: {report_path} ===")

if __name__ == "__main__":
    main()
