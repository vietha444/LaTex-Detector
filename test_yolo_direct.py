"""Test trực tiếp YOLO layout detection."""
from PIL import Image, ImageDraw
from formula_extractor import FormulaExtractor

def main():
    print("Khởi tạo FormulaExtractor...")
    extractor = FormulaExtractor()
    
    # Tạo ảnh test có chữ
    img = Image.new('RGB', (1200, 800), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((100, 100), "This is a paragraph of normal text.", fill='black')
    draw.text((100, 300), "E = mc^2", fill='black')
    draw.text((100, 500), "x = (-b +/- sqrt(b^2 - 4ac)) / 2a", fill='black')
    
    print("\nChạy process_page...")
    annotated, results = extractor.process_page(img, 0, conf_threshold=0.1, extract_all=True)
    
    print(f"\nSố kết quả: {len(results)}")
    for r in results:
        print(f"  [{r['box_idx']}] type={r['type']} conf={r['confidence']:.2f} bbox={r['bbox']} latex={r['latex'][:60]}")
    
    annotated.save("test_annotated.png")
    print("\nĐã lưu test_annotated.png")

    # Test với file PDF thật nếu có
    import os
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    if pdf_files:
        print(f"\n=== Test PDF thật: {pdf_files[0]} ===")
        from pdf_processor import PDFProcessor
        proc = PDFProcessor(dpi=300, poppler_path=r"C:\Users\admin\Downloads\Release-26.09.0-0\poppler-26.09.0\Library\bin")
        pages = proc.process(pdf_files[0], output_dir=None)
        if pages:
            annotated2, results2 = extractor.process_page(pages[0], 0, conf_threshold=0.25)
            print(f"Trang 1: {len(results2)} công thức phát hiện")
            for r in results2:
                print(f"  [{r['box_idx']}] type={r['type']} conf={r['confidence']:.2f} bbox={[int(b) for b in r['bbox']]}")
                print(f"    LaTeX: {r['latex'][:100]}")
            annotated2.save("test_pdf_annotated.png")
            print("Đã lưu test_pdf_annotated.png")

if __name__ == "__main__":
    main()
