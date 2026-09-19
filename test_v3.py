from PIL import Image, ImageDraw
from formula_extractor import FormulaExtractor

def test():
    print("Khởi tạo FormulaExtractor...")
    extractor = FormulaExtractor()
    
    img = Image.new('RGB', (1200, 800), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((100, 100), "This is a paragraph of normal text.", fill='black')
    draw.text((100, 300), "E = mc^2", fill='black')
    
    print("Chạy process_page...")
    annotated, results = extractor.process_page(img, 0, conf_threshold=0.1, extract_all=True)
    
    print(f"Số kết quả: {len(results)}")
    for r in results:
        print(f"  [{r['box_idx']}] type={r['type']} bbox={r['bbox']}")
        print(f"    LaTeX: {r['latex'][:60]}")

if __name__ == "__main__":
    test()
