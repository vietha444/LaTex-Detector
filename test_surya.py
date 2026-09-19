import os
os.environ["TORCH_DEVICE"] = "cuda"
import traceback
from PIL import Image
from surya_extractor import SuryaExtractor

def test():
    try:
        extractor = SuryaExtractor()
        img = Image.new('RGB', (800, 800), color='white')
        _, results = extractor.process_page(img, 0, extract_all=True)
        print("Chạy thành công. Kết quả:", len(results))
    except Exception as e:
        print("Lỗi nghiêm trọng:")
        traceback.print_exc()

if __name__ == "__main__":
    test()
