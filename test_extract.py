import traceback
from smart_extractor import SmartExtractor
from PIL import Image

def test():
    try:
        extractor = SmartExtractor()
        print("Model loaded.")
        img = Image.new('RGB', (1000, 1000), color='white')
        res = extractor.process_page(img, 0)
        print("SUCCESS:", res)
    except Exception as e:
        print("ERROR:")
        traceback.print_exc()

if __name__ == "__main__":
    test()
