from pix2text import Pix2Text
from PIL import Image

def test():
    try:
        p2t = Pix2Text.from_config(device='cpu')
        img = Image.new('RGB', (800, 800), color='white')
        out = p2t.recognize_page(img)
        print("Page object:", type(out))
        if hasattr(out, 'elements'):
            for el in out.elements:
                print("Element type:", type(el))
                print("Dir:", dir(el))
                break
        else:
            print("No elements attribute.")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    test()
