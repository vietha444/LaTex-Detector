from pix2text import Pix2Text
from PIL import Image

def test():
    try:
        p2t = Pix2Text.from_config(device='cuda', mfd_config={'backend': 'pytorch'})
        print("Model loaded with PyTorch backend!")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test()
