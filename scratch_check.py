from pix2text import Pix2Text
import numpy as np
from PIL import Image

p2t = Pix2Text.from_config(device='cpu')
img = Image.new('RGB', (100, 100), color = 'white')
out = p2t.recognize_page(img)
print(type(out))
print(dir(out))
