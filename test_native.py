from transformers import AutoModelForObjectDetection, AutoProcessor
from PIL import Image
import torch

def test():
    try:
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained("vikp/surya_layout2")
        print("Loading model...")
        model = AutoModelForObjectDetection.from_pretrained("vikp/surya_layout2")
        print("Model loaded successfully!")
        
        img = Image.new('RGB', (800, 800), color='white')
        inputs = processor(images=img, return_tensors="pt")
        outputs = model(**inputs)
        print("Inference successful!")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
