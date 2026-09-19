"""
Test chi tiet Pix2Text: recognize_page va recognize
"""
from pix2text import Pix2Text
import torch
from PIL import Image, ImageDraw

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("Loading Pix2Text with pytorch backend...")
    p2t = Pix2Text.from_config(device=device, mfd_config={'backend': 'pytorch'})
    print("Model loaded OK")

    # Tao anh co noi dung giong cong thuc
    img = Image.new('RGB', (1200, 400), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((100, 100), "E = mc^2", fill='black')
    draw.text((100, 200), "x = (-b +/- sqrt(b^2 - 4ac)) / 2a", fill='black')
    draw.text((100, 300), "integral from 0 to infinity e^(-x) dx = 1", fill='black')

    print("\n=== Test 1: recognize_page ===")
    out = p2t.recognize_page(img)
    print(f"Out type: {type(out)}")
    elements = getattr(out, 'elements', [])
    print(f"Number of elements: {len(elements)}")
    for i, el in enumerate(elements):
        el_type = getattr(el, 'type', '?')
        el_text = getattr(el, 'text', '?')
        el_pos = getattr(el, 'position', None)
        print(f"  [{i}] type={el_type} | text={str(el_text)[:80]} | pos={el_pos}")

    print("\n=== Test 2: recognize (single) ===")
    try:
        out2 = p2t.recognize(img)
        print(f"Type: {type(out2)}")
        print(f"Result: {str(out2)[:300]}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n=== Test 3: Dir of Page object ===")
    print([attr for attr in dir(out) if not attr.startswith('_')])

    print("\n=== Test 4: to_markdown ===")
    try:
        md = out.to_markdown()
        print(f"Markdown output:\n{md}")
    except Exception as e:
        print(f"Error: {e}")

    # Test voi mot anh PDF that su
    import os
    from pdf_processor import PDFProcessor
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    if pdf_files:
        print(f"\n=== Test 5: Real PDF ({pdf_files[0]}) ===")
        proc = PDFProcessor(dpi=300, poppler_path=r"C:\Users\admin\Downloads\Release-26.09.0-0\poppler-26.09.0\Library\bin")
        pages = proc.process(pdf_files[0], output_dir=None)
        if pages:
            page = pages[0]
            print(f"Page size: {page.size}")
            out_real = p2t.recognize_page(page)
            elements_real = getattr(out_real, 'elements', [])
            print(f"Elements found: {len(elements_real)}")
            for i, el in enumerate(elements_real[:10]):
                el_type = getattr(el, 'type', '?')
                el_text = getattr(el, 'text', '?')
                el_pos = getattr(el, 'position', None)
                print(f"  [{i}] type={el_type} | text={str(el_text)[:80]} | has_pos={el_pos is not None}")
    else:
        print("\nKhong tim thay file PDF de test.")

if __name__ == "__main__":
    main()
