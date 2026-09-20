"""
YoloRouter — Điều phối YOLO layout detection → OCR routing → LaTeX assembly.

Thiết kế cho bố cục IEEE 2 cột:
  - Cột trái được đọc trước (top→bottom), sau đó cột phải.
  - Header/Footer bị bỏ qua hoàn toàn.
  - Công thức → Pix2Text Math OCR
  - Văn bản Tiếng Anh (paragraph, title…) → Nougat
  - Văn bản Tiếng Việt → Pix2Text vi+en
  - Bảng → Pix2Text Table OCR
  - Hình → Giữ lại placeholder
"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from lang_detector import detect_language
from latex_postprocessor import clean_latex

# ── Màu bounding box theo loại block ──────────────────────────────────────────
BLOCK_COLORS: Dict[str, Tuple[str, str]] = {
    # (outline_color, label_bg_color)
    "title":     ("#22c55e", "#22c55e"),   # 🟢 Xanh lá
    "paragraph": ("#3b82f6", "#3b82f6"),   # 🔵 Xanh dương
    "text":      ("#3b82f6", "#3b82f6"),
    "isolated_formula": ("#ef4444", "#ef4444"),  # 🔴 Đỏ
    "embedding_formula": ("#f97316", "#f97316"), # 🟠 Cam (inline)
    "formula":   ("#ef4444", "#ef4444"),
    "equation":  ("#ef4444", "#ef4444"),
    "math":      ("#ef4444", "#ef4444"),
    "table":     ("#eab308", "#eab308"),   # 🟡 Vàng
    "figure":    ("#a855f7", "#a855f7"),   # 🟣 Tím
    "image":     ("#a855f7", "#a855f7"),
    "list":      ("#06b6d4", "#06b6d4"),   # 🩵 Xanh nhạt
    "header":    ("#9ca3af", "#9ca3af"),   # ⚪ Xám (sẽ bỏ qua)
    "footer":    ("#9ca3af", "#9ca3af"),
}

# Các block loại này sẽ bị BỎ QUA (header/footer)
SKIP_TYPES = {"header", "footer", "page_number"}

# Công thức
MATH_TYPES = {"isolated_formula", "embedding_formula", "formula", "equation", "math"}

# Hình ảnh
FIGURE_TYPES = {"figure", "image", "picture"}

# Bảng
TABLE_TYPES = {"table"}

# Tiêu đề
TITLE_TYPES = {"title", "section", "subsection"}


def _color_for(b_type: str) -> Tuple[str, str]:
    for key, val in BLOCK_COLORS.items():
        if key in b_type:
            return val
    return ("#6b7280", "#6b7280")  # Xám mặc định


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def draw_annotated(image: Image.Image, blocks: List[Dict]) -> Image.Image:
    """
    Vẽ bounding box màu sắc theo loại lên ảnh gốc.
    Nhãn được vẽ BÊN TRONG góc trên-trái của box để tránh đè nhau.
    Trả về ảnh mới (không sửa gốc).
    """
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated, "RGBA")

    # Thử font có kích thước phù hợp
    font_size = max(12, min(16, image.width // 80))
    font = None
    for font_name in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]:
        try:
            font = ImageFont.truetype(font_name, font_size)
            break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()

    PAD  = 3   # padding trong label
    LINE = 3   # độ dày viền box

    for idx, block in enumerate(blocks):
        b_type = block["type"]
        x1, y1, x2, y2 = [float(v) for v in block["bbox"]]
        # Đảm bảo tọa độ không âm và không vượt kích thước ảnh
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(image.width - 1, x2); y2 = min(image.height - 1, y2)

        outline, bg = _color_for(b_type)

        if b_type in SKIP_TYPES:
            # Vẽ nét đứt xám nhạt, nhãn nhỏ bên trong
            draw.rectangle([x1, y1, x2, y2], outline="#9ca3af", width=1)
            if x2 - x1 > 40 and y2 - y1 > 12:
                draw.text((x1 + PAD, y1 + PAD), f"skip/{b_type}", fill="#9ca3af", font=font)
            continue

        # ── Vẽ viền bounding box ──────────────────────────────────────────────
        draw.rectangle([x1, y1, x2, y2], outline=outline, width=LINE)

        # ── Vẽ nhãn BÊN TRONG góc trên-trái ─────────────────────────────────
        order  = block.get("order", idx + 1)
        # Nhãn ngắn gọn: số thứ tự + loại (không cần confidence để tránh dài)
        label  = f"#{order} {b_type}"
        conf   = block.get("confidence", 1.0)
        if conf < 0.99:
            label += f" {conf:.0%}"

        # Đo kích thước text
        try:
            bbox_text = font.getbbox(label)
            tw = bbox_text[2] - bbox_text[0]
            th = bbox_text[3] - bbox_text[1]
        except Exception:
            tw = len(label) * (font_size // 2 + 1)
            th = font_size + 2

        # Chỉ vẽ nhãn nếu box đủ lớn để chứa
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w > tw + PAD * 2 + 4 and box_h > th + PAD * 2:
            # Nền nhãn: hình chữ nhật bên trong, góc trên-trái
            lx1 = x1 + LINE
            ly1 = y1 + LINE
            lx2 = lx1 + tw + PAD * 2
            ly2 = ly1 + th + PAD * 2
            # Màu nền bán trong suốt
            r, g, b = _hex_to_rgb(bg)
            draw.rectangle([lx1, ly1, lx2, ly2], fill=(r, g, b, 200))
            draw.text((lx1 + PAD, ly1 + PAD), label, fill="white", font=font)
        elif box_w > 20 and box_h > 12:
            # Box quá nhỏ: chỉ in số thứ tự
            short = f"#{order}"
            draw.text((x1 + LINE + 1, y1 + LINE + 1), short, fill=outline, font=font)

    return annotated


def sort_reading_order(blocks: List[Dict], image_width: int, two_column: bool = True) -> List[Dict]:
    """
    Sắp xếp blocks theo thứ tự đọc.

    two_column=True  → Kiểu IEEE 2 cột:
        Cột trái (x1 < midpoint) đọc trước từ trên xuống,
        sau đó cột phải (x1 >= midpoint) từ trên xuống.

    two_column=False → Văn xuôi 1 cột:
        Tất cả blocks xếp thẳng theo y1 (trên→dưới),
        không phân biệt cột trái/phải.
    """
    if not two_column:
        # Đơn giản: sắp xếp toàn bộ theo y1
        return sorted(blocks, key=lambda b: b["bbox"][1])

    # IEEE 2 cột
    midpoint = image_width / 2
    left_col  = [b for b in blocks if b["bbox"][0] < midpoint]
    right_col = [b for b in blocks if b["bbox"][0] >= midpoint]
    left_col.sort(key=lambda b: b["bbox"][1])
    right_col.sort(key=lambda b: b["bbox"][1])
    return left_col + right_col


def _ocr_math(p2t, crop: Image.Image) -> str:
    """Nhận diện công thức toán học qua Pix2Text."""
    try:
        res = p2t.recognize_formula(crop)
        if isinstance(res, dict):
            return res.get("text", str(res))
        return str(res)
    except Exception as e:
        try:
            res = p2t.recognize(crop)
            if isinstance(res, dict):
                return res.get("text", str(res))
            return str(res)
        except Exception:
            return f"% Lỗi nhận diện công thức: {e}"


def _ocr_text_pix2text(p2t, crop: Image.Image) -> str:
    """OCR văn bản tiếng Việt bằng Pix2Text."""
    try:
        res = p2t.recognize(crop)
        if isinstance(res, dict):
            return res.get("text", str(res))
        return str(res)
    except Exception as e:
        return f"% Lỗi Pix2Text: {e}"


def _ocr_text_nougat(nougat_model, nougat_device, crop: Image.Image) -> str:
    """OCR văn bản / paragraph Tiếng Anh bằng Nougat (đã load trong bộ nhớ)."""
    try:
        import torch
        from nougat.postprocessing import markdown_compatible

        img_rgb = crop.convert("RGB")
        tensor = nougat_model.encoder.prepare_input(img_rgb).unsqueeze(0).to(nougat_device)
        with torch.no_grad():
            output = nougat_model.inference(image_tensors=tensor, early_stopping=True)
        prediction = output["predictions"][0]
        return clean_latex(markdown_compatible(prediction))
    except Exception as e:
        return f"% Lỗi Nougat: {e}"


def _ocr_table(p2t, crop: Image.Image) -> str:
    """OCR bảng biểu qua Pix2Text."""
    try:
        res = p2t.recognize(crop)
        if isinstance(res, dict):
            return res.get("text", str(res))
        return str(res)
    except Exception as e:
        return f"% Lỗi nhận diện bảng: {e}"


def _wrap_latex(b_type: str, text: str, block_idx: int) -> str:
    """Bọc text nhận diện được vào cú pháp LaTeX phù hợp với loại block."""
    text = text.strip()
    if not text:
        return ""

    if b_type in MATH_TYPES:
        # Nếu chưa có dấu $$, bọc lại
        if not (text.startswith("$$") or text.startswith("\\[")):
            return f"\\[\n{text}\n\\]"
        return text

    if b_type in TABLE_TYPES:
        # Pix2Text có thể trả về Markdown bảng — giữ nguyên dạng Markdown
        return text

    if b_type in FIGURE_TYPES:
        return f"% [FIGURE #{block_idx} — xem ảnh đính kèm]"

    if b_type in TITLE_TYPES:
        return f"\\section*{{{text}}}"

    # Paragraph / text thông thường
    return text


class YoloRouter:
    """
    Điều phối YOLO → OCR → LaTeX cho Tab 2.

    Cách dùng:
        router = YoloRouter(extractor, nougat_model, nougat_device)
        result = router.preview(image, conf=0.25)
        # → result["annotated_img"], result["blocks"]

        result2 = router.convert(image, blocks, progress_callback)
        # → result2["latex"], result2["blocks_with_text"]
    """

    def __init__(self, extractor, nougat_model=None, nougat_device="cpu"):
        self.extractor = extractor          # FormulaExtractor (chứa p2t và layout_parser)
        self.nougat_model = nougat_model
        self.nougat_device = nougat_device

    def preview(
        self,
        image: Image.Image,
        conf_threshold: float = 0.25,
        two_column: bool = True,
    ) -> Dict[str, Any]:
        """
        Bước 1: Chạy YOLO, trả về ảnh annotated + danh sách blocks.
        Không chạy OCR.

        two_column=True  → sắp xếp kiểu IEEE 2 cột
        two_column=False → sắp xếp văn xuôi 1 cột (y từ trên xuống dưới)
        """
        image = image.convert("RGB")

        raw_blocks = self.extractor.get_blocks(image, conf_threshold=conf_threshold)

        # Sắp xếp theo thứ tự đọc (IEEE 2 cột hoặc 1 cột thuần)
        sorted_blocks = sort_reading_order(raw_blocks, image.width, two_column=two_column)

        for idx, b in enumerate(sorted_blocks):
            b["order"] = idx + 1

        annotated = draw_annotated(image, sorted_blocks)

        stats: Dict[str, int] = {}
        for b in sorted_blocks:
            key = "skip" if b["type"] in SKIP_TYPES else b["type"]
            stats[key] = stats.get(key, 0) + 1

        return {
            "annotated_img": annotated,
            "blocks": sorted_blocks,
            "stats": stats,
            "two_column": two_column,
        }

    def convert(
        self,
        image: Image.Image,
        blocks: List[Dict],
        progress_callback=None,        # callable(current, total, msg) hoặc None
        lang_hint: str = "auto",       # 'auto', 'vi', 'en'
        output_format: str = "ieee",   # 'ieee' | 'plain'
    ) -> Dict[str, Any]:
        """
        Bước 2: OCR từng block theo loại, ghép thành LaTeX hoàn chỉnh.

        output_format:
            'ieee'  → Bọc trong \\documentclass{IEEEtran}, dùng \\section, equation env.
            'plain' → Văn bản xuôi dạng Markdown/LaTeX tối giản, không có preamble.
        """
        image = image.convert("RGB")
        p2t = self.extractor.p2t
        total = len(blocks)
        latex_parts: List[str] = []
        blocks_with_text: List[Dict] = []

        for idx, block in enumerate(blocks):
            b_type = block["type"]
            crop = block["image"]
            order = block.get("order", idx + 1)

            if progress_callback:
                progress_callback(idx + 1, total, f"Block #{order} — {b_type}")

            # Bỏ qua header / footer
            if b_type in SKIP_TYPES:
                block_copy = dict(block)
                block_copy["text"] = ""
                block_copy["skipped"] = True
                blocks_with_text.append(block_copy)
                continue

            text = ""

            try:
                if b_type in MATH_TYPES:
                    raw = _ocr_math(p2t, crop)
                    if output_format == "plain":
                        # Dạng xuôi: bọc bằng $$...$$
                        inner = raw.strip().strip("$").strip("\\[").strip("\\]").strip()
                        text = f"$$\n{inner}\n$$" if inner else raw
                    else:
                        # IEEE: dùng \begin{equation}
                        inner = raw.strip().strip("$").strip("\\[").strip("\\]").strip()
                        text = f"\\begin{{equation}}\n{inner}\n\\end{{equation}}" if inner else raw

                elif b_type in FIGURE_TYPES:
                    if output_format == "plain":
                        text = f"> 🖼️ *[Hình #{order} — xem ảnh đính kèm]*"
                    else:
                        text = (
                            f"\\begin{{figure}}[h]\n"
                            f"  \\centering\n"
                            f"  % \\includegraphics{{fig{order}}}\n"
                            f"  \\caption{{Figure {order}}}\n"
                            f"\\end{{figure}}"
                        )

                elif b_type in TABLE_TYPES:
                    raw = _ocr_table(p2t, crop)
                    if output_format == "plain":
                        text = raw  # Giữ dạng Markdown table
                    else:
                        text = raw  # Pix2Text đã trả về dạng tabular

                else:
                    # Tiêu đề / Paragraph / Text
                    if lang_hint == "vi":
                        detected = "vi"
                    elif lang_hint == "en":
                        detected = "en"
                    else:
                        try:
                            sample = _ocr_text_pix2text(p2t, crop)
                            detected = detect_language(sample)
                        except Exception:
                            detected = "en"

                    if detected == "vi":
                        text = _ocr_text_pix2text(p2t, crop)
                    else:
                        if self.nougat_model is not None:
                            text = _ocr_text_nougat(self.nougat_model, self.nougat_device, crop)
                            # FALLBACK: Nougat thường trả về rỗng khi crop quá nhỏ
                            if not text.strip():
                                text = _ocr_text_pix2text(p2t, crop)
                        else:
                            text = _ocr_text_pix2text(p2t, crop)

                    # Xử lý tiêu đề theo format
                    if b_type in TITLE_TYPES:
                        import re
                        # Xóa dấu '#' markdown nếu có
                        t = re.sub(r'^#+\s*', '', text.strip())
                        if output_format == "plain":
                            text = f"## {t}"
                        else:
                            text = f"\\section*{{{t}}}"
                    else:
                        # Thoát ký tự đặc biệt cho LaTeX để tránh mất chữ (vd: %)
                        if output_format == "ieee":
                            # Chỉ escape % nếu nó không đứng sau \ (không phải \%)
                            import re
                            text = re.sub(r'(?<!\\)%', r'\%', text)
                            # Xóa '#' markdown đầu dòng để LaTeX khỏi lỗi
                            text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)

            except Exception as e:
                text = f"% Lỗi block #{order}: {e}"

            latex_parts.append(text)

            block_copy = dict(block)
            block_copy["text"] = text
            block_copy["skipped"] = False
            blocks_with_text.append(block_copy)

        # ── Ghép đầu ra theo format ─────────────────────────────────────────
        latex_body = "\n\n".join(p for p in latex_parts if p.strip())

        if output_format == "ieee":
            latex_full = (
                "% === Được tạo bởi YOLO-Router (IEEEtran) ===\n"
                "\\documentclass{IEEEtran}\n"
                "\\usepackage{amsmath,amssymb,graphicx,booktabs}\n"
                "\\begin{document}\n\n"
                + latex_body
                + "\n\n\\end{document}\n"
            )
            file_ext = ".tex"
            lang_label = "latex"
        else:
            # Plain: Markdown + LaTeX inline, không cần preamble
            latex_full = (
                "<!-- Được tạo bởi YOLO-Router (Plain) -->\n\n"
                + latex_body
            )
            file_ext = ".md"
            lang_label = "markdown"

        return {
            "latex": latex_full,
            "latex_body": latex_body,
            "blocks_with_text": blocks_with_text,
            "file_ext": file_ext,
            "lang_label": lang_label,
            "output_format": output_format,
        }



