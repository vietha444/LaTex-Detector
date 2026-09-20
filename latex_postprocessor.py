import re

def clean_latex(text: str) -> str:
    """
    Chuẩn hóa kết quả đầu ra của OCR.
    """
    if not text:
        return ""
        
    # Xóa các chuỗi hallucination thường gặp của Nougat
    hallucinations = [
        "[MISSING_PAGE_POST]",
        "[MISSING_PAGE_EMPTY]",
        "[MISSING_PAGE_FAIL]"
    ]
    for h in hallucinations:
        text = text.replace(h, "")
        
    return text.strip()
