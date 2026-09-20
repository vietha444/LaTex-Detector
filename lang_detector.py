import re

def is_vietnamese(text: str) -> bool:
    """
    Phát hiện tiếng Việt thông qua các ký tự có dấu đặc trưng.
    """
    # Tập các ký tự có dấu đặc trưng của tiếng Việt
    vn_chars = set("àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ")
    
    text_lower = text.lower()
    vn_count = sum(1 for char in text_lower if char in vn_chars)
    
    # Nếu có ký tự đặc trưng, khả năng cao là tiếng Việt
    return vn_count >= 1

def detect_language(text: str) -> str:
    """
    Trả về 'vi' nếu là tiếng Việt, ngược lại 'en'.
    """
    if not text:
        return 'en'
    if is_vietnamese(text):
        return 'vi'
    return 'en'
