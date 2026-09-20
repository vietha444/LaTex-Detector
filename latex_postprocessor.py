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


def markdown_table_to_latex(md_table: str) -> str:
    """
    Chuyển đổi bảng Markdown cơ bản thành dạng LaTeX \\begin{tabular}.
    """
    lines = md_table.strip().split('\n')
    if not lines or len(lines) < 2:
        return md_table
        
    latex_lines = []
    # Tìm số cột dựa trên dòng phân cách (dòng 2)
    sep_line = lines[1]
    if '|' not in sep_line:
        return md_table # Không phải bảng hợp lệ
        
    cols_count = sep_line.count('|') - 1
    if cols_count <= 0:
        return md_table
        
    col_format = "c" * cols_count
    
    latex_lines.append(f"\\begin{{tabular}}{{{col_format}}}")
    latex_lines.append("\\hline")
    
    for i, line in enumerate(lines):
        if i == 1: # Bỏ qua dòng phân cách |---|---|
            continue
            
        # Lấy nội dung các ô
        cells = [cell.strip() for cell in line.split('|')][1:-1]
        if not cells:
            continue
            
        # Nối ô bằng & và kết thúc dòng bằng \\
        row_str = " & ".join(cells) + " \\\\"
        latex_lines.append(row_str)
        
        if i == 0:
            latex_lines.append("\\hline") # Gạch dưới header
            
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    
    return "\n".join(latex_lines)


def markdown_to_pure_latex(text: str, is_table: bool = False) -> str:
    """
    Dịch các phần tử Markdown (Heading, Bold, Italic) thành mã LaTeX tương ứng.
    Bảo vệ các khối toán học để không bị regex sửa nhầm.
    """
    if not text.strip():
        return ""

    # --- 1. Trích xuất và bảo vệ công thức toán học ---
    math_blocks = []
    
    # Placeholder an toàn (tuyệt đối không chứa _, *, # để tránh xung đột regex)
    PLACEHOLDER = "@@@MATHBLOCK{}@@@"
    
    # Regex tìm các khối toán học: $$...$$, \[...\], \(...\), $...$
    math_patterns = [
        r'\$\$.*?\$\$',
        r'\\\[.*?\\\]',
        r'\\\(.*?\\\)',
        r'(?<!\\)\$.*?(?<!\\)\$', # $...$ nhưng không phải \$
        r'\\begin\{.*?\}.*?\\end\{.*?\}'
    ]
    
    temp_text = text
    for pattern in math_patterns:
        # re.DOTALL để khớp qua nhiều dòng
        matches = re.finditer(pattern, temp_text, re.DOTALL)
        offset = 0
        new_text = ""
        last_idx = 0
        
        for match in matches:
            block = match.group(0)
            math_blocks.append(block)
            idx = len(math_blocks) - 1
            
            start, end = match.span()
            new_text += temp_text[last_idx:start] + PLACEHOLDER.format(idx)
            last_idx = end
            
        new_text += temp_text[last_idx:]
        temp_text = new_text

    # --- 2. Xử lý Markdown Table ---
    # Luôn quét bảng qua Regex (kể cả khi YOLO đã tag là is_table hay phân loại nhầm)
    # Điều này đảm bảo toán học bên trong bảng vẫn được bảo vệ an toàn
    def replace_md_table(m):
        return "\n" + markdown_table_to_latex(m.group(1)) + "\n"
    tbl_pattern = r'((?:^|\n)\|.*\|\n\|[-:| ]+\|\n(?:\|.*\|(?:\n|$))*)'
    temp_text = re.sub(tbl_pattern, replace_md_table, temp_text)

    # --- 3. Headings ---
    def replace_heading(m):
        level = len(m.group(1))
        content = m.group(2).strip()
        if level == 1:
            return f"\\section*{{{content}}}"
        elif level == 2:
            return f"\\subsection*{{{content}}}"
        else:
            return f"\\subsubsection*{{{content}}}"
            
    temp_text = re.sub(r'^(#{1,6})\s+(.+)$', replace_heading, temp_text, flags=re.MULTILINE)
    
    # --- 4. Bold (**text**) ---
    temp_text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', temp_text)
    
    # --- 5. Italic (*text* nhưng KHÔNG dùng _text_) ---
    # CẢNH BÁO: Không bao giờ dùng _ để chuyển italic trong văn bản Khoa học/Vật lý,
    # vì nó sẽ phá nát các chỉ số dưới (e.g. \omega_{1} -> \omega\textit{{1}})
    temp_text = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r'\\textit{\1}', temp_text)
    
    # --- 6. Thoát ký tự đặc biệt cho Text thường ---
    # Tránh lỗi Missing $ inserted
    # Chỉ thực hiện trên phần text còn lại, không ảnh hưởng đến placeholder toán học
    temp_text = re.sub(r'(?<!\\)%', r'\%', temp_text)
    temp_text = re.sub(r'(?<!\\)&', r'\&', temp_text)
    temp_text = re.sub(r'(?<!\\)_', r'\_', temp_text)
    temp_text = re.sub(r'(?<!\\)#', r'\#', temp_text)

    # --- 7. Phục hồi công thức toán học ---
    for i, block in enumerate(math_blocks):
        temp_text = temp_text.replace(PLACEHOLDER.format(i), block)
        
    return temp_text
