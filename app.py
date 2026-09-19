import streamlit as st
import os
import json
import tempfile
from PIL import Image

from pdf_processor import PDFProcessor
from formula_extractor import FormulaExtractor

st.set_page_config(page_title="PDF Formula → LaTeX", page_icon="📐", layout="wide")

st.title("📐 Hệ thống Trích xuất Công thức từ PDF sang LaTeX")
st.markdown("Sử dụng **YOLO Layout Detection** + **Pix2Text OCR** để trích xuất chính xác.")

# --- Sidebar ---
st.sidebar.header("⚙️ Cấu hình")
poppler_path = st.sidebar.text_input(
    "Đường dẫn Poppler (bin):",
    value=r"C:\Users\admin\Downloads\Release-26.09.0-0\poppler-26.09.0\Library\bin"
)
conf_threshold = st.sidebar.slider("Ngưỡng tin cậy YOLO", 0.1, 0.9, 0.25, 0.05)
extract_all = st.sidebar.checkbox("Trích xuất tất cả (bao gồm text)", value=False)
dpi = st.sidebar.selectbox("DPI render PDF", [150, 200, 300], index=2)

# --- Load Engine ---
@st.cache_resource
def load_engine():
    return FormulaExtractor()

with st.spinner("🔄 Đang tải mô hình AI (YOLO + Pix2Text)..."):
    try:
        extractor = load_engine()
        engine_ok = True
    except Exception as e:
        st.error(f"Lỗi khởi tạo engine: {e}")
        engine_ok = False

# --- Upload ---
uploaded_file = st.file_uploader("📄 Tải lên file PDF", type=["pdf"])

if uploaded_file is not None and engine_ok:
    if st.button("🚀 Quét và Phân tích PDF", type="primary", use_container_width=True):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_pdf_path = tmp.name

        try:
            processor = PDFProcessor(dpi=dpi, poppler_path=poppler_path)

            with st.spinner("Đang render PDF thành ảnh..."):
                pages = processor.process(tmp_pdf_path, output_dir=None)

            st.info(f"Đã render **{len(pages)} trang**. Đang phân tích...")

            all_results = []
            progress = st.progress(0, text="Bắt đầu phân tích...")

            # ==========================================
            # PHẦN 1: TOÀN CẢNH BOUNDING BOX
            # ==========================================
            st.header("🖼️ Toàn cảnh Bounding Box")
            st.caption("🔴 Đỏ = Công thức toán học &nbsp;&nbsp; 🔵 Xanh = Văn bản / Khác")

            for page_idx, page_img in enumerate(pages):
                progress.progress(
                    (page_idx + 1) / len(pages),
                    text=f"Trang {page_idx + 1}/{len(pages)}..."
                )

                annotated_img, formulas = extractor.process_page(
                    page_img, page_idx,
                    conf_threshold=conf_threshold,
                    extract_all=extract_all
                )
                all_results.extend(formulas)

                with st.expander(f"📄 Trang {page_idx + 1}  —  {len(formulas)} công thức phát hiện", expanded=(page_idx == 0)):
                    st.image(annotated_img, use_column_width=True)

            progress.progress(1.0, text="✅ Hoàn tất!")

            # ==========================================
            # PHẦN 2: CHI TIẾT CÔNG THỨC ĐÃ CẮT + LATEX
            # ==========================================
            if not all_results:
                st.warning("Không phát hiện công thức nào. Thử giảm ngưỡng tin cậy trong Sidebar hoặc bật 'Trích xuất tất cả'.")
            else:
                st.header(f"✂️ Chi tiết {len(all_results)} Công thức")

                report_data = []
                for item in all_results:
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.image(
                            item['image'],
                            caption=f"Trang {item['page']} • Box #{item['box_idx']} • {item['type']} ({item['confidence']:.0%})",
                            use_column_width=True
                        )

                    with col2:
                        latex_str = item.get('latex', '')
                        st.markdown("**Mã LaTeX:**")
                        st.code(latex_str, language='latex')

                        # Render LaTeX
                        try:
                            clean = latex_str.strip()
                            # Loại bỏ $$ wrapper nếu có
                            if clean.startswith('$$') and clean.endswith('$$'):
                                clean = clean[2:-2].strip()
                            elif clean.startswith('$') and clean.endswith('$'):
                                clean = clean[1:-1].strip()
                            if clean:
                                st.markdown("**Hiển thị:**")
                                st.latex(clean)
                        except Exception:
                            pass

                    report_data.append({
                        "page": item["page"],
                        "box_idx": item["box_idx"],
                        "bbox": item["bbox"],
                        "confidence": item["confidence"],
                        "type": item["type"],
                        "latex": latex_str
                    })
                    st.divider()

                # ==========================================
                # PHẦN 3: TẢI BÁO CÁO
                # ==========================================
                st.header("💾 Xuất Báo cáo")
                json_str = json.dumps(report_data, ensure_ascii=False, indent=2)
                st.download_button(
                    "⬇️ Tải báo cáo JSON",
                    data=json_str,
                    file_name="formula_report.json",
                    mime="application/json",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Lỗi: {e}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            try:
                os.remove(tmp_pdf_path)
            except Exception:
                pass
