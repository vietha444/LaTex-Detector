import streamlit as st
import os
import sys
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
doc_language = st.sidebar.radio(
    "Ngôn ngữ tài liệu:",
    ("Tiếng Anh (Mặc định)", "Tiếng Việt + Anh")
)
poppler_path = st.sidebar.text_input(
    "Đường dẫn Poppler (bin):",
    value=r"C:\Users\admin\Downloads\Release-26.09.0-0\poppler-26.09.0\Library\bin"
)
conf_threshold = st.sidebar.slider("Ngưỡng tin cậy YOLO", 0.1, 0.9, 0.25, 0.05)
extract_all = st.sidebar.checkbox("Trích xuất tất cả (bao gồm text)", value=False)
dpi = st.sidebar.selectbox("DPI render PDF", [150, 200, 300], index=2)

# Xác định tham số ngôn ngữ cho Engine
lang_tuple = ('vi', 'en') if "Việt" in doc_language else ('en',)

# --- Load Engine ---
@st.cache_resource
def load_engine(languages):
    return FormulaExtractor(languages=languages)

with st.spinner("🔄 Đang tải mô hình AI (YOLO + Pix2Text)..."):
    try:
        extractor = load_engine(lang_tuple)
        engine_ok = True
    except Exception as e:
        st.error(f"Lỗi khởi tạo engine: {e}")
        engine_ok = False

# --- Load Nougat ---
@st.cache_resource
def load_nougat():
    import torch
    import transformers.generation.utils as gen_utils
    def noop_validate(*args, **kwargs):
        pass
    gen_utils.GenerationMixin._validate_model_kwargs = noop_validate

    from nougat.model import BARTDecoder
    original_prepare = BARTDecoder.prepare_inputs_for_inference
    def new_prepare(self, *args, **kwargs):
        kwargs.pop('cache_position', None)
        return original_prepare(self, *args, **kwargs)
    BARTDecoder.prepare_inputs_for_inference = new_prepare
    
    from nougat import NougatModel
    from nougat.utils.checkpoint import get_checkpoint
    
    # Sử dụng Base model (khoảng 900MB - lớn hơn Small nhưng cực kỳ chính xác)
    checkpoint = get_checkpoint(model_tag="0.1.0-base")
    model = NougatModel.from_pretrained(checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return model, device

# --- Upload ---
uploaded_file = st.file_uploader("📄 Tải lên file PDF", type=["pdf"])

if uploaded_file is not None and engine_ok:
    if "pdf_pages" not in st.session_state or st.session_state.get("last_uploaded_file") != uploaded_file.name:
        with st.spinner("Đang render PDF thành ảnh..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_pdf_path = tmp.name
            
            try:
                processor = PDFProcessor(dpi=dpi, poppler_path=poppler_path)
                st.session_state.pdf_pages = processor.process(tmp_pdf_path, output_dir=None)
                st.session_state.last_uploaded_file = uploaded_file.name
            except Exception as e:
                st.error(f"Lỗi đọc PDF: {e}")
                st.session_state.pdf_pages = []
            finally:
                try:
                    os.remove(tmp_pdf_path)
                except Exception:
                    pass

    pages = st.session_state.get("pdf_pages", [])
    if pages:
        st.success(f"✅ Đã tải xong tài liệu gồm **{len(pages)} trang**.")

        tab_formulas, tab_fullpage = st.tabs([
            "✂️ Trích xuất Công thức (Bản chuẩn)", 
            "📄 Chuyển đổi Toàn trang (Auto-Router)"
        ])

        with tab_formulas:
            if st.button("🚀 Quét và Bóc tách Công thức", type="primary", use_container_width=True):
                try:
                    all_results = []
                    progress = st.progress(0, text="Bắt đầu phân tích...")

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

                                try:
                                    clean = latex_str.strip()
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

        with tab_fullpage:
            st.info(
                "💡 **YOLO-Router (IEEE 2 cột):** YOLO phân tích bố cục → "
                "từng block được định tuyến đến đúng engine → ghép LaTeX theo thứ tự đọc."
            )

            col_pg, col_conf, col_lang, col_fmt = st.columns([1, 1, 1, 1])
            with col_pg:
                page_to_process = st.number_input(
                    "Chọn trang:", min_value=1, max_value=len(pages), value=1, key="tab2_page"
                )
            with col_conf:
                conf_tab2 = st.slider("Ngưỡng YOLO:", 0.1, 0.9, 0.25, 0.05, key="tab2_conf")
            with col_lang:
                lang_tab2 = st.radio(
                    "Gợi ý ngôn ngữ:",
                    ["Tự động (Auto)", "Tiếng Việt", "Tiếng Anh"],
                    index=2, key="tab2_lang",
                    help="Tiếng Anh → Nougat cho paragraph, Tiếng Việt → Pix2Text"
                )
            with col_fmt:
                layout_tab2 = st.radio(
                    "Bố cục đọc:",
                    ["🗞️ IEEE (2 cột)", "📄 Văn xuôi (1 cột)"],
                    index=0, key="tab2_layout",
                    help=(
                        "IEEE 2 cột: đọc cột trái hết, rồi cột phải (đúng thứ tự tạp chí IEEE).\n"
                        "Văn xuôi 1 cột: tất cả block xếp thẳng từ trên xuống theo vị trí y."
                    )
                )

            lang_hint_map   = {"Tự động (Auto)": "auto", "Tiếng Việt": "vi", "Tiếng Anh": "en"}
            output_fmt_map  = {"📄 IEEE LaTeX (.tex)": "ieee", "📝 Văn xuôi (.md)": "plain"}
            two_column      = (layout_tab2 == "🗞️ IEEE (2 cột)")
            lang_hint       = lang_hint_map[lang_tab2]

            # ── Bước 1: YOLO Preview ─────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 🔍 Bước 1 — Phân tích Bố cục (YOLO Preview)")

            if st.button("🔍 Phân tích Layout", use_container_width=True, key="btn_preview"):
                page_img = pages[page_to_process - 1]
                with st.spinner("Đang chạy YOLO phân tích bố cục..."):
                    try:
                        from yolo_router import YoloRouter
                        router = YoloRouter(extractor)
                        preview = router.preview(
                            page_img,
                            conf_threshold=conf_tab2,
                            two_column=two_column,
                        )
                        st.session_state["yolo_preview"]  = preview
                        st.session_state["yolo_page_img"] = page_img
                        st.session_state["yolo_page_idx"] = page_to_process
                    except Exception as e:
                        st.error(f"Lỗi phân tích layout: {e}")
                        import traceback
                        st.code(traceback.format_exc())

            if "yolo_preview" in st.session_state and st.session_state.get("yolo_page_idx") == page_to_process:
                preview = st.session_state["yolo_preview"]
                blocks = preview["blocks"]
                stats = preview["stats"]

                st.image(
                    preview["annotated_img"],
                    caption=f"Bounding Box YOLO — {'IEEE 2 cột (trái → phải)' if preview.get('two_column', True) else 'Văn xuôi 1 cột (trên → dưới)'}",
                    use_column_width=True
                )
                st.markdown(
                    "🔴 Công thức &nbsp;|&nbsp; 🔵 Văn bản/Paragraph &nbsp;|&nbsp; "
                    "🟢 Tiêu đề &nbsp;|&nbsp; 🟡 Bảng &nbsp;|&nbsp; "
                    "🟣 Hình &nbsp;|&nbsp; ⚪ Bỏ qua (Header/Footer)"
                )
                stat_str = " &nbsp;|&nbsp; ".join(f"**{k}**: {v}" for k, v in stats.items())
                st.markdown(f"📊 {stat_str}")

                st.markdown("#### Chi tiết Blocks theo thứ tự đọc")
                from yolo_router import SKIP_TYPES, MATH_TYPES, FIGURE_TYPES, TABLE_TYPES

                header_cols = st.columns([0.4, 1.4, 1.8, 2.2, 1.5])
                header_cols[0].markdown("**#**")
                header_cols[1].markdown("**Loại**")
                header_cols[2].markdown("**Engine**")
                header_cols[3].markdown("**Bbox**")
                header_cols[4].markdown("**Preview**")

                for b in blocks:
                    skip = b["type"] in SKIP_TYPES
                    if skip:
                        eng = "⚪ Bỏ qua"
                    elif any(k in b["type"] for k in MATH_TYPES):
                        eng = "🔴 Pix2Text Math"
                    elif any(k in b["type"] for k in TABLE_TYPES):
                        eng = "🟡 Pix2Text Table"
                    elif any(k in b["type"] for k in FIGURE_TYPES):
                        eng = "🟣 Placeholder"
                    else:
                        eng = "🔵 Nougat" if lang_hint in ("auto", "en") else "🔵 Pix2Text (vi)"

                    row = st.columns([0.4, 1.4, 1.8, 2.2, 1.5])
                    row[0].write(b["order"])
                    row[1].write(b["type"])
                    row[2].write(eng)
                    x1, y1, x2, y2 = [int(v) for v in b["bbox"]]
                    row[3].caption(f"({x1},{y1})→({x2},{y2})")
                    row[4].image(b["image"], use_column_width=True)

                # ── Bước 2: Convert ──────────────────────────────────────────────
                st.markdown("---")
                st.markdown("### 🚀 Bước 2 — Chuyển đổi LaTeX từng Block")

                if st.button("🚀 Bắt đầu Chuyển đổi", type="primary", use_container_width=True, key="btn_convert"):
                    page_img2 = st.session_state["yolo_page_img"]
                    prog_bar  = st.progress(0, text="Chuẩn bị...")
                    status_ph = st.empty()

                    try:
                        nougat_model_inst = None
                        nougat_dev = "cpu"
                        if lang_hint in ("auto", "en"):
                            with st.spinner("Đang tải Nougat Base vào bộ nhớ (lần đầu ~2 phút)..."):
                                nougat_model_inst, nougat_dev = load_nougat()

                        from yolo_router import YoloRouter
                        router2 = YoloRouter(extractor, nougat_model_inst, nougat_dev)

                        def on_progress(current, total, msg):
                            prog_bar.progress(current / total, text=f"[{current}/{total}] {msg}")
                            status_ph.markdown(f"⏳ **{msg}**")

                        result = router2.convert(
                            page_img2, blocks,
                            progress_callback=on_progress,
                            lang_hint=lang_hint,
                            output_format="ieee"
                        )

                        prog_bar.progress(1.0, text="✅ Hoàn tất!")
                        status_ph.success("✅ Chuyển đổi hoàn tất!")

                        latex_full  = result["latex"]
                        latex_body  = result["latex_body"]
                        file_ext    = result.get("file_ext", ".tex")
                        lang_label  = result.get("lang_label", "latex")
                        out_fmt     = result.get("output_format", "ieee")

                        out_label = "📄 LaTeX IEEEtran" if out_fmt == "ieee" else "📝 Văn xuôi Markdown"
                        st.markdown(f"### {out_label}")
                        st.code(latex_full, language=lang_label)

                        st.markdown("### 👀 Xem trước nội dung")
                        st.markdown(latex_body)

                        with st.expander("🔬 Chi tiết từng Block"):
                            for b in result["blocks_with_text"]:
                                if b.get("skipped"):
                                    continue
                                st.markdown(f"**Block #{b['order']} — {b['type']}**")
                                ca, cb = st.columns([1, 2])
                                ca.image(b["image"], use_column_width=True)
                                cb.code(b.get("text", ""), language=lang_label)
                                st.divider()

                        c1, c2 = st.columns(2)
                        c1.download_button(
                            f"⬇️ Tải đầy đủ ({file_ext})",
                            data=latex_full,
                            file_name=f"page_{page_to_process}_yolo{file_ext}",
                            mime="text/plain" if file_ext == ".tex" else "text/markdown",
                            use_container_width=True,
                            key="btn_dl_tex"
                        )
                        c2.download_button(
                            "⬇️ Tải Body (.md)",
                            data=latex_body,
                            file_name=f"page_{page_to_process}_body.md",
                            mime="text/markdown",
                            use_container_width=True,
                            key="btn_dl_md2"
                        )

                    except Exception as e:
                        st.error(f"Lỗi chuyển đổi: {e}")
                        import traceback
                        st.code(traceback.format_exc())

