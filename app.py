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

            st.markdown("---")
            col_fmt_1, col_fmt_2 = st.columns(2)
            with col_fmt_1:
                out_fmt_choice = st.radio(
                    "Định dạng mã xuất ra:",
                    [
                        "📄 Thuần LaTeX (Tạp chí - IEEEtran)", 
                        "📄 Thuần LaTeX (Thường - Article)", 
                        "🤖 Markdown thô (Dành cho AI đọc)"
                    ],
                    index=2
                )
            with col_fmt_2:
                engine_choice = st.radio(
                    "Động cơ OCR (Model):",
                    [
                        "🌟 Kết hợp (Nougat cho chữ, Pix2Text cho Toán/Bảng)", 
                        "🧪 Chỉ dùng Nougat (Toàn bộ bằng Nougat)",
                        "⚡ Chỉ dùng Pix2Text (Nhanh, Toàn bộ bằng Pix2Text)"
                    ],
                    index=0,
                    help="Nougat thuần có thể nhận diện tốt hơn các công thức vật lý (Dirac, Feynman) nhưng dễ bị ảo giác nếu vùng cắt quá nhỏ."
                )

            lang_hint_map   = {"Tự động (Auto)": "auto", "Tiếng Việt": "vi", "Tiếng Anh": "en"}
            output_fmt_map  = {
                "📄 Thuần LaTeX (Tạp chí - IEEEtran)": "pure_latex_ieee", 
                "📄 Thuần LaTeX (Thường - Article)": "pure_latex_article",
                "🤖 Markdown thô (Dành cho AI đọc)": "markdown"
            }
            if "Chỉ dùng Nougat" in engine_choice:
                engine_mode = "nougat"
            elif "Chỉ dùng Pix2Text" in engine_choice:
                engine_mode = "pix2text"
            else:
                engine_mode = "hybrid"
                
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
                            two_column=two_column
                        )
                        st.session_state["yolo_blocks"] = preview["blocks"]
                        st.session_state["yolo_preview_img"] = preview["annotated_img"]
                        st.session_state["yolo_page_img"] = page_img
                    except Exception as e:
                        st.error(f"Lỗi: {e}")

            if "yolo_blocks" in st.session_state and "yolo_preview_img" in st.session_state:
                st.image(st.session_state["yolo_preview_img"], use_column_width=True)

                blocks = st.session_state["yolo_blocks"]
                from yolo_router import SKIP_TYPES, MATH_TYPES, FIGURE_TYPES, TABLE_TYPES
                with st.expander(f"📋 Bảng thứ tự Blocks ({len(blocks)})"):
                    for b in blocks:
                        if b["type"] in SKIP_TYPES:
                            eng = "🚫 Bỏ qua"
                        elif b["type"] in MATH_TYPES or b["type"] in TABLE_TYPES:
                            eng = "🟢 Nougat" if engine_mode == "nougat" else "🟢 Pix2Text"
                        elif any(k in b["type"] for k in FIGURE_TYPES):
                            eng = "🟣 Placeholder"
                        else:
                            if engine_mode == "nougat":
                                eng = "🔵 Nougat"
                            else:
                                eng = "🔵 Nougat" if lang_hint in ("auto", "en") else "🔵 Pix2Text (vi)"

                        row = st.columns([0.4, 1.4, 1.8, 2.2, 1.5])
                        row[0].write(b["order"])
                        row[1].write(b["type"])
                        row[2].write(eng)
                        x1, y1, x2, y2 = [int(v) for v in b["bbox"]]
                        row[3].caption(f"({x1},{y1})→({x2},{y2})")
                        row[4].image(b["image"], use_column_width=True)

                # ── Cấu hình LLM ──────────────────────────────────────────────
                st.markdown("---")
                st.markdown("### ✨ Chuẩn hóa LaTeX bằng AI (Ollama Local)")
                use_ollama = st.checkbox("Kích hoạt tự động sửa lỗi và dọn dẹp ảo giác bằng LLM", value=False)
                col_m, col_u = st.columns(2)
                with col_m:
                    ollama_model = st.text_input("Tên Model (vd: gemma, llama3, qwen2):", value="gemma", disabled=not use_ollama)
                with col_u:
                    ollama_url = st.text_input("Ollama API URL:", value="http://chatbot.tail36da8e.ts.net:11434", disabled=not use_ollama)

                # ── Bước 2: Convert ──────────────────────────────────────────────
                st.markdown("---")
                st.markdown("### 🚀 Bước 2 — Chuyển đổi LaTeX")

                c_btn1, c_btn2 = st.columns(2)
                run_single = c_btn1.button("🚀 Chuyển đổi Trang hiện tại", type="primary", use_container_width=True, key="btn_convert_single")
                run_all = c_btn2.button(f"📚 Chuyển đổi TOÀN BỘ ({len(pages)} trang)", type="primary", use_container_width=True, key="btn_convert_all")

                if run_single or run_all:
                    prog_bar  = st.progress(0, text="Chuẩn bị...")
                    status_ph = st.empty()

                    try:
                        nougat_model_inst = None
                        nougat_dev = "cpu"
                        # Luôn load Nougat nếu engine_mode là nougat hoặc (hybrid + text TA)
                        if engine_mode == "nougat" or lang_hint in ("auto", "en"):
                            with st.spinner("Đang tải Nougat Base vào bộ nhớ (lần đầu ~2 phút)..."):
                                nougat_model_inst, nougat_dev = load_nougat()

                        from yolo_router import YoloRouter
                        router2 = YoloRouter(extractor, nougat_model_inst, nougat_dev)

                        def on_progress(current, total, msg):
                            prog_bar.progress(current / total, text=f"[{current}/{total}] {msg}")
                            status_ph.markdown(f"⏳ **{msg}**")

                        if run_single:
                            page_img2 = st.session_state["yolo_page_img"]
                            result = router2.convert(
                                page_img2, blocks,
                                progress_callback=on_progress,
                                lang_hint=lang_hint,
                                output_format=output_fmt_map[out_fmt_choice],
                                engine_mode=engine_mode
                            )
                        else:
                            result = router2.convert_batch(
                                pages,
                                conf_threshold=conf_tab2,
                                two_column=two_column,
                                progress_callback=on_progress,
                                lang_hint=lang_hint,
                                output_format=output_fmt_map[out_fmt_choice],
                                engine_mode=engine_mode
                            )

                        prog_bar.progress(1.0, text="✅ Hoàn tất OCR!")
                        
                        latex_full  = result["latex"]
                        latex_body  = result["latex_body"]
                        file_ext    = result.get("file_ext", ".tex")
                        lang_label  = result.get("lang_label", "latex")
                        out_fmt     = result.get("output_format", "pure_latex_ieee")
                        
                        latex_full_download = latex_full

                        if use_ollama:
                            status_ph.info("⏳ Đang gửi sang Ollama để AI tự động dọn dẹp ảo giác và sửa cú pháp... (Có thể mất vài phút)")
                            with st.spinner(f"Chờ phản hồi từ Ollama ({ollama_model})..."):
                                try:
                                    from ollama_corrector import fix_latex_with_ollama
                                    latex_full_corrected = fix_latex_with_ollama(latex_full, ollama_model, ollama_url)
                                    status_ph.success(f"✅ Chuyển đổi và Sửa lỗi AI hoàn tất!")
                                    
                                    st.markdown(f"### Mã nguồn xuất ra ({file_ext})")
                                    tab_ai, tab_raw = st.tabs(["✨ Đã sửa lỗi (Ollama)", "📄 Bản gốc (OCR)"])
                                    with tab_ai:
                                        st.code(latex_full_corrected, language=lang_label)
                                    with tab_raw:
                                        st.code(latex_full, language=lang_label)
                                        
                                    latex_full_download = latex_full_corrected
                                except Exception as ollama_err:
                                    status_ph.error(f"⚠️ Lỗi kết nối Ollama: {ollama_err}")
                                    st.markdown(f"### Mã nguồn xuất ra ({file_ext}) - Bản Gốc")
                                    st.code(latex_full, language=lang_label)
                        else:
                            status_ph.success(f"✅ Chuyển đổi hoàn tất {'toàn bộ file' if run_all else 'trang này'}!")
                            st.markdown(f"### Mã nguồn xuất ra ({file_ext})")
                            st.code(latex_full, language=lang_label)

                        st.markdown("### 👀 Xem trước nội dung (Bản gốc)")
                        st.markdown(latex_body)

                        if run_single:
                            with st.expander("🔬 Chi tiết từng Block (chỉ hiển thị khi chạy 1 trang)"):
                                for b in result.get("blocks_with_text", []):
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
                            data=latex_full_download,
                            file_name=f"yolo_output{file_ext}",
                            mime="text/plain",
                            use_container_width=True
                        )
                        c2.download_button(
                            f"⬇️ Tải phần Body ({file_ext})",
                            data=latex_body,
                            file_name=f"yolo_body{file_ext}",
                            mime="text/plain",
                            use_container_width=True
                        )

                    except Exception as e:
                        st.error(f"Lỗi khi chuyển đổi OCR: {e}")
                        import traceback
                        st.code(traceback.format_exc())
