# Hệ thống Nhận diện và Trích xuất Công thức từ PDF sang LaTeX

Dự án này sử dụng YOLOv8 để tìm vị trí các công thức toán học trong file PDF và Pix2Tex (LaTeX-OCR) để dịch các hình ảnh đã cắt thành mã LaTeX.

## Yêu cầu hệ thống
- Python 3.9 trở lên
- (Dành cho Windows) Cần cài đặt Poppler để thư viện `pdf2image` có thể chuyển đổi PDF sang ảnh. Tải tại: [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/)

## Hướng dẫn cài đặt
1. Cài đặt các thư viện Python:
```bash
pip install -r requirements.txt
```

2. (Tùy chọn) Download trọng số YOLOv8 đã fine-tune cho công thức toán học và đổi tên thành `yolov8n.pt` (hoặc chỉ định qua tham số command line).

## Cách sử dụng

Chạy lệnh sau trong Terminal/Command Prompt:

```bash
python main.py -i <đường_dẫn_file_pdf> -o <thư_mục_kết_quả> --poppler <đường_dẫn_đến_poppler_bin>
```

**Ví dụ trên Windows:**
```bash
python main.py -i sample_math.pdf -o output_data --poppler "C:\Users\admin\Downloads\Release-26.09.0-0\poppler-26.09.0\Library\bin"
```

## Sử dụng qua Giao diện Web (Web UI)
Hệ thống đã được tích hợp Web UI bằng Streamlit, cho phép bạn tải file lên và xem kết quả trực quan (ảnh công thức đối chiếu với mã LaTeX render).

Chạy lệnh sau:
```bash
streamlit run app.py
```
Trình duyệt sẽ tự động mở tab mới tại địa chỉ `http://localhost:8501`.

## Kết quả
Hệ thống sẽ tạo ra một thư mục kết quả (`output_data`), bao gồm:
- Thư mục `pages/`: Chứa các trang PDF đã render thành ảnh PNG.
- Thư mục `crops/`: Chứa từng công thức đã được cắt rời (để tiện kiểm chứng).
- File `formulas_report.json`: Chứa tọa độ bounding box và chuỗi LaTeX kết quả tương ứng.
