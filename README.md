# Chat_box_LLM

## Giới thiệu

Dự án **Chat_box_LLM** xây dựng một hệ thống chat ứng dụng mô hình ngôn ngữ lớn (LLM) để trả lời tin nhắn người dùng. Toàn bộ hệ thống được triển khai trên Jupyter Notebook, dễ dàng thử nghiệm, chỉnh sửa và mở rộng.

## Cấu trúc dự án

Dự án bao gồm các thành phần chính sau:

### 1. Tiền xử lý dữ liệu (Preprocessing)
- **Chức năng:** Làm sạch, chuẩn hóa dữ liệu đầu vào từ người dùng trước khi gửi tới mô hình LLM.
- **Code liên quan:** Cụ thể là các hàm loại bỏ ký tự đặc biệt, kiểm tra lỗi chính tả, tách câu, chuẩn hóa tiếng Việt/Anh.

### 2. Tương tác với mô hình LLM
- **Chức năng:** Kết nối, gửi yêu cầu và nhận phản hồi từ mô hình ngôn ngữ lớn (ví dụ: GPT, Llama, v.v.).
- **Code liên quan:** Khởi tạo đối tượng mô hình, định nghĩa hàm gửi prompt và nhận response.

### 3. Giao diện chat (UI)
- **Chức năng:** Tạo giao diện chat trên Jupyter hoặc web (Streamlit, Gradio…), tương tác trực tiếp với người dùng.
- **Code liên quan:** Cell Jupyter với widget nhập liệu, nút gửi, hiển thị lịch sử hội thoại.

### 4. Xử lý và lưu trữ lịch sử hội thoại
- **Chức năng:** Lưu lại các tin nhắn, phản hồi trong một phiên làm việc để phục vụ phân tích hoặc huấn luyện lại.
- **Code liên quan:** Sử dụng list hoặc pandas DataFrame để lưu trữ, có thể xuất ra file .csv.

### 5. Các hàm tiện ích và cấu hình
- **Chức năng:** Đọc config, cài đặt các thông số hệ thống, hàm format dữ liệu, logging.
- **Code liên quan:** Đọc file cấu hình, thiết lập API key, định nghĩa biến môi trường.

## Luồng hoạt động của hệ thống

1. Người dùng nhập tin nhắn.
2. Hệ thống tiền xử lý nội dung vừa nhập.
3. Gửi nội dung đến mô hình LLM và nhận phản hồi.
4. Hiển thị phản hồi lên giao diện chat.
5. Lưu lại lịch sử hội thoại.

