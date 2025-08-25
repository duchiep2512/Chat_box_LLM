# Hệ thống Chatbot RAG cho 1 trang web

Hệ thống chatbot thông minh sử dụng công nghệ RAG (Retrieval-Augmented Generation) để trả lời các câu hỏi về chính sách bảo mật của Presight, được xây dựng với Streamlit, Google Gemini AI, và Sentence Transformers.

## Tổng quan dự án

Dự án này triển khai một hệ thống RAG hoàn chỉnh bao gồm: crawl dữ liệu chính sách bảo mật từ website Presight, xử lý và cấu trúc hóa dữ liệu, sau đó tạo giao diện chatbot thông minh để trả lời các câu hỏi của người dùng về quyền riêng tư và sử dụng dữ liệu.

## Quy trình làm việc

1. **Crawl dữ liệu**: Trích xuất nội dung chính sách bảo mật từ https://www.presight.io/privacy-policy.html
2. **Xử lý dữ liệu**: Làm sạch, cấu trúc và đánh index nội dung đã crawl
3. **Xây dựng RAG**: Tạo pipeline retrieval và generation
4. **Giao diện chat**: Phát triển ứng dụng web Streamlit cho người dùng

## Cấu trúc dự án

```
Chat_box_LLM/
├── app.py                    # Ứng dụng Streamlit chính
├── rag_system.py            # Hệ thống RAG với embedding và generation  
├── ui_components.py         # Các component giao diện Streamlit
├── config.py                # Cấu hình và thiết lập hằng số
├── indexed_list.json        # Dữ liệu chính sách đã xử lý và đánh index
├── notebooks/               # Jupyter notebooks cho phân tích và test
├── requirements.txt         # Các package Python cần thiết
└── README.md               # Tài liệu dự án
```

## Mô tả chi tiết từng file

### Các file ứng dụng chính

**`app.py`**
- Điểm khởi đầu chính của ứng dụng Streamlit
- Điều phối toàn bộ workflow của chatbot
- Khởi tạo hệ thống RAG và các component UI
- Xử lý input từ người dùng và hiển thị kết quả
- Quản lý session state và luồng hoạt động của ứng dụng

**`rag_system.py`** 
- Triển khai pipeline RAG (Retrieval-Augmented Generation) hoàn chỉnh
- Load và xử lý dữ liệu chính sách bảo mật đã được index
- Thực hiện text embedding sử dụng Sentence Transformers (all-MiniLM-L6-v2)
- Tìm kiếm similarity để lấy các phần nội dung liên quan
- Tích hợp với Google Gemini API để sinh ngôn ngữ tự nhiên
- Quản lý việc xây dựng context và tạo câu trả lời từ documents

**`ui_components.py`**
- Chứa tất cả các component UI và styling functions của Streamlit
- Triển khai CSS tùy chỉnh cho giao diện chat chuyên nghiệp
- Quản lý hiển thị tin nhắn với style khác nhau cho user/bot
- Tạo sidebar với controls, thống kê và các câu hỏi mẫu
- Xử lý khởi tạo và quản lý session state
- Cung cấp tính năng export/import lịch sử chat
- Render welcome screen, form input và hiển thị phản hồi

**`config.py`**
- File cấu hình trung tâm cho tất cả thiết lập hệ thống
- Chứa API keys và cấu hình models
- Định nghĩa các tham số RAG (giá trị top-k, ngưỡng similarity)
- Thiết lập tham số hiển thị UI (độ dài preview, giới hạn tin nhắn)
- Lưu trữ các câu hỏi mẫu và metadata ứng dụng
- Quản lý đường dẫn files và tên models

### Dữ liệu và tài liệu

**`indexed_list.json`**
- Dữ liệu chính sách bảo mật đã được xử lý và cấu trúc hóa
- Chứa các sections, headings và nội dung từ privacy policy của Presight
- Bao gồm metadata để thực hiện retrieval hiệu quả
- Có thể lưu trữ pre-computed embeddings để tìm kiếm nhanh hơn
- Format cấu trúc được tối ưu cho việc sử dụng trong hệ thống RAG

**`notebooks/`**
- Các Jupyter notebooks phục vụ phát triển và phân tích
- Notebooks xử lý và khám phá dữ liệu
- Scripts test và đánh giá models
- Phân tích hiệu suất hệ thống RAG
- Các tính năng thử nghiệm và cải tiến

## Triển khai kỹ thuật

### Quy trình crawl dữ liệu
Hệ thống trích xuất nội dung từ https://www.presight.io/privacy-policy.html, phân tích cấu trúc HTML để xác định các sections chính sách, headings và nội dung text liên quan.

### Pipeline RAG
1. **Embedding**: Nội dung text được chuyển đổi thành vectors 384 chiều sử dụng all-MiniLM-L6-v2
2. **Retrieval**: Tìm kiếm cosine similarity để tìm các sections chính sách liên quan nhất
3. **Context Building**: Tập hợp top-K sections liên quan thành context mạch lạc
4. **Generation**: Google Gemini xử lý context và query của user để tạo response

### Giao diện người dùng
Ứng dụng web dựa trên Streamlit cung cấp:
- Giao diện chat real-time với lịch sử hội thoại
- Các tham số retrieval có thể cấu hình (top-k sections)
- Câu hỏi mẫu để test nhanh
- Hiển thị các sections liên quan với similarity scores
- Tính năng export chat và thống kê hệ thống

## Cài đặt và sử dụng

### Yêu cầu hệ thống
- Python 3.8+
- Google Gemini API key
- Kết nối Internet để download models lần đầu

### Thiết lập
```bash
git clone https://github.com/duchiep2512/Chat_box_LLM.git
cd Chat_box_LLM
pip install -r requirements.txt
```

### Cấu hình
Chỉnh sửa `config.py` với Google Gemini API key của bạn:
```python
GOOGLE_API_KEY = "your-api-key-here"
GENERATION_MODEL_NAME = "gemini-1.5-flash-8b-latest"
```

### Chạy ứng dụng
```bash
streamlit run app.py
```

Truy cập chatbot tại http://localhost:8501

## Tính năng chính

- **Tìm kiếm ngữ nghĩa**: Tìm các sections chính sách liên quan sử dụng NLP embeddings tiên tiến
- **Phản hồi có ngữ cảnh**: Tạo câu trả lời dựa trên nội dung chính sách bảo mật cụ thể
- **Giao diện tương tác**: UI chat hiện đại với phản hồi real-time
- **Retrieval có thể cấu hình**: Tham số độ liên quan có thể điều chỉnh
- **Minh bạch nội dung**: Hiển thị các sections nguồn được sử dụng cho mỗi phản hồi
- **Khả năng Export**: Lưu lịch sử hội thoại để phân tích

## Stack công nghệ

- **Frontend**: Streamlit với CSS tùy chỉnh
- **Backend**: Python với sentence-transformers và google-generativeai
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Generation Model**: Google Gemini 1.5 Flash 8B Latest
- **Xử lý dữ liệu**: Lưu trữ cấu trúc dựa trên JSON
- **Similarity Metric**: Cosine similarity cho document retrieval

## Hiệu suất

- Thời gian phản hồi trung bình: 1-2 giây
- Tạo embedding: ~100ms mỗi query
- Retrieval context: ~50ms cho top-5 sections
- Tạo answer: 0.5-1 giây qua Gemini API