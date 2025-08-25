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
├── main.ipynb               # Crawl dữ liệu, làm sạch, phân tích và test
├── requirements.txt         # Các package Python cần thiết
└── README.md               # Tài liệu dự án
```

## Mô tả chi tiết từng file

### File xử lý dữ liệu

**`main.ipynb`**
- Jupyter notebook chính để crawl dữ liệu từ website
- Web scraping từ https://www.presight.io/privacy-policy.html
- Xử lý và làm sạch HTML content thành text có cấu trúc
- Tạo embeddings và phân tích dữ liệu
- Export kết quả xử lý vào indexed_list.json
- Test và đánh giá chất lượng RAG system

### Các file ứng dụng chính

**`app.py`**
- Điểm khởi đầu chính của ứng dụng Streamlit
- Điều phối toàn bộ workflow của chatbot
- Khởi tạo hệ thống RAG và các component UI
- Xử lý input từ người dùng và hiển thị kết quả

**`rag_system.py`** 
- Triển khai pipeline RAG hoàn chỉnh
- Text embedding sử dụng Sentence Transformers (all-MiniLM-L6-v2)
- Tìm kiếm similarity để lấy các phần nội dung liên quan
- Tích hợp với Google Gemini API để sinh ngôn ngữ tự nhiên

**`ui_components.py`**
- Các component UI và styling functions của Streamlit
- CSS tùy chỉnh cho giao diện chat chuyên nghiệp
- Quản lý hiển thị tin nhắn và sidebar controls
- Tính năng export/import lịch sử chat

**`config.py`**
- File cấu hình trung tâm cho toàn bộ hệ thống
- API keys và cấu hình models
- Các tham số RAG và UI settings

**`indexed_list.json`**
- Dữ liệu privacy policy đã xử lý từ main.ipynb
- Cấu trúc sections, headings và nội dung
- Format tối ưu cho hệ thống RAG

## Cài đặt và sử dụng

### Yêu cầu hệ thống
- Python 3.8+
- Google Gemini API key
- Jupyter Notebook

### Thiết lập
```bash
git clone https://github.com/duchiep2512/Chat_box_LLM.git
cd Chat_box_LLM
pip install -r requirements.txt
```

### Cấu hình
Chỉnh sửa `config.py`:
```python
GOOGLE_API_KEY = "your-api-key-here"
GENERATION_MODEL_NAME = "gemini-1.5-flash-8b-latest"
```

### Chạy hệ thống
1. **Xử lý dữ liệu**: Chạy `main.ipynb` để crawl và tạo indexed_list.json
2. **Chạy chatbot**: `streamlit run app.py`
3. **Truy cập**: http://localhost:8501

## Tính năng chính

- **Tìm kiếm ngữ nghĩa**: NLP embeddings để tìm sections liên quan
- **Phản hồi có ngữ cảnh**: Câu trả lời dựa trên nội dung thực tế
- **Giao diện tương tác**: Chat UI hiện đại với real-time responses
- **Minh bạch nguồn**: Hiển thị sections nguồn với similarity scores
- **Export capability**: Lưu lịch sử hội thoại

## Stack công nghệ

- **Frontend**: Streamlit với CSS tùy chỉnh
- **Backend**: Python với sentence-transformers và google-generativeai
- **Models**: all-MiniLM-L6-v2 embeddings + Gemini 1.5 Flash 8B Latest
- **Storage**: JSON-based structured data

## Hiệu suất

- Thời gian phản hồi: 1-2 giây
- Embedding generation: ~100ms per query
- Context retrieval: ~50ms cho top-5 sections
- Answer generation: 0.5-1 giây via Gemini API

