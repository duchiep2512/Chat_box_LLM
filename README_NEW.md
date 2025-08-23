# Enhanced RAG Chatbot System

A complete Retrieval-Augmented Generation (RAG) chatbot system built with Python, featuring FAISS vector search, Google Gemini integration, and a modern Streamlit web interface.

## 🚀 Features

### Core RAG Pipeline
- **FAISS Vector Database** - Efficient similarity search with multiple index types
- **Advanced Document Processing** - Semantic chunking with metadata preservation
- **Multiple Retrieval Strategies** - Semantic search with ranking and filtering
- **Google Gemini Integration** - Powered by Gemini-1.5-pro for answer generation

### Enhanced Chatbot Interface
- **Conversation History Management** - Maintains context across interactions
- **Source Citations** - Shows references and similarity scores for each answer
- **Bilingual Support** - Handles both English and Vietnamese queries
- **Performance Monitoring** - Tracks response times and confidence scores

### Modern Web Interface
- **Streamlit Web App** - Interactive web interface with real-time chat
- **System Monitoring** - Live status updates and configuration controls
- **Export Capabilities** - Save conversations and analysis data
- **Responsive Design** - Clean, user-friendly interface

### Modular Architecture
- **Separated Concerns** - Individual modules for processing, RAG, and chat
- **Configuration Management** - Centralized settings with easy customization
- **Caching System** - Optimized embeddings and vector storage
- **Error Handling** - Comprehensive logging and error recovery

## 📁 Project Structure

```
Chat_box_LLM/
├── config.py              # Configuration management
├── data_processor.py      # Document processing and chunking
├── rag_pipeline.py        # RAG system with FAISS integration
├── chatbot.py            # Enhanced chatbot with conversation history
├── streamlit_app.py      # Web interface using Streamlit
├── utils.py              # Utility functions and helpers
├── main.py               # CLI interface and demonstrations
├── requirements.txt      # Python dependencies
├── indexed_list.json     # Processed document data
└── cache/                # Cached embeddings and FAISS index
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/duchiep2512/Chat_box_LLM.git
   cd Chat_box_LLM
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API key:**
   - Get a Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Set the environment variable:
     ```bash
     export GEMINI_API_KEY="your_api_key_here"
     ```
   - Or modify the API key in `config.py`

## 🚀 Quick Start

### 1. Run System Diagnostics
```bash
python main.py --mode diagnostics
```

### 2. Test the System
```bash
python main.py --mode demo
```

### 3. Start Interactive Chat
```bash
python main.py --mode chat
```

### 4. Launch Web Interface
```bash
streamlit run streamlit_app.py
```
or
```bash
python main.py --mode streamlit
```

## 💬 Usage Examples

### Command Line Interface
```bash
# Run full system demonstration
python main.py --mode demo

# Start interactive chat session
python main.py --mode chat

# Test data processing only
python main.py --mode data

# Test RAG pipeline only
python main.py --mode rag
```

### Python API Usage
```python
from chatbot import EnhancedChatbot

# Initialize chatbot
chatbot = EnhancedChatbot()

# Initialize system with data
result = chatbot.initialize_system('indexed_list.json')

# Start chat session
session_id = chatbot.start_chat_session()

# Chat with the bot
response = chatbot.chat("What is the privacy policy about?")
print(response['response'])
```

### Web Interface
1. Run: `streamlit run streamlit_app.py`
2. Open browser to `http://localhost:8501`
3. Initialize system using sidebar
4. Start chatting!

## ⚙️ Configuration

The system can be configured through `config.py`:

```python
# Embedding configuration
config.embedding.model_name = 'all-MiniLM-L6-v2'
config.embedding.cache_embeddings = True

# FAISS configuration
config.faiss.index_type = 'IndexFlatIP'
config.faiss.save_index = True

# Retrieval configuration
config.retrieval.top_k = 5
config.retrieval.similarity_threshold = 0.3
config.retrieval.max_chunk_size = 512

# Chat configuration
config.chat.max_history_length = 10
config.chat.show_sources = True
```

## 🔧 Advanced Features

### Custom Data Processing
```python
from data_processor import DataProcessor

processor = DataProcessor(max_chunk_size=1024, chunk_overlap=100)
chunks = processor.create_document_chunks(your_data)
```

### FAISS Vector Store
```python
from rag_pipeline import FAISSVectorStore, EmbeddingManager

embedding_manager = EmbeddingManager()
vector_store = FAISSVectorStore(embedding_manager.embedding_dim)
```

### Performance Monitoring
```python
from utils import PerformanceMonitor

with PerformanceMonitor("Query Processing"):
    result = rag_pipeline.query("your question")
```

## 📊 System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Input    │────│  Data Processor  │────│  Document       │
│ (indexed_list)  │    │  - Chunking      │    │  Chunks         │
└─────────────────┘    │  - Cleaning      │    └─────────────────┘
                       │  - Metadata      │            │
                       └──────────────────┘            │
                                                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │────│   RAG Pipeline   │────│  FAISS Vector   │
│                 │    │  - Embedding     │    │  Database       │
└─────────────────┘    │  - Retrieval     │    └─────────────────┘
        │               │  - Ranking       │            │
        │               └──────────────────┘            │
        │                         │                     │
        │                         ▼                     │
        │               ┌──────────────────┐            │
        │               │  Answer Gen.     │◄───────────┘
        │               │  (Gemini API)    │
        │               └──────────────────┘
        │                         │
        ▼                         ▼
┌─────────────────┐    ┌──────────────────┐
│  Chat Interface │────│   Response +     │
│  - History      │    │   Sources        │
│  - Web UI       │    └──────────────────┘
└─────────────────┘
```

## 🧪 Testing

Run the test suite to verify system functionality:

```bash
# Test individual components
python main.py --mode data      # Data processing
python main.py --mode rag       # RAG pipeline
python main.py --mode chat      # Interactive chat

# Full system test
python main.py --mode demo
```

## 📈 Performance

### Benchmarks
- **Document Processing**: ~14 chunks from 11 documents in <1 second
- **Embedding Generation**: ~384-dimensional vectors with caching
- **FAISS Search**: Sub-millisecond similarity search
- **End-to-End Response**: Typically 1-3 seconds including LLM generation

### Optimization Features
- **Embedding Caching**: Persistent cache for generated embeddings
- **FAISS Index Persistence**: Save and reload vector indices
- **Chunking Strategy**: Semantic chunking preserves context
- **Memory Management**: Efficient handling of large document sets

## 🔍 Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```
   Error: Invalid API key
   Solution: Set GEMINI_API_KEY environment variable
   ```

2. **Missing Dependencies**
   ```
   Error: ModuleNotFoundError
   Solution: pip install -r requirements.txt
   ```

3. **Data File Not Found**
   ```
   Error: indexed_list.json not found
   Solution: Ensure data file exists in project directory
   ```

4. **FAISS Installation Issues**
   ```
   For CPU-only: pip install faiss-cpu
   For GPU support: pip install faiss-gpu
   ```

### Debug Mode
Enable verbose logging:
```bash
python main.py --mode demo --verbose
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source. Please check the repository for license details.

## 🙏 Acknowledgments

- **Google Gemini API** - For powerful language model capabilities
- **FAISS** - For efficient vector similarity search
- **Sentence Transformers** - For high-quality text embeddings
- **Streamlit** - For the modern web interface
- **Hugging Face** - For the transformer models and utilities

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Run system diagnostics: `python main.py --mode diagnostics`
3. Check the logs in the `logs/` directory
4. Open an issue on GitHub

---

**Built with ❤️ for enhanced conversational AI experiences**