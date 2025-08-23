"""
Enhanced Chatbot with Conversation History and Source Citations

This module provides an enhanced chatbot interface that maintains conversation
history and provides detailed source citations for answers.
"""

import time
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from config import config
from rag_pipeline import RAGPipeline
from data_processor import DataProcessor, DocumentChunk

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    """Represents a chat message with metadata"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass 
class ChatSession:
    """Represents a chat session with history"""
    session_id: str
    messages: List[ChatMessage]
    created_at: datetime
    last_activity: datetime
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None) -> None:
        """Add a message to the session"""
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.last_activity = datetime.now()

class ConversationManager:
    """Manages conversation history and context"""
    
    def __init__(self, max_history_length: int = None):
        self.max_history_length = max_history_length or config.chat.max_history_length
        self.sessions: Dict[str, ChatSession] = {}
        self.current_session_id: Optional[str] = None
    
    def start_new_session(self, session_id: str = None) -> str:
        """Start a new chat session"""
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        session = ChatSession(
            session_id=session_id,
            messages=[],
            created_at=datetime.now(),
            last_activity=datetime.now()
        )
        
        self.sessions[session_id] = session
        self.current_session_id = session_id
        
        logger.info(f"Started new chat session: {session_id}")
        return session_id
    
    def get_current_session(self) -> Optional[ChatSession]:
        """Get the current chat session"""
        if self.current_session_id and self.current_session_id in self.sessions:
            return self.sessions[self.current_session_id]
        return None
    
    def add_user_message(self, content: str, metadata: Dict[str, Any] = None) -> None:
        """Add a user message to current session"""
        session = self.get_current_session()
        if session:
            session.add_message('user', content, metadata)
            self._maintain_history_limit(session)
    
    def add_assistant_message(self, content: str, metadata: Dict[str, Any] = None) -> None:
        """Add an assistant message to current session"""
        session = self.get_current_session()
        if session:
            session.add_message('assistant', content, metadata)
            self._maintain_history_limit(session)
    
    def _maintain_history_limit(self, session: ChatSession) -> None:
        """Maintain maximum history length"""
        if len(session.messages) > self.max_history_length:
            # Remove oldest messages but keep pairs (user + assistant)
            messages_to_remove = len(session.messages) - self.max_history_length
            if messages_to_remove % 2 == 1:  # Keep even number
                messages_to_remove += 1
            session.messages = session.messages[messages_to_remove:]
    
    def get_conversation_context(self, include_last_n: int = 3) -> str:
        """Get recent conversation context for better responses"""
        session = self.get_current_session()
        if not session or not session.messages:
            return ""
        
        # Get last n pairs of messages
        recent_messages = session.messages[-(include_last_n * 2):]
        
        context_parts = []
        for message in recent_messages:
            role_label = "User" if message.role == 'user' else "Assistant"
            context_parts.append(f"{role_label}: {message.content}")
        
        return "\n".join(context_parts)
    
    def export_session(self, session_id: str, file_path: str) -> None:
        """Export session to JSON file"""
        if session_id not in self.sessions:
            logger.warning(f"Session {session_id} not found")
            return
        
        session = self.sessions[session_id]
        export_data = {
            'session_id': session.session_id,
            'created_at': session.created_at.isoformat(),
            'last_activity': session.last_activity.isoformat(),
            'messages': []
        }
        
        for message in session.messages:
            message_data = {
                'role': message.role,
                'content': message.content,
                'timestamp': message.timestamp.isoformat(),
                'metadata': message.metadata
            }
            export_data['messages'].append(message_data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Exported session {session_id} to {file_path}")

class EnhancedChatbot:
    """Enhanced chatbot with RAG capabilities and conversation management"""
    
    def __init__(self):
        self.rag_pipeline = RAGPipeline()
        self.conversation_manager = ConversationManager()
        self.response_timeout = config.chat.response_timeout
        self.show_sources = config.chat.show_sources
        self.show_similarity_scores = config.chat.show_similarity_scores
        
        logger.info("Enhanced Chatbot initialized")
    
    def initialize_system(self, data_file: str = None) -> Dict[str, Any]:
        """Initialize the chatbot system with document indexing"""
        data_file = data_file or config.data_file
        
        try:
            # Check if already indexed
            if self.rag_pipeline.is_indexed:
                logger.info("System already indexed")
                return {
                    'status': 'already_indexed',
                    'message': 'System is ready to use',
                    'system_info': self.rag_pipeline.get_system_info()
                }
            
            # Load and process data
            logger.info(f"Loading data from {data_file}")
            processor = DataProcessor(
                max_chunk_size=config.retrieval.max_chunk_size,
                chunk_overlap=config.retrieval.chunk_overlap
            )
            
            # Load indexed data
            indexed_data = processor.load_indexed_data(data_file)
            
            # Create document chunks
            chunks = processor.create_document_chunks(indexed_data)
            
            # Prepare texts for embedding
            texts = processor.prepare_text_for_embedding(chunks)
            
            # Index documents
            self.rag_pipeline.index_documents(chunks, texts)
            
            # Get summary
            summary = processor.get_chunk_metadata_summary(chunks)
            
            return {
                'status': 'success',
                'message': 'System initialized successfully',
                'summary': summary,
                'system_info': self.rag_pipeline.get_system_info()
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            return {
                'status': 'error',
                'message': f'Failed to initialize: {str(e)}',
                'error': str(e)
            }
    
    def start_chat_session(self) -> str:
        """Start a new chat session"""
        session_id = self.conversation_manager.start_new_session()
        
        # Add welcome message
        welcome_msg = (
            "🤖 Chào bạn! Tôi là trợ lý AI được nâng cấp với khả năng RAG.\n"
            "Tôi có thể trả lời câu hỏi của bạn dựa trên dữ liệu đã được lập chỉ mục.\n"
            "Hãy đặt câu hỏi và tôi sẽ cung cấp câu trả lời chi tiết kèm theo nguồn tham khảo!"
        )
        
        self.conversation_manager.add_assistant_message(
            welcome_msg, 
            {'type': 'welcome', 'features': ['rag', 'source_citation', 'conversation_history']}
        )
        
        return session_id
    
    def chat(self, user_input: str) -> Dict[str, Any]:
        """Process a chat message and return response with metadata"""
        if not user_input.strip():
            return {
                'response': 'Vui lòng nhập câu hỏi để tôi có thể trả lời bạn.',
                'error': 'empty_input'
            }
        
        # Start timing
        start_time = time.time()
        
        # Add user message to conversation
        self.conversation_manager.add_user_message(user_input)
        
        # Get response from RAG pipeline
        rag_result = self.rag_pipeline.query(
            user_input, 
            top_k=config.retrieval.top_k,
            show_sources=self.show_sources
        )
        
        # Calculate timing
        response_time = time.time() - start_time
        
        # Format response
        formatted_response = self._format_response(rag_result, response_time)
        
        # Add assistant response to conversation
        assistant_metadata = {
            'response_time': response_time,
            'confidence': rag_result.get('confidence', 0.0),
            'sources_count': len(rag_result.get('sources', [])),
            'rag_result': rag_result
        }
        
        self.conversation_manager.add_assistant_message(
            formatted_response['response'], 
            assistant_metadata
        )
        
        return formatted_response
    
    def _format_response(self, rag_result: Dict[str, Any], response_time: float) -> Dict[str, Any]:
        """Format the response with sources and timing information"""
        
        answer = rag_result.get('answer', 'Không thể tạo câu trả lời.')
        sources = rag_result.get('sources', [])
        confidence = rag_result.get('confidence', 0.0)
        
        # Build response
        response_parts = [answer]
        
        # Add sources if available and enabled
        if sources and self.show_sources:
            response_parts.append("\n📚 **Nguồn tham khảo:**")
            for source in sources:
                source_text = f"• [{source['index']}] {source['heading']}"
                if source.get('subheading'):
                    source_text += f" - {source['subheading']}"
                
                if self.show_similarity_scores:
                    source_text += f" (Độ liên quan: {source['similarity_score']:.3f})"
                
                response_parts.append(source_text)
        
        # Add timing and confidence info
        response_parts.append(f"\n⏱️ Thời gian xử lý: {response_time:.2f} giây")
        
        if confidence > 0:
            confidence_level = "Cao" if confidence > 0.7 else "Trung bình" if confidence > 0.5 else "Thấp"
            response_parts.append(f"🎯 Độ tin cậy: {confidence_level} ({confidence:.2f})")
        
        formatted_response = "\n".join(response_parts)
        
        return {
            'response': formatted_response,
            'raw_answer': answer,
            'sources': sources,
            'confidence': confidence,
            'response_time': response_time,
            'status': 'success'
        }
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history"""
        session = self.conversation_manager.get_current_session()
        if not session:
            return []
        
        recent_messages = session.messages[-limit:] if limit > 0 else session.messages
        
        history = []
        for message in recent_messages:
            history.append({
                'role': message.role,
                'content': message.content,
                'timestamp': message.timestamp.isoformat(),
                'metadata': message.metadata
            })
        
        return history
    
    def clear_conversation(self) -> str:
        """Clear current conversation and start new session"""
        new_session_id = self.start_chat_session()
        return f"Đã bắt đầu cuộc trò chuyện mới! Session ID: {new_session_id}"
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        rag_info = self.rag_pipeline.get_system_info()
        
        current_session = self.conversation_manager.get_current_session()
        session_info = {
            'current_session_id': self.conversation_manager.current_session_id,
            'total_sessions': len(self.conversation_manager.sessions),
            'current_session_messages': len(current_session.messages) if current_session else 0
        }
        
        return {
            'rag_system': rag_info,
            'conversation': session_info,
            'configuration': {
                'max_history_length': self.conversation_manager.max_history_length,
                'response_timeout': self.response_timeout,
                'show_sources': self.show_sources,
                'show_similarity_scores': self.show_similarity_scores
            }
        }
    
    def export_conversation(self, file_path: str = None) -> str:
        """Export current conversation to file"""
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"conversation_{timestamp}.json"
        
        session_id = self.conversation_manager.current_session_id
        if session_id:
            self.conversation_manager.export_session(session_id, file_path)
            return f"Conversation exported to {file_path}"
        else:
            return "No active conversation to export"

# Console interface for backwards compatibility
class ConsoleInterface:
    """Console interface for the enhanced chatbot"""
    
    def __init__(self):
        self.chatbot = EnhancedChatbot()
    
    def run(self, data_file: str = None):
        """Run the console interface"""
        print("🚀 Initializing Enhanced RAG Chatbot System...")
        
        # Initialize system
        init_result = self.chatbot.initialize_system(data_file)
        
        if init_result['status'] == 'error':
            print(f"❌ Initialization failed: {init_result['message']}")
            return
        
        print(f"✅ {init_result['message']}")
        
        if 'summary' in init_result:
            summary = init_result['summary']
            print(f"📊 Processed {summary['total_chunks']} chunks from {summary['unique_headings']} headings")
        
        # Start chat session
        session_id = self.chatbot.start_chat_session()
        print(f"💬 Started chat session: {session_id}")
        print("=" * 60)
        
        # Get welcome message
        history = self.chatbot.get_conversation_history(1)
        if history:
            print("\n🤖 Assistant:")
            print(history[0]['content'])
        
        print("\n" + "=" * 60)
        print("Commands: 'exit' to quit, 'clear' to start new conversation, 'status' for system info")
        print("=" * 60)
        
        # Chat loop
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() == 'exit':
                    print("\n👋 Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    result = self.chatbot.clear_conversation()
                    print(f"\n🔄 {result}")
                    continue
                elif user_input.lower() == 'status':
                    status = self.chatbot.get_system_status()
                    print(f"\n📊 System Status:")
                    print(f"  • RAG System: {'Ready' if status['rag_system']['is_indexed'] else 'Not Ready'}")
                    print(f"  • Total Chunks: {status['rag_system']['total_chunks']}")
                    print(f"  • Current Session: {status['conversation']['current_session_id']}")
                    print(f"  • Messages in Session: {status['conversation']['current_session_messages']}")
                    continue
                
                # Get response
                response = self.chatbot.chat(user_input)
                
                if response.get('status') == 'success':
                    print(f"\n🤖 Assistant:")
                    print(response['response'])
                else:
                    print(f"\n❌ Error: {response.get('response', 'Unknown error')}")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                logger.error(f"Console interface error: {e}")

if __name__ == "__main__":
    # Run console interface for testing
    console = ConsoleInterface()
    console.run()