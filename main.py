"""
Main script to demonstrate the Enhanced RAG Chatbot System

This script provides a command-line interface to test and demonstrate
the complete RAG system with various modes of operation.
"""

import argparse
import sys
import os
from typing import Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from data_processor import DataProcessor
from rag_pipeline import RAGPipeline
from chatbot import EnhancedChatbot, ConsoleInterface
from utils import setup_logging, validate_system_requirements, get_system_summary

def check_requirements() -> bool:
    """Check if all requirements are met"""
    print("🔍 Checking system requirements...")
    
    requirements = validate_system_requirements()
    all_met = True
    
    for req, available in requirements.items():
        status = "✅" if available else "❌"
        print(f"  {status} {req}: {'Available' if available else 'Missing'}")
        if not available:
            all_met = False
    
    if not all_met:
        print("\n❌ Some requirements are missing. Please install them using:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All requirements met!\n")
    return True

def demonstrate_data_processing(data_file: str = None) -> Dict[str, Any]:
    """Demonstrate data processing capabilities"""
    print("🔄 Demonstrating data processing...")
    
    data_file = data_file or config.data_file
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("   Please ensure 'indexed_list.json' exists in the current directory.")
        return {'status': 'error', 'message': 'Data file not found'}
    
    try:
        processor = DataProcessor(
            max_chunk_size=config.retrieval.max_chunk_size,
            chunk_overlap=config.retrieval.chunk_overlap
        )
        
        # Load data
        indexed_data = processor.load_indexed_data(data_file)
        print(f"📊 Loaded {len(indexed_data)} indexed items")
        
        # Create chunks
        chunks = processor.create_document_chunks(indexed_data)
        print(f"✂️ Created {len(chunks)} document chunks")
        
        # Get summary
        summary = processor.get_chunk_metadata_summary(chunks)
        
        print("📈 Processing Summary:")
        for key, value in summary.items():
            print(f"   • {key}: {value}")
        
        # Export analysis
        analysis_file = "chunk_analysis.json"
        processor.export_chunks_for_analysis(chunks, analysis_file)
        print(f"📝 Exported analysis to {analysis_file}")
        
        return {
            'status': 'success',
            'chunks': chunks,
            'summary': summary
        }
        
    except Exception as e:
        print(f"❌ Error in data processing: {e}")
        return {'status': 'error', 'message': str(e)}

def demonstrate_rag_pipeline(chunks = None, data_file: str = None) -> Dict[str, Any]:
    """Demonstrate RAG pipeline capabilities"""
    print("🚀 Demonstrating RAG pipeline...")
    
    try:
        # Initialize pipeline
        pipeline = RAGPipeline()
        
        # Check if already indexed
        if not pipeline.is_indexed:
            if chunks is None:
                # Need to process data first
                result = demonstrate_data_processing(data_file)
                if result['status'] != 'success':
                    return result
                chunks = result['chunks']
            
            # Prepare texts for embedding
            processor = DataProcessor()
            texts = processor.prepare_text_for_embedding(chunks)
            
            # Index documents
            print("🔍 Indexing documents...")
            pipeline.index_documents(chunks, texts)
        else:
            print("✅ Documents already indexed")
        
        # Test queries
        test_queries = [
            "What is the privacy policy about?",
            "How is personal data collected?",
            "What are the contact details?",
            "Chính sách bảo mật là gì?"  # Vietnamese query
        ]
        
        print("\n🧪 Testing sample queries:")
        for query in test_queries:
            print(f"\n❓ Query: {query}")
            result = pipeline.query(query, top_k=3)
            
            if result.get('confidence', 0) > 0.3:
                print(f"   ✅ Answer: {result['answer'][:100]}...")
                print(f"   📊 Confidence: {result['confidence']:.2f}")
                print(f"   📚 Sources: {len(result['sources'])}")
            else:
                print(f"   ⚠️ Low confidence answer: {result['answer'][:100]}...")
        
        system_info = pipeline.get_system_info()
        print(f"\n📊 System Info:")
        for key, value in system_info.items():
            print(f"   • {key}: {value}")
        
        return {
            'status': 'success',
            'pipeline': pipeline,
            'system_info': system_info
        }
        
    except Exception as e:
        print(f"❌ Error in RAG pipeline: {e}")
        return {'status': 'error', 'message': str(e)}

def demonstrate_chatbot(data_file: str = None):
    """Demonstrate chatbot capabilities"""
    print("🤖 Starting Enhanced Chatbot demonstration...")
    
    try:
        console = ConsoleInterface()
        console.run(data_file)
        
    except KeyboardInterrupt:
        print("\n👋 Chatbot demonstration ended.")
    except Exception as e:
        print(f"❌ Error in chatbot: {e}")

def run_system_diagnostics():
    """Run comprehensive system diagnostics"""
    print("🔧 Running system diagnostics...\n")
    
    # System summary
    summary = get_system_summary()
    
    print("💻 System Information:")
    print(f"   • OS: {summary['platform']['system']} {summary['platform']['release']}")
    print(f"   • Machine: {summary['platform']['machine']}")
    print(f"   • Python: {summary['python']['version'].split()[0]}")
    
    print("\n📦 Package Requirements:")
    requirements = summary['requirements']
    for req, available in requirements.items():
        status = "✅" if available else "❌"
        print(f"   {status} {req}")
    
    # Configuration check
    print("\n⚙️ Configuration:")
    print(f"   • Embedding model: {config.embedding.model_name}")
    print(f"   • Vector store: {config.faiss.index_type}")
    print(f"   • Max chunk size: {config.retrieval.max_chunk_size}")
    print(f"   • Top-k retrieval: {config.retrieval.top_k}")
    
    # API key check
    api_key_status = "✅ Set" if config.llm.api_key and "Your_API" not in config.llm.api_key else "❌ Not set"
    print(f"   • Gemini API key: {api_key_status}")
    
    # File checks
    print("\n📁 File System:")
    data_exists = "✅" if os.path.exists(config.data_file) else "❌"
    print(f"   • Data file ({config.data_file}): {data_exists}")
    
    cache_exists = "✅" if os.path.exists(config.cache_dir) else "❌"
    print(f"   • Cache directory: {cache_exists}")
    
    logs_exists = "✅" if os.path.exists(config.logs_dir) else "❌"
    print(f"   • Logs directory: {logs_exists}")

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Enhanced RAG Chatbot System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode diagnostics          # Run system diagnostics
  python main.py --mode data                 # Test data processing
  python main.py --mode rag                  # Test RAG pipeline
  python main.py --mode chat                 # Start interactive chat
  python main.py --mode demo                 # Full demonstration
  python main.py --mode streamlit            # Launch Streamlit web app
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['diagnostics', 'data', 'rag', 'chat', 'demo', 'streamlit'],
        default='demo',
        help='Mode of operation (default: demo)'
    )
    
    parser.add_argument(
        '--data-file',
        default=None,
        help='Path to data file (default: indexed_list.json)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        import logging
        setup_logging(level=logging.DEBUG)
    
    print("🚀 Enhanced RAG Chatbot System")
    print("=" * 50)
    
    if args.mode == 'diagnostics':
        run_system_diagnostics()
    
    elif args.mode == 'data':
        if not check_requirements():
            sys.exit(1)
        demonstrate_data_processing(args.data_file)
    
    elif args.mode == 'rag':
        if not check_requirements():
            sys.exit(1)
        demonstrate_rag_pipeline(data_file=args.data_file)
    
    elif args.mode == 'chat':
        if not check_requirements():
            sys.exit(1)
        demonstrate_chatbot(args.data_file)
    
    elif args.mode == 'demo':
        if not check_requirements():
            sys.exit(1)
        
        print("🎯 Running full system demonstration...\n")
        
        # Step 1: Data processing
        data_result = demonstrate_data_processing(args.data_file)
        if data_result['status'] != 'success':
            print("❌ Demo failed at data processing step")
            sys.exit(1)
        
        print("\n" + "="*50)
        
        # Step 2: RAG pipeline
        rag_result = demonstrate_rag_pipeline(data_result.get('chunks'), args.data_file)
        if rag_result['status'] != 'success':
            print("❌ Demo failed at RAG pipeline step")
            sys.exit(1)
        
        print("\n" + "="*50)
        print("🎉 Demonstration completed successfully!")
        print("\nYou can now:")
        print("  • Run 'python main.py --mode chat' for interactive chat")
        print("  • Run 'streamlit run streamlit_app.py' for web interface")
        print("  • Run 'python main.py --mode diagnostics' for system info")
    
    elif args.mode == 'streamlit':
        try:
            import streamlit
            import subprocess
            
            print("🌐 Launching Streamlit web interface...")
            print("   The web interface will open in your browser")
            print("   Press Ctrl+C to stop the server")
            print("=" * 50)
            
            # Launch streamlit app
            subprocess.run([
                sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py'
            ])
            
        except ImportError:
            print("❌ Streamlit not installed. Install it with:")
            print("   pip install streamlit")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n👋 Streamlit server stopped.")
        except Exception as e:
            print(f"❌ Error launching Streamlit: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()