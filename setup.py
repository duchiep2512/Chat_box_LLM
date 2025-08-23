#!/usr/bin/env python3
"""
Setup script for Enhanced RAG Chatbot System

This script helps users set up the system and verify everything is working correctly.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_step(step_num, title):
    """Print a step indicator"""
    print(f"\n🔹 Step {step_num}: {title}")

def run_command(command, description="", check=True):
    """Run a shell command with error handling"""
    print(f"   Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if e.stderr:
            print(f"   Error details: {e.stderr}")
        return False

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Please use Python 3.8+")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is supported")
    return True

def install_requirements():
    """Install Python requirements"""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    print("📦 Installing Python packages...")
    success = run_command(f"{sys.executable} -m pip install -r requirements.txt")
    if success:
        print("✅ All packages installed successfully")
    else:
        print("❌ Failed to install packages")
    return success

def create_data_file():
    """Create the indexed_list.json data file if it doesn't exist"""
    data_file = Path("indexed_list.json")
    if data_file.exists():
        print("✅ Data file already exists")
        return True
    
    print("📝 Creating sample data file...")
    
    # Sample data (privacy policy content)
    sample_data = [
        {
            "heading": "PRIVACY POLICY",
            "content": "",
            "subheaders": []
        },
        {
            "heading": "Last updated 15 Sep 2023",
            "content": "At Presight, we are committed to protecting the privacy of our customers and visitors to our website. This Privacy Policy explains how we collect, use, and disclose information about our customers and visitors.",
            "subheaders": []
        },
        {
            "heading": "Information Collection and Use",
            "content": "For a better experience, while using our Service, we may require you to provide us with certain personally identifiable information, including but not limited to, your name, phone number, and postal address. The information that we request will be retained by us and used as described in this privacy policy.",
            "subheaders": []
        },
        {
            "heading": "Types of Data Collected",
            "content": "",
            "subheaders": [
                {
                    "Title": "Personal Data",
                    "Content": "While using our Service, we may ask you to provide us with certain personally identifiable information that can be used to contact or identify you. Personally identifiable information may include, but is not limited to: Email address, First name and last name, Phone number, Address.",
                    "List": []
                },
                {
                    "Title": "Usage Data",
                    "Content": "We may also collect information that your browser sends whenever you visit our Service. This Usage Data may include information such as your computer's Internet Protocol address, browser type, browser version, the pages of our Service that you visit.",
                    "List": []
                }
            ]
        },
        {
            "heading": "Contact Us",
            "content": "If you have any questions about this Privacy Policy, please contact us through the customer portal or by email at presight@presight.io.",
            "subheaders": []
        }
    ]
    
    try:
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        print("✅ Sample data file created")
        return True
    except Exception as e:
        print(f"❌ Failed to create data file: {e}")
        return False

def setup_directories():
    """Create necessary directories"""
    directories = ["cache", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✅ Directories created")
    return True

def check_api_key():
    """Check if API key is configured"""
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key and "Your_API" not in api_key:
        print("✅ Gemini API key is configured")
        return True
    else:
        print("⚠️  Gemini API key not configured")
        print("   You can:")
        print("   1. Set environment variable: export GEMINI_API_KEY='your_key'")
        print("   2. Or modify config.py to set the API key directly")
        print("   3. Get a key from: https://makersuite.google.com/app/apikey")
        return False

def test_system():
    """Test the system functionality"""
    print("🧪 Testing system functionality...")
    
    # Test basic imports
    try:
        import numpy
        import sentence_transformers
        import faiss
        import google.generativeai
        print("✅ Core packages imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test main script
    success = run_command(f"{sys.executable} main.py --mode diagnostics", check=False)
    if success:
        print("✅ System diagnostics passed")
    else:
        print("⚠️  System diagnostics had issues (this might be due to API key)")
    
    return True

def main():
    """Main setup function"""
    print_header("🚀 Enhanced RAG Chatbot Setup")
    
    print("This setup script will help you configure the Enhanced RAG Chatbot System.")
    print("Please ensure you have Python 3.8+ installed and an internet connection.")
    
    # Step 1: Check Python version
    print_step(1, "Checking Python version")
    if not check_python_version():
        print("❌ Setup failed: Python version not supported")
        sys.exit(1)
    
    # Step 2: Install requirements
    print_step(2, "Installing Python packages")
    if not install_requirements():
        print("❌ Setup failed: Could not install required packages")
        sys.exit(1)
    
    # Step 3: Create directories
    print_step(3, "Creating directories")
    setup_directories()
    
    # Step 4: Create data file
    print_step(4, "Setting up data file")
    create_data_file()
    
    # Step 5: Check API key
    print_step(5, "Checking API configuration")
    api_key_configured = check_api_key()
    
    # Step 6: Test system
    print_step(6, "Testing system")
    test_system()
    
    # Final summary
    print_header("🎉 Setup Complete!")
    
    if api_key_configured:
        status = "✅ READY TO USE"
        next_steps = [
            "python main.py --mode demo           # Run full demonstration",
            "python main.py --mode chat           # Start interactive chat",
            "streamlit run streamlit_app.py       # Launch web interface"
        ]
    else:
        status = "⚠️  NEEDS API KEY"
        next_steps = [
            "1. Set your Gemini API key (see instructions above)",
            "2. python main.py --mode demo        # Run demonstration",
            "3. python main.py --mode chat        # Start chat",
            "4. streamlit run streamlit_app.py    # Web interface"
        ]
    
    print(f"\nSystem Status: {status}")
    print("\nNext Steps:")
    for step in next_steps:
        print(f"   • {step}")
    
    print("\n📚 For more information, see README_NEW.md")
    print("🐛 For issues, run: python main.py --mode diagnostics")

if __name__ == "__main__":
    main()