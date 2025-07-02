#!/usr/bin/env python3
"""
Property Labeling Bot Setup Script
This script helps users set up the project for the first time.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_step(step_num, description):
    """Print a formatted step message"""
    print(f"\n{'='*50}")
    print(f"STEP {step_num}: {description}")
    print(f"{'='*50}")

def check_prerequisites():
    """Check if required software is installed"""
    print_step(1, "Checking Prerequisites")
    
    # Check Python
    try:
        python_version = sys.version_info
        if python_version.major >= 3 and python_version.minor >= 8:
            print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} - OK")
        else:
            print(f"❌ Python {python_version.major}.{python_version.minor}.{python_version.micro} - Need Python 3.8+")
            return False
    except:
        print("❌ Python not found - Please install Python 3.8+")
        return False
    
    # Check Node.js
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js {result.stdout.strip()} - OK")
        else:
            print("❌ Node.js not found - Please install Node.js 14+")
            return False
    except:
        print("❌ Node.js not found - Please install Node.js 14+")
        return False
    
    # Check npm
    try:
        result = subprocess.run(['npm', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ npm {result.stdout.strip()} - OK")
        else:
            print("❌ npm not found - Please install npm")
            return False
    except:
        print("❌ npm not found - Please install npm")
        return False
    
    return True

def setup_backend():
    """Set up the backend"""
    print_step(2, "Setting Up Backend")
    
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("❌ Backend directory not found")
        return False
    
    # Check for model files
    model_files = [
        "best_room_classifier.pth",
        "yolov8x.pt", 
        "class_mapping.json"
    ]
    
    missing_files = []
    for file in model_files:
        if not (backend_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("⚠️  Missing model files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n📥 Please download these files from:")
        print("   https://huggingface.co/Mahathig/property-labeling-models")
        print("   and place them in the backend/ folder")
        return False
    
    print("✅ All model files found")
    
    # Create .env file if it doesn't exist
    env_file = backend_dir / ".env"
    env_example = backend_dir / ".env.example"
    
    if not env_file.exists():
        if env_example.exists():
            shutil.copy(env_example, env_file)
            print("✅ Created .env file from template")
        else:
            # Create basic .env file
            with open(env_file, 'w') as f:
                f.write("# Google Gemini API Key\n")
                f.write("# Get your API key from: https://makersuite.google.com/app/apikey\n")
                f.write("GEMINI_API_KEY=your_gemini_api_key_here\n\n")
                f.write("# Server Configuration\n")
                f.write("HOST=127.0.0.1\n")
                f.write("PORT=8000\n\n")
                f.write("# CORS Settings\n")
                f.write("ALLOWED_ORIGINS=http://localhost:3001,http://127.0.0.1:3001\n")
            print("✅ Created .env file")
    
    # Install Python dependencies
    print("📦 Installing Python dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      cwd=backend_dir, check=True)
        print("✅ Python dependencies installed")
    except subprocess.CalledProcessError:
        print("❌ Failed to install Python dependencies")
        return False
    
    return True

def setup_frontend():
    """Set up the frontend"""
    print_step(3, "Setting Up Frontend")
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return False
    
    # Install Node.js dependencies
    print("📦 Installing Node.js dependencies...")
    try:
        subprocess.run(['npm', 'install'], cwd=frontend_dir, check=True)
        print("✅ Node.js dependencies installed")
    except subprocess.CalledProcessError:
        print("❌ Failed to install Node.js dependencies")
        return False
    
    return True

def print_next_steps():
    """Print instructions for next steps"""
    print_step(4, "Next Steps")
    
    print("🎉 Setup completed successfully!")
    print("\n📋 To run the application:")
    print("\n1. Configure your API key:")
    print("   - Edit backend/.env file")
    print("   - Replace 'your_gemini_api_key_here' with your actual Gemini API key")
    print("   - Get your API key from: https://makersuite.google.com/app/apikey")
    
    print("\n2. Start the backend server:")
    print("   cd backend")
    print("   uvicorn main:app --reload --port 8000")
    
    print("\n3. Start the frontend server (in a new terminal):")
    print("   cd frontend")
    print("   npm start")
    
    print("\n4. Open your browser and go to: http://localhost:3001")
    
    print("\n📚 For detailed instructions, see README.md")

def main():
    """Main setup function"""
    print("🏠 Property Labeling Bot Setup")
    print("This script will help you set up the project for the first time.")
    
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please install the required software.")
        return
    
    if not setup_backend():
        print("\n❌ Backend setup failed.")
        return
    
    if not setup_frontend():
        print("\n❌ Frontend setup failed.")
        return
    
    print_next_steps()

if __name__ == "__main__":
    main() 