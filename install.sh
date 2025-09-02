#!/bin/bash

# Consiglio Agent Installation Script
# This script installs the Consiglio agent and its dependencies

set -e

echo "🤖 Consiglio Agent Installation"
echo "================================"

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

echo "\nTo run the web server:"
echo "  python -m src.web"

# Install in development mode
echo "🔨 Installing in development mode..."
pip install -e .

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs data work work/reports work/summaries data/vectorstore data/memory

# Copy environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys and preferences"
fi

# Make main script executable
echo "🔐 Making main script executable..."
chmod +x src/main.py

echo ""
echo "🎉 Installation complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Activate virtual environment: source venv/bin/activate"
echo "3. Run demo: python src/main.py demo"
echo "4. Check status: python src/main.py status"
echo ""
echo "For help: python src/main.py --help"
echo ""
echo "Happy coding with Consiglio! 🤖✨"