#!/bin/bash

echo "🚀 Starting LLM Fine-tuning Pipeline..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "myenv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv myenv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source myenv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt
pip install -e .

# Run the pipeline
echo "🚀 Running pipeline..."
python run_pipeline.py

echo
echo "✅ Pipeline completed!" 