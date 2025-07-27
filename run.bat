@echo off
echo 🚀 Starting LLM Fine-tuning Pipeline...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "myenv\Scripts\activate.bat" (
    echo 📦 Creating virtual environment...
    python -m venv myenv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call myenv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt
pip install -e .

REM Run the pipeline
echo 🚀 Running pipeline...
python run_pipeline.py

echo.
echo ✅ Pipeline completed!
pause 