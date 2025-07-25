# Interview Task: Fine-Tune & Serve Small Language Model

## Overview
This end-to-end pipeline collects domain-specific data for **electric vehicle charging stations**, fine-tunes a small language model (Llama-3-7B) using QLoRA, evaluates it against a baseline with comprehensive metrics, and deploys it for production use. The pipeline is designed for question-answering tasks and includes automated data processing, model versioning, performance monitoring, and CI/CD integration.

## 🎯 Target Domain
- **Domain**: Electric Vehicle Charging Stations
- **Use Case**: Question-Answering
- **Data Sources**: Web scraping, PDF extraction
- **Base Model**: Llama-3-7B (≤7B parameters)

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Processing    │    │   Training      │
│                 │    │                 │    │                 │
│ • Web Scraping  │───▶│ • Cleaning      │───▶│ • QLoRA         │
│ • PDF Extraction│    │ • Augmentation  │    │ • Experiment    │
│ • Metadata      │    │ • Tokenization  │    │   Tracking      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Deployment    │    │   Evaluation    │    │   Monitoring    │
│                 │    │                 │    │                 │
│ • FastAPI       │◀───│ • ROUGE/BLEU    │◀───│ • Performance   │
│ • Authentication│    │ • Benchmark     │    │ • Latency       │
│ • Versioning    │    │ • Comparison    │    │ • Throughput    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- 6GB+ VRAM (for QLoRA training)
- Git

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd llm_fine_tuning

# Create virtual environment
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Create necessary directories
mkdir -p data/{raw,processed,training,evaluation} logs models
```

### Configuration
1. Update environment variables in `src/llm_fine_tuning/config/settings.py`:
   ```python
   # Model Configuration
   os.environ["MODEL_NAME"] = "meta-llama/Llama-3-7B"
   
   # API Configuration
   os.environ["API_TOKEN"] = "your-secret-token"
   
   # Training Configuration
   os.environ["BATCH_SIZE"] = "1"
   os.environ["NUM_EPOCHS"] = "3"
   ```

2. Add your data sources:
   - Update URLs in `src/llm_fine_tuning/pipeline/main.py`
   - Place PDF files in the project directory

### Run the Pipeline
```bash
# Execute the complete pipeline
python scripts/run_pipeline.py

# Or run individual components
python scripts/debug_pipeline.py  # Test individual components
```

### Access the API
```bash
# Start the API server
uvicorn src.llm_fine_tuning.deployment.api_server:app --host 0.0.0.0 --port 8000

# Test the API
curl -X POST "http://localhost:8000/predict" \
     -H "Authorization: Bearer your-secret-token" \
     -H "Content-Type: application/json" \
     -d '{"input_text": "What is the charging speed of Level 2 chargers?"}'
```

## 📋 Requirements Compliance

### ✅ **Config**
- **Target Domain**: Electric vehicle charging stations (configurable)
- **Environment Variables**: Comprehensive configuration system
- **Prompt Management**: Domain-specific prompts for data augmentation

### ✅ **Data Collection**
- **Web Scraping**: Automated collection from EV charging websites
- **PDF Extraction**: Layout-aware extraction with metadata
- **Source Attribution**: Full metadata tracking and attribution

### ✅ **Data Processing**
- **Cleaning & Deduplication**: Automated quality filtering
- **Normalization**: Standardized data formats
- **Tokenization**: Model-specific tokenization
- **Storage**: Structured data storage system

### ✅ **Training Dataset**
- **LLM API Integration**: Automated dataset augmentation
- **Q&A Generation**: Domain-specific question-answer pairs
- **Formatting**: Training-ready dataset preparation

### ✅ **Fine-tuning**
- **≤7B Parameters**: Llama-3-7B base model
- **QLoRA**: Memory-efficient training
- **Experiment Tracking**: MLflow integration

### ✅ **Evaluation & Benchmarking**
- **Domain-Specific Benchmark**: 100+ EV charging questions
- **Automated Metrics**: ROUGE, BLEU, performance comparison
- **Baseline Comparison**: Performance vs. base model
- **Latency & Throughput**: Comprehensive performance monitoring

### ✅ **Deployment & Serving**
- **Model Registration**: MLflow model versioning
- **Lightweight Inference**: Optimized deployment
- **API Endpoint**: FastAPI with authentication
- **Monitoring**: Real-time endpoint monitoring

### ✅ **Orchestration**
- **Workflow Automation**: ZenML pipeline orchestration
- **Manual Triggers**: Script-based execution
- **Scheduled Triggers**: CI/CD automation

### ✅ **MLOps - CI/CD**
- **GitHub Actions**: Complete CI/CD pipeline
- **Automated Testing**: Unit, integration, security tests
- **Code Quality**: Linting, formatting, coverage
- **Deployment**: Automated staging deployment

## 🔧 Configuration

### Environment Variables
```bash
# Model Configuration
MODEL_NAME=meta-llama/Llama-3-7B
BASE_MODEL_PATH=./models/base
FINE_TUNED_MODEL_PATH=./models/fine_tuned

# Data Configuration
DATA_PATH=data/
RAW_DATA_PATH=data/raw/
PROCESSED_DATA_PATH=data/processed/

# API Configuration
API_TOKEN=your-secret-token
API_HOST=0.0.0.0
API_PORT=8000

# Training Configuration
BATCH_SIZE=1
LEARNING_RATE=2e-4
NUM_EPOCHS=3
MAX_LENGTH=512
```

### Domain Configuration
```python
TARGET_DOMAIN = "electric vehicle charging stations"
USE_CASE = "question-answering"
DATA_SOURCES = ["web_scraping", "pdf_extraction"]

# Benchmark categories
BENCHMARK_CONFIG = {
    "num_questions": 100,
    "categories": ["charging_speed", "connector_types", "installation", "pricing", "availability"],
    "difficulty_levels": ["easy", "medium", "hard"]
}
```

## 📈 Evaluation Results

The pipeline automatically generates comprehensive evaluation reports:

### Metrics Comparison
- **ROUGE Scores**: Text similarity and overlap
- **BLEU Scores**: Translation quality assessment
- **Performance Metrics**: Latency and throughput
- **Resource Usage**: Memory and CPU utilization

### Benchmark Results
- **Domain-Specific Questions**: 100+ EV charging questions
- **Category Performance**: Performance by question type
- **Difficulty Analysis**: Performance by difficulty level

## 🚀 CI/CD Pipeline

### GitHub Actions Workflow
- **Automated Testing**: Unit tests, integration tests
- **Code Quality**: Linting, formatting, security scanning
- **Data Validation**: Pipeline component validation
- **Model Training**: Automated training on main branch
- **Deployment**: Staging deployment with testing

### Workflow Triggers
- **Push to main/develop**: Full pipeline execution
- **Pull Requests**: Testing and validation
- **Scheduled**: Weekly automated runs

## 📝 Prompting Approach

### Code Generation Strategy
1. **Modular Design**: Request specific components with clear interfaces
2. **Iterative Refinement**: Build components step-by-step
3. **Error Handling**: Include robust error handling in prompts
4. **Documentation**: Request inline documentation and examples

### Example Prompts
```
"Create a modular data processing pipeline with:
- Data cleaning and deduplication
- Quality filtering with configurable thresholds
- Structured logging for monitoring
- Error handling for edge cases
- Unit tests for validation"
```

## 🔍 Monitoring and Logging

### Logging Structure
```
logs/
├── pipeline.log              # Main pipeline logs
├── performance_metrics.json  # Performance data
├── evaluation_results.json   # Evaluation results
└── benchmark_dataset.json    # Generated benchmark
```

### Monitoring Metrics
- **Pipeline Execution**: Step-by-step progress tracking
- **Performance**: Latency, throughput, resource usage
- **Quality**: Model performance improvements
- **Errors**: Error tracking and alerting

## 🛠️ Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch size or use gradient accumulation
2. **Model Loading**: Ensure sufficient disk space for model downloads
3. **API Errors**: Check authentication token and network connectivity
4. **Training Issues**: Verify data format and model compatibility

### Performance Optimization
- **Memory**: Use QLoRA for efficient training
- **Speed**: Optimize data loading and preprocessing
- **Quality**: Tune hyperparameters for better results

## 📚 Dependencies

### Core Dependencies
```
crawl4ai          # Web scraping
docling           # PDF extraction
transformers      # Hugging Face models
peft              # Parameter-efficient fine-tuning
zenml             # Pipeline orchestration
fastapi           # API framework
mlflow            # Experiment tracking
torch             # Deep learning framework
```

### Development Dependencies
```
pytest            # Testing framework
black             # Code formatting
flake8            # Linting
pre-commit        # Git hooks
safety            # Security scanning
bandit            # Security analysis
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For questions or issues, please contact: contact@energyai.berlin