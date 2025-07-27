# 🚀 LLM Fine-tuning Pipeline - Run Instructions

## 📋 **Quick Start**

### **Option 1: Windows (Easiest)**
```bash
# Double-click or run:
run.bat
```

### **Option 2: Linux/Mac**
```bash
# Make script executable and run:
chmod +x run.sh
./run.sh
```

### **Option 3: Manual Setup**
```bash
# 1. Create virtual environment
python -m venv myenv

# 2. Activate virtual environment
# Windows:
myenv\Scripts\activate
# Linux/Mac:
source myenv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .

# 4. Run pipeline
python run_pipeline.py
```

### **Option 4: Docker (Production)**
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run individual services
docker-compose up llm-pipeline
docker-compose up api-server
```

## 🎯 **What the Pipeline Does**

1. **📊 Data Collection**: Loads sample EV charging data
2. **🔄 Data Processing**: Cleans, augments, and splits data
3. **📈 Benchmark Generation**: Creates 90 domain-specific questions
4. **🤖 Model Training**: Fine-tunes the model (simulated for demo)
5. **📊 Evaluation**: Calculates ROUGE and BLEU scores
6. **⚡ Performance Monitoring**: Tracks system metrics
7. **🚀 API Server**: Starts FastAPI server for predictions

## 📁 **Generated Files**

After running, you'll find:

```
data/
├── raw/sample_ev_data.txt          # Input data
├── processed/cleaned_data.csv       # Cleaned data
├── training/train_data.csv          # Training data
├── evaluation/
│   ├── benchmark_dataset.json       # 90 benchmark questions
│   └── evaluation_results.json      # Model evaluation scores
└── models/training_info.json        # Training metadata

logs/
└── performance_metrics.json         # Performance data

results/                              # Training outputs
```

## 🌐 **API Access**

Once running, access the API at:
- **API Server**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### **Example API Call**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Authorization: Bearer your-secret-token" \
     -H "Content-Type: application/json" \
     -d '{"input_text": "What is Level 2 charging?"}'
```

## ⚙️ **Configuration**

Edit `src/llm_fine_tuning/config/settings.py` to customize:

- **Model**: Change `MODEL_NAME` (default: microsoft/DialoGPT-medium)
- **Domain**: Change `DOMAIN_TARGET` (default: electric vehicle charging stations)
- **API Token**: Change `API_TOKEN` for security
- **Training Parameters**: Adjust batch size, learning rate, etc.

## 🔧 **Troubleshooting**

### **Common Issues**

1. **Import Errors**
   ```bash
   # Make sure you're in the project root
   cd llm_fine_tuning
   
   # Install package in development mode
   pip install -e .
   ```

2. **Missing Dependencies**
   ```bash
   # Update requirements
   pip install -r requirements.txt --upgrade
   ```

3. **Port Already in Use**
   ```bash
   # Change port in run_pipeline.py or docker-compose.yml
   # Default: 8000
   ```

4. **Memory Issues**
   ```bash
   # Use smaller model
   export MODEL_NAME="microsoft/DialoGPT-small"
   ```

### **Logs**

Check logs in:
- `logs/` directory
- Console output
- Docker logs: `docker-compose logs`

## 🎯 **Expected Output**

```
🚀 STARTING LLM FINE-TUNING PIPELINE
==================================================
📊 Collecting sample data...
✅ Loaded sample data: 1 items
🔄 Processing data...
✅ Cleaned data: 1 records
✅ Augmented data: 1 records
📊 Generating benchmark dataset...
✅ Generated benchmark: 90 questions
🤖 Training model...
✅ Model training completed (simulated)
📈 Evaluating model...
✅ Saved evaluation results
⚡ Monitoring performance...
✅ Performance summary: Latency=150.0ms, Throughput=8.5 req/s
🚀 Starting API server...
✅ API server started on http://localhost:8000

🎉 PIPELINE COMPLETED SUCCESSFULLY!
==================================================
📊 Data processed: 1 training, 0 validation
📈 Benchmark questions: 90
🤖 Model trained: microsoft/DialoGPT-medium
📊 Evaluation scores: ROUGE-1=0.450, BLEU=0.580
⚡ Performance: 150.0ms latency, 8.5 req/s
🚀 API Server: http://localhost:8000
```

## 🚀 **Next Steps**

1. **Add Real Data**: Replace `data/raw/sample_ev_data.txt` with your data
2. **Enable Real Training**: Uncomment real training code in `run_pipeline.py`
3. **Customize Domain**: Modify benchmark questions for your use case
4. **Scale Up**: Use Docker for production deployment
5. **Monitor**: Set up logging and monitoring for production

## 📞 **Support**

For issues or questions:
1. Check the logs in `logs/` directory
2. Review the configuration in `src/llm_fine_tuning/config/settings.py`
3. Test individual components with `scripts/quick_test.py`

---

**🎉 Your LLM Fine-tuning Pipeline is ready to run!** 