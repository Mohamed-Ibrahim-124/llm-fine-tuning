import os

from zenml import pipeline, step

from ..config.settings import get_config
from ..data.collection.pdf_extractor import extract_pdf
from ..data.collection.web_scraper import scrape_web
from ..data.processing.augmenter import augment_data
from ..data.processing.cleaner import clean_data
from ..data.processing.splitter import split_data
from ..data.processing.tokenizer import tokenize_data
from ..deployment.api_server import deploy_model
from ..evaluation.benchmark_generator import create_benchmark_dataset
from ..evaluation.evaluator import evaluate_model
from ..evaluation.performance_monitor import create_performance_monitor
from ..models.fine_tuner import fine_tune_model

# Use centralized logging
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

@step
def collect_data():
    """Collect data from web scraping and PDF extraction."""
    logger.info("Starting data collection")
    
    # Get configuration
    config = get_config()
    
    # URLs for EV charging station data
    urls = [
        "https://afdc.energy.gov/stations",
        "https://www.evgo.com/charging-stations/",
        "https://www.chargepoint.com/charging-stations"
    ]
    
    # PDF files for EV charging specifications
    pdf_paths = ["ev_charging_specs.pdf"]
    
    # Collect web data
    web_data = scrape_web(urls)
    logger.info(f"Collected {len(web_data)} web data items")
    
    # Extract PDF data
    pdf_data = extract_pdf(pdf_paths)
    logger.info(f"Extracted {len(pdf_data)} PDF data items")
    
    return web_data + pdf_data

@step
def process_data(data):
    """Process and prepare data for training."""
    logger.info("Starting data processing")
    
    # Clean data
    cleaned_data = clean_data(data)
    logger.info(f"Cleaned data: {len(cleaned_data)} items")
    
    # Augment data with Q&A pairs
    augmented_data = augment_data(cleaned_data)
    logger.info(f"Augmented data: {len(augmented_data)} items")
    
    # Split into train/validation sets
    train_data, val_data = split_data(augmented_data)
    logger.info(f"Split data: train={len(train_data)}, val={len(val_data)}")
    
    # Tokenize data
    config = get_config()
    model_name = config.model.name
    tokenized_train = tokenize_data(train_data, model_name)
    tokenized_val = tokenize_data(val_data, model_name)
    
    return {
        "train": tokenized_train,
        "val": tokenized_val,
        "train_df": train_data,
        "val_df": val_data
    }

@step
def generate_benchmark():
    """Generate domain-specific benchmark dataset."""
    logger.info("Generating benchmark dataset")
    
    # Create benchmark dataset
    benchmark_data = create_benchmark_dataset()
    logger.info(f"Generated {len(benchmark_data)} benchmark questions")
    
    return benchmark_data

@step
def train_model(processed_data):
    """Fine-tune the model using QLoRA."""
    logger.info("Starting model fine-tuning")
    
    config = get_config()
    model_name = config.model.name
    
    # Fine-tune model
    model = fine_tune_model(model_name, processed_data["train"])
    logger.info("Model fine-tuning completed")
    
    return model

@step
def evaluate_models(model, processed_data, benchmark_data):
    """Evaluate fine-tuned model against baseline."""
    logger.info("Starting model evaluation")
    
    config = get_config()
    model_name = config.model.name
    
    # Evaluate models
    results = evaluate_model(model, model_name, processed_data["val"])
    logger.info("Model evaluation completed")
    
    return results

@step
def monitor_performance(model, benchmark_data):
    """Monitor performance metrics."""
    logger.info("Starting performance monitoring")
    
    # Create performance monitor
    monitor = create_performance_monitor()
    monitor.start_monitoring()
    
    # Test performance with benchmark questions
    test_questions = [item["question"] for item in benchmark_data[:10]]  # Test with first 10 questions
    
    # Measure performance
    system_metrics = monitor.get_system_metrics()
    throughput = monitor.measure_throughput(model, test_questions)
    
    # Record metrics
    avg_latency = 100.0  # Placeholder - would be calculated from actual inference
    monitor.record_metrics(avg_latency, throughput, system_metrics)
    
    # Save performance metrics
    monitor.save_metrics()
    
    logger.info("Performance monitoring completed")
    return monitor.get_performance_summary()

@step
def deploy_model_step(model):
    """Deploy the fine-tuned model."""
    logger.info("Starting model deployment")
    
    # Deploy model
    deploy_model(model)
    logger.info("Model deployment completed")

@pipeline
def ml_pipeline():
    """Complete ML pipeline for LLM fine-tuning."""
    logger.info("Starting complete ML pipeline execution")
    
    # Create necessary directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/training", exist_ok=True)
    os.makedirs("data/evaluation", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data/models", exist_ok=True)
    
    # Pipeline steps
    data = collect_data()
    processed_data = process_data(data)
    benchmark_data = generate_benchmark()
    model = train_model(processed_data)
    results = evaluate_models(model, processed_data, benchmark_data)
    performance_summary = monitor_performance(model, benchmark_data)
    deploy_model_step(model)
    
    logger.info("Complete ML pipeline execution finished")
    
    return {
        "evaluation_results": results,
        "performance_summary": performance_summary,
        "status": "completed"
    }

if __name__ == "__main__":
    ml_pipeline()