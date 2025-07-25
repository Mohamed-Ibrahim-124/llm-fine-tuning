"""
Configuration settings for the LLM Fine-tuning Pipeline.

This module provides centralized configuration management with environment variables
and type-safe configuration access.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ModelConfig:
    """Model configuration settings."""
    name: str
    base_path: str
    fine_tuned_path: str


@dataclass
class DataConfig:
    """Data configuration settings."""
    path: str
    raw_path: str
    processed_path: str
    training_path: str
    evaluation_path: str


@dataclass
class APIConfig:
    """API configuration settings."""
    token: str
    host: str
    port: int


@dataclass
class TrainingConfig:
    """Training configuration settings."""
    batch_size: int
    learning_rate: float
    num_epochs: int
    max_length: int
    gradient_accumulation_steps: int


@dataclass
class EvaluationConfig:
    """Evaluation configuration settings."""
    batch_size: int
    metrics: List[str]


@dataclass
class DomainConfig:
    """Domain-specific configuration settings."""
    target: str
    use_case: str
    sources: List[str]


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig
    data: DataConfig
    api: APIConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    domain: DomainConfig


# Environment variable setup
def _setup_environment():
    """Set up environment variables with defaults."""
    # Model Configuration
    os.environ.setdefault("MODEL_NAME", "meta-llama/Llama-3-7B")
    os.environ.setdefault("BASE_MODEL_PATH", "./models/base")
    os.environ.setdefault("FINE_TUNED_MODEL_PATH", "./models/fine_tuned")

    # Data Configuration
    os.environ.setdefault("DATA_PATH", "data/")
    os.environ.setdefault("RAW_DATA_PATH", "data/raw/")
    os.environ.setdefault("PROCESSED_DATA_PATH", "data/processed/")
    os.environ.setdefault("TRAINING_DATA_PATH", "data/training/")
    os.environ.setdefault("EVALUATION_DATA_PATH", "data/evaluation/")

    # API Configuration
    os.environ.setdefault("API_TOKEN", "your-secret-token")
    os.environ.setdefault("API_HOST", "0.0.0.0")
    os.environ.setdefault("API_PORT", "8000")

    # Training Configuration
    os.environ.setdefault("BATCH_SIZE", "1")
    os.environ.setdefault("LEARNING_RATE", "2e-4")
    os.environ.setdefault("NUM_EPOCHS", "3")
    os.environ.setdefault("MAX_LENGTH", "512")
    os.environ.setdefault("GRADIENT_ACCUMULATION_STEPS", "4")

    # Evaluation Configuration
    os.environ.setdefault("EVAL_BATCH_SIZE", "4")
    os.environ.setdefault("METRICS", "rouge,bleu")

    # Logging Configuration
    os.environ.setdefault("LOG_LEVEL", "INFO")
    os.environ.setdefault("LOG_FILE", "logs/pipeline.log")


# Domain-specific configuration
TARGET_DOMAIN = "electric vehicle charging stations"
USE_CASE = "question-answering"
DATA_SOURCES = ["web_scraping", "pdf_extraction"]

# Domain-specific prompts for data augmentation
DOMAIN_PROMPTS = {
    "qa_generation": "Generate a question-answer pair about electric vehicle charging stations based on this text: {text}",
    "summarization": "Summarize this information about EV charging: {text}",
    "classification": "Classify this EV charging information into categories: {text}"
}

# Benchmark dataset configuration
BENCHMARK_CONFIG = {
    "num_questions": 100,
    "categories": ["charging_speed", "connector_types", "installation", "pricing", "availability"],
    "difficulty_levels": ["easy", "medium", "hard"]
}


def get_config() -> Config:
    """
    Get configuration as a structured Config object.
    
    Returns:
        Config: Structured configuration object
    """
    _setup_environment()
    
    return Config(
        model=ModelConfig(
            name=os.environ["MODEL_NAME"],
            base_path=os.environ["BASE_MODEL_PATH"],
            fine_tuned_path=os.environ["FINE_TUNED_MODEL_PATH"]
        ),
        data=DataConfig(
            path=os.environ["DATA_PATH"],
            raw_path=os.environ["RAW_DATA_PATH"],
            processed_path=os.environ["PROCESSED_DATA_PATH"],
            training_path=os.environ["TRAINING_DATA_PATH"],
            evaluation_path=os.environ["EVALUATION_DATA_PATH"]
        ),
        api=APIConfig(
            token=os.environ["API_TOKEN"],
            host=os.environ["API_HOST"],
            port=int(os.environ["API_PORT"])
        ),
        training=TrainingConfig(
            batch_size=int(os.environ["BATCH_SIZE"]),
            learning_rate=float(os.environ["LEARNING_RATE"]),
            num_epochs=int(os.environ["NUM_EPOCHS"]),
            max_length=int(os.environ["MAX_LENGTH"]),
            gradient_accumulation_steps=int(os.environ["GRADIENT_ACCUMULATION_STEPS"])
        ),
        evaluation=EvaluationConfig(
            batch_size=int(os.environ["EVAL_BATCH_SIZE"]),
            metrics=os.environ["METRICS"].split(",")
        ),
        domain=DomainConfig(
            target=TARGET_DOMAIN,
            use_case=USE_CASE,
            sources=DATA_SOURCES
        )
    )


 