"""
LLM Fine-tuning Pipeline Package

A comprehensive pipeline for fine-tuning small language models for domain-specific tasks.
"""

__version__ = "1.0.0"
__author__ = "LLM Fine-tuning Team"

# Main package imports
from .config.settings import get_config
from .pipeline.main import ml_pipeline

__all__ = [
    "get_config",
    "ml_pipeline",
    "__version__",
    "__author__"
] 