#!/usr/bin/env python3
"""
Main pipeline execution script for the LLM Fine-tuning Pipeline.

This script orchestrates the complete pipeline from data collection
to model deployment.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_fine_tuning.pipeline.main import ml_pipeline
from llm_fine_tuning.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Execute the complete ML pipeline."""
    try:
        logger.info("Starting LLM Fine-tuning Pipeline")
        
        # Run the pipeline
        results = ml_pipeline()
        
        logger.info("Pipeline execution completed successfully")
        logger.info(f"Results: {results}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 