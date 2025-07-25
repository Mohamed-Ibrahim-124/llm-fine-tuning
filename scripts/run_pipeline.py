#!/usr/bin/env python3
"""
Main script to run the complete LLM fine-tuning pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_fine_tuning.config.settings import get_config
from llm_fine_tuning.pipeline.main import ml_pipeline
from llm_fine_tuning.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Run the complete ML pipeline."""
    try:
        logger.info("Starting LLM Fine-tuning Pipeline")

        # Get configuration
        config = get_config()
        logger.info(f"Using model: {config.model.name}")
        logger.info(f"Target domain: {config.domain.target}")
        # Run the pipeline
        ml_pipeline()

        logger.info("Pipeline completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
