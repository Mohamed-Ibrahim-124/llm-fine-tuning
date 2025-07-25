#!/usr/bin/env python3
"""
Debug script for testing individual pipeline components.

This script allows testing and debugging of specific pipeline components
without running the entire pipeline.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_fine_tuning.config.settings import get_config
from llm_fine_tuning.data.collection.pdf_extractor import extract_pdf
from llm_fine_tuning.data.collection.web_scraper import scrape_web
from llm_fine_tuning.data.processing.augmenter import augment_data
from llm_fine_tuning.data.processing.cleaner import clean_data
from llm_fine_tuning.data.processing.splitter import split_data
from llm_fine_tuning.data.processing.tokenizer import tokenize_data
from llm_fine_tuning.utils.logger import setup_logger

logger = setup_logger(__name__)


def test_data_collection():
    """Test data collection components."""
    logger.info("Testing data collection components")

    # Test web scraping
    test_urls = ["https://example.com"]
    web_data = scrape_web(test_urls)
    logger.info(f"Web scraping test: {len(web_data)} items collected")

    # Test PDF extraction
    test_pdfs = ["ev_charging_specs.pdf"]
    pdf_data = extract_pdf(test_pdfs)
    logger.info(f"PDF extraction test: {len(pdf_data)} items collected")

    return web_data + pdf_data


def test_data_processing(data):
    """Test data processing components."""
    logger.info("Testing data processing components")

    # Test data cleaning
    cleaned_df = clean_data(data)
    logger.info(f"Data cleaning test: {len(cleaned_df)} records after cleaning")

    # Test data augmentation
    augmented_df = augment_data(cleaned_df)
    logger.info(f"Data augmentation test: {len(augmented_df)} Q&A pairs generated")

    # Test data splitting
    train_df, val_df = split_data(augmented_df)
    logger.info(f"Data splitting test: train={len(train_df)}, val={len(val_df)}")

    return train_df, val_df


def test_tokenization(train_df):
    """Test tokenization component."""
    logger.info("Testing tokenization component")

    config = get_config()
    model_name = config.model.name

    try:
        tokenized_data = tokenize_data(train_df, model_name)
        logger.info("Tokenization test: Success")
        return tokenized_data
    except Exception as e:
        logger.error(f"Tokenization test failed: {str(e)}")
        return None


def main():
    """Run debug tests for pipeline components."""
    try:
        logger.info("Starting pipeline component debugging")

        # Test data collection
        data = test_data_collection()

        # Test data processing
        train_df, val_df = test_data_processing(data)

        # Test tokenization
        tokenized_data = test_tokenization(train_df)

        logger.info("All component tests completed")

        return 0

    except Exception as e:
        logger.error(f"Debug testing failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
