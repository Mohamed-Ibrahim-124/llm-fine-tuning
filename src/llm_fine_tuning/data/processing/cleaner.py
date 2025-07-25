"""
Data cleaning module for the LLM Fine-tuning Pipeline.

This module provides functionality to clean and preprocess collected data
with deduplication, quality filtering, and normalization.
"""

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class DataCleaner:
    """Data cleaner class for preprocessing collected data."""
    
    def __init__(self, min_text_length: int = 10, max_text_length: int = 10000):
        """
        Initialize the data cleaner.
        
        Args:
            min_text_length: Minimum text length to keep
            max_text_length: Maximum text length to keep
        """
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
    
    def clean_text(self, text: str) -> str:
        """
        Clean a single text entry.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove very short or very long texts
        if len(text) < self.min_text_length or len(text) > self.max_text_length:
            return ""
        
        return text
    
    def clean_data(self, data: List[Dict[str, Any]], output_path: str = None) -> pd.DataFrame:
        """
        Clean the collected data.
        
        Args:
            data: List of dictionaries containing raw data
            output_path: Optional path to save cleaned data
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning process")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        initial_count = len(df)
        logger.info(f"Initial data count: {initial_count}")
        
        # Remove rows with missing text
        df = df.dropna(subset=['text'])
        logger.info(f"After removing nulls: {len(df)}")
        
        # Clean text content
        df['text'] = df['text'].apply(self.clean_text)
        
        # Remove empty texts after cleaning
        df = df[df['text'].str.len() > 0]
        logger.info(f"After text cleaning: {len(df)}")
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['text'])
        logger.info(f"After deduplication: {len(df)}")
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Save cleaned data if output path is provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Cleaned data saved to: {output_path}")
        
        final_count = len(df)
        logger.info(f"Data cleaning completed: {final_count}/{initial_count} records kept")
        
        return df


def clean_data(data: List[Dict[str, Any]], output_path: str = None) -> pd.DataFrame:
    """
    Clean the collected data.
    
    Args:
        data: List of dictionaries containing raw data
        output_path: Optional path to save cleaned data
        
    Returns:
        Cleaned DataFrame
    """
    cleaner = DataCleaner()
    return cleaner.clean_data(data, output_path) 