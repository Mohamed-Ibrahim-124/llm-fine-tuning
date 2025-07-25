"""
Data splitting module for the LLM Fine-tuning Pipeline.

This module provides functionality to split data into training and validation sets
with proper stratification and reproducibility.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class DataSplitter:
    """Data splitter class for creating train/validation splits."""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the data splitter.
        
        Args:
            test_size: Proportion of data to use for validation
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
    
    def split_data(self, df: pd.DataFrame, output_dir: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and validation sets.
        
        Args:
            df: DataFrame to split
            output_dir: Optional directory to save split datasets
            
        Returns:
            Tuple of (train_df, val_df)
        """
        logger.info("Starting data splitting process")
        
        initial_count = len(df)
        logger.info(f"Total data count: {initial_count}")
        
        # Perform the split
        train_df, val_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True
        )
        
        # Reset indices
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        
        logger.info(f"Data split completed: train={len(train_df)}, val={len(val_df)}")
        
        # Save split datasets if output directory is provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            train_path = output_dir / "train_data.csv"
            val_path = output_dir / "val_data.csv"
            
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            
            logger.info(f"Split datasets saved to: {output_dir}")
        
        return train_df, val_df


def split_data(df: pd.DataFrame, output_dir: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and validation sets.
    
    Args:
        df: DataFrame to split
        output_dir: Optional directory to save split datasets
        
    Returns:
        Tuple of (train_df, val_df)
    """
    splitter = DataSplitter()
    return splitter.split_data(df, output_dir) 