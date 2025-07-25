"""
Tokenization module for the LLM Fine-tuning Pipeline.

This module provides functionality to tokenize text data for model training
using Hugging Face tokenizers.
"""

from typing import Dict, List

import pandas as pd
import torch
from transformers import AutoTokenizer

from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class DataTokenizer:
    """Data tokenizer class for text tokenization."""

    def __init__(self, model_name: str, max_length: int = 512):
        """
        Initialize the data tokenizer.

        Args:
            model_name: Name of the pre-trained model to use for tokenization
            max_length: Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self._load_tokenizer()

    def _load_tokenizer(self):
        """Load the tokenizer for the specified model."""
        try:
            logger.info(f"Loading tokenizer for model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Set padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("Tokenizer loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}")
            raise

    def tokenize_texts(
        self, questions: List[str], answers: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize question and answer texts.

        Args:
            questions: List of question texts
            answers: List of answer texts

        Returns:
            Dictionary containing tokenized inputs
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded")

        logger.info("Starting text tokenization")

        try:
            # Tokenize questions and answers
            tokenized = self.tokenizer(
                questions,
                answers,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            logger.info("Tokenization completed successfully")
            logger.info(f"Tokenized shape: {tokenized['input_ids'].shape}")

            return tokenized

        except Exception as e:
            logger.error(f"Tokenization failed: {str(e)}")
            raise

    def tokenize_dataframe(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Tokenize data from a DataFrame.

        Args:
            df: DataFrame containing 'question' and 'answer' columns

        Returns:
            Dictionary containing tokenized inputs
        """
        if "question" not in df.columns or "answer" not in df.columns:
            raise ValueError("DataFrame must contain 'question' and 'answer' columns")

        questions = df["question"].tolist()
        answers = df["answer"].tolist()

        return self.tokenize_texts(questions, answers)


def tokenize_data(df: pd.DataFrame, model_name: str) -> Dict[str, torch.Tensor]:
    """
    Tokenize data for model training.

    Args:
        df: DataFrame containing Q&A pairs
        model_name: Name of the pre-trained model

    Returns:
        Dictionary containing tokenized inputs
    """
    tokenizer = DataTokenizer(model_name)
    return tokenizer.tokenize_dataframe(df)
