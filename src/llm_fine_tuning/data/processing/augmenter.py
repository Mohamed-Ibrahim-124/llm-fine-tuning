"""
Data augmentation module for the LLM Fine-tuning Pipeline.

This module provides functionality to augment data by generating
question-answer pairs from collected text content.
"""

from pathlib import Path
from typing import Dict

import pandas as pd

from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class DataAugmenter:
    """Data augmenter class for generating Q&A pairs."""

    def __init__(self, domain_prompts: Dict[str, str] = None):
        """
        Initialize the data augmenter.

        Args:
            domain_prompts: Dictionary of domain-specific prompts for augmentation
        """
        self.domain_prompts = domain_prompts or {
            "qa_generation": (
                "Generate a question-answer pair about electric vehicle charging "
                "stations based on this text: {text}"
            ),
            "summarization": "Summarize this information about EV charging: {text}",
            "classification": "Classify this EV charging information into categories: {text}",
        }

    def generate_qa_pair(self, text: str) -> Dict[str, str]:
        """
        Generate a question-answer pair from text.

        Args:
            text: Input text to generate Q&A from

        Returns:
            Dictionary containing question and answer
        """
        # Simple rule-based augmentation
        # In a real implementation, you would use an LLM API here

        # Split text into sentences
        sentences = text.split(".")
        if not sentences or len(sentences[0].strip()) < 10:
            return {"question": "What is this text about?", "answer": text}

        # Use first sentence to generate a question
        first_sentence = sentences[0].strip()
        question = f"What is {first_sentence.lower()}?"

        return {"question": question, "answer": text}

    def augment_data(self, df, output_path: str = None):
        """
        Augment data by generating Q&A pairs.

        Args:
            df: DataFrame or list containing cleaned text data
            output_path: Optional path to save augmented data

        Returns:
            Augmented DataFrame with Q&A pairs
        """
        logger.info("Starting data augmentation process")

        # Convert list to DataFrame if needed
        if isinstance(df, list):
            df = pd.DataFrame(df)

        qa_pairs = []
        initial_count = len(df)

        for idx, row in df.iterrows():
            text = row["text"]
            source = row.get("source", "unknown")

            # Generate Q&A pair
            qa_pair = self.generate_qa_pair(text)
            qa_pair["source"] = source
            qa_pair["original_text"] = text

            qa_pairs.append(qa_pair)

            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{initial_count} records")

        # Create augmented DataFrame
        augmented_df = pd.DataFrame(qa_pairs)

        # Save augmented data if output path is provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            augmented_df.to_csv(output_path, index=False)
            logger.info(f"Augmented data saved to: {output_path}")

        logger.info(
            f"Data augmentation completed: {len(augmented_df)} Q&A pairs generated"
        )

        return augmented_df


def augment_data(df: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
    """
    Augment data by generating Q&A pairs.

    Args:
        df: DataFrame containing cleaned text data
        output_path: Optional path to save augmented data

    Returns:
        Augmented DataFrame with Q&A pairs
    """
    augmenter = DataAugmenter()
    return augmenter.augment_data(df, output_path)
