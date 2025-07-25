import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_fine_tuning.config.settings import get_config
from llm_fine_tuning.data.processing.augmenter import augment_data
from llm_fine_tuning.data.processing.cleaner import clean_data
from llm_fine_tuning.data.processing.splitter import split_data
from llm_fine_tuning.data.processing.tokenizer import tokenize_data

# Import pipeline components
from llm_fine_tuning.evaluation.benchmark_generator import (
    BenchmarkGenerator,
    create_benchmark_dataset,
)
from llm_fine_tuning.evaluation.performance_monitor import (
    PerformanceMonitor,
    create_performance_monitor,
)


class TestDataProcessing:
    """Test data processing components."""

    def test_clean_data(self):
        """Test data cleaning functionality."""
        test_data = [
            {"text": "Test text 1", "source": "test1"},
            {"text": "Test text 2", "source": "test2"},
            {"text": "Test text 1", "source": "test1"},  # Duplicate
            {"text": "", "source": "test3"},  # Empty text
        ]

        cleaned = clean_data(test_data)

        assert len(cleaned) == 2  # Should remove duplicate and empty
        assert "Test text 1" in cleaned["text"].values
        assert "Test text 2" in cleaned["text"].values

    def test_augment_data(self):
        """Test data augmentation functionality."""
        test_data = [
            {"text": "Level 2 chargers provide 3-19 kW power.", "source": "test1"},
            {
                "text": "DC fast chargers can add 60-80 miles in 20 minutes.",
                "source": "test2",
            },
        ]

        augmented = augment_data(test_data)

        assert len(augmented) == 2
        assert "question" in augmented.columns
        assert "answer" in augmented.columns
        assert "source" in augmented.columns

    def test_split_data(self):
        """Test data splitting functionality."""
        test_data = pd.DataFrame(
            {"text": [f"Text {i}" for i in range(10)], "source": ["test"] * 10}
        )

        train_data, val_data = split_data(test_data)

        assert len(train_data) + len(val_data) == 10
        assert len(train_data) > 0
        assert len(val_data) > 0


class TestBenchmarkGenerator:
    """Test benchmark generation components."""

    def test_benchmark_generator_creation(self):
        """Test benchmark generator initialization."""
        generator = BenchmarkGenerator()

        assert generator.categories == [
            "charging_speed",
            "connector_types",
            "installation",
            "pricing",
            "availability",
        ]
        assert generator.difficulty_levels == ["easy", "medium", "hard"]
        assert generator.num_questions == 100

    def test_question_generation(self):
        """Test question generation for different categories."""
        generator = BenchmarkGenerator()

        question_data = generator._generate_question(
            "charging_speed", "easy", "test_001"
        )

        assert "id" in question_data
        assert "category" in question_data
        assert "difficulty" in question_data
        assert "question" in question_data
        assert "expected_answer" in question_data
        assert question_data["category"] == "charging_speed"
        assert question_data["difficulty"] == "easy"

    def test_benchmark_dataset_creation(self):
        """Test full benchmark dataset creation."""
        # Create a smaller dataset for testing
        with patch(
            "llm_fine_tuning.evaluation.benchmark_generator.BENCHMARK_CONFIG"
        ) as mock_config:
            mock_config.__getitem__.side_effect = lambda x: {
                "num_questions": 10,
                "categories": ["charging_speed"],
                "difficulty_levels": ["easy"],
            }.get(x)

            benchmark_data = create_benchmark_dataset()

            assert len(benchmark_data) > 0
            assert all("question" in item for item in benchmark_data)
            assert all("expected_answer" in item for item in benchmark_data)


class TestPerformanceMonitoring:
    """Test performance monitoring components."""

    def test_performance_monitor_creation(self):
        """Test performance monitor initialization."""
        monitor = create_performance_monitor()

        assert isinstance(monitor, PerformanceMonitor)
        assert monitor.metrics_history == []
        assert monitor.start_time is None

    def test_system_metrics(self):
        """Test system metrics collection."""
        monitor = create_performance_monitor()

        metrics = monitor.get_system_metrics()

        assert "memory_usage_mb" in metrics
        assert "cpu_usage_percent" in metrics
        assert metrics["memory_usage_mb"] > 0
        assert 0 <= metrics["cpu_usage_percent"] <= 100

    def test_metrics_recording(self):
        """Test metrics recording functionality."""
        monitor = create_performance_monitor()
        monitor.start_monitoring()

        system_metrics = monitor.get_system_metrics()
        monitor.record_metrics(100.0, 10.0, system_metrics)

        assert len(monitor.metrics_history) == 1
        assert monitor.metrics_history[0].latency_ms == 100.0
        assert monitor.metrics_history[0].throughput_requests_per_second == 10.0

    def test_performance_summary(self):
        """Test performance summary generation."""
        monitor = create_performance_monitor()
        monitor.start_monitoring()

        # Record some test metrics
        system_metrics = monitor.get_system_metrics()
        monitor.record_metrics(100.0, 10.0, system_metrics)
        monitor.record_metrics(200.0, 5.0, system_metrics)

        summary = monitor.get_performance_summary()

        assert "total_requests" in summary
        assert "latency" in summary
        assert "throughput" in summary
        assert summary["total_requests"] == 2
        assert summary["latency"]["mean_ms"] == 150.0


class TestConfiguration:
    """Test configuration management."""

    def test_config_structure(self):
        """Test configuration structure and values."""
        config = get_config()

        assert hasattr(config, "model")
        assert hasattr(config, "data")
        assert hasattr(config, "api")
        assert hasattr(config, "training")
        assert hasattr(config, "evaluation")
        assert hasattr(config, "domain")

    def test_model_config(self):
        """Test model configuration values."""
        config = get_config()

        assert config.model.name == "microsoft/DialoGPT-medium"
        assert hasattr(config.model, "base_path")
        assert hasattr(config.model, "fine_tuned_path")

    def test_training_config(self):
        """Test training configuration values."""
        config = get_config()

        assert config.training.batch_size == 1
        assert config.training.num_epochs == 3
        assert config.training.learning_rate == 2e-4


class TestIntegration:
    """Integration tests for pipeline components."""

    def test_end_to_end_data_processing(self):
        """Test complete data processing pipeline."""
        # Test data
        test_data = [
            {"text": "Level 2 chargers provide 3-19 kW power.", "source": "test1"},
            {
                "text": "DC fast chargers can add 60-80 miles in 20 minutes.",
                "source": "test2",
            },
            {
                "text": "Level 2 chargers provide 3-19 kW power.",
                "source": "test1",
            },  # Duplicate
        ]

        # Clean data
        cleaned = clean_data(test_data)
        assert len(cleaned) == 2

        # Augment data
        augmented = augment_data(cleaned)
        assert len(augmented) == 2
        assert "question" in augmented.columns

        # Split data
        train_data, val_data = split_data(augmented)
        assert len(train_data) + len(val_data) == 2

    def test_benchmark_and_monitoring_integration(self):
        """Test integration between benchmark generation and performance monitoring."""
        # Create benchmark
        generator = BenchmarkGenerator()
        benchmark_data = generator.generate_benchmark_dataset()

        # Create performance monitor
        monitor = create_performance_monitor()
        monitor.start_monitoring()

        # Simulate some performance metrics
        system_metrics = monitor.get_system_metrics()
        monitor.record_metrics(150.0, 8.0, system_metrics)

        # Verify both components work together
        assert len(benchmark_data) > 0
        assert len(monitor.metrics_history) == 1

        # Test saving metrics
        monitor.save_metrics("logs/test_performance_metrics.json")
        assert os.path.exists("logs/test_performance_metrics.json")

        # Cleanup
        if os.path.exists("logs/test_performance_metrics.json"):
            os.remove("logs/test_performance_metrics.json")


if __name__ == "__main__":
    pytest.main([__file__])
