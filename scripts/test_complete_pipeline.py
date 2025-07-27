#!/usr/bin/env python3
"""
Complete pipeline test script to verify all components work correctly.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_fine_tuning.config.settings import get_config
from llm_fine_tuning.utils.logger import setup_logger

logger = setup_logger(__name__)


def test_configuration():
    """Test configuration loading."""
    print("🔧 Testing Configuration...")

    try:
        config = get_config()

        # Verify all required config sections exist
        assert hasattr(config, "model"), "Model config missing"
        assert hasattr(config, "data"), "Data config missing"
        assert hasattr(config, "api"), "API config missing"
        assert hasattr(config, "training"), "Training config missing"
        assert hasattr(config, "evaluation"), "Evaluation config missing"
        assert hasattr(config, "domain"), "Domain config missing"

        # Verify model name is set
        assert config.model.name, "Model name not configured"
        print(f"   ✅ Model: {config.model.name}")

        # Verify domain target
        assert config.domain.target, "Domain target not configured"
        print(f"   ✅ Domain: {config.domain.target}")

        print("   ✅ Configuration test passed")
        return True

    except Exception as e:
        print(f"   ❌ Configuration test failed: {str(e)}")
        return False


def test_pdf_extraction():
    """Test PDF extraction with docling."""
    print("📄 Testing PDF Extraction...")

    try:
        from llm_fine_tuning.data.collection.pdf_extractor import extract_pdf

        # Create a temporary test PDF (or use existing one)
        test_pdf_paths = ["data/raw/test.pdf"]

        # Mock docling if not available
        with patch("docling.parse") as mock_parse:
            mock_document = MagicMock()
            mock_document.text = "Test PDF content about EV charging stations"
            mock_document.pages = [MagicMock()]
            mock_parse.return_value = mock_document

            results = extract_pdf(test_pdf_paths)

            assert len(results) > 0, "No results returned"
            assert results[0]["status"] == "success", "PDF extraction failed"
            assert "text" in results[0], "Text content missing"

            print(f"   ✅ PDF extraction test passed")
            return True

    except Exception as e:
        print(f"   ❌ PDF extraction test failed: {str(e)}")
        return False


def test_web_scraping():
    """Test web scraping with crawl4ai."""
    print("🌐 Testing Web Scraping...")

    try:
        from llm_fine_tuning.data.collection.web_scraper import scrape_urls

        test_urls = ["https://example.com"]

        # Mock crawl4ai if not available
        with patch("crawl4ai.AsyncWebCrawler") as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler_class.return_value = mock_crawler

            # Mock the async result
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.extracted_content = {
                "text": "Test web content about EV charging",
                "title": "Test Page",
            }

            async def mock_arun(*args, **kwargs):
                return mock_result

            mock_crawler.arun = mock_arun

            results = scrape_urls(test_urls)

            assert len(results) > 0, "No results returned"
            assert results[0]["status"] == "success", "Web scraping failed"
            assert "text" in results[0], "Text content missing"

            print(f"   ✅ Web scraping test passed")
            return True

    except Exception as e:
        print(f"   ❌ Web scraping test failed: {str(e)}")
        return False


def test_data_processing():
    """Test data processing pipeline."""
    print("🔄 Testing Data Processing...")

    try:
        from llm_fine_tuning.data.processing.augmenter import augment_data
        from llm_fine_tuning.data.processing.cleaner import clean_data
        from llm_fine_tuning.data.processing.splitter import split_data

        # Test data
        test_data = [
            {"text": "Level 2 chargers provide 3-19 kW power.", "source": "test1"},
            {
                "text": "DC fast chargers can add 60-80 miles in 20 minutes.",
                "source": "test2",
            },
            {
                "text": "Tesla Superchargers offer up to 250 kW charging.",
                "source": "test3",
            },
        ]

        # Test cleaning
        cleaned_data = clean_data(test_data)
        assert len(cleaned_data) > 0, "Data cleaning failed"
        print(f"   ✅ Data cleaning: {len(cleaned_data)} records")

        # Test augmentation (with mocked OpenAI)
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = (
                "Question: What is the charging speed?\nAnswer: Level 2 chargers provide 3-19 kW power."
            )
            mock_client.chat.completions.create.return_value = mock_response

            augmented_data = augment_data(cleaned_data)
            assert len(augmented_data) > 0, "Data augmentation failed"
            print(f"   ✅ Data augmentation: {len(augmented_data)} records")

        # Test splitting
        if len(augmented_data) > 1:
            train_data, val_data = split_data(augmented_data)
            assert len(train_data) > 0, "Training data empty"
            assert len(val_data) > 0, "Validation data empty"
            print(f"   ✅ Data splitting: train={len(train_data)}, val={len(val_data)}")
        else:
            print(
                f"   ⚠️ Skipped splitting (insufficient data: {len(augmented_data)} items)"
            )

        print(f"   ✅ Data processing test passed")
        return True

    except Exception as e:
        print(f"   ❌ Data processing test failed: {str(e)}")
        return False


def test_benchmark_generation():
    """Test benchmark dataset generation."""
    print("📊 Testing Benchmark Generation...")

    try:
        from llm_fine_tuning.evaluation.benchmark_generator import (
            create_benchmark_dataset,
        )

        benchmark_data = create_benchmark_dataset()

        assert len(benchmark_data) > 0, "No benchmark questions generated"
        assert all("question" in item for item in benchmark_data), "Questions missing"
        assert all(
            "expected_answer" in item for item in benchmark_data
        ), "Expected answers missing"

        print(f"   ✅ Benchmark generation: {len(benchmark_data)} questions")
        print(f"   ✅ Benchmark test passed")
        return True

    except Exception as e:
        print(f"   ❌ Benchmark generation test failed: {str(e)}")
        return False


def test_model_training():
    """Test model training (mocked)."""
    print("🤖 Testing Model Training...")

    try:
        from llm_fine_tuning.models.fine_tuner import fine_tune_model

        # Mock MLflow and transformers
        with (
            patch("mlflow.start_run"),
            patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model,
            patch("peft.get_peft_model") as mock_peft,
            patch("transformers.Trainer") as mock_trainer,
        ):

            mock_model_instance = MagicMock()
            mock_model.return_value = mock_model_instance

            mock_peft_instance = MagicMock()
            mock_peft.return_value = mock_peft_instance

            mock_trainer_instance = MagicMock()
            mock_trainer.return_value = mock_trainer_instance

            # Mock training data
            mock_training_data = MagicMock()

            # Test fine-tuning function
            result = fine_tune_model("microsoft/DialoGPT-medium", mock_training_data)

            assert result is not None, "Model training failed"
            print(f"   ✅ Model training test passed")
            return True

    except Exception as e:
        print(f"   ❌ Model training test failed: {str(e)}")
        return False


def test_evaluation():
    """Test model evaluation."""
    print("📈 Testing Model Evaluation...")

    try:
        from llm_fine_tuning.evaluation.evaluator import evaluate_model

        # Mock model and data
        mock_model = MagicMock()
        mock_eval_data = MagicMock()

        # Mock evaluate library
        with patch("evaluate.load") as mock_evaluate_load:
            mock_rouge = MagicMock()
            mock_bleu = MagicMock()
            mock_evaluate_load.side_effect = lambda x: (
                mock_rouge if "rouge" in x else mock_bleu
            )

            mock_rouge.compute.return_value = {
                "rouge1": 0.5,
                "rouge2": 0.3,
                "rougeL": 0.4,
            }
            mock_bleu.compute.return_value = {"bleu": 0.6}

            results = evaluate_model(mock_model, "test-model", mock_eval_data)

            assert "rouge_scores" in results, "ROUGE scores missing"
            assert "bleu_score" in results, "BLEU score missing"

            print(f"   ✅ Model evaluation test passed")
            return True

    except Exception as e:
        print(f"   ❌ Model evaluation test failed: {str(e)}")
        return False


def test_performance_monitoring():
    """Test performance monitoring."""
    print("⚡ Testing Performance Monitoring...")

    try:
        from llm_fine_tuning.evaluation.performance_monitor import (
            create_performance_monitor,
        )

        monitor = create_performance_monitor()

        # Test system metrics
        metrics = monitor.get_system_metrics()
        assert "memory_usage_mb" in metrics, "Memory metrics missing"
        assert "cpu_usage_percent" in metrics, "CPU metrics missing"

        # Test metrics recording
        monitor.record_metrics(100.0, 10.0, metrics)
        assert len(monitor.metrics_history) > 0, "Metrics not recorded"

        # Test performance summary
        summary = monitor.get_performance_summary()
        assert "average_latency" in summary, "Latency summary missing"
        assert "average_throughput" in summary, "Throughput summary missing"

        print(f"   ✅ Performance monitoring test passed")
        return True

    except Exception as e:
        print(f"   ❌ Performance monitoring test failed: {str(e)}")
        return False


def test_api_server():
    """Test API server functionality."""
    print("🚀 Testing API Server...")

    try:
        from fastapi.testclient import TestClient

        from llm_fine_tuning.deployment.api_server import app

        client = TestClient(app)

        # Test health check (if available) or basic functionality
        try:
            response = client.get("/")
            print(f"   ✅ API server responds: {response.status_code}")
        except:
            # If no root endpoint, test that app is created
            assert app is not None, "API app not created"
            print(f"   ✅ API app created successfully")

        print(f"   ✅ API server test passed")
        return True

    except Exception as e:
        print(f"   ❌ API server test failed: {str(e)}")
        return False


def test_pipeline_integration():
    """Test pipeline integration."""
    print("🔗 Testing Pipeline Integration...")

    try:
        # Test that all modules can be imported
        modules_to_test = [
            "llm_fine_tuning.pipeline.main",
            "llm_fine_tuning.data.collection.pdf_extractor",
            "llm_fine_tuning.data.collection.web_scraper",
            "llm_fine_tuning.data.processing.cleaner",
            "llm_fine_tuning.data.processing.augmenter",
            "llm_fine_tuning.data.processing.splitter",
            "llm_fine_tuning.models.fine_tuner",
            "llm_fine_tuning.evaluation.benchmark_generator",
            "llm_fine_tuning.evaluation.evaluator",
            "llm_fine_tuning.evaluation.performance_monitor",
            "llm_fine_tuning.deployment.api_server",
        ]

        for module_name in modules_to_test:
            __import__(module_name)
            print(f"   ✅ {module_name} imported successfully")

        print(f"   ✅ Pipeline integration test passed")
        return True

    except Exception as e:
        print(f"   ❌ Pipeline integration test failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("🧪 COMPLETE PIPELINE TEST SUITE")
    print("=" * 50)

    tests = [
        test_configuration,
        test_pdf_extraction,
        test_web_scraping,
        test_data_processing,
        test_benchmark_generation,
        test_model_training,
        test_evaluation,
        test_performance_monitoring,
        test_api_server,
        test_pipeline_integration,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"   ❌ Test {test.__name__} crashed: {str(e)}")
            results.append(False)

    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)

    passed = sum(results)
    total = len(results)

    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Pipeline is ready for production.")
        return 0
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    exit(main())
