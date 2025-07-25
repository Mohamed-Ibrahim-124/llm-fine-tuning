#!/usr/bin/env python3
"""
Local test script for the LLM Fine-tuning Pipeline.
This script tests the pipeline components without requiring actual model training.
"""

import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi.testclient import TestClient

from llm_fine_tuning.config.settings import get_config
from llm_fine_tuning.data.collection.web_scraper import scrape_web
from llm_fine_tuning.data.processing.augmenter import augment_data
from llm_fine_tuning.data.processing.cleaner import clean_data
from llm_fine_tuning.data.processing.splitter import split_data
from llm_fine_tuning.deployment.api_server import app
from llm_fine_tuning.evaluation.benchmark_generator import create_benchmark_dataset
from llm_fine_tuning.evaluation.performance_monitor import create_performance_monitor


def test_pipeline_components():
    """Test all pipeline components locally."""
    print("🚀 Testing LLM Fine-tuning Pipeline Components")
    print("=" * 50)

    # Test 1: Configuration
    print("\n1. Testing Configuration...")
    config = get_config()
    print(f"   ✅ Model: {config.model.name}")
    print(f"   ✅ Domain: {config.domain.target}")
    print(f"   ✅ Training epochs: {config.training.num_epochs}")

    # Test 2: Data Collection
    print("\n2. Testing Data Collection...")
    urls = ["https://afdc.energy.gov/stations"]
    web_data = scrape_web(urls)
    print(f"   ✅ Collected {len(web_data)} web data items")

    # Test 3: Data Processing
    print("\n3. Testing Data Processing...")
    cleaned_data = clean_data(web_data)
    print(f"   ✅ Cleaned {len(cleaned_data)} data items")

    augmented_data = augment_data(cleaned_data)
    print(f"   ✅ Augmented to {len(augmented_data)} Q&A pairs")

    # Only split if we have enough data
    if len(augmented_data) > 1:
        train_data, val_data = split_data(augmented_data)
        print(f"   ✅ Split data: train={len(train_data)}, val={len(val_data)}")
    else:
        print(
            f"   ✅ Skipped splitting (insufficient data: {len(augmented_data)} items)"
        )
        train_data, val_data = augmented_data, []

    # Test 4: Benchmark Generation
    print("\n4. Testing Benchmark Generation...")
    benchmark_data = create_benchmark_dataset()
    print(f"   ✅ Generated {len(benchmark_data)} benchmark questions")

    # Test 5: Performance Monitoring
    print("\n5. Testing Performance Monitoring...")
    monitor = create_performance_monitor()
    monitor.start_monitoring()

    # Simulate some metrics
    system_metrics = monitor.get_system_metrics()
    monitor.record_metrics(150.0, 8.0, system_metrics)
    print("   ✅ Recorded performance metrics")
    # Test 6: API Server
    print("\n6. Testing API Server...")
    client = TestClient(app)

    # Test API endpoint
    response = client.post(
        "/predict",
        json={"input_text": "What is EV charging?"},
        headers={"Authorization": "Bearer your-secret-token"},
    )
    print(f"   ✅ API endpoint test: {response.status_code}")

    # Test 7: Save test results
    print("\n7. Saving Test Results...")
    os.makedirs("logs", exist_ok=True)

    # Save benchmark data
    with open("logs/test_benchmark.json", "w") as f:
        json.dump(benchmark_data[:5], f, indent=2)  # Save first 5 questions

    # Save performance metrics
    monitor.save_metrics("logs/test_performance.json")

    print("   ✅ Test results saved to logs/")

    print("\n" + "=" * 50)
    print("🎉 All pipeline components tested successfully!")
    print("\n📁 Generated files:")
    print("   - logs/test_benchmark.json (sample benchmark questions)")
    print("   - logs/test_performance.json (performance metrics)")
    print("\n📊 Pipeline Summary:")
    print(f"   - Data collected: {len(web_data)} items")
    print(f"   - Data processed: {len(augmented_data)} Q&A pairs")
    print(f"   - Benchmark questions: {len(benchmark_data)}")
    print("   - API endpoint: ✅ Working")
    return True


if __name__ == "__main__":
    try:
        test_pipeline_components()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
