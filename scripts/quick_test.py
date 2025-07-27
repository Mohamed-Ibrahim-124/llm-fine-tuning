#!/usr/bin/env python3
"""
Quick test script to verify pipeline components work.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("🔧 Testing imports...")
    
    try:
        from llm_fine_tuning.config.settings import get_config
        print("   ✅ Config imported")
        
        from llm_fine_tuning.data.collection.web_scraper import scrape_urls
        print("   ✅ Web scraper imported")
        
        from llm_fine_tuning.data.processing.cleaner import clean_data
        print("   ✅ Data cleaner imported")
        
        from llm_fine_tuning.data.processing.augmenter import augment_data
        print("   ✅ Data augmenter imported")
        
        from llm_fine_tuning.evaluation.benchmark_generator import (
            create_benchmark_dataset,
        )
        print("   ✅ Benchmark generator imported")
        
        from llm_fine_tuning.models.fine_tuner import fine_tune_model
        print("   ✅ Model fine-tuner imported")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import failed: {str(e)}")
        return False

def test_config():
    """Test configuration."""
    print("⚙️ Testing configuration...")
    
    try:
        from llm_fine_tuning.config.settings import get_config
        config = get_config()
        
        print(f"   ✅ Model: {config.model.name}")
        print(f"   ✅ Domain: {config.domain.target}")
        return True
        
    except Exception as e:
        print(f"   ❌ Config failed: {str(e)}")
        return False

def test_data_processing():
    """Test data processing pipeline."""
    print("🔄 Testing data processing...")
    
    try:
        from llm_fine_tuning.data.processing.augmenter import augment_data
        from llm_fine_tuning.data.processing.cleaner import clean_data

        # Test data
        test_data = [
            {"text": "Level 2 chargers provide 3-19 kW power.", "source": "test1"},
            {"text": "DC fast chargers can add 60-80 miles in 20 minutes.", "source": "test2"}
        ]
        
        # Test cleaning
        cleaned = clean_data(test_data)
        print(f"   ✅ Data cleaning: {len(cleaned)} records")
        
        # Test augmentation (with fallback)
        augmented = augment_data(cleaned)
        print(f"   ✅ Data augmentation: {len(augmented)} records")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Data processing failed: {str(e)}")
        return False

def test_benchmark():
    """Test benchmark generation."""
    print("📊 Testing benchmark generation...")
    
    try:
        from llm_fine_tuning.evaluation.benchmark_generator import (
            create_benchmark_dataset,
        )
        
        benchmark = create_benchmark_dataset()
        print(f"   ✅ Benchmark: {len(benchmark)} questions")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Benchmark failed: {str(e)}")
        return False

def main():
    """Run quick tests."""
    print("🧪 QUICK PIPELINE TEST")
    print("=" * 30)
    
    tests = [
        test_imports,
        test_config,
        test_data_processing,
        test_benchmark
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
    print("\n" + "=" * 30)
    print("📋 SUMMARY")
    print("=" * 30)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All basic tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed.")
        return 1

if __name__ == "__main__":
    exit(main()) 