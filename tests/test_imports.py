"""
Simple test to verify all imports work correctly.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_basic_imports():
    """Test that basic imports work."""
    try:
        from llm_fine_tuning.config.settings import get_config
        from llm_fine_tuning.data.processing.augmenter import augment_data
        from llm_fine_tuning.data.processing.cleaner import clean_data
        from llm_fine_tuning.data.processing.splitter import split_data
        from llm_fine_tuning.evaluation.benchmark_generator import (
            create_benchmark_dataset,
        )
        from llm_fine_tuning.evaluation.performance_monitor import (
            create_performance_monitor,
        )
        from llm_fine_tuning.models.fine_tuner import fine_tune_model

        print("✅ All imports successful")
        assert True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        assert False, f"Import failed: {e}"


def test_config_import():
    """Test configuration import specifically."""
    try:
        from llm_fine_tuning.config.settings import get_config

        config = get_config()
        assert config is not None
        print("✅ Config import successful")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        assert False, f"Config import failed: {e}"


def test_models_import():
    """Test models import specifically."""
    try:
        from llm_fine_tuning.models.fine_tuner import fine_tune_model

        assert fine_tune_model is not None
        print("✅ Models import successful")
    except Exception as e:
        print(f"❌ Models import failed: {e}")
        assert False, f"Models import failed: {e}"


if __name__ == "__main__":
    test_basic_imports()
    test_config_import()
    test_models_import()
    print("🎉 All import tests passed!")
