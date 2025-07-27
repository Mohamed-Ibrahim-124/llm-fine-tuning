#!/usr/bin/env python3
"""
Script to switch between mock and real components.
"""

import os
import sys
from pathlib import Path


def switch_to_real_components():
    """Switch to real component implementations."""
    print("🔄 Switching to real components...")

    # Set environment variables
    os.environ["USE_REAL_PDF_EXTRACTION"] = "true"
    os.environ["USE_REAL_WEB_SCRAPING"] = "true"
    os.environ["USE_REAL_AUGMENTATION"] = "true"

    print("✅ Switched to real components!")
    print("   - Real PDF extraction enabled")
    print("   - Advanced web scraping enabled")
    print("   - LLM-based augmentation enabled")


def switch_to_mock_components():
    """Switch to mock component implementations."""
    print("🔄 Switching to mock components...")

    # Set environment variables
    os.environ["USE_REAL_PDF_EXTRACTION"] = "false"
    os.environ["USE_REAL_WEB_SCRAPING"] = "false"
    os.environ["USE_REAL_AUGMENTATION"] = "false"

    print("✅ Switched to mock components!")
    print("   - Mock PDF extraction enabled")
    print("   - Basic web scraping enabled")
    print("   - Rule-based augmentation enabled")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "real":
        switch_to_real_components()
    else:
        switch_to_mock_components()
