"""
Setup script for the LLM Fine-tuning Pipeline package.
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
install_requires = []
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        install_requires = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]

setup(
    name="llm-fine-tuning",
    version="1.0.0",
    author="LLM Fine-tuning Team",
    author_email="team@example.com",
    description="A comprehensive pipeline for fine-tuning small language models for domain-specific tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/llm-fine-tuning",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "pre-commit>=2.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-fine-tuning=llm_fine_tuning.pipeline.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
