"""
Data collection package for web scraping and PDF extraction.
"""

from .pdf_extractor import extract_pdf
from .web_scraper import scrape_web

__all__ = ["scrape_web", "extract_pdf"] 
