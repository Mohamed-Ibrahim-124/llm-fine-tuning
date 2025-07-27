"""
Web scraping module for collecting domain-specific data.

This module provides functionality to scrape web content from URLs
using requests and BeautifulSoup with proper error handling and logging.
"""

from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup

from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class WebScraper:
    """Web scraper class for collecting content from URLs."""

    def __init__(self, timeout: int = 10, user_agent: str = None):
        """
        Initialize the web scraper.

        Args:
            timeout: Request timeout in seconds
            user_agent: Custom user agent string
        """
        self.timeout = timeout
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
        self.headers = {"User-Agent": self.user_agent}

    def scrape_url(self, url: str) -> Dict[str, Any]:
        """
        Scrape content from a single URL using crawl4ai.

        Args:
            url: URL to scrape

        Returns:
            Dictionary containing scraped content and metadata
        """
        try:
            logger.info(f"Scraping URL with crawl4ai: {url}")

            # Use crawl4ai for AI-powered web scraping (no API key required)
            import asyncio

            import crawl4ai
            from crawl4ai import AsyncWebCrawler

            # Initialize crawler without API key
            crawler = AsyncWebCrawler()

            # Perform the crawl using basic extraction
            async def crawl():
                result = await crawler.arun(
                    url=url,
                    # Use basic extraction without LLM
                    extraction_strategy="markdown",
                )
                return result

            crawl_result = asyncio.run(crawl())

            # Extract content from result
            if crawl_result.success:
                text_content = crawl_result.extracted_content.get("text", "")
                title = crawl_result.extracted_content.get("title", "No title")

                result = {
                    "url": url,
                    "text": text_content,
                    "source": url,
                    "title": title,
                    "status": "success",
                    "metadata": {
                        "extraction_method": "crawl4ai",
                        "crawl_status": "success",
                    },
                }
            else:
                result = {
                    "url": url,
                    "text": f"Failed to scrape content from {url}",
                    "source": url,
                    "title": "Error",
                    "status": "error",
                    "error": crawl_result.error_message or "Unknown error",
                }

            logger.info(f"Successfully scraped with crawl4ai: {url}")
            return result

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to scrape {url}: {str(e)}")
            return {
                "url": url,
                "text": f"Failed to scrape content from {url}: {str(e)}",
                "source": url,
                "title": "Error",
                "status": "error",
                "error": str(e),
            }
        except Exception as e:
            logger.error(f"Unexpected error scraping {url}: {str(e)}")
            return {
                "url": url,
                "text": f"Unexpected error scraping {url}: {str(e)}",
                "source": url,
                "title": "Error",
                "status": "error",
                "error": str(e),
            }

    def scrape_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Scrape content from multiple URLs.

        Args:
            urls: List of URLs to scrape

        Returns:
            List of dictionaries containing scraped content
        """
        logger.info(f"Starting web scraping for {len(urls)} URLs")
        results = []

        for url in urls:
            result = self.scrape_url(url)
            results.append(result)

        successful_scrapes = sum(1 for r in results if r["status"] == "success")
        logger.info(
            f"Web scraping completed: {successful_scrapes}/{len(urls)} successful"
        )

        return results


def scrape_web(urls: List[str]) -> List[Dict[str, Any]]:
    """
    Scrape web content from URLs using requests and BeautifulSoup.

    Args:
        urls: List of URLs to scrape

    Returns:
        List of dictionaries containing scraped content and metadata
    """
    scraper = WebScraper()
    return scraper.scrape_urls(urls)


def scrape_urls(urls: List[str]) -> List[Dict[str, Any]]:
    """
    Alias for scrape_web for backward compatibility.
    """
    return scrape_web(urls)
