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
        Scrape content from a single URL.

        Args:
            url: URL to scrape

        Returns:
            Dictionary containing scraped content and metadata
        """
        try:
            logger.info(f"Scraping URL: {url}")

            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()

            # Parse HTML content
            soup = BeautifulSoup(response.content, "html.parser")

            # Extract text content
            text_content = soup.get_text(separator=" ", strip=True)

            # Clean up text
            text_content = " ".join(text_content.split())

            # Create result object
            result = {
                "url": url,
                "text": text_content,
                "source": url,
                "title": soup.title.string if soup.title else "No title",
                "status": "success",
            }

            logger.info(f"Successfully scraped: {url}")
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
