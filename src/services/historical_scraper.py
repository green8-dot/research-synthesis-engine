"""
Historical Web Scraping Service for Knowledge Base Building
Scrapes historical articles and content to build comprehensive knowledge base
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class HistoricalSource:
    """Configuration for historical data sources"""
    name: str
    base_url: str
    archive_pattern: str  # URL pattern for accessing archives
    date_format: str  # How dates appear in URLs
    content_selector: str  # CSS selector for main content
    title_selector: str  # CSS selector for article title
    date_selector: str  # CSS selector for article date
    max_pages: int = 50  # Maximum pages to scrape per source
    delay_seconds: float = 2.0  # Delay between requests

class HistoricalScraper:
    """Scrapes historical content to build knowledge base"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.scraped_urls = set()
        
        # Configure historical sources for space industry
        self.sources = [
            HistoricalSource(
                name="SpaceNews Archives",
                base_url="https://spacenews.com",
                archive_pattern="/page/{page}/",
                date_format="%Y-%m-%d",
                content_selector="article .entry-content",
                title_selector="h1.entry-title",
                date_selector=".entry-meta time",
                max_pages=100,
                delay_seconds=2.0
            ),
            HistoricalSource(
                name="SpaceflightNow Archives",
                base_url="https://spaceflightnow.com",
                archive_pattern="/news/page/{page}/",
                date_format="%Y/%m/%d",
                content_selector=".post-content",
                title_selector="h1.post-title",
                date_selector=".post-meta .date",
                max_pages=75,
                delay_seconds=2.5
            ),
            HistoricalSource(
                name="Space.com Archives", 
                base_url="https://space.com",
                archive_pattern="/news/page/{page}",
                date_format="%Y-%m-%d",
                content_selector="article .content",
                title_selector="h1#article-title",
                date_selector=".byline time",
                max_pages=80,
                delay_seconds=1.5
            ),
            # NASA Archives (public domain)
            HistoricalSource(
                name="NASA News Archives",
                base_url="https://nasa.gov",
                archive_pattern="/news/page/{page}/",
                date_format="%Y/%m/%d",
                content_selector=".entry-content",
                title_selector="h1.entry-title",
                date_selector=".entry-date",
                max_pages=200,
                delay_seconds=1.0
            ),
            # ESA Archives
            HistoricalSource(
                name="ESA News Archives",
                base_url="https://esa.int",
                archive_pattern="/News/page/{page}",
                date_format="%Y-%m-%d",
                content_selector=".article-content",
                title_selector="h1.article-title",
                date_selector=".article-meta .date",
                max_pages=50,
                delay_seconds=2.0
            )
        ]
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Research Synthesis Engine Historical Scraper 1.0 (Educational/Research)'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def scrape_historical_data(self, 
                                   sources: List[str] = None,
                                   date_range_days: int = 365,
                                   max_articles_per_source: int = 100) -> Dict[str, Any]:
        """
        Scrape historical data from specified sources
        
        Args:
            sources: List of source names to scrape (None for all)
            date_range_days: How many days back to scrape
            max_articles_per_source: Maximum articles per source
            
        Returns:
            Dictionary with scraping results
        """
        start_time = datetime.now()
        results = {
            "started_at": start_time.isoformat(),
            "sources_scraped": [],
            "total_articles_found": 0,
            "total_articles_processed": 0,
            "errors": [],
            "performance_stats": {}
        }
        
        # Filter sources if specified
        target_sources = self.sources
        if sources:
            target_sources = [s for s in self.sources if s.name in sources]
        
        logger.info(f"Starting historical scraping of {len(target_sources)} sources")
        
        # Scrape each source
        for source in target_sources:
            logger.info(f"Scraping historical data from {source.name}")
            
            try:
                source_results = await self._scrape_source_historical(
                    source, date_range_days, max_articles_per_source
                )
                
                results["sources_scraped"].append({
                    "name": source.name,
                    "articles_found": source_results["articles_found"],
                    "articles_processed": source_results["articles_processed"],
                    "processing_time_seconds": source_results["processing_time"],
                    "errors": source_results["errors"]
                })
                
                results["total_articles_found"] += source_results["articles_found"]
                results["total_articles_processed"] += source_results["articles_processed"]
                results["errors"].extend(source_results["errors"])
                
            except Exception as e:
                error_msg = f"Failed to scrape {source.name}: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
        
        # Calculate final stats
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        results.update({
            "completed_at": end_time.isoformat(),
            "total_duration_seconds": total_duration,
            "average_articles_per_minute": (results["total_articles_processed"] / (total_duration / 60)) if total_duration > 0 else 0,
            "success_rate": (results["total_articles_processed"] / max(results["total_articles_found"], 1)) * 100
        })
        
        logger.info(f"Historical scraping completed: {results['total_articles_processed']} articles in {total_duration:.1f}s")
        return results
    
    async def _scrape_source_historical(self, 
                                      source: HistoricalSource,
                                      date_range_days: int,
                                      max_articles: int) -> Dict[str, Any]:
        """Scrape historical articles from a single source"""
        source_start = datetime.now()
        articles_found = 0
        articles_processed = 0
        errors = []
        
        try:
            # Generate archive URLs to scrape
            archive_urls = self._generate_archive_urls(source, date_range_days)
            
            for page_num, archive_url in enumerate(archive_urls[:source.max_pages], 1):
                if articles_processed >= max_articles:
                    break
                
                try:
                    logger.debug(f"Scraping {source.name} page {page_num}: {archive_url}")
                    
                    # Fetch archive page
                    async with self.session.get(archive_url) as response:
                        if response.status != 200:
                            logger.warning(f"Archive page returned {response.status}: {archive_url}")
                            continue
                        
                        content = await response.text()
                        
                    # Extract article links from archive page
                    article_links = self._extract_article_links(content, source.base_url)
                    articles_found += len(article_links)
                    
                    # Process each article
                    for article_url in article_links:
                        if articles_processed >= max_articles:
                            break
                            
                        if article_url in self.scraped_urls:
                            continue  # Skip already processed URLs
                        
                        try:
                            article_data = await self._scrape_article(article_url, source)
                            if article_data:
                                # Store article in database
                                await self._store_article(article_data)
                                articles_processed += 1
                                self.scraped_urls.add(article_url)
                                
                                logger.debug(f"Processed article: {article_data.get('title', 'Untitled')}")
                        
                        except Exception as e:
                            error_msg = f"Failed to process article {article_url}: {str(e)}"
                            errors.append(error_msg)
                            logger.debug(error_msg)
                    
                    # Respect rate limiting
                    await asyncio.sleep(source.delay_seconds)
                
                except Exception as e:
                    error_msg = f"Failed to process archive page {archive_url}: {str(e)}"
                    errors.append(error_msg)
                    logger.debug(error_msg)
        
        except Exception as e:
            error_msg = f"Failed to scrape source {source.name}: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
        
        processing_time = (datetime.now() - source_start).total_seconds()
        
        return {
            "articles_found": articles_found,
            "articles_processed": articles_processed,
            "processing_time": processing_time,
            "errors": errors
        }
    
    def _generate_archive_urls(self, source: HistoricalSource, date_range_days: int) -> List[str]:
        """Generate URLs for archive pages to scrape"""
        urls = []
        
        # Simple pagination-based approach (most common)
        for page in range(1, source.max_pages + 1):
            archive_url = urljoin(source.base_url, source.archive_pattern.format(page=page))
            urls.append(archive_url)
        
        return urls
    
    def _extract_article_links(self, html_content: str, base_url: str) -> List[str]:
        """Extract article links from archive page HTML"""
        # Simple regex-based link extraction
        # In production, would use BeautifulSoup or similar
        link_pattern = r'href=["/']([^"/']*(?:news|article|post)[^"/']*)["/']'
        matches = re.findall(link_pattern, html_content, re.IGNORECASE)
        
        article_links = []
        for link in matches:
            if link.startswith('http'):
                article_links.append(link)
            elif link.startswith('/'):
                article_links.append(urljoin(base_url, link))
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(article_links))
    
    async def _scrape_article(self, url: str, source: HistoricalSource) -> Optional[Dict[str, Any]]:
        """Scrape individual article content"""
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return None
                
                content = await response.text()
            
            # Extract article data (simplified - would use BeautifulSoup in production)
            article_data = {
                "url": url,
                "source": source.name,
                "scraped_at": datetime.now().isoformat(),
                "title": self._extract_text(content, source.title_selector),
                "content": self._extract_text(content, source.content_selector),
                "published_date": self._extract_date(content, source.date_selector),
                "source_type": "historical_scrape"
            }
            
            # Validate required fields
            if not article_data["title"] or not article_data["content"]:
                return None
            
            return article_data
        
        except Exception as e:
            logger.debug(f"Failed to scrape article {url}: {e}")
            return None
    
    def _extract_text(self, html: str, selector: str) -> str:
        """Extract text content using simple regex (placeholder for BeautifulSoup)"""
        # This is a simplified implementation
        # In production, would use proper HTML parsing
        
        if selector.startswith('h1'):
            # Extract title from h1 tags
            h1_pattern = r'<h1[^>]*>(.*?)</h1>'
            matches = re.findall(h1_pattern, html, re.DOTALL | re.IGNORECASE)
            if matches:
                return re.sub(r'<[^>]+>', '', matches[0]).strip()
        
        elif selector.startswith('article') or selector.startswith('.content'):
            # Extract main content
            content_pattern = r'<(?:article|div)[^>]*class=["/'][^"/']*content[^"/']*["/'][^>]*>(.*?)</(?:article|div)>'
            matches = re.findall(content_pattern, html, re.DOTALL | re.IGNORECASE)
            if matches:
                text = re.sub(r'<[^>]+>', ' ', matches[0])
                return ' '.join(text.split()).strip()
        
        return ""
    
    def _extract_date(self, html: str, selector: str) -> Optional[str]:
        """Extract publication date from article"""
        # Simple date extraction
        date_patterns = [
            r'<time[^>]*datetime=["/']([^"/']*)["/']',
            r'<meta[^>]*property=["/']article:published_time["/'][^>]*content=["/']([^"/']*)["/']',
            r'(/d{4}-/d{2}-/d{2})',
            r'(/d{1,2}//d{1,2}//d{4})'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, html)
            if matches:
                return matches[0]
        
        return None
    
    async def _store_article(self, article_data: Dict[str, Any]):
        """Store scraped article in database"""
        try:
            from research_synthesis.database.connection import get_postgres_session
            from research_synthesis.database.models import Article
            from sqlalchemy import select
            
            async for db_session in get_postgres_session():
                # Check if article already exists
                existing = await db_session.execute(
                    select(Article).filter(Article.url == article_data["url"])
                )
                
                if existing.first():
                    logger.debug(f"Article already exists: {article_data['url']}")
                    return
                
                # Create new article
                article = Article(
                    title=article_data["title"],
                    content=article_data["content"],
                    url=article_data["url"],
                    summary=article_data["content"][:500] + "..." if len(article_data["content"]) > 500 else article_data["content"],
                    source=article_data["source"],
                    created_at=datetime.now()
                )
                
                db_session.add(article)
                await db_session.commit()
                logger.debug(f"Stored article: {article_data['title']}")
                break
        
        except Exception as e:
            logger.error(f"Failed to store article: {e}")


# Convenience functions
async def run_historical_scraping(sources: List[str] = None, 
                                days_back: int = 30,
                                max_articles_per_source: int = 50) -> Dict[str, Any]:
    """Run historical scraping with default configuration"""
    async with HistoricalScraper() as scraper:
        return await scraper.scrape_historical_data(
            sources=sources,
            date_range_days=days_back,
            max_articles_per_source=max_articles_per_source
        )

async def build_knowledge_base(target_articles: int = 500) -> Dict[str, Any]:
    """Build knowledge base with target number of articles"""
    articles_per_source = target_articles // 5  # Distribute across 5 sources
    
    async with HistoricalScraper() as scraper:
        return await scraper.scrape_historical_data(
            sources=None,  # All sources
            date_range_days=180,  # 6 months back
            max_articles_per_source=articles_per_source
        )