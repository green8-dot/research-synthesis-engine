"""
Space News Spider - Scrapes major space industry news sources
"""
import scrapy
from scrapy import signals
from datetime import datetime
import json
from urllib.parse import urljoin, urlparse
from typing import Dict, Any, Optional
import hashlib
from loguru import logger


class SpaceNewsSpider(scrapy.Spider):
    """Spider for scraping space industry news"""
    
    name = 'space_news'
    allowed_domains = [
        'spacenews.com',
        'spaceflightnow.com',
        'space.com',
        'nasaspaceflight.com',
        'spaceref.com'
    ]
    
    start_urls = [
        'https://spacenews.com/segment/news/',
        'https://www.spaceflightnow.com/',
        'https://www.space.com/news',
        'https://www.nasaspaceflight.com/',
        'https://spaceref.com/'
    ]
    
    custom_settings = {
        'DOWNLOAD_DELAY': 2,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,
    }
    
    # Manufacturing-related keywords
    manufacturing_keywords = [
        'manufacturing', '3d printing', 'additive', 'production',
        'factory', 'assembly', 'industrial', 'fabrication',
        'microgravity', 'orbital', 'in-space', 'processing'
    ]
    
    def parse(self, response):
        """Parse main page and extract article links"""
        domain = urlparse(response.url).netloc
        
        # Domain-specific parsing
        if 'spacenews.com' in domain:
            yield from self.parse_spacenews(response)
        elif 'spaceflightnow.com' in domain:
            yield from self.parse_spaceflightnow(response)
        elif 'space.com' in domain:
            yield from self.parse_spacecom(response)
        elif 'nasaspaceflight.com' in domain:
            yield from self.parse_nasaspaceflight(response)
        elif 'spaceref.com' in domain:
            yield from self.parse_spaceref(response)
    
    def parse_spacenews(self, response):
        """Parse SpaceNews articles"""
        articles = response.css('article.post')
        
        for article in articles:
            url = article.css('h2.post-title a::attr(href)').get()
            if url:
                yield response.follow(url, self.parse_article, meta={
                    'source': 'SpaceNews',
                    'source_url': response.url
                })
        
        # Follow pagination
        next_page = response.css('a.next::attr(href)').get()
        if next_page:
            yield response.follow(next_page, self.parse)
    
    def parse_spaceflightnow(self, response):
        """Parse Spaceflight Now articles"""
        articles = response.css('div.entry')
        
        for article in articles:
            url = article.css('h2 a::attr(href)').get()
            if url:
                yield response.follow(url, self.parse_article, meta={
                    'source': 'Spaceflight Now',
                    'source_url': response.url
                })
    
    def parse_spacecom(self, response):
        """Parse Space.com articles"""
        articles = response.css('article.search-result-news')
        
        for article in articles:
            url = article.css('a.article-link::attr(href)').get()
            if url:
                yield response.follow(url, self.parse_article, meta={
                    'source': 'Space.com',
                    'source_url': response.url
                })
    
    def parse_nasaspaceflight(self, response):
        """Parse NASASpaceflight articles"""
        articles = response.css('article.post')
        
        for article in articles:
            url = article.css('h2 a::attr(href)').get()
            if url:
                yield response.follow(url, self.parse_article, meta={
                    'source': 'NASASpaceflight',
                    'source_url': response.url
                })
    
    def parse_spaceref(self, response):
        """Parse SpaceRef articles"""
        articles = response.css('div.article-item')
        
        for article in articles:
            url = article.css('h3 a::attr(href)').get()
            if url:
                yield response.follow(url, self.parse_article, meta={
                    'source': 'SpaceRef',
                    'source_url': response.url
                })
    
    def parse_article(self, response):
        """Parse individual article page"""
        source = response.meta.get('source', 'Unknown')
        
        # Extract common elements
        title = self.extract_title(response, source)
        author = self.extract_author(response, source)
        date = self.extract_date(response, source)
        content = self.extract_content(response, source)
        tags = self.extract_tags(response, source)
        
        # Check if manufacturing-related
        is_manufacturing = self.check_manufacturing_relevance(title, content, tags)
        
        # Generate unique ID
        article_id = self.generate_article_id(response.url)
        
        # Extract images
        images = response.css('article img::attr(src)').getall()
        images = [urljoin(response.url, img) for img in images]
        
        # Extract links
        external_links = self.extract_external_links(response)
        
        # Build article item
        article = {
            'id': article_id,
            'url': response.url,
            'source': source,
            'title': title,
            'author': author,
            'published_date': date,
            'content': content,
            'summary': self.generate_summary(content),
            'tags': tags,
            'categories': self.extract_categories(response, source),
            'images': images,
            'external_links': external_links,
            'is_manufacturing_related': is_manufacturing,
            'manufacturing_keywords': self.extract_manufacturing_keywords(content) if is_manufacturing else [],
            'word_count': len(content.split()) if content else 0,
            'scraped_at': datetime.utcnow().isoformat(),
            'metadata': {
                'source_domain': urlparse(response.url).netloc,
                'response_time': response.meta.get('download_latency', 0),
                'depth': response.meta.get('depth', 0)
            }
        }
        
        yield article
    
    def extract_title(self, response, source: str) -> str:
        """Extract article title based on source"""
        selectors = {
            'SpaceNews': 'h1.post-title::text',
            'Spaceflight Now': 'h1.entry-title::text',
            'Space.com': 'h1::text',
            'NASASpaceflight': 'h1.entry-title::text',
            'SpaceRef': 'h1.article-title::text'
        }
        
        selector = selectors.get(source, 'h1::text')
        title = response.css(selector).get()
        
        if not title:
            title = response.css('title::text').get()
        
        return title.strip() if title else ''
    
    def extract_author(self, response, source: str) -> str:
        """Extract article author based on source"""
        selectors = {
            'SpaceNews': 'span.author-name::text',
            'Spaceflight Now': 'span.by-author::text',
            'Space.com': 'span.by-author::text',
            'NASASpaceflight': 'span.author::text',
            'SpaceRef': 'div.author::text'
        }
        
        selector = selectors.get(source, 'span.author::text')
        author = response.css(selector).get()
        
        return author.strip() if author else ''
    
    def extract_date(self, response, source: str) -> Optional[str]:
        """Extract publication date"""
        selectors = {
            'SpaceNews': 'time::attr(datetime)',
            'Spaceflight Now': 'time.entry-date::attr(datetime)',
            'Space.com': 'time::attr(datetime)',
            'NASASpaceflight': 'time.published::attr(datetime)',
            'SpaceRef': 'span.date::text'
        }
        
        selector = selectors.get(source)
        if selector:
            date = response.css(selector).get()
            if date:
                try:
                    # Parse and standardize date format
                    if not date.startswith('20'):  # Not ISO format
                        from dateutil import parser
                        date = parser.parse(date).isoformat()
                    return date
                except:
                    pass
        
        return None
    
    def extract_content(self, response, source: str) -> str:
        """Extract article content"""
        selectors = {
            'SpaceNews': 'div.post-content p::text',
            'Spaceflight Now': 'div.entry-content p::text',
            'Space.com': 'div.article-content p::text',
            'NASASpaceflight': 'div.entry-content p::text',
            'SpaceRef': 'div.article-body p::text'
        }
        
        selector = selectors.get(source, 'article p::text')
        paragraphs = response.css(selector).getall()
        
        content = ' '.join([p.strip() for p in paragraphs if p.strip()])
        return content
    
    def extract_tags(self, response, source: str) -> list:
        """Extract article tags"""
        tags = response.css('a.tag::text').getall()
        if not tags:
            tags = response.css('a[rel="tag"]::text').getall()
        
        return [tag.strip() for tag in tags if tag.strip()]
    
    def extract_categories(self, response, source: str) -> list:
        """Extract article categories"""
        categories = response.css('a[rel="category"]::text').getall()
        if not categories:
            categories = response.css('span.category::text').getall()
        
        return [cat.strip() for cat in categories if cat.strip()]
    
    def extract_external_links(self, response) -> list:
        """Extract external links from article"""
        all_links = response.css('article a::attr(href)').getall()
        external_links = []
        
        current_domain = urlparse(response.url).netloc
        
        for link in all_links:
            if link.startswith('http'):
                link_domain = urlparse(link).netloc
                if link_domain != current_domain:
                    external_links.append(link)
        
        return list(set(external_links))
    
    def check_manufacturing_relevance(self, title: str, content: str, tags: list) -> bool:
        """Check if article is related to space manufacturing"""
        text_to_check = f"{title} {content} {' '.join(tags)}".lower()
        
        for keyword in self.manufacturing_keywords:
            if keyword in text_to_check:
                return True
        
        return False
    
    def extract_manufacturing_keywords(self, content: str) -> list:
        """Extract manufacturing-related keywords found in content"""
        found_keywords = []
        content_lower = content.lower()
        
        for keyword in self.manufacturing_keywords:
            if keyword in content_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def generate_summary(self, content: str, max_sentences: int = 3) -> str:
        """Generate a simple summary (first N sentences)"""
        if not content:
            return ''
        
        sentences = content.split('.')[:max_sentences]
        summary = '. '.join(sentences)
        
        if summary and not summary.endswith('.'):
            summary += '.'
        
        return summary
    
    def generate_article_id(self, url: str) -> str:
        """Generate unique article ID from URL"""
        return hashlib.md5(url.encode()).hexdigest()