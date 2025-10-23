#!/usr/bin/env python3
"""
INTELLIGENT WEB SCRAPER - Phase 3 Implementation
AI-powered adaptive web scraping with content understanding and anti-detection
"""

import asyncio
import json
import re
import time
import random
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from urllib.parse import urljoin, urlparse
from pathlib import Path

# Web scraping and AI imports
try:
    import requests
    from requests_html import HTMLSession, AsyncHTMLSession
    from bs4 import BeautifulSoup
    import selenium
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    import torch
    INTELLIGENT_SCRAPING_AVAILABLE = True
except ImportError as e:
    logging.error(f"Intelligent web scraping dependencies not available: {e}")
    INTELLIGENT_SCRAPING_AVAILABLE = False

from loguru import logger

class ContentType(Enum):
    """Content types for intelligent classification"""
    NEWS_ARTICLE = "news_article"
    RESEARCH_PAPER = "research_paper"
    FINANCIAL_DATA = "financial_data"
    PRODUCT_INFO = "product_info"
    SOCIAL_MEDIA = "social_media"
    FORUM_POST = "forum_post"
    COMPANY_INFO = "company_info"
    MARKET_DATA = "market_data"
    TECHNICAL_DOC = "technical_doc"
    UNKNOWN = "unknown"

class ScrapingStrategy(Enum):
    """Scraping strategies for different content types"""
    BASIC_HTTP = "basic_http"
    JAVASCRIPT_RENDER = "javascript_render"
    BROWSER_AUTOMATION = "browser_automation"
    API_EXTRACTION = "api_extraction"
    HYBRID_APPROACH = "hybrid_approach"

class ContentQuality(Enum):
    """Content quality assessment"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SPAM = "spam"

@dataclass
class ScrapingTarget:
    """Target URL and associated metadata"""
    url: str
    content_type: ContentType
    priority: int  # 1-10 scale
    expected_update_frequency: timedelta
    last_scraped: Optional[datetime]
    scraping_strategy: ScrapingStrategy
    custom_selectors: Dict[str, str]
    headers: Dict[str, str]
    cookies: Dict[str, str]

@dataclass
class ScrapedContent:
    """Scraped content with AI-generated metadata"""
    url: str
    title: str
    content: str
    extracted_data: Dict[str, Any]
    content_type: ContentType
    content_quality: ContentQuality
    confidence_score: float
    scraped_at: datetime
    processing_time: float
    word_count: int
    language: str
    key_entities: List[str]
    summary: str

@dataclass
class ScrapingResult:
    """Complete scraping operation result"""
    target: ScrapingTarget
    success: bool
    content: Optional[ScrapedContent]
    error_message: Optional[str]
    response_time: float
    status_code: Optional[int]
    content_changed: bool
    next_scrape_recommended: datetime

class IntelligentWebScraper:
    """AI-powered adaptive web scraper with content understanding"""
    
    def __init__(self):
        self.session = None
        self.async_session = None
        self.browser = None
        
        # AI Components
        self.content_classifier = None
        self.quality_assessor = None
        self.text_vectorizer = None
        self.strategy_predictor = None
        
        # Anti-detection mechanisms
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]
        
        # Content patterns for classification
        self.content_patterns = {
            ContentType.NEWS_ARTICLE: [
                r'/b(?:breaking|news|report|update|announced|confirms)/b',
                r'/b(?:source|reporter|journalist|correspondent)/b',
                r'/b(?:today|yesterday|this week|this month)/b'
            ],
            ContentType.FINANCIAL_DATA: [
                r'/$[/d,]+/.?/d*[MBK]?',
                r'/b(?:stock|shares|market|trading|price|revenue|profit|loss)/b',
                r'/b(?:NYSE|NASDAQ|S&P|Dow|index)/b'
            ],
            ContentType.RESEARCH_PAPER: [
                r'/b(?:abstract|methodology|results|conclusion|references)/b',
                r'/b(?:study|research|analysis|findings|hypothesis)/b',
                r'/b(?:published|journal|peer.reviewed|citation)/b'
            ],
            ContentType.COMPANY_INFO: [
                r'/b(?:company|corporation|inc|ltd|llc)/b',
                r'/b(?:founded|headquarters|employees|revenue|CEO)/b',
                r'/b(?:products|services|mission|vision)/b'
            ]
        }
        
        # Quality indicators
        self.quality_indicators = {
            'high_quality': [
                r'/b(?:source|citation|reference|study|data|research)/b',
                r'/b(?:published|authored|verified|confirmed)/b',
                r'/b/d{4}[-/]/d{2}[-/]/d{2}/b',  # Date patterns
            ],
            'low_quality': [
                r'/b(?:click here|amazing|incredible|you won/'t believe)/b',
                r'/b(?:spam|advertisement|promotion|affiliate)/b',
                r'[!]{3,}|[?]{3,}'  # Excessive punctuation
            ]
        }
        
        # Scraping statistics
        self.stats = {
            'total_requests': 0,
            'successful_scrapes': 0,
            'failed_scrapes': 0,
            'content_types_discovered': {},
            'average_response_time': 0,
            'last_reset': datetime.now()
        }
        
        if INTELLIGENT_SCRAPING_AVAILABLE:
            self._initialize_components()
    
    def _initialize_components(self):
        """Initialize AI components and web scraping tools"""
        try:
            logger.info("Initializing intelligent web scraper components...")
            
            # Initialize sessions
            self.session = HTMLSession()
            self.async_session = AsyncHTMLSession()
            
            # Set up anti-detection headers
            default_headers = {
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            self.session.headers.update(default_headers)
            
            # Initialize AI components
            self.text_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Initialize content classifier
            self.content_classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            
            # Initialize quality assessor
            self.quality_assessor = RandomForestClassifier(
                n_estimators=50,
                random_state=42
            )
            
            logger.info("Intelligent web scraper initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing intelligent web scraper: {e}")
    
    async def scrape_intelligently(self, target: ScrapingTarget) -> ScrapingResult:
        """Perform intelligent scraping with adaptive strategy selection"""
        start_time = time.time()
        
        try:
            # 1. Analyze target and select optimal strategy
            optimal_strategy = await self._select_scraping_strategy(target)
            
            # 2. Apply anti-detection measures
            await self._apply_anti_detection(target)
            
            # 3. Execute scraping with selected strategy
            raw_content = await self._execute_scraping(target, optimal_strategy)
            
            if not raw_content:
                return ScrapingResult(
                    target=target,
                    success=False,
                    content=None,
                    error_message="Failed to retrieve content",
                    response_time=time.time() - start_time,
                    status_code=None,
                    content_changed=False,
                    next_scrape_recommended=datetime.now() + target.expected_update_frequency
                )
            
            # 4. Process and understand content with AI
            processed_content = await self._process_content_with_ai(target, raw_content)
            
            # 5. Assess content quality and relevance
            quality_score = await self._assess_content_quality(processed_content)
            
            # 6. Update scraping statistics and learning
            await self._update_learning_data(target, processed_content, True)
            
            # 7. Determine if content has changed
            content_changed = await self._detect_content_changes(target, processed_content)
            
            # 8. Calculate next optimal scrape time
            next_scrape = await self._predict_next_scrape_time(target, content_changed)
            
            return ScrapingResult(
                target=target,
                success=True,
                content=processed_content,
                error_message=None,
                response_time=time.time() - start_time,
                status_code=200,
                content_changed=content_changed,
                next_scrape_recommended=next_scrape
            )
            
        except Exception as e:
            logger.error(f"Error in intelligent scraping for {target.url}: {e}")
            
            await self._update_learning_data(target, None, False)
            
            return ScrapingResult(
                target=target,
                success=False,
                content=None,
                error_message=str(e),
                response_time=time.time() - start_time,
                status_code=None,
                content_changed=False,
                next_scrape_recommended=datetime.now() + target.expected_update_frequency * 2
            )
    
    async def _select_scraping_strategy(self, target: ScrapingTarget) -> ScrapingStrategy:
        """Use AI to select optimal scraping strategy"""
        
        # If strategy is explicitly set, use it
        if target.scraping_strategy != ScrapingStrategy.BASIC_HTTP:
            return target.scraping_strategy
        
        # Analyze URL patterns to predict best strategy
        url_features = self._extract_url_features(target.url)
        
        # Simple rule-based strategy selection (can be enhanced with ML)
        if any(js_indicator in target.url.lower() for js_indicator in ['spa', 'react', 'angular', 'vue']):
            return ScrapingStrategy.JAVASCRIPT_RENDER
        
        elif any(api_indicator in target.url.lower() for api_indicator in ['api', 'json', 'rest']):
            return ScrapingStrategy.API_EXTRACTION
        
        elif target.content_type in [ContentType.SOCIAL_MEDIA, ContentType.FORUM_POST]:
            return ScrapingStrategy.BROWSER_AUTOMATION
        
        else:
            return ScrapingStrategy.BASIC_HTTP
    
    def _extract_url_features(self, url: str) -> Dict[str, Any]:
        """Extract features from URL for strategy prediction"""
        parsed = urlparse(url)
        
        return {
            'domain': parsed.netloc,
            'path_depth': len([p for p in parsed.path.split('/') if p]),
            'has_query': bool(parsed.query),
            'subdomain_count': len(parsed.netloc.split('.')) - 2,
            'url_length': len(url),
            'https': url.startswith('https')
        }
    
    async def _apply_anti_detection(self, target: ScrapingTarget):
        """Apply anti-detection measures"""
        
        # Randomize user agent
        user_agent = random.choice(self.user_agents)
        self.session.headers.update({'User-Agent': user_agent})
        
        # Random delays to mimic human behavior
        delay = random.uniform(1, 3)
        await asyncio.sleep(delay)
        
        # Apply custom headers if provided
        if target.headers:
            self.session.headers.update(target.headers)
        
        # Apply cookies if provided
        if target.cookies:
            for name, value in target.cookies.items():
                self.session.cookies.set(name, value)
    
    async def _execute_scraping(self, target: ScrapingTarget, strategy: ScrapingStrategy) -> Optional[Dict[str, Any]]:
        """Execute scraping based on selected strategy"""
        
        try:
            if strategy == ScrapingStrategy.BASIC_HTTP:
                return await self._scrape_with_requests(target)
            
            elif strategy == ScrapingStrategy.JAVASCRIPT_RENDER:
                return await self._scrape_with_js_render(target)
            
            elif strategy == ScrapingStrategy.BROWSER_AUTOMATION:
                return await self._scrape_with_browser(target)
            
            elif strategy == ScrapingStrategy.API_EXTRACTION:
                return await self._scrape_api_endpoint(target)
            
            else:
                # Hybrid approach - try multiple strategies
                return await self._scrape_hybrid(target)
                
        except Exception as e:
            logger.error(f"Scraping execution failed with strategy {strategy}: {e}")
            return None
    
    async def _scrape_with_requests(self, target: ScrapingTarget) -> Dict[str, Any]:
        """Basic HTTP scraping with requests"""
        
        response = self.session.get(target.url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract basic content
        title = ""
        if soup.title:
            title = soup.title.string.strip() if soup.title.string else ""
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text_content = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text_content.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text_content = ' '.join(chunk for chunk in chunks if chunk)
        
        return {
            'title': title,
            'content': text_content,
            'html': str(soup),
            'status_code': response.status_code,
            'headers': dict(response.headers),
            'url': response.url
        }
    
    async def _scrape_with_js_render(self, target: ScrapingTarget) -> Dict[str, Any]:
        """JavaScript rendering scraping"""
        
        r = await self.async_session.get(target.url)
        await r.html.arender(timeout=20)
        
        # Extract content after JS rendering
        title = ""
        if r.html.find('title', first=True):
            title = r.html.find('title', first=True).text
        
        # Get rendered text
        text_content = r.html.text
        
        return {
            'title': title,
            'content': text_content,
            'html': r.html.html,
            'status_code': r.status_code,
            'headers': dict(r.headers),
            'url': str(r.url)
        }
    
    async def _scrape_with_browser(self, target: ScrapingTarget) -> Dict[str, Any]:
        """Browser automation scraping (Selenium)"""
        
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument(f'--user-agent={random.choice(self.user_agents)}')
        
        driver = webdriver.Chrome(options=options)
        
        try:
            driver.get(target.url)
            
            # Wait for page to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Get page content
            title = driver.title
            content = driver.find_element(By.TAG_NAME, "body").text
            html = driver.page_source
            
            return {
                'title': title,
                'content': content,
                'html': html,
                'status_code': 200,
                'headers': {},
                'url': driver.current_url
            }
            
        finally:
            driver.quit()
    
    async def _scrape_api_endpoint(self, target: ScrapingTarget) -> Dict[str, Any]:
        """API endpoint scraping"""
        
        headers = {'Accept': 'application/json'}
        headers.update(target.headers)
        
        response = self.session.get(target.url, headers=headers)
        response.raise_for_status()
        
        try:
            json_data = response.json()
            
            # Convert JSON to text for processing
            content = json.dumps(json_data, indent=2)
            
            return {
                'title': f"API Data from {urlparse(target.url).netloc}",
                'content': content,
                'html': '',
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'url': response.url,
                'json_data': json_data
            }
            
        except json.JSONDecodeError:
            # Fallback to text content
            return {
                'title': f"Data from {urlparse(target.url).netloc}",
                'content': response.text,
                'html': '',
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'url': response.url
            }
    
    async def _scrape_hybrid(self, target: ScrapingTarget) -> Dict[str, Any]:
        """Hybrid scraping approach - try multiple strategies"""
        
        strategies = [
            ScrapingStrategy.BASIC_HTTP,
            ScrapingStrategy.JAVASCRIPT_RENDER,
            ScrapingStrategy.BROWSER_AUTOMATION
        ]
        
        for strategy in strategies:
            try:
                result = await self._execute_scraping(target, strategy)
                if result and result.get('content'):
                    logger.info(f"Hybrid scraping succeeded with strategy: {strategy}")
                    return result
            except Exception as e:
                logger.warning(f"Hybrid strategy {strategy} failed: {e}")
                continue
        
        raise Exception("All hybrid scraping strategies failed")
    
    async def _process_content_with_ai(self, target: ScrapingTarget, raw_content: Dict[str, Any]) -> ScrapedContent:
        """Process scraped content with AI understanding"""
        
        title = raw_content.get('title', '')
        content = raw_content.get('content', '')
        
        # 1. Classify content type
        detected_type = await self._classify_content_type(title, content)
        
        # 2. Extract key entities
        entities = await self._extract_entities(content)
        
        # 3. Generate summary
        summary = await self._generate_content_summary(content)
        
        # 4. Assess language
        language = await self._detect_language(content)
        
        # 5. Calculate confidence score
        confidence = await self._calculate_confidence_score(content, detected_type)
        
        # 6. Assess quality
        quality = await self._assess_content_quality_level(content)
        
        return ScrapedContent(
            url=target.url,
            title=title,
            content=content,
            extracted_data=raw_content,
            content_type=detected_type,
            content_quality=quality,
            confidence_score=confidence,
            scraped_at=datetime.now(),
            processing_time=0.0,  # Will be set by caller
            word_count=len(content.split()),
            language=language,
            key_entities=entities,
            summary=summary
        )
    
    async def _classify_content_type(self, title: str, content: str) -> ContentType:
        """Classify content type using pattern matching and AI"""
        
        text = f"{title} {content}".lower()
        
        # Pattern-based classification
        type_scores = {}
        
        for content_type, patterns in self.content_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches
            
            type_scores[content_type] = score
        
        # Find best match
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            if type_scores[best_type] > 0:
                return best_type
        
        return ContentType.UNKNOWN
    
    async def _extract_entities(self, content: str) -> List[str]:
        """Extract key entities from content"""
        
        # Simple entity extraction using patterns
        entities = []
        
        # Extract organizations (capitalized words)
        org_pattern = r'/b[A-Z][A-Za-z]+(?:/s+[A-Z][A-Za-z]+)*(?:/s+(?:Inc|Corp|LLC|Ltd))/b'
        entities.extend(re.findall(org_pattern, content))
        
        # Extract monetary amounts
        money_pattern = r'/$[/d,]+(?:/./d{2})?(?:/s*(?:million|billion|trillion))?'
        entities.extend(re.findall(money_pattern, content, re.IGNORECASE))
        
        # Extract dates
        date_pattern = r'/b/d{1,2}[-/]/d{1,2}[-/]/d{4}/b|/b/d{4}[-/]/d{1,2}[-/]/d{1,2}/b'
        entities.extend(re.findall(date_pattern, content))
        
        # Extract percentages
        percent_pattern = r'/d+(?:/./d+)?%'
        entities.extend(re.findall(percent_pattern, content))
        
        return list(set(entities))[:20]  # Return unique entities, max 20
    
    async def _generate_content_summary(self, content: str) -> str:
        """Generate content summary"""
        
        sentences = content.split('. ')
        if len(sentences) <= 3:
            return content
        
        # Simple extractive summarization - take first and most informative sentences
        first_sentence = sentences[0] if sentences else ""
        
        # Find sentence with most entities/numbers
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences[1:6]:  # Check first 5 sentences after first
            score = len(re.findall(r'/b[A-Z][A-Za-z]+/b', sentence))  # Count capitalized words
            score += len(re.findall(r'/d+', sentence))  # Count numbers
            
            if score > best_score:
                best_score = score
                best_sentence = sentence
        
        summary = first_sentence
        if best_sentence and best_sentence != first_sentence:
            summary += f". {best_sentence}"
        
        return summary[:500]  # Limit to 500 characters
    
    async def _detect_language(self, content: str) -> str:
        """Simple language detection"""
        
        # Basic English detection - can be enhanced with proper language detection
        english_indicators = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        
        words = content.lower().split()[:100]  # Check first 100 words
        english_count = sum(1 for word in words if word in english_indicators)
        
        if english_count > len(words) * 0.1:  # If more than 10% are common English words
            return "en"
        else:
            return "unknown"
    
    async def _calculate_confidence_score(self, content: str, content_type: ContentType) -> float:
        """Calculate confidence score for classification"""
        
        base_score = 0.5
        
        # Content length factor
        if len(content.split()) > 50:
            base_score += 0.2
        
        # Type-specific patterns
        if content_type != ContentType.UNKNOWN:
            patterns = self.content_patterns.get(content_type, [])
            matches = sum(len(re.findall(pattern, content.lower())) for pattern in patterns)
            base_score += min(0.3, matches * 0.05)
        
        # Structure indicators
        if any(indicator in content.lower() for indicator in ['title', 'author', 'date', 'source']):
            base_score += 0.1
        
        return min(1.0, base_score)
    
    async def _assess_content_quality_level(self, content: str) -> ContentQuality:
        """Assess content quality"""
        
        text = content.lower()
        
        # Check for spam indicators
        spam_score = 0
        for pattern in self.quality_indicators['low_quality']:
            spam_score += len(re.findall(pattern, text))
        
        if spam_score > 3:
            return ContentQuality.SPAM
        
        # Check for quality indicators
        quality_score = 0
        for pattern in self.quality_indicators['high_quality']:
            quality_score += len(re.findall(pattern, text))
        
        # Length and structure assessment
        word_count = len(content.split())
        
        if quality_score > 3 and word_count > 200:
            return ContentQuality.HIGH
        elif quality_score > 1 and word_count > 100:
            return ContentQuality.MEDIUM
        else:
            return ContentQuality.LOW
    
    async def _assess_content_quality(self, content: ScrapedContent) -> float:
        """Assess overall content quality score"""
        
        score = 0.0
        
        # Quality level factor
        quality_weights = {
            ContentQuality.HIGH: 1.0,
            ContentQuality.MEDIUM: 0.7,
            ContentQuality.LOW: 0.4,
            ContentQuality.SPAM: 0.0
        }
        score += quality_weights.get(content.content_quality, 0.5) * 0.4
        
        # Confidence factor
        score += content.confidence_score * 0.3
        
        # Content richness factor
        if content.word_count > 200:
            score += 0.1
        if len(content.key_entities) > 5:
            score += 0.1
        if content.summary:
            score += 0.1
        
        return min(1.0, score)
    
    async def _detect_content_changes(self, target: ScrapingTarget, content: ScrapedContent) -> bool:
        """Detect if content has changed since last scrape"""
        
        # For now, assume content has changed if it's a new scrape
        # In production, this would compare against stored content hashes
        return True
    
    async def _predict_next_scrape_time(self, target: ScrapingTarget, content_changed: bool) -> datetime:
        """Predict optimal next scrape time"""
        
        base_interval = target.expected_update_frequency
        
        # Adjust based on content change patterns
        if content_changed:
            # Content is actively updating - scrape more frequently
            multiplier = 0.8
        else:
            # Content stable - scrape less frequently
            multiplier = 1.5
        
        # Add some randomization to avoid detection
        randomization = random.uniform(0.9, 1.1)
        
        next_interval = base_interval * multiplier * randomization
        
        return datetime.now() + next_interval
    
    async def _update_learning_data(self, target: ScrapingTarget, content: Optional[ScrapedContent], success: bool):
        """Update learning data for continuous improvement"""
        
        self.stats['total_requests'] += 1
        
        if success:
            self.stats['successful_scrapes'] += 1
            
            if content:
                content_type = content.content_type.value
                self.stats['content_types_discovered'][content_type] = /
                    self.stats['content_types_discovered'].get(content_type, 0) + 1
        else:
            self.stats['failed_scrapes'] += 1
    
    def get_scraping_statistics(self) -> Dict[str, Any]:
        """Get comprehensive scraping statistics"""
        
        total = self.stats['total_requests']
        success_rate = (self.stats['successful_scrapes'] / total * 100) if total > 0 else 0
        
        return {
            'total_requests': total,
            'successful_scrapes': self.stats['successful_scrapes'],
            'failed_scrapes': self.stats['failed_scrapes'],
            'success_rate': f"{success_rate:.1f}%",
            'content_types_discovered': self.stats['content_types_discovered'],
            'average_response_time': self.stats['average_response_time'],
            'last_reset': self.stats['last_reset'],
            'uptime_hours': (datetime.now() - self.stats['last_reset']).total_seconds() / 3600
        }
    
    async def batch_scrape(self, targets: List[ScrapingTarget], max_concurrent: int = 5) -> List[ScrapingResult]:
        """Perform batch scraping with concurrency control"""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_with_semaphore(target):
            async with semaphore:
                return await self.scrape_intelligently(target)
        
        tasks = [scrape_with_semaphore(target) for target in targets]
        results = await asyncio.gather(*tasks)
        
        return results
    
    def __del__(self):
        """Cleanup resources"""
        try:
            if self.session:
                self.session.close()
            if self.async_session:
                asyncio.create_task(self.async_session.close())
        except:
            pass

# Global instance
intelligent_web_scraper = IntelligentWebScraper()

def get_intelligent_web_scraper() -> IntelligentWebScraper:
    """Get the global intelligent web scraper instance"""
    return intelligent_web_scraper