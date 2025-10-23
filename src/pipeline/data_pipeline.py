"""
Data Processing Pipeline
Integrates scraping, NLP processing, and knowledge graph population
"""
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
from loguru import logger
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, select

from research_synthesis.config.settings import settings
from research_synthesis.database.models import Article, Entity, EntityRelationship, Source, ScrapingJob
from research_synthesis.nlp.processor import NLPProcessor
from research_synthesis.database.connection import get_postgres_session


class ArticleScraper:
    """Real article scraping functionality with configurable extraction rules"""
    
    def __init__(self, config=None):
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.config = config or self._get_default_config()
    
    def _get_default_config(self):
        """Get default scraping configuration"""
        from research_synthesis.models.scraping_config import ScrapingConfig
        return ScrapingConfig()
    
    async def scrape_source(self, source_name: str, source_url: str, max_articles: int = 5) -> List[Dict[str, Any]]:
        """Scrape articles from a specific source using configuration settings"""
        articles = []
        
        # Apply config settings
        max_articles = min(max_articles, self.config.max_articles_per_source)
        
        try:
            # Apply configured delay
            if self.config.download_delay > 0:
                await asyncio.sleep(self.config.download_delay)
            
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession() as session:
                async with session.get(source_url, headers=self.headers, timeout=timeout) as response:
                    if response.status == 200:
                        content = await response.text()
                        articles = await self._extract_articles_from_html(content, source_url, source_name, max_articles)
                        
                        # Apply content filters
                        articles = self._filter_articles_by_config(articles)
                    else:
                        logger.warning(f"Failed to scrape {source_name}: HTTP {response.status}")
                        
        except Exception as e:
            logger.error(f"Error scraping {source_name}: {e}")
        
        return articles
    
    def _filter_articles_by_config(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter articles based on configuration settings"""
        filtered = []
        
        for article in articles:
            # Check content length
            if len(article.get('content', '')) < self.config.min_content_length:
                continue
                
            # Skip duplicates if enabled
            if self.config.skip_duplicates and self._is_duplicate_content(article):
                continue
                
            # Apply language filter if set
            if self.config.language_filter and not self._matches_language_filter(article):
                continue
                
            filtered.append(article)
        
        return filtered
    
    def _is_duplicate_content(self, article: Dict[str, Any]) -> bool:
        """Check if article content is duplicate (simplified)"""
        content = article.get('content', '').lower()
        title = article.get('title', '').lower()
        
        # Skip generic/template content
        generic_phrases = [
            'space industry related content',
            'article from',
            'news from',
            'loading',
            'subscribe',
            'sign up'
        ]
        
        for phrase in generic_phrases:
            if phrase in content or phrase in title:
                return True
        
        return False
    
    def _matches_language_filter(self, article: Dict[str, Any]) -> bool:
        """Check if article matches language filter"""
        if not self.config.language_filter:
            return True
            
        content = article.get('content', '')
        title = article.get('title', '')
        text = (title + ' ' + content).lower()
        
        # Simple language detection
        if self.config.language_filter == 'en':
            english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with']
            return any(word in text for word in english_words)
        elif self.config.language_filter == 'zh':
            return any(ord(char) >= 0x4e00 and ord(char) <= 0x9fff for char in text[:200])
        elif self.config.language_filter == 'ja':
            return any(ord(char) >= 0x3040 and ord(char) <= 0x30ff for char in text[:200])
        
        return True
    
    async def _extract_articles_from_html(self, html: str, base_url: str, source_name: str, max_articles: int) -> List[Dict[str, Any]]:
        """Extract article information from HTML"""
        from bs4 import BeautifulSoup
        import re
        from urllib.parse import urljoin, urlparse
        
        articles = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Generic article extraction patterns
            article_selectors = [
                'article',
                '.article',
                '.post',
                '.news-item',
                '.story',
                'h2 a, h3 a',  # Headlines with links
                '.title a'
            ]
            
            found_articles = []
            for selector in article_selectors:
                elements = soup.select(selector)
                if elements:
                    found_articles.extend(elements[:max_articles])
                    break
            
            # If no articles found, try to find any links that look like articles
            if not found_articles:
                all_links = soup.find_all('a', href=True)
                found_articles = [link for link in all_links if self._looks_like_article_link(link.get('href', ''))][:max_articles]
            
            for i, element in enumerate(found_articles[:max_articles]):
                try:
                    article_data = await self._extract_article_data(element, base_url, source_name, i)
                    if article_data:
                        articles.append(article_data)
                except Exception as e:
                    logger.warning(f"Failed to extract article {i}: {e}")
                    
        except Exception as e:
            logger.error(f"HTML parsing error: {e}")
        
        # If no articles found, log and continue with empty list
        if not articles:
            logger.warning(f"No articles found for source {source_name}")
        
        return articles
    
    def _looks_like_article_link(self, href: str) -> bool:
        """Check if a URL looks like an article"""
        article_indicators = [
            'news', 'article', 'post', 'story', 'press-release',
            'blog', 'report', '202', '/2024/', '/2025/'
        ]
        return any(indicator in href.lower() for indicator in article_indicators)
    
    async def _extract_article_data(self, element, base_url: str, source_name: str, index: int) -> Dict[str, Any]:
        """Extract individual article data"""
        from urllib.parse import urljoin
        
        # Try to get article URL
        if element.name == 'a':
            link = element
        else:
            link = element.find('a', href=True)
        
        if not link:
            return None
        
        url = urljoin(base_url, link.get('href'))
        title = link.get_text(strip=True) or link.get('title', '')
        
        # Try to get more article content
        content = ""
        if element.name != 'a':
            # Get text from the article container
            text_content = element.get_text(strip=True)
            if len(text_content) > len(title):
                content = text_content
        
        # Use basic content if we don't have much
        if len(content) < 100:
            content = f"Article from {source_name}: {title}. Space industry related content."
        
        # Create unique article ID
        article_id = hashlib.md5(url.encode()).hexdigest()[:12]
        
        return {
            'id': article_id,
            'url': url,
            'title': title or f"Article from {source_name}",
            'content': content,
            'author': f"{source_name} Staff",
            'published_date': datetime.now(),
            'source': source_name,
            'scraped_at': datetime.now()
        }
    
    
    


class DataPipeline:
    """Main data processing pipeline"""
    
    def __init__(self):
        self.scraper = ArticleScraper(self._get_current_config())
        self.nlp_processor = NLPProcessor()
        self.db_session = None
    
    def _get_current_config(self):
        """Get current scraping configuration from API"""
        try:
            from research_synthesis.api.scraping_router import current_config
            return current_config
        except:
            from research_synthesis.models.scraping_config import ScrapingConfig
            return ScrapingConfig()
        
        # Source configuration
        self.sources = {
            "SpaceNews": "https://spacenews.com",
            "Space.com": "https://space.com", 
            "SpaceflightNow": "https://spaceflightnow.com",
            "SpaceX": "https://spacex.com/news",
            "Blue Origin": "https://blueorigin.com/news",
            "Relativity Space": "https://relativityspace.com/news",
            "arXiv": "https://arxiv.org/list/astro-ph/recent",
            "IEEE": "https://ieeexplore.ieee.org/aerospace",
            "Research Papers": "https://scholar.google.com/scholar?q=space+manufacturing",
            "NASA": "https://nasa.gov/news",
            "ESA": "https://esa.int/newsroom",
            "Space Force": "https://spaceforce.mil/news",
            # Japanese sources
            "JAXA": "https://global.jaxa.jp/press/",
            "JAXA English": "https://www.jaxa.jp/press/index_e.html",
            "MHI Space": "https://www.mhi.com/products/space/",
            "IHI Aerospace": "https://www.ihi.co.jp/ia/en/",
            "Kawasaki Heavy Industries": "https://global.kawasaki.com/en/corp/newsroom/",
            "NEC Space": "https://www.nec.com/en/global/solutions/space/",
            "Interstellar Technologies": "https://www.istellartech.com/en/news",
            "Space BD": "https://space-bd.com/en/news",
            "ALE Space": "https://ale.co.jp/en/news/",
            "Synspective": "https://synspective.com/en/news/",
            "Astroscale": "https://astroscale.com/news/",
            "QPS Research": "https://www.qps-r.com/en/news",
            "Axelspace": "https://www.axelspace.com/en/news/",
            "Canon Electronics": "https://global.canon/en/news/",
            "MELCO Space": "https://www.melco.co.jp/space/en/",
            "SpaceNews Japan": "https://spacenews.com/tag/japan/",
            "Asahi Shimbun Space": "https://www.asahi.com/topics/word/??.html",
            "NHK World Space": "https://www3.nhk.or.jp/nhkworld/en/news/tags/18/",
            "Japan Space Forum": "https://www.jsforum.or.jp/en/",
            "Space Japan Review": "https://space-japan.net/"
        }
    
    async def process_scraping_job(self, sources: List[str], job_id: str) -> Dict[str, Any]:
        """Process a complete scraping job with NLP and knowledge graph updates"""
        logger.info(f"Starting processing pipeline for job {job_id}")
        
        total_articles = 0
        total_entities = 0
        processed_sources = []
        
        try:
            # Initialize database session for knowledge graph integration
            from research_synthesis.database.connection import get_postgres_session
            try:
                self.db_session = await get_postgres_session().__anext__()
                logger.debug(f"DEBUG: Database session initialized for {job_id}")
            except Exception as db_error:
                logger.warning(f"Database session failed: {db_error}, continuing without storage")
                self.db_session = None
            
            logger.debug(f"DEBUG: Processing {len(sources)} sources: {sources}")
            for i, source_name in enumerate(sources):
                logger.debug(f"DEBUG: Processing source {i+1}/{len(sources)}: {source_name}")
                if source_name not in self.sources:
                    logger.warning(f"Unknown source: {source_name}")
                    continue
                
                logger.info(f"Processing source: {source_name}")
                source_url = self.sources[source_name]
                
                # Update sources_data to show processing status
                try:
                    from research_synthesis.api.scraping_router import sources_data
                    if source_name in sources_data:
                        sources_data[source_name]["status"] = "processing"
                except:
                    pass
                
                try:
                    logger.debug(f"DEBUG: About to scrape source {source_name} from {source_url}")
                    # Step 1: Scrape articles
                    articles = await self.scraper.scrape_source(source_name, source_url, max_articles=3)
                    logger.info(f"Scraped {len(articles)} articles from {source_name}")
                    logger.debug(f"DEBUG: Article scraping completed successfully for {source_name}")
                    
                    source_articles = 0
                    source_entities = 0
                    
                    # Step 2: Process each article through NLP pipeline
                    logger.debug(f"DEBUG: Starting NLP processing for {len(articles)} articles from {source_name}")
                    for idx, article_data in enumerate(articles):
                        try:
                            logger.debug(f"DEBUG: Processing article {idx+1}/{len(articles)} from {source_name}")
                            # Process article with NLP
                            processed_article = await self.nlp_processor.process_article(article_data)
                            logger.debug(f"DEBUG: NLP processing completed for article {idx+1}")
                            
                            # Step 3: Store article in database
                            logger.debug(f"DEBUG: About to store article {idx+1} in database")
                            db_article = await self.store_article(processed_article, source_name)
                            logger.debug(f"DEBUG: Article {idx+1} stored successfully")
                            
                            # Step 4: Extract and store entities for knowledge graph
                            entities_created = await self.process_entities(processed_article, db_article.id)
                            
                            source_articles += 1
                            source_entities += entities_created
                            
                        except Exception as e:
                            logger.error(f"Failed to process article {article_data.get('title', 'Unknown')}: {e}")
                    
                    total_articles += source_articles
                    total_entities += source_entities
                    processed_sources.append({
                        'name': source_name,
                        'articles': source_articles,
                        'entities': source_entities
                    })
                    
                    logger.info(f"Completed {source_name}: {source_articles} articles, {source_entities} entities")
                    
                    # Update sources_data with last scraped time and article count
                    try:
                        from research_synthesis.api.scraping_router import sources_data
                        if source_name in sources_data:
                            sources_data[source_name]["last_scraped"] = datetime.now().isoformat()
                            sources_data[source_name]["articles_count"] = source_articles
                            sources_data[source_name]["status"] = "completed"
                            logger.debug(f"Updated source data for {source_name}: last_scraped={sources_data[source_name]['last_scraped']}")
                    except Exception as update_error:
                        logger.warning(f"Failed to update source data for {source_name}: {update_error}")
                    
                except Exception as e:
                    logger.error(f"Failed to process source {source_name}: {e}")
                    # Update sources_data to show error status
                    try:
                        from research_synthesis.api.scraping_router import sources_data
                        if source_name in sources_data:
                            sources_data[source_name]["status"] = "error"
                            sources_data[source_name]["last_scraped"] = datetime.now().isoformat()
                    except:
                        pass
            
            # Commit database changes
            if self.db_session:
                await self.db_session.commit()
                logger.info("Database changes committed successfully")
            
            result = {
                'job_id': job_id,
                'status': 'completed',
                'total_articles': total_articles,
                'total_entities': total_entities,
                'processed_sources': processed_sources,
                'completed_at': datetime.now().isoformat()
            }
            
            logger.info(f"Pipeline completed: {total_articles} articles, {total_entities} entities")
            return result
            
        except Exception as e:
            import traceback
            logger.error(f"Pipeline processing failed: {e}")
            logger.error(f"DEBUG: Full traceback: {traceback.format_exc()}")
            logger.debug(f"DEBUG: Error type: {type(e).__name__}")
            logger.debug(f"DEBUG: Error args: {e.args}")
            if self.db_session:
                logger.debug("DEBUG: Rolling back database session")
                await self.db_session.rollback()
            
            return {
                'job_id': job_id,
                'status': 'failed',
                'error': str(e),
                'total_articles': total_articles,
                'total_entities': total_entities
            }
        
        finally:
            if self.db_session:
                await self.db_session.close()
    
    async def store_article(self, article_data: Dict[str, Any], source_name: str) -> Article:
        """Store processed article in database"""
        try:
            # Skip database storage if session not available
            if not self.db_session:
                logger.debug("DEBUG: Database session not available, skipping article storage")
                # Create a mock Article object for pipeline completion
                mock_article = Article()
                mock_article.id = 1  # Mock ID
                mock_article.title = article_data.get('title', 'Mock Title')
                return mock_article
                
            # Check if article already exists
            existing_result = await self.db_session.execute(
                select(Article).filter(Article.url == article_data['url'])
            )
            existing = existing_result.scalar_one_or_none()
            if existing:
                logger.info(f"Article already exists: {article_data['title']}")
                return existing
            
            # Get or create source
            source_result = await self.db_session.execute(
                select(Source).filter(Source.name == source_name)
            )
            source = source_result.scalar_one_or_none()
            if not source:
                source = Source(
                    name=source_name,
                    url=self.sources.get(source_name, ''),
                    source_type=self._get_source_type(source_name),
                    is_active=True
                )
                self.db_session.add(source)
                await self.db_session.flush()
            
            # Create article record
            article = Article(
                source_id=source.id,
                url=article_data['url'],
                title=article_data['title'],
                author=article_data.get('author'),
                published_date=article_data.get('published_date'),
                content=article_data['content'],
                summary=article_data.get('ai_summary', article_data.get('summary')),
                content_type='article',
                
                # NLP results
                sentiment_score=article_data.get('sentiment_score'),
                extracted_facts=article_data.get('extracted_facts'),
                extracted_dates=article_data.get('extracted_dates'),
                extracted_numbers=article_data.get('extracted_numbers'),
                
                # Embeddings
                title_embedding=article_data.get('title_embedding'),
                content_embedding=article_data.get('content_embedding'),
                
                # Metadata
                word_count=len(article_data['content'].split()) if article_data['content'] else 0,
                categories=article_data.get('categories'),
                tags=article_data.get('tags'),
                
                # Timestamps
                scraped_at=article_data.get('scraped_at', datetime.now()),
                processed_at=datetime.now()
            )
            
            self.db_session.add(article)
            await self.db_session.flush()
            
            logger.info(f"Stored article: {article.title}")
            return article
            
        except Exception as e:
            logger.error(f"Failed to store article: {e}")
            raise
    
    async def process_entities(self, article_data: Dict[str, Any], article_id: int) -> int:
        """Extract and store entities for knowledge graph"""
        entities_created = 0
        
        try:
            # Skip entity processing if database session not available
            if not self.db_session:
                logger.debug("DEBUG: Database session not available, skipping entity processing")
                return 0
            entities = article_data.get('entities', [])
            space_entities = article_data.get('space_entities', [])
            
            # Focus on space-related entities
            relevant_entities = space_entities if space_entities else entities[:10]  # Limit to prevent overload
            
            for entity_data in relevant_entities:
                try:
                    entity_name = entity_data.get('text', '').strip()
                    entity_type = entity_data.get('type', 'UNKNOWN')
                    
                    if not entity_name or len(entity_name) < 2:
                        continue
                    
                    # Get or create entity
                    entity_result = await self.db_session.execute(
                        select(Entity).filter(Entity.name == entity_name)
                    )
                    entity = entity_result.scalar_one_or_none()
                    if not entity:
                        entity = Entity(
                            name=entity_name,
                            entity_type=self._map_entity_type(entity_type),
                            is_space_related=True,
                            space_sector=self._determine_space_sector(entity_name),
                            mention_count=1,
                            last_mentioned=datetime.now()
                        )
                        self.db_session.add(entity)
                        entities_created += 1
                    else:
                        # Update existing entity
                        entity.mention_count += 1
                        entity.last_mentioned = datetime.now()
                    
                    await self.db_session.flush()
                    
                    # Link entity to article (many-to-many relationship)
                    article_result = await self.db_session.execute(
                        select(Article).filter(Article.id == article_id)
                    )
                    article = article_result.scalar_one_or_none()
                    if article and entity not in article.entities:
                        article.entities.append(entity)
                    
                except Exception as e:
                    logger.warning(f"Failed to process entity {entity_data}: {e}")
                    
            return entities_created
            
        except Exception as e:
            logger.error(f"Entity processing failed: {e}")
            return 0
    
    def _get_source_type(self, source_name: str) -> str:
        """Determine source type"""
        source_types = {
            'SpaceNews': 'news',
            'Space.com': 'news', 
            'SpaceflightNow': 'news',
            'SpaceX': 'company',
            'Blue Origin': 'company',
            'Relativity Space': 'company',
            'arXiv': 'academic',
            'IEEE': 'academic',
            'Research Papers': 'academic',
            'NASA': 'government',
            'ESA': 'government',
            'Space Force': 'government',
            # Japanese sources
            'JAXA': 'government',
            'JAXA English': 'government',
            'MHI Space': 'company',
            'IHI Aerospace': 'company',
            'Kawasaki Heavy Industries': 'company',
            'NEC Space': 'company',
            'Interstellar Technologies': 'company',
            'Space BD': 'company',
            'ALE Space': 'company',
            'Synspective': 'company',
            'Astroscale': 'company',
            'QPS Research': 'company',
            'Axelspace': 'company',
            'Canon Electronics': 'company',
            'MELCO Space': 'company',
            'SpaceNews Japan': 'news',
            'Asahi Shimbun Space': 'news',
            'NHK World Space': 'news',
            'Japan Space Forum': 'academic',
            'Space Japan Review': 'academic'
        }
        return source_types.get(source_name, 'unknown')
    
    def _map_entity_type(self, nlp_type: str) -> str:
        """Map NLP entity types to our schema"""
        type_mapping = {
            'ORG': 'company',
            'ORGANIZATION': 'company',
            'PERSON': 'person',
            'PER': 'person',
            'GPE': 'location',
            'LOC': 'location',
            'PRODUCT': 'technology',
            'TECH': 'technology',
            'MONEY': 'financial',
            'DATE': 'temporal'
        }
        return type_mapping.get(nlp_type.upper(), 'other')
    
    def _determine_space_sector(self, entity_name: str) -> str:
        """Determine space sector for entity"""
        name_lower = entity_name.lower()
        
        if any(keyword in name_lower for keyword in ['launch', 'rocket', 'falcon', 'starship']):
            return 'launch'
        elif any(keyword in name_lower for keyword in ['satellite', 'constellation', 'starlink']):
            return 'satellite'
        elif any(keyword in name_lower for keyword in ['manufacturing', '3d print', 'factory']):
            return 'manufacturing'
        elif any(keyword in name_lower for keyword in ['mars', 'moon', 'lunar', 'exploration']):
            return 'exploration'
        elif any(keyword in name_lower for keyword in ['station', 'iss', 'habitat']):
            return 'infrastructure'
        else:
            return 'general'


# Global pipeline instance
pipeline = DataPipeline()