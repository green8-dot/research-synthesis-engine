"""
Reddit-specific scraper for JSON API responses
"""
import aiohttp
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from loguru import logger
import re
from urllib.parse import urljoin, urlparse

class RedditScraper:
    """Specialized scraper for Reddit JSON API"""
    
    def __init__(self):
        self.session = None
        self.rate_limit_delay = 2  # seconds between requests
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Research Synthesis Bot 1.0 (Space Industry Analysis)',
                'Accept': 'application/json'
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            
    async def scrape_subreddit(self, url: str, limit: int = 25) -> List[Dict[str, Any]]:
        """
        Scrape a subreddit using Reddit's JSON API
        
        Args:
            url: Reddit subreddit URL (should end with .json)
            limit: Maximum number of posts to fetch
            
        Returns:
            List of article dictionaries
        """
        articles = []
        
        try:
            # Ensure URL uses JSON endpoint
            if not url.endswith('.json'):
                url = url.rstrip('/') + '.json'
                
            # Add limit parameter
            if '?' in url:
                url += f'&limit={limit}'
            else:
                url += f'?limit={limit}'
                
            logger.info(f"Scraping Reddit: {url}")
            
            async with self.session.get(url) as response:
                if response.status == 429:
                    logger.warning("Reddit rate limit hit, waiting...")
                    await asyncio.sleep(self.rate_limit_delay * 2)
                    return articles
                    
                if response.status != 200:
                    logger.error(f"Reddit API returned status {response.status}")
                    return articles
                    
                data = await response.json()
                
                # Parse Reddit response structure
                if not isinstance(data, dict) or 'data' not in data:
                    logger.error("Invalid Reddit API response structure")
                    return articles
                    
                posts = data['data'].get('children', [])
                logger.info(f"Found {len(posts)} posts from Reddit")
                
                for post_wrapper in posts:
                    if not isinstance(post_wrapper, dict) or 'data' not in post_wrapper:
                        continue
                        
                    post = post_wrapper['data']
                    article = await self._parse_reddit_post(post)
                    
                    if article:
                        articles.append(article)
                        
            logger.info(f"Successfully parsed {len(articles)} articles from Reddit")
            
        except Exception as e:
            logger.error(f"Error scraping Reddit {url}: {e}")
            
        return articles
    
    async def _parse_reddit_post(self, post: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse a single Reddit post into article format"""
        try:
            # Skip removed/deleted posts
            if post.get('removed_by_category') or post.get('title', '').startswith('[deleted]'):
                return None
                
            # Skip non-text posts unless they have good titles
            if post.get('is_video', False) and len(post.get('title', '')) < 20:
                return None
                
            # Extract basic information
            title = post.get('title', '').strip()
            if not title or len(title) < 10:
                return None
                
            # Get post content
            content = self._extract_post_content(post)
            if not content:
                return None
                
            # Parse timestamp
            created_utc = post.get('created_utc', 0)
            published_date = datetime.fromtimestamp(created_utc, tz=timezone.utc) if created_utc else datetime.now(tz=timezone.utc)
            
            # Extract URL
            post_url = f"https://reddit.com{post.get('permalink', '')}"
            
            # Check if it's space-related based on title and content
            if not self._is_space_related(title + " " + content):
                return None
                
            article = {
                'title': title,
                'content': content,
                'url': post_url,
                'published_date': published_date,
                'source': 'Reddit',
                'author': post.get('author', 'Unknown'),
                'metadata': {
                    'subreddit': post.get('subreddit', ''),
                    'score': post.get('score', 0),
                    'num_comments': post.get('num_comments', 0),
                    'upvote_ratio': post.get('upvote_ratio', 0.0),
                    'post_type': self._determine_post_type(post),
                    'flair': post.get('link_flair_text', ''),
                    'reddit_id': post.get('id', '')
                }
            }
            
            return article
            
        except Exception as e:
            logger.error(f"Error parsing Reddit post: {e}")
            return None
    
    def _extract_post_content(self, post: Dict[str, Any]) -> str:
        """Extract meaningful content from Reddit post"""
        content_parts = []
        
        # Get selftext (for text posts)
        selftext = post.get('selftext', '').strip()
        if selftext and selftext != '[removed]' and selftext != '[deleted]':
            content_parts.append(selftext)
            
        # Get URL content description if it's a link post
        url = post.get('url', '')
        if url and url != post.get('permalink', ''):
            # Check if it's an external link
            if not url.startswith('https://reddit.com') and not url.startswith('/r/'):
                domain = urlparse(url).netloc
                content_parts.append(f"Link to: {domain}")
                
        # Use title as content if no other content available
        if not content_parts:
            title = post.get('title', '')
            if len(title) > 20:  # Only use substantial titles
                content_parts.append(title)
                
        return ' '.join(content_parts).strip()
    
    def _determine_post_type(self, post: Dict[str, Any]) -> str:
        """Determine the type of Reddit post"""
        if post.get('is_video'):
            return 'video'
        elif post.get('post_hint') == 'image':
            return 'image'
        elif post.get('selftext'):
            return 'text'
        elif post.get('url'):
            return 'link'
        else:
            return 'unknown'
    
    def _is_space_related(self, text: str) -> bool:
        """Check if content is space-related"""
        space_keywords = [
            # Space agencies and organizations
            'nasa', 'spacex', 'blue origin', 'virgin galactic', 'esa', 'roscosmos', 'isro', 'jaxa',
            'boeing', 'lockheed martin', 'northrop grumman', 'astranis', 'planet labs', 'maxar',
            
            # Space missions and programs
            'artemis', 'apollo', 'mars', 'moon', 'lunar', 'iss', 'international space station',
            'hubble', 'webb telescope', 'jwst', 'perseverance', 'curiosity', 'ingenuity',
            'voyager', 'cassini', 'juno', 'new horizons', 'parker solar probe',
            
            # Space technology and concepts  
            'rocket', 'launch', 'satellite', 'spacecraft', 'space station', 'astronaut', 'cosmonaut',
            'orbit', 'orbital', 'interplanetary', 'interstellar', 'space exploration', 'space travel',
            'space industry', 'commercial space', 'space economy', 'space mining', 'space manufacturing',
            'space debris', 'space junk', 'reusable rocket', 'falcon heavy', 'starship', 'sls',
            
            # Celestial bodies and phenomena
            'asteroid', 'comet', 'planet', 'exoplanet', 'galaxy', 'solar system', 'universe',
            'black hole', 'neutron star', 'supernova', 'cosmic', 'astronomy', 'astrophysics',
            
            # Space-related terms
            'zero gravity', 'microgravity', 'spacewalk', 'eva', 'docking', 'landing', 'reentry',
            'heat shield', 'propulsion', 'thruster', 'fuel', 'oxidizer', 'payload', 'cargo'
        ]
        
        text_lower = text.lower()
        
        # Check for direct keyword matches
        for keyword in space_keywords:
            if keyword in text_lower:
                return True
                
        # Check for space-related patterns
        space_patterns = [
            r'/b(space|mars|moon|lunar)/s+(mission|program|project|exploration)',
            r'/b(rocket|satellite|spacecraft)/s+(launch|development|technology)',
            r'/b(astronaut|cosmonaut|crew)/s+(training|mission|flight)',
            r'/b(orbital|interplanetary|deep/s+space)/s+',
        ]
        
        for pattern in space_patterns:
            if re.search(pattern, text_lower):
                return True
                
        return False

# Convenience function for external use
async def scrape_reddit_json(url: str, limit: int = 25) -> List[Dict[str, Any]]:
    """
    Convenience function to scrape Reddit using JSON API
    
    Args:
        url: Reddit URL (will be converted to JSON endpoint if needed)
        limit: Maximum posts to fetch
        
    Returns:
        List of parsed articles
    """
    async with RedditScraper() as scraper:
        return await scraper.scrape_subreddit(url, limit)