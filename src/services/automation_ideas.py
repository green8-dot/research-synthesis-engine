"""
Automation Ideas Discovery System
Scrapes automation sources to find new productivity ideas and improvements
"""
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import re
import json
from dataclasses import dataclass
from enum import Enum

@dataclass
class AutomationIdea:
    """Represents an automation idea with productivity metrics"""
    title: str
    description: str
    source: str
    url: str
    category: str
    difficulty: str  # "easy", "medium", "hard"
    productivity_estimate: int  # Percentage improvement (0-100)
    implementation_time_days: int
    priority_score: float  # Calculated priority score
    tags: List[str]
    discovered_at: datetime
    upvotes: int = 0
    engagement_score: int = 0

class AutomationCategory(Enum):
    """Categories for automation ideas"""
    DATA_PROCESSING = "Data Processing"
    WORKFLOW_AUTOMATION = "Workflow Automation"
    MONITORING = "Monitoring & Alerts"
    DEPLOYMENT = "Deployment & DevOps"
    COMMUNICATION = "Communication & Notifications"
    ANALYTICS = "Analytics & Reporting"
    TESTING = "Testing & QA"
    SECURITY = "Security & Compliance"
    INTEGRATION = "System Integration"
    GENERAL = "General Automation"

class AutomationIdeasDiscovery:
    """Main class for discovering automation ideas"""
    
    def __init__(self):
        self.session = None
        self.discovered_subreddits = set()
        self.automation_sources = {
            # Primary automation sources
            "Reddit r/Automate": {
                "url": "https://old.reddit.com/r/automate/.json",
                "type": "reddit",
                "weight": 0.9
            },
            "Reddit r/productivity": {
                "url": "https://old.reddit.com/r/productivity/.json", 
                "type": "reddit",
                "weight": 0.8
            },
            
            # Technical automation communities
            "Reddit r/selfhosted": {
                "url": "https://old.reddit.com/r/selfhosted/.json",
                "type": "reddit", 
                "weight": 0.7
            },
            "Reddit r/sysadmin": {
                "url": "https://old.reddit.com/r/sysadmin/.json",
                "type": "reddit",
                "weight": 0.9
            },
            "Reddit r/DevOps": {
                "url": "https://old.reddit.com/r/devops/.json",
                "type": "reddit",
                "weight": 0.8
            },
            
            # Programming and scripting automation
            "Reddit r/Python": {
                "url": "https://old.reddit.com/r/python/.json",
                "type": "reddit",
                "weight": 0.7
            },
            "Reddit r/PowerShell": {
                "url": "https://old.reddit.com/r/powershell/.json",
                "type": "reddit",
                "weight": 0.6
            },
            "Reddit r/bash": {
                "url": "https://old.reddit.com/r/bash/.json",
                "type": "reddit",
                "weight": 0.6
            },
            
            # Home and personal automation
            "Reddit r/homeautomation": {
                "url": "https://old.reddit.com/r/homeautomation/.json",
                "type": "reddit",
                "weight": 0.7
            },
            "Reddit r/homelab": {
                "url": "https://old.reddit.com/r/homelab/.json",
                "type": "reddit",
                "weight": 0.6
            },
            "Reddit r/HomeAssistant": {
                "url": "https://old.reddit.com/r/homeassistant/.json",
                "type": "reddit",
                "weight": 0.6
            },
            
            # Workflow and business automation
            "Reddit r/workflow": {
                "url": "https://old.reddit.com/r/workflow/.json",
                "type": "reddit",
                "weight": 0.7
            },
            "Reddit r/IFTTT": {
                "url": "https://old.reddit.com/r/ifttt/.json",
                "type": "reddit",
                "weight": 0.5
            },
            "Reddit r/zapier": {
                "url": "https://old.reddit.com/r/zapier/.json",
                "type": "reddit",
                "weight": 0.5
            },
            
            # GitHub automation repositories
            "GitHub Awesome Automation": {
                "url": "https://api.github.com/repos/awesome-lists/awesome-automation/contents",
                "type": "github_repo",
                "weight": 0.9
            },
            "GitHub Automation Scripts": {
                "url": "https://api.github.com/search/repositories?q=automation+scripts+sort:stars",
                "type": "github_search",
                "weight": 0.8
            },
            "GitHub DevOps Tools": {
                "url": "https://api.github.com/search/repositories?q=devops+automation+sort:stars",
                "type": "github_search", 
                "weight": 0.8
            },
            "GitHub Python Automation": {
                "url": "https://api.github.com/search/repositories?q=python+automation+sort:stars",
                "type": "github_search",
                "weight": 0.7
            },
            "GitHub Workflow Automation": {
                "url": "https://api.github.com/search/repositories?q=workflow+automation+sort:stars",
                "type": "github_search",
                "weight": 0.7
            },
            
            # Stack Overflow automation questions
            "StackOverflow Automation": {
                "url": "https://api.stackexchange.com/2.3/questions?order=desc&sort=votes&tagged=automation&site=stackoverflow",
                "type": "stackoverflow",
                "weight": 0.8
            },
            "StackOverflow DevOps": {
                "url": "https://api.stackexchange.com/2.3/questions?order=desc&sort=votes&tagged=devops&site=stackoverflow", 
                "type": "stackoverflow",
                "weight": 0.7
            },
            "StackOverflow CI/CD": {
                "url": "https://api.stackexchange.com/2.3/questions?order=desc&sort=votes&tagged=continuous-integration&site=stackoverflow",
                "type": "stackoverflow",
                "weight": 0.7
            },
            
            # Developer communities
            "Dev.to Automation": {
                "url": "https://dev.to/api/articles?tag=automation",
                "type": "dev_to",
                "weight": 0.6
            },
            "Dev.to DevOps": {
                "url": "https://dev.to/api/articles?tag=devops",
                "type": "dev_to", 
                "weight": 0.6
            },
            "Dev.to Productivity": {
                "url": "https://dev.to/api/articles?tag=productivity",
                "type": "dev_to",
                "weight": 0.5
            },
            
            # Tech forums and communities  
            "Hacker News Automation": {
                "url": "https://hn.algolia.com/api/v1/search?query=automation&tags=story",
                "type": "hackernews",
                "weight": 0.7
            },
            "Hacker News DevOps": {
                "url": "https://hn.algolia.com/api/v1/search?query=devops&tags=story",
                "type": "hackernews",
                "weight": 0.6
            },
            
            # Product Hunt automation tools
            "ProductHunt Automation": {
                "url": "https://api.producthunt.com/v1/categories/developer-tools/posts",
                "type": "producthunt",
                "weight": 0.5
            },
            
            # Mobile automation
            "Reddit r/tasker": {
                "url": "https://old.reddit.com/r/tasker/.json",
                "type": "reddit",
                "weight": 0.6
            },
            "Reddit r/shortcuts": {
                "url": "https://old.reddit.com/r/shortcuts/.json",
                "type": "reddit",
                "weight": 0.6
            },
            
            # Data science automation
            "Reddit r/MachineLearning": {
                "url": "https://old.reddit.com/r/machinelearning/.json",
                "type": "reddit",
                "weight": 0.6
            },
            "Reddit r/datascience": {
                "url": "https://old.reddit.com/r/datascience/.json",
                "type": "reddit",
                "weight": 0.6
            },
            
            # General tech and tools
            "Reddit r/technology": {
                "url": "https://old.reddit.com/r/technology/.json",
                "type": "reddit",
                "weight": 0.4
            },
            "Reddit r/software": {
                "url": "https://old.reddit.com/r/software/.json",
                "type": "reddit",
                "weight": 0.5
            }
        }
        
    async def discover_related_subreddits(self) -> List[str]:
        """Discover related subreddits from r/Automate sidebar and other sources"""
        try:
            related_subreddits = []
            
            # First, get the main r/Automate page to find related subreddits
            async with self.session.get("https://old.reddit.com/r/automate/", timeout=30) as response:
                if response.status == 200:
                    html_content = await response.text()
                    
                    # Find subreddit mentions in the sidebar and content
                    subreddit_pattern = r'/r/([a-zA-Z0-9_]+)'
                    matches = re.findall(subreddit_pattern, html_content)
                    
                    # Filter for automation/productivity related subreddits
                    automation_keywords = [
                        'automate', 'automation', 'productivity', 'workflow', 'script', 'bot',
                        'devops', 'sysadmin', 'homelab', 'selfhosted', 'python', 'powershell',
                        'efficiency', 'tools', 'software', 'programming', 'tech', 'ifttt',
                        'zapier', 'homeassistant', 'tasker', 'shortcuts'
                    ]
                    
                    for subreddit in set(matches):
                        subreddit_lower = subreddit.lower()
                        # Check if subreddit name contains automation-related keywords
                        if (any(keyword in subreddit_lower for keyword in automation_keywords) or 
                            len(subreddit) > 4):  # Include longer subreddit names
                            related_subreddits.append(subreddit)
            
            # Add some known automation/productivity subreddits
            known_automation_subs = [
                'MacroRecorder', 'AutoHotkey', 'homeassistant', 'shortcuts', 'tasker',
                'IFTTT', 'workflow', 'PowerShell', 'Python', 'learnpython',
                'devops', 'linuxadmin', 'homelab', 'selfhosted', 'DataEngineering',
                'ETL', 'businessintelligence', 'excel', 'googlesheets',
                'automation', 'productivityporn', 'getmotivated', 'efficiency',
                'lifehacks', 'organization', 'gtd', 'bujo', 'notion',
                'Zapier', 'MicrosoftFlow', 'node_red'
            ]
            
            related_subreddits.extend(known_automation_subs)
            
            # Remove duplicates and filter out common ones we already have
            existing_subs = {'automate', 'productivity', 'selfhosted', 'sysadmin', 'devops'}
            new_subreddits = [sub for sub in set(related_subreddits) 
                            if sub.lower() not in existing_subs and len(sub) >= 3]
            
            self.discovered_subreddits.update(new_subreddits)
            logger.info(f"Discovered {len(new_subreddits)} related automation subreddits")
            
            return new_subreddits[:15]  # Limit to top 15 new subreddits
            
        except Exception as e:
            logger.error(f"Error discovering related subreddits: {e}")
            return []

    async def expand_automation_sources(self):
        """Expand automation sources with discovered subreddits"""
        try:
            related_subs = await self.discover_related_subreddits()
            
            for subreddit in related_subs:
                source_name = f"Reddit r/{subreddit}"
                if source_name not in self.automation_sources:
                    self.automation_sources[source_name] = {
                        "url": f"https://old.reddit.com/r/{subreddit}/.json",
                        "type": "reddit",
                        "weight": 0.5,  # Lower weight for discovered sources
                        "discovered": True
                    }
                    
            logger.info(f"Expanded automation sources to {len(self.automation_sources)} total sources")
            
        except Exception as e:
            logger.error(f"Error expanding automation sources: {e}")
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'AutomationIdeas Discovery Bot 1.0'
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def discover_automation_ideas(self, limit_per_source: int = 25) -> List[AutomationIdea]:
        """Discover automation ideas from all sources"""
        all_ideas = []
        
        # First expand sources by discovering related subreddits
        logger.info("Expanding automation sources by discovering related subreddits...")
        await self.expand_automation_sources()
        
        logger.info(f"Discovering automation ideas from {len(self.automation_sources)} sources...")
        
        for source_name, source_config in self.automation_sources.items():
            try:
                logger.info(f"Scraping automation ideas from {source_name}...")
                
                # Route to appropriate scraper based on source type
                source_type = source_config.get("type", "reddit")
                ideas = []
                
                if source_type == "reddit":
                    ideas = await self._scrape_reddit_automation(
                        source_name, source_config["url"], limit_per_source
                    )
                elif source_type == "github_search":
                    ideas = await self._scrape_github_search(
                        source_name, source_config["url"], limit_per_source
                    )
                elif source_type == "github_repo":
                    ideas = await self._scrape_github_repo(
                        source_name, source_config["url"], limit_per_source
                    )
                elif source_type == "stackoverflow":
                    ideas = await self._scrape_stackoverflow(
                        source_name, source_config["url"], limit_per_source
                    )
                elif source_type == "dev_to":
                    ideas = await self._scrape_dev_to(
                        source_name, source_config["url"], limit_per_source
                    )
                elif source_type == "hackernews":
                    ideas = await self._scrape_hackernews(
                        source_name, source_config["url"], limit_per_source
                    )
                elif source_type == "producthunt":
                    ideas = await self._scrape_producthunt(
                        source_name, source_config["url"], limit_per_source
                    )
                
                # Apply source weight to priority scores
                for idea in ideas:
                    idea.priority_score *= source_config["weight"]
                    
                all_ideas.extend(ideas)
                logger.info(f"Found {len(ideas)} ideas from {source_name}")
                
                # Rate limiting - more conservative for API sources
                if source_type in ["github_search", "github_repo", "stackoverflow"]:
                    await asyncio.sleep(5)  # Longer delay for APIs
                else:
                    await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error scraping {source_name}: {e}")
                continue
        
        # Sort by priority score
        all_ideas.sort(key=lambda x: x.priority_score, reverse=True)
        
        logger.info(f"Discovered {len(all_ideas)} total automation ideas")
        return all_ideas

    async def _scrape_reddit_automation(self, source_name: str, url: str, limit: int) -> List[AutomationIdea]:
        """Scrape automation ideas from Reddit"""
        ideas = []
        
        try:
            # Add limit parameter
            url_with_limit = f"{url}?limit={limit}"
            
            async with self.session.get(url_with_limit) as response:
                if response.status != 200:
                    logger.error(f"Reddit API returned status {response.status} for {source_name}")
                    return ideas
                    
                data = await response.json()
                posts = data.get('data', {}).get('children', [])
                
                for post_wrapper in posts:
                    post = post_wrapper.get('data', {})
                    idea = await self._parse_reddit_automation_post(post, source_name)
                    
                    if idea:
                        ideas.append(idea)
                        
        except Exception as e:
            logger.error(f"Error scraping Reddit automation from {url}: {e}")
            
        return ideas

    async def _parse_reddit_automation_post(self, post: Dict[str, Any], source_name: str) -> Optional[AutomationIdea]:
        """Parse a Reddit post into an automation idea"""
        try:
            title = post.get('title', '').strip()
            if not title or len(title) < 10:
                return None
                
            # Filter for automation-related content
            if not self._is_automation_related(title + " " + post.get('selftext', '')):
                return None
                
            # Extract content
            description = self._extract_post_description(post)
            if not description:
                return None
                
            # Categorize the idea
            category = self._categorize_automation_idea(title, description)
            
            # Estimate productivity impact
            productivity_estimate = self._estimate_productivity_impact(title, description, post.get('score', 0))
            
            # Estimate implementation difficulty and time
            difficulty, implementation_days = self._estimate_implementation_effort(title, description)
            
            # Calculate priority score
            priority_score = self._calculate_priority_score(
                productivity_estimate, 
                difficulty,
                post.get('score', 0),
                post.get('num_comments', 0)
            )
            
            # Extract tags
            tags = self._extract_automation_tags(title, description)
            
            return AutomationIdea(
                title=title,
                description=description,
                source=source_name,
                url=f"https://reddit.com{post.get('permalink', '')}",
                category=category.value,
                difficulty=difficulty,
                productivity_estimate=productivity_estimate,
                implementation_time_days=implementation_days,
                priority_score=priority_score,
                tags=tags,
                discovered_at=datetime.now(),
                upvotes=post.get('score', 0),
                engagement_score=post.get('num_comments', 0)
            )
            
        except Exception as e:
            logger.error(f"Error parsing automation post: {e}")
            return None

    async def _scrape_github_search(self, source_name: str, url: str, limit: int) -> List[AutomationIdea]:
        """Scrape automation ideas from GitHub search results"""
        ideas = []
        try:
            async with self.session.get(url, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    repos = data.get('items', [])[:limit]
                    
                    for repo in repos:
                        idea = await self._parse_github_repo(repo, source_name)
                        if idea:
                            ideas.append(idea)
                            
        except Exception as e:
            logger.error(f"Error scraping GitHub search from {url}: {e}")
            
        return ideas

    async def _scrape_github_repo(self, source_name: str, url: str, limit: int) -> List[AutomationIdea]:
        """Scrape automation ideas from GitHub repository contents"""
        ideas = []
        try:
            async with self.session.get(url, timeout=30) as response:
                if response.status == 200:
                    contents = await response.json()
                    
                    # Look for README files and automation scripts
                    for item in contents[:limit]:
                        if item.get('name', '').lower() in ['readme.md', 'readme.txt', 'readme']:
                            readme_url = item.get('download_url')
                            if readme_url:
                                async with self.session.get(readme_url, timeout=30) as readme_response:
                                    if readme_response.status == 200:
                                        readme_content = await readme_response.text()
                                        idea = self._parse_github_readme(readme_content, source_name, item.get('html_url'))
                                        if idea:
                                            ideas.append(idea)
                            
        except Exception as e:
            logger.error(f"Error scraping GitHub repo from {url}: {e}")
            
        return ideas

    async def _scrape_stackoverflow(self, source_name: str, url: str, limit: int) -> List[AutomationIdea]:
        """Scrape automation ideas from Stack Overflow questions"""
        ideas = []
        try:
            async with self.session.get(url, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    questions = data.get('items', [])[:limit]
                    
                    for question in questions:
                        idea = await self._parse_stackoverflow_question(question, source_name)
                        if idea:
                            ideas.append(idea)
                            
        except Exception as e:
            logger.error(f"Error scraping StackOverflow from {url}: {e}")
            
        return ideas

    async def _scrape_dev_to(self, source_name: str, url: str, limit: int) -> List[AutomationIdea]:
        """Scrape automation ideas from Dev.to articles"""
        ideas = []
        try:
            async with self.session.get(url, timeout=30) as response:
                if response.status == 200:
                    articles = await response.json()
                    
                    for article in articles[:limit]:
                        idea = await self._parse_dev_to_article(article, source_name)
                        if idea:
                            ideas.append(idea)
                            
        except Exception as e:
            logger.error(f"Error scraping Dev.to from {url}: {e}")
            
        return ideas

    async def _scrape_hackernews(self, source_name: str, url: str, limit: int) -> List[AutomationIdea]:
        """Scrape automation ideas from Hacker News"""
        ideas = []
        try:
            async with self.session.get(url, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    hits = data.get('hits', [])[:limit]
                    
                    for hit in hits:
                        idea = await self._parse_hackernews_story(hit, source_name)
                        if idea:
                            ideas.append(idea)
                            
        except Exception as e:
            logger.error(f"Error scraping Hacker News from {url}: {e}")
            
        return ideas

    async def _scrape_producthunt(self, source_name: str, url: str, limit: int) -> List[AutomationIdea]:
        """Scrape automation ideas from Product Hunt"""
        ideas = []
        try:
            headers = {'Authorization': 'Bearer YOUR_PRODUCTHUNT_TOKEN'}  # Would need API token
            async with self.session.get(url, headers=headers, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    posts = data.get('posts', [])[:limit]
                    
                    for post in posts:
                        idea = await self._parse_producthunt_post(post, source_name)
                        if idea:
                            ideas.append(idea)
                            
        except Exception as e:
            logger.error(f"Error scraping Product Hunt from {url}: {e}")
            
        return ideas

    async def _parse_github_repo(self, repo: Dict[str, Any], source_name: str) -> Optional[AutomationIdea]:
        """Parse a GitHub repository into an automation idea"""
        try:
            name = repo.get('name', '')
            description = repo.get('description', '')
            
            if not self._is_automation_related(name + " " + description):
                return None
                
            title = f"Automation Tool: {name}"
            full_description = description or f"GitHub repository for {name}"
            
            return AutomationIdea(
                title=title,
                description=full_description,
                source=source_name,
                url=repo.get('html_url', ''),
                category=self._categorize_automation_idea(title, full_description).value,
                difficulty="medium",
                productivity_estimate=self._estimate_productivity_from_github(repo),
                implementation_time_days=7,
                priority_score=self._calculate_github_priority(repo),
                tags=self._extract_github_tags(repo),
                discovered_at=datetime.now(),
                upvotes=repo.get('stargazers_count', 0),
                engagement_score=repo.get('forks_count', 0)
            )
            
        except Exception as e:
            logger.error(f"Error parsing GitHub repo: {e}")
            return None

    def _parse_github_readme(self, readme_content: str, source_name: str, url: str) -> Optional[AutomationIdea]:
        """Parse GitHub README content into automation idea"""
        try:
            lines = readme_content.split('/n')
            title = lines[0].strip('#').strip() if lines else "Automation Project"
            
            if not self._is_automation_related(readme_content[:500]):
                return None
                
            description = readme_content[:300] + "..." if len(readme_content) > 300 else readme_content
            
            return AutomationIdea(
                title=title,
                description=description,
                source=source_name,
                url=url or '',
                category=self._categorize_automation_idea(title, description).value,
                difficulty="medium",
                productivity_estimate=25,
                implementation_time_days=5,
                priority_score=0.7,
                tags=self._extract_automation_tags(title, description),
                discovered_at=datetime.now(),
                upvotes=0,
                engagement_score=0
            )
            
        except Exception as e:
            logger.error(f"Error parsing GitHub README: {e}")
            return None

    async def _parse_stackoverflow_question(self, question: Dict[str, Any], source_name: str) -> Optional[AutomationIdea]:
        """Parse a Stack Overflow question into an automation idea"""
        try:
            title = question.get('title', '')
            
            if not self._is_automation_related(title):
                return None
                
            # Convert question to actionable automation idea
            idea_title = f"Automation Solution: {title}"
            description = f"Stack Overflow discussion about {title.lower()}"
            
            return AutomationIdea(
                title=idea_title,
                description=description,
                source=source_name,
                url=question.get('link', ''),
                category=self._categorize_automation_idea(title, description).value,
                difficulty="medium",
                productivity_estimate=20,
                implementation_time_days=3,
                priority_score=self._calculate_stackoverflow_priority(question),
                tags=question.get('tags', []),
                discovered_at=datetime.now(),
                upvotes=question.get('score', 0),
                engagement_score=question.get('view_count', 0) // 100
            )
            
        except Exception as e:
            logger.error(f"Error parsing StackOverflow question: {e}")
            return None

    async def _parse_dev_to_article(self, article: Dict[str, Any], source_name: str) -> Optional[AutomationIdea]:
        """Parse a Dev.to article into an automation idea"""
        try:
            title = article.get('title', '')
            description = article.get('description', '')
            
            if not self._is_automation_related(title + " " + description):
                return None
                
            return AutomationIdea(
                title=title,
                description=description,
                source=source_name,
                url=article.get('url', ''),
                category=self._categorize_automation_idea(title, description).value,
                difficulty="easy",
                productivity_estimate=15,
                implementation_time_days=2,
                priority_score=self._calculate_dev_to_priority(article),
                tags=article.get('tag_list', []),
                discovered_at=datetime.now(),
                upvotes=article.get('positive_reactions_count', 0),
                engagement_score=article.get('comments_count', 0)
            )
            
        except Exception as e:
            logger.error(f"Error parsing Dev.to article: {e}")
            return None

    async def _parse_hackernews_story(self, story: Dict[str, Any], source_name: str) -> Optional[AutomationIdea]:
        """Parse a Hacker News story into an automation idea"""
        try:
            title = story.get('title', '')
            
            if not self._is_automation_related(title):
                return None
                
            url = story.get('url', f"https://news.ycombinator.com/item?id={story.get('objectID')}")
            
            return AutomationIdea(
                title=title,
                description=f"Hacker News discussion: {title}",
                source=source_name,
                url=url,
                category=self._categorize_automation_idea(title, title).value,
                difficulty="medium",
                productivity_estimate=self._estimate_hn_productivity(story),
                implementation_time_days=4,
                priority_score=self._calculate_hn_priority(story),
                tags=self._extract_automation_tags(title, title),
                discovered_at=datetime.now(),
                upvotes=story.get('points', 0),
                engagement_score=story.get('num_comments', 0)
            )
            
        except Exception as e:
            logger.error(f"Error parsing Hacker News story: {e}")
            return None

    async def _parse_producthunt_post(self, post: Dict[str, Any], source_name: str) -> Optional[AutomationIdea]:
        """Parse a Product Hunt post into an automation idea"""
        try:
            name = post.get('name', '')
            tagline = post.get('tagline', '')
            
            if not self._is_automation_related(name + " " + tagline):
                return None
                
            return AutomationIdea(
                title=f"Tool: {name}",
                description=tagline,
                source=source_name,
                url=post.get('redirect_url', ''),
                category=self._categorize_automation_idea(name, tagline).value,
                difficulty="easy",
                productivity_estimate=20,
                implementation_time_days=1,
                priority_score=self._calculate_ph_priority(post),
                tags=[name.lower()],
                discovered_at=datetime.now(),
                upvotes=post.get('votes_count', 0),
                engagement_score=post.get('comments_count', 0)
            )
            
        except Exception as e:
            logger.error(f"Error parsing Product Hunt post: {e}")
            return None

    def _estimate_productivity_from_github(self, repo: Dict[str, Any]) -> int:
        """Estimate productivity impact from GitHub repo metrics"""
        stars = repo.get('stargazers_count', 0)
        forks = repo.get('forks_count', 0)
        
        if stars > 1000:
            return 40
        elif stars > 500:
            return 30
        elif stars > 100:
            return 20
        else:
            return 15

    def _calculate_github_priority(self, repo: Dict[str, Any]) -> float:
        """Calculate priority score for GitHub repository"""
        stars = repo.get('stargazers_count', 0)
        forks = repo.get('forks_count', 0)
        issues = repo.get('open_issues_count', 0)
        
        # Higher stars = higher priority
        star_score = min(stars / 1000, 1.0)
        
        # More forks = more useful
        fork_score = min(forks / 100, 1.0)
        
        # Fewer open issues = more stable
        issue_penalty = min(issues / 50, 0.3)
        
        return max(0.1, star_score + fork_score - issue_penalty)

    def _extract_github_tags(self, repo: Dict[str, Any]) -> List[str]:
        """Extract relevant tags from GitHub repository"""
        tags = []
        
        language = repo.get('language', '').lower()
        if language:
            tags.append(language)
            
        topics = repo.get('topics', [])
        tags.extend(topics)
        
        name_words = repo.get('name', '').lower().split('-')
        tags.extend([word for word in name_words if len(word) > 2])
        
        return tags[:10]

    def _calculate_stackoverflow_priority(self, question: Dict[str, Any]) -> float:
        """Calculate priority for Stack Overflow question"""
        score = question.get('score', 0)
        views = question.get('view_count', 0)
        
        return min(0.8, (score + views/1000) / 100)

    def _calculate_dev_to_priority(self, article: Dict[str, Any]) -> float:
        """Calculate priority for Dev.to article"""
        reactions = article.get('positive_reactions_count', 0)
        comments = article.get('comments_count', 0)
        
        return min(0.7, (reactions + comments) / 50)

    def _estimate_hn_productivity(self, story: Dict[str, Any]) -> int:
        """Estimate productivity impact from HN story"""
        points = story.get('points', 0)
        
        if points > 100:
            return 25
        elif points > 50:
            return 20
        else:
            return 15

    def _calculate_hn_priority(self, story: Dict[str, Any]) -> float:
        """Calculate priority for Hacker News story"""
        points = story.get('points', 0)
        comments = story.get('num_comments', 0)
        
        return min(0.8, (points + comments) / 200)

    def _calculate_ph_priority(self, post: Dict[str, Any]) -> float:
        """Calculate priority for Product Hunt post"""
        votes = post.get('votes_count', 0)
        comments = post.get('comments_count', 0)
        
        return min(0.6, (votes + comments) / 100)

    def _is_automation_related(self, text: str) -> bool:
        """Check if content is automation-related"""
        automation_keywords = [
            # Direct automation terms
            'automate', 'automation', 'automated', 'automatically', 'auto-',
            'script', 'scripting', 'batch', 'cron', 'scheduler', 'workflow',
            
            # Tools and technologies
            'python', 'bash', 'powershell', 'ansible', 'terraform', 'docker',
            'jenkins', 'github actions', 'gitlab ci', 'ci/cd', 'devops',
            'zapier', 'ifttt', 'n8n', 'node-red', 'make.com',
            
            # Process improvements
            'efficiency', 'productivity', 'streamline', 'optimize', 'reduce manual',
            'eliminate repetitive', 'save time', 'speed up', 'faster workflow',
            
            # Monitoring and alerts
            'monitor', 'alert', 'notification', 'dashboard', 'metrics',
            'logging', 'tracking', 'status check', 'health check',
            
            # Integration terms  
            'api', 'webhook', 'integration', 'sync', 'connect', 'bridge',
            'pipeline', 'etl', 'data flow', 'orchestration'
        ]
        
        text_lower = text.lower()
        
        # Check for keyword matches
        for keyword in automation_keywords:
            if keyword in text_lower:
                return True
                
        # Check for automation patterns
        automation_patterns = [
            r'/bauto[- ]?/w+',  # auto-deploy, automate, etc.
            r'/bscript/w*/b',   # script, scripting, scripts
            r'/bbot/b|/bbots/b', # bot, bots
            r'/bcron/b|/btask/s+scheduler',
            r'/bworkflow/b|/bpipeline/b',
            r'/bci[/-]cd/b|/bdevops/b'
        ]
        
        for pattern in automation_patterns:
            if re.search(pattern, text_lower):
                return True
                
        return False

    def _extract_post_description(self, post: Dict[str, Any]) -> str:
        """Extract meaningful description from Reddit post"""
        description_parts = []
        
        # Get selftext (for text posts)
        selftext = post.get('selftext', '').strip()
        if selftext and selftext not in ['[removed]', '[deleted]']:
            description_parts.append(selftext[:500])  # Limit length
            
        # Get URL context for link posts
        url = post.get('url', '')
        if url and not url.startswith('https://reddit.com'):
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            description_parts.append(f"Link to: {domain}")
            
        # Use title if no other content
        if not description_parts:
            title = post.get('title', '')
            if len(title) > 20:
                description_parts.append(title)
                
        return ' '.join(description_parts).strip()

    def _categorize_automation_idea(self, title: str, description: str) -> AutomationCategory:
        """Categorize the automation idea"""
        text = (title + " " + description).lower()
        
        category_keywords = {
            AutomationCategory.DATA_PROCESSING: [
                'data', 'etl', 'processing', 'parsing', 'csv', 'json', 'database',
                'transform', 'clean', 'migrate', 'import', 'export'
            ],
            AutomationCategory.WORKFLOW_AUTOMATION: [
                'workflow', 'task', 'process', 'sequence', 'chain', 'orchestration',
                'zapier', 'ifttt', 'n8n', 'make.com', 'flow'
            ],
            AutomationCategory.MONITORING: [
                'monitor', 'alert', 'notification', 'dashboard', 'status', 'health',
                'uptime', 'metric', 'log', 'tracking', 'watch', 'check'
            ],
            AutomationCategory.DEPLOYMENT: [
                'deploy', 'deployment', 'ci/cd', 'jenkins', 'github actions', 'gitlab',
                'docker', 'container', 'kubernetes', 'ansible', 'terraform'
            ],
            AutomationCategory.COMMUNICATION: [
                'slack', 'discord', 'teams', 'email', 'sms', 'notification',
                'message', 'chat', 'communicate', 'inform'
            ],
            AutomationCategory.ANALYTICS: [
                'analytics', 'report', 'dashboard', 'chart', 'graph', 'metric',
                'kpi', 'business intelligence', 'visualization', 'insights'
            ],
            AutomationCategory.TESTING: [
                'test', 'testing', 'qa', 'quality', 'validation', 'check',
                'verify', 'assert', 'selenium', 'pytest', 'unit test'
            ],
            AutomationCategory.SECURITY: [
                'security', 'backup', 'encryption', 'auth', 'login', 'password',
                'ssl', 'certificate', 'vulnerability', 'scan'
            ],
            AutomationCategory.INTEGRATION: [
                'api', 'webhook', 'integration', 'sync', 'connect', 'bridge',
                'third-party', 'service', 'platform'
            ]
        }
        
        # Score each category
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                category_scores[category] = score
                
        # Return highest scoring category, or GENERAL if none match
        if category_scores:
            return max(category_scores.keys(), key=lambda k: category_scores[k])
        else:
            return AutomationCategory.GENERAL

    def _estimate_productivity_impact(self, title: str, description: str, upvotes: int) -> int:
        """Estimate productivity improvement percentage"""
        text = (title + " " + description).lower()
        
        # Base score from content analysis
        impact_indicators = {
            # High impact keywords
            'eliminate manual': 15,
            'fully automated': 20,
            'save hours': 18,
            'reduce time by': 16,
            'streamline': 12,
            'optimize': 10,
            '24/7': 15,
            'real-time': 12,
            
            # Medium impact
            'faster': 8,
            'efficient': 8,
            'improve': 6,
            'speed up': 10,
            'automate': 12,
            'script': 8,
            
            # Technology indicators  
            'ci/cd': 15,
            'docker': 10,
            'kubernetes': 12,
            'ansible': 14,
            'terraform': 12,
            'monitoring': 10,
            'dashboard': 8
        }
        
        base_score = 5  # Minimum baseline
        
        for indicator, score in impact_indicators.items():
            if indicator in text:
                base_score += score
                
        # Boost based on community engagement
        if upvotes > 50:
            base_score *= 1.2
        elif upvotes > 20:
            base_score *= 1.1
        elif upvotes > 10:
            base_score *= 1.05
            
        # Cap at reasonable maximum
        return min(int(base_score), 85)

    def _estimate_implementation_effort(self, title: str, description: str) -> Tuple[str, int]:
        """Estimate implementation difficulty and time"""
        text = (title + " " + description).lower()
        
        # Complexity indicators
        easy_indicators = [
            'simple', 'basic', 'quick', 'easy', 'beginner', 'script',
            'one-liner', 'cron', 'batch file', 'shortcut'
        ]
        
        hard_indicators = [
            'complex', 'advanced', 'enterprise', 'scale', 'kubernetes',
            'microservices', 'distributed', 'architecture', 'framework',
            'machine learning', 'ai', 'deep learning'
        ]
        
        medium_indicators = [
            'docker', 'api', 'integration', 'database', 'web scraping',
            'monitoring', 'dashboard', 'automation', 'workflow'
        ]
        
        easy_count = sum(1 for indicator in easy_indicators if indicator in text)
        hard_count = sum(1 for indicator in hard_indicators if indicator in text)
        medium_count = sum(1 for indicator in medium_indicators if indicator in text)
        
        # Determine difficulty and time
        if hard_count > easy_count + medium_count:
            return "hard", 14  # 2 weeks
        elif easy_count > hard_count + medium_count:
            return "easy", 2   # 2 days
        else:
            return "medium", 5  # 1 week

    def _calculate_priority_score(self, productivity: int, difficulty: str, upvotes: int, comments: int) -> float:
        """Calculate priority score for ranking ideas"""
        
        # Base score from productivity impact
        score = productivity
        
        # Adjust for difficulty (easier = higher priority)
        difficulty_multipliers = {
            "easy": 1.3,
            "medium": 1.0,
            "hard": 0.7
        }
        score *= difficulty_multipliers.get(difficulty, 1.0)
        
        # Boost for community engagement
        engagement_boost = min((upvotes * 0.1) + (comments * 0.05), 10)
        score += engagement_boost
        
        # ROI factor (productivity impact / implementation time)
        implementation_days = {"easy": 2, "medium": 5, "hard": 14}.get(difficulty, 5)
        roi_factor = productivity / implementation_days
        score += roi_factor * 0.5
        
        return round(score, 2)

    def _extract_automation_tags(self, title: str, description: str) -> List[str]:
        """Extract relevant tags from the content"""
        text = (title + " " + description).lower()
        
        tag_keywords = {
            'python': ['python', 'py', 'pip'],
            'bash': ['bash', 'shell', 'sh'],
            'docker': ['docker', 'container'],
            'api': ['api', 'rest', 'webhook'],
            'monitoring': ['monitor', 'alert', 'uptime'],
            'ci/cd': ['ci/cd', 'jenkins', 'github actions'],
            'database': ['database', 'sql', 'postgres', 'mysql'],
            'cloud': ['aws', 'azure', 'gcp', 'cloud'],
            'notification': ['slack', 'email', 'sms', 'notification'],
            'web-scraping': ['scraping', 'crawling', 'selenium'],
            'data-processing': ['csv', 'json', 'etl', 'data'],
            'backup': ['backup', 'sync', 'copy'],
            'security': ['security', 'auth', 'ssl', 'encrypt']
        }
        
        found_tags = []
        for tag, keywords in tag_keywords.items():
            if any(keyword in text for keyword in keywords):
                found_tags.append(tag)
                
        return found_tags[:6]  # Limit to 6 tags

    def rank_automation_sources(self, ideas: List[AutomationIdea]) -> Dict[str, Any]:
        """Rank automation sources by quality and usefulness"""
        source_stats = {}
        
        for idea in ideas:
            source = idea.source
            if source not in source_stats:
                source_stats[source] = {
                    'total_ideas': 0,
                    'avg_productivity': 0,
                    'avg_priority_score': 0,
                    'total_upvotes': 0,
                    'avg_engagement': 0,
                    'categories': {},
                    'difficulties': {'easy': 0, 'medium': 0, 'hard': 0}
                }
            
            stats = source_stats[source]
            stats['total_ideas'] += 1
            stats['avg_productivity'] += idea.productivity_estimate
            stats['avg_priority_score'] += idea.priority_score
            stats['total_upvotes'] += idea.upvotes
            stats['avg_engagement'] += idea.engagement_score
            
            # Track categories
            if idea.category not in stats['categories']:
                stats['categories'][idea.category] = 0
            stats['categories'][idea.category] += 1
            
            # Track difficulties
            stats['difficulties'][idea.difficulty] += 1
        
        # Calculate averages and rankings
        for source, stats in source_stats.items():
            if stats['total_ideas'] > 0:
                stats['avg_productivity'] = round(stats['avg_productivity'] / stats['total_ideas'], 1)
                stats['avg_priority_score'] = round(stats['avg_priority_score'] / stats['total_ideas'], 2)
                stats['avg_engagement'] = round(stats['avg_engagement'] / stats['total_ideas'], 1)
                
                # Calculate source quality score
                stats['quality_score'] = (
                    stats['avg_productivity'] * 0.4 +
                    stats['avg_priority_score'] * 0.3 +
                    min(stats['avg_engagement'], 20) * 0.3  # Cap engagement impact
                )
        
        # Sort sources by quality score
        ranked_sources = sorted(
            source_stats.items(),
            key=lambda x: x[1]['quality_score'],
            reverse=True
        )
        
        return {
            'rankings': ranked_sources,
            'summary': {
                'total_sources': len(source_stats),
                'total_ideas': len(ideas),
                'best_source': ranked_sources[0][0] if ranked_sources else None,
                'avg_productivity_all': round(sum(idea.productivity_estimate for idea in ideas) / len(ideas), 1) if ideas else 0
            }
        }

# Convenience function for external use
async def discover_automation_ideas(limit_per_source: int = 25) -> Tuple[List[AutomationIdea], Dict[str, Any]]:
    """
    Discover automation ideas and rank sources
    
    Returns:
        Tuple of (ideas_list, source_rankings)
    """
    async with AutomationIdeasDiscovery() as discovery:
        ideas = await discovery.discover_automation_ideas(limit_per_source)
        rankings = discovery.rank_automation_sources(ideas)
        return ideas, rankings