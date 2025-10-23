"""
Scraping Configuration Models
"""
from pydantic import BaseModel
from typing import Optional, Dict, Any


class ScrapingConfig(BaseModel):
    """Scraping configuration settings"""
    # Basic settings
    concurrent_requests: int = 4
    download_delay: float = 1.0
    max_articles_per_source: int = 10
    
    # Historical search settings
    days_back: int = 7  # How many days back to search
    search_depth: str = "recent"  # "recent", "weekly", "monthly", "deep"
    
    # Content quality filters
    min_content_length: int = 100
    skip_duplicates: bool = True
    language_filter: Optional[str] = None  # None = all languages, "en", "zh", "ja"


class ConfigResponse(BaseModel):
    """Response model for configuration updates"""
    message: str
    config: ScrapingConfig