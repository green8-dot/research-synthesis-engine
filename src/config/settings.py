"""
OrbitScope Intelligence Suite Configuration Settings
"""
import os
from typing import List, Dict, Any
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Main configuration for OrbitScope Intelligence Suite"""
    
    # Application Settings
    APP_NAME: str = "OrbitScope Intelligence Suite"
    APP_VERSION: str = "2.0.0"  # Major integration release
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # API Settings
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8001, env="API_PORT")
    API_PREFIX: str = "/api/v1"
    
    # Database Settings - PostgreSQL
    USE_SQLITE: bool = Field(default=True, env="USE_SQLITE")
    POSTGRES_HOST: str = Field(default="localhost", env="POSTGRES_HOST")
    POSTGRES_PORT: int = Field(default=5432, env="POSTGRES_PORT")
    POSTGRES_DB: str = Field(default="research_synthesis", env="POSTGRES_DB")
    POSTGRES_USER: str = Field(default="postgres", env="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field(default="", env="POSTGRES_PASSWORD")
    
    # MongoDB Settings (preserved from original)
    MONGODB_URI: str = Field(default="mongodb://localhost:27017/", env="MONGODB_URI")
    MONGODB_DB: str = Field(default="orbitscope_ml", env="MONGODB_DB")
    
    # Redis Settings
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # Celery Settings
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND")
    
    # Elasticsearch Settings
    ELASTICSEARCH_HOST: str = Field(default="localhost", env="ELASTICSEARCH_HOST")
    ELASTICSEARCH_PORT: int = Field(default=9200, env="ELASTICSEARCH_PORT")
    ELASTICSEARCH_INDEX_PREFIX: str = "research_"
    ELASTICSEARCH_URL: str = Field(default="http://localhost:9200", env="ELASTICSEARCH_URL")
    
    # Security Settings
    CREDENTIAL_MASTER_KEY: str = Field(default="", env="CREDENTIAL_MASTER_KEY")
    JWT_SECRET: str = Field(default="", env="JWT_SECRET")
    ENCRYPTION_ENABLED: bool = Field(default=True, env="ENCRYPTION_ENABLED")
    
    # Plaid Integration Settings
    PLAID_ENV: str = Field(default="sandbox", env="PLAID_ENV")
    PLAID_PRODUCTS: str = Field(default="transactions,auth,identity,assets", env="PLAID_PRODUCTS")
    PLAID_COUNTRY_CODES: str = Field(default="US", env="PLAID_COUNTRY_CODES")
    
    # AI Services Configuration
    AI_SERVICES_ENABLED: bool = Field(default=True, env="AI_SERVICES_ENABLED")
    SEMANTIC_SEARCH_ENABLED: bool = Field(default=True, env="SEMANTIC_SEARCH_ENABLED")
    MARKET_INTELLIGENCE_ENABLED: bool = Field(default=True, env="MARKET_INTELLIGENCE_ENABLED")
    FINANCIAL_ANALYSIS_ENABLED: bool = Field(default=True, env="FINANCIAL_ANALYSIS_ENABLED")
    
    # Performance Settings
    WORKER_PROCESSES: int = Field(default=4, env="WORKER_PROCESSES")
    MAX_CONNECTIONS: int = Field(default=100, env="MAX_CONNECTIONS")
    CACHE_ENABLED: bool = Field(default=True, env="CACHE_ENABLED")
    BACKGROUND_PROCESSING: bool = Field(default=True, env="BACKGROUND_PROCESSING")
    
    # System Information
    OPERATIONAL_MOTTO: str = Field(default="Organized and Optimized", env="OPERATIONAL_MOTTO")
    SYSTEM_VERSION: str = Field(default="3.1.0", env="SYSTEM_VERSION")
    ARCHITECTURE: str = Field(default="unified_intelligence_suite", env="ARCHITECTURE")
    AUDIT_LOGGING: bool = Field(default=True, env="AUDIT_LOGGING")
    
    # Vector Database Settings (ChromaDB)
    CHROMA_HOST: str = Field(default="localhost", env="CHROMA_HOST")
    CHROMA_PORT: int = Field(default=8000, env="CHROMA_PORT")
    CHROMA_PERSIST_DIR: str = Field(default="./chroma_db", env="CHROMA_PERSIST_DIR")
    
    # OpenAI Settings
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")
    OPENAI_MODEL: str = Field(default="gpt-4-turbo-preview", env="OPENAI_MODEL")
    OPENAI_EMBEDDING_MODEL: str = Field(default="text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    
    # Scraping Settings
    SCRAPING_USER_AGENT: str = "Research Synthesis Bot 1.0"
    SCRAPING_CONCURRENT_REQUESTS: int = 16
    SCRAPING_DOWNLOAD_DELAY: float = 1.0
    SCRAPING_ROBOTSTXT_OBEY: bool = True
    SCRAPING_AUTOTHROTTLE_ENABLED: bool = True
    
    # Space Industry Sources
    SPACE_NEWS_SOURCES: List[str] = [
        "https://spacenews.com",
        "https://www.spaceflightnow.com",
        "https://www.space.com",
        "https://spaceref.com",
        "https://www.nasaspaceflight.com"
    ]
    
    SPACE_COMPANY_SOURCES: List[str] = [
        "https://www.spacex.com/news",
        "https://www.blueorigin.com/news",
        "https://www.virgingalactic.com/media",
        "https://www.rocketlabusa.com/updates",
        "https://relativityspace.com/news"
    ]
    
    GOVERNMENT_SOURCES: List[str] = [
        "https://www.nasa.gov/news",
        "https://www.esa.int/Newsroom",
        "https://www.spaceforce.mil/News",
        "https://www.faa.gov/space"
    ]
    
    # Manufacturing Focus Areas
    MANUFACTURING_KEYWORDS: List[str] = [
        "in-space manufacturing",
        "orbital factory",
        "3D printing in space",
        "additive manufacturing",
        "microgravity production",
        "space-based production",
        "orbital assembly",
        "space industrialization",
        "fiber optic manufacturing",
        "pharmaceutical manufacturing in space"
    ]
    
    # Report Generation Settings
    REPORT_TEMPLATE_DIR: str = Field(default="./templates", env="REPORT_TEMPLATE_DIR")
    REPORT_OUTPUT_DIR: str = Field(default="./reports", env="REPORT_OUTPUT_DIR")
    REPORT_MAX_LENGTH: int = 10000
    
    # Monitoring Settings
    PROMETHEUS_ENABLED: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    PROMETHEUS_PORT: int = Field(default=9090, env="PROMETHEUS_PORT")
    
    # Security Settings (preserved)
    SECRET_KEY: str = Field(default="", env="SECRET_KEY")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Logging Settings
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: str = Field(default="research_synthesis.log", env="LOG_FILE")
    
    # Data Processing Settings
    BATCH_SIZE: int = 1000
    MAX_WORKERS: int = 4
    CACHE_TTL: int = 3600  # seconds
    
    # Quality Thresholds
    MIN_SOURCE_CREDIBILITY: float = 0.6
    MIN_CONFIDENCE_SCORE: float = 0.7
    DUPLICATE_THRESHOLD: float = 0.85
    
    @property
    def postgres_url(self) -> str:
        """Generate database connection URL (SQLite for local development)"""
        # Use SQLite for local development if PostgreSQL is not available
        if self.USE_SQLITE:
            return "sqlite+aiosqlite:///./research_synthesis.db"
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    @property
    def redis_url(self) -> str:
        """Generate Redis connection URL"""
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings()