"""
Scrapy Settings for Research Synthesis Spiders
"""

BOT_NAME = 'research_synthesis'

SPIDER_MODULES = ['research_synthesis.scrapers.spiders']
NEWSPIDER_MODULE = 'research_synthesis.scrapers.spiders'

# User agent
USER_AGENT = 'Research Synthesis Bot 1.0 (+http://orbitscope.ai/bot)'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure maximum concurrent requests
CONCURRENT_REQUESTS = 16
CONCURRENT_REQUESTS_PER_DOMAIN = 4

# Configure delays
DOWNLOAD_DELAY = 1.0
RANDOMIZE_DOWNLOAD_DELAY = True

# AutoThrottle
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1
AUTOTHROTTLE_MAX_DELAY = 10
AUTOTHROTTLE_TARGET_CONCURRENCY = 8.0
AUTOTHROTTLE_DEBUG = False

# Retry configuration
RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429]

# Download timeout
DOWNLOAD_TIMEOUT = 30

# Disable cookies
COOKIES_ENABLED = False

# Configure pipelines
ITEM_PIPELINES = {
    'research_synthesis.scrapers.pipelines.ValidationPipeline': 100,
    'research_synthesis.scrapers.pipelines.DuplicateFilterPipeline': 200,
    'research_synthesis.scrapers.pipelines.ProcessingPipeline': 300,
    'research_synthesis.scrapers.pipelines.DatabasePipeline': 400,
    'research_synthesis.scrapers.pipelines.ElasticsearchPipeline': 500,
}

# Configure middlewares
DOWNLOADER_MIDDLEWARES = {
    'research_synthesis.scrapers.middlewares.RotateUserAgentMiddleware': 400,
    'research_synthesis.scrapers.middlewares.ProxyMiddleware': 410,
    'scrapy.downloadermiddlewares.retry.RetryMiddleware': 550,
}

# User agent rotation list
USER_AGENT_LIST = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
]

# Proxy configuration (optional)
PROXY_LIST = []  # Add proxy URLs if needed

# Feed exports
FEEDS = {
    'data/scraped/%(name)s_%(time)s.json': {
        'format': 'json',
        'encoding': 'utf8',
        'indent': 2,
    },
}

# Cache configuration
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 3600
HTTPCACHE_DIR = 'httpcache'
HTTPCACHE_IGNORE_HTTP_CODES = [503, 500, 400]

# Memory usage limits
MEMUSAGE_LIMIT_MB = 2048
MEMUSAGE_WARNING_MB = 1536

# Request fingerprinting
DUPEFILTER_CLASS = 'scrapy.dupefilters.RFPDupeFilter'

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(levelname)s: %(message)s'

# Stats collection
STATS_CLASS = 'scrapy.statscollectors.MemoryStatsCollector'

# Telnet console (for debugging)
TELNETCONSOLE_ENABLED = False

# DNS settings
DNSCACHE_ENABLED = True
DNSCACHE_SIZE = 10000
DNSCACHE_TIMEOUT = 60

# Connection pool
REACTOR_THREADPOOL_MAXSIZE = 20

# Download handlers
DOWNLOAD_HANDLERS = {
    'http': 'scrapy.core.downloader.handlers.http.HTTPDownloadHandler',
    'https': 'scrapy.core.downloader.handlers.http.HTTPDownloadHandler',
}

# Media pipeline (for downloading images/documents)
IMAGES_STORE = 'data/images'
FILES_STORE = 'data/files'

# Custom settings for specific spiders
CUSTOM_SETTINGS = {
    'space_news': {
        'DOWNLOAD_DELAY': 2,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,
    },
    'academic_papers': {
        'DOWNLOAD_DELAY': 3,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
    },
}