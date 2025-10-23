"""
Database Connection Management
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
try:
    from motor.motor_asyncio import AsyncIOMotorClient
except ImportError:
    AsyncIOMotorClient = None

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

try:
    from elasticsearch import AsyncElasticsearch
except ImportError:
    AsyncElasticsearch = None
from loguru import logger

from research_synthesis.config.settings import settings
from research_synthesis.database.models import Base


# PostgreSQL
postgres_engine = None
AsyncSessionLocal = None

# MongoDB (preserved from original)
mongodb_client = None
mongodb_database = None

# Redis
redis_client = None

# Elasticsearch
elasticsearch_client = None

# ChromaDB
chroma_client = None


async def init_postgres():
    """Initialize database connection"""
    global postgres_engine, AsyncSessionLocal
    
    try:
        # Create async engine (SQLite or PostgreSQL)
        if "sqlite" in settings.postgres_url:
            postgres_engine = create_async_engine(
                settings.postgres_url,
                echo=settings.DEBUG
            )
        else:
            postgres_engine = create_async_engine(
                settings.postgres_url,
                echo=settings.DEBUG,
                pool_size=20,
                max_overflow=0
            )
        
        # Create session factory
        AsyncSessionLocal = sessionmaker(
            postgres_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Create tables
        async with postgres_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        db_type = "SQLite" if "sqlite" in settings.postgres_url else "PostgreSQL"
        logger.info(f"{db_type} connected successfully")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise


async def init_mongodb():
    """Initialize MongoDB connection"""
    global mongodb_client, mongodb_database
    
    if AsyncIOMotorClient is None:
        logger.warning("motor not installed - MongoDB unavailable")
        mongodb_client = None
        mongodb_database = None
        return
    
    try:
        mongodb_client = AsyncIOMotorClient(settings.MONGODB_URI)
        mongodb_database = mongodb_client[settings.MONGODB_DB]
        
        # Test connection
        await mongodb_client.admin.command('ping')
        logger.info("MongoDB connected successfully")
    except Exception as e:
        logger.warning(f"MongoDB not available: {e}")
        mongodb_client = None
        mongodb_database = None


async def init_redis():
    """Initialize Redis connection"""
    global redis_client
    
    if redis is None:
        logger.warning("redis not installed - Redis unavailable")
        redis_client = None
        return
    
    try:
        redis_client = await redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
            health_check_interval=30
        )
        
        # Test connection
        await redis_client.ping()
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.warning(f"Redis not available: {e}")
        redis_client = None


async def init_elasticsearch():
    """Initialize Elasticsearch connection"""
    global elasticsearch_client
    
    if AsyncElasticsearch is None:
        logger.warning("elasticsearch not installed - Elasticsearch unavailable")
        elasticsearch_client = None
        return
    
    try:
        elasticsearch_client = AsyncElasticsearch(
            [f"http://{settings.ELASTICSEARCH_HOST}:{settings.ELASTICSEARCH_PORT}"]
        )
        
        # Test connection
        info = await elasticsearch_client.info()
        logger.info(f"Elasticsearch connected: {info['version']['number']}")
    except Exception as e:
        logger.warning(f"Elasticsearch not available: {e}")
        elasticsearch_client = None


def init_chromadb():
    """Initialize ChromaDB connection"""
    global chroma_client
    
    try:
        import chromadb
        chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        logger.info("ChromaDB initialized successfully")
    except ImportError:
        logger.warning("ChromaDB not installed - vector search unavailable")
        chroma_client = None
    except Exception as e:
        logger.warning(f"ChromaDB initialization failed: {e}")
        chroma_client = None


async def init_databases():
    """Initialize all database connections"""
    await init_postgres()
    await init_mongodb()
    # await init_redis()  # Disabled for single-user app
    await init_elasticsearch()
    init_chromadb()


async def close_databases():
    """Close all database connections"""
    global postgres_engine, mongodb_client, redis_client, elasticsearch_client
    
    if postgres_engine:
        await postgres_engine.dispose()
        logger.info("PostgreSQL connection closed")
    
    if mongodb_client:
        mongodb_client.close()
        logger.info("MongoDB connection closed")
    
    # Redis disabled for single-user app
    # if redis_client:
    #     await redis_client.close()
    #     logger.info("Redis connection closed")
    
    if elasticsearch_client:
        await elasticsearch_client.close()
        logger.info("Elasticsearch connection closed")


async def get_postgres_session():
    """Get PostgreSQL session with proper cleanup"""
    if AsyncSessionLocal is None:
        raise RuntimeError("Database not initialized - call init_databases() first")
    
    session = AsyncSessionLocal()
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error(f"Database session error: {e}")
        raise e
    finally:
        await session.close()


def get_mongodb():
    """Get MongoDB database instance"""
    return mongodb_database


def get_redis():
    """Get Redis client"""
    return redis_client


def get_elasticsearch():
    """Get Elasticsearch client"""
    return elasticsearch_client


def get_chromadb():
    """Get ChromaDB client"""
    return chroma_client