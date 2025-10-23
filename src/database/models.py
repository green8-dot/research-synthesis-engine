"""
Database Models for Research Synthesis Engine
"""
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, Boolean, JSON, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


# Association tables for many-to-many relationships
article_entity_association = Table(
    'article_entities',
    Base.metadata,
    Column('article_id', Integer, ForeignKey('articles.id')),
    Column('entity_id', Integer, ForeignKey('entities.id'))
)

report_source_association = Table(
    'report_sources',
    Base.metadata,
    Column('report_id', Integer, ForeignKey('reports.id')),
    Column('source_id', Integer, ForeignKey('sources.id'))
)


class Source(Base):
    """Data source model"""
    __tablename__ = 'sources'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    url = Column(String(500), unique=True)
    source_type = Column(String(50))  # news, academic, government, industry
    credibility_score = Column(Float, default=0.5)
    last_scraped = Column(DateTime)
    scraping_frequency = Column(Integer, default=3600)  # seconds
    is_active = Column(Boolean, default=True)
    source_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    articles = relationship("Article", back_populates="source")


class AutomationIdea(Base):
    """Automation ideas and suggestions model"""
    __tablename__ = 'automation_ideas'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(500), nullable=False)
    description = Column(Text)
    source = Column(String(255))
    source_url = Column(String(500))
    category = Column(String(100))  # workflow, data, integration, monitoring, etc.
    complexity = Column(String(50))  # simple, moderate, complex
    potential_impact = Column(String(50))  # low, medium, high
    tools_required = Column(JSON)  # List of tools/technologies
    implementation_notes = Column(Text)
    priority_score = Column(Float, default=0.5)
    feasibility_score = Column(Float, default=0.5)
    votes = Column(Integer, default=0)
    implemented = Column(Boolean, default=False)
    implementation_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Article(Base):
    """Scraped article/document model"""
    __tablename__ = 'articles'
    
    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey('sources.id'))
    url = Column(String(500), unique=True)
    title = Column(String(500))
    author = Column(String(255))
    published_date = Column(DateTime)
    content = Column(Text)
    summary = Column(Text)
    content_type = Column(String(50))  # article, paper, report, patent
    language = Column(String(10), default='en')
    
    # Analysis results
    sentiment_score = Column(Float)
    relevance_score = Column(Float)
    quality_score = Column(Float)
    
    # Extracted data
    extracted_facts = Column(JSON)
    extracted_dates = Column(JSON)
    extracted_numbers = Column(JSON)
    citations = Column(JSON)
    
    # Metadata
    word_count = Column(Integer)
    reading_time = Column(Integer)  # minutes
    tags = Column(JSON)
    categories = Column(JSON)
    
    # Embeddings
    title_embedding = Column(JSON)  # Store as list of floats
    content_embedding = Column(JSON)
    
    # Timestamps
    scraped_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    source = relationship("Source", back_populates="articles")
    entities = relationship("Entity", secondary=article_entity_association, back_populates="articles")
    insights = relationship("Insight", back_populates="article")


class Entity(Base):
    """Named entity model (companies, people, technologies, etc.)"""
    __tablename__ = 'entities'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True)
    entity_type = Column(String(50))  # company, person, technology, location, etc.
    description = Column(Text)
    aliases = Column(JSON)
    
    # Space industry specific
    is_space_related = Column(Boolean, default=False)
    space_sector = Column(String(100))  # launch, satellite, manufacturing, etc.
    
    # Metadata
    wikipedia_url = Column(String(500))
    official_website = Column(String(500))
    social_media = Column(JSON)
    founded_date = Column(DateTime)
    headquarters = Column(String(255))
    
    # Analytics
    mention_count = Column(Integer, default=0)
    sentiment_average = Column(Float)
    last_mentioned = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    articles = relationship("Article", secondary=article_entity_association, back_populates="entities")
    relationships = relationship("EntityRelationship", 
                                foreign_keys="EntityRelationship.entity1_id",
                                back_populates="entity1")


class EntityRelationship(Base):
    """Relationships between entities"""
    __tablename__ = 'entity_relationships'
    
    id = Column(Integer, primary_key=True)
    entity1_id = Column(Integer, ForeignKey('entities.id'))
    entity2_id = Column(Integer, ForeignKey('entities.id'))
    relationship_type = Column(String(100))  # partnership, competitor, subsidiary, etc.
    confidence = Column(Float)
    evidence = Column(JSON)  # List of article IDs supporting this relationship
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    entity1 = relationship("Entity", foreign_keys=[entity1_id], back_populates="relationships")
    entity2 = relationship("Entity", foreign_keys=[entity2_id])


class Insight(Base):
    """Extracted insights and trends"""
    __tablename__ = 'insights'
    
    id = Column(Integer, primary_key=True)
    article_id = Column(Integer, ForeignKey('articles.id'))
    insight_type = Column(String(50))  # trend, anomaly, prediction, fact
    title = Column(String(500))
    description = Column(Text)
    confidence = Column(Float)
    importance = Column(Float)
    
    # Space manufacturing specific
    is_manufacturing_related = Column(Boolean, default=False)
    manufacturing_category = Column(String(100))
    
    # Supporting data
    supporting_facts = Column(JSON)
    related_entities = Column(JSON)
    timeframe = Column(String(50))
    
    # Validation
    is_verified = Column(Boolean, default=False)
    verification_sources = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    article = relationship("Article", back_populates="insights")


class Report(Base):
    """Generated research reports"""
    __tablename__ = 'reports'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(500))
    report_type = Column(String(50))  # executive, technical, market, competitive
    executive_summary = Column(Text)
    full_content = Column(Text)
    
    # Report metadata
    word_count = Column(Integer)
    reading_time = Column(Integer)
    confidence_score = Column(Float)
    
    # Sections
    sections = Column(JSON)  # List of section titles and content
    key_findings = Column(JSON)
    recommendations = Column(JSON)
    risks = Column(JSON)
    opportunities = Column(JSON)
    
    # Visualizations
    charts = Column(JSON)  # Chart configurations
    tables = Column(JSON)  # Table data
    
    # Generation details
    generation_params = Column(JSON)
    generation_time = Column(Float)  # seconds
    model_used = Column(String(100))
    
    # Export info
    exported_formats = Column(JSON)  # pdf, docx, pptx, etc.
    export_urls = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sources = relationship("Source", secondary=report_source_association)


class SearchQuery(Base):
    """User search queries for analytics"""
    __tablename__ = 'search_queries'
    
    id = Column(Integer, primary_key=True)
    query_text = Column(Text)
    query_type = Column(String(50))  # keyword, semantic, entity
    filters = Column(JSON)
    results_count = Column(Integer)
    user_id = Column(String(100))
    session_id = Column(String(100))
    response_time = Column(Float)  # milliseconds
    created_at = Column(DateTime, default=datetime.utcnow)


class ScrapingJob(Base):
    """Scraping job tracking"""
    __tablename__ = 'scraping_jobs'
    
    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey('sources.id'))
    job_type = Column(String(50))  # scheduled, manual, real-time
    status = Column(String(50))  # pending, running, completed, failed
    
    # Statistics
    pages_scraped = Column(Integer, default=0)
    articles_found = Column(Integer, default=0)
    new_articles = Column(Integer, default=0)
    errors_count = Column(Integer, default=0)
    
    # Timing
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    duration = Column(Float)  # seconds
    
    # Error handling
    error_messages = Column(JSON)
    retry_count = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AuditLog(Base):
    """Comprehensive audit log for all system activities"""
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), default='system')
    session_id = Column(String(100))
    action = Column(String(100))  # create, read, update, delete, api_call, report_generate
    resource_type = Column(String(100))  # report, article, entity, usage_tracking
    resource_id = Column(String(100))
    
    # Activity details
    activity_description = Column(Text)
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    endpoint = Column(String(200))
    method = Column(String(10))  # GET, POST, PUT, DELETE
    
    # Data changes
    old_values = Column(JSON)
    new_values = Column(JSON)
    
    # Cost tracking
    operation_cost_usd = Column(Float, default=0.0)
    tokens_used = Column(Integer, default=0)
    
    # Result info
    status = Column(String(50))  # success, error, warning
    error_message = Column(Text)
    response_time_ms = Column(Float)
    
    # Compliance tracking
    compliance_checked = Column(Boolean, default=False)
    data_classification = Column(String(50))  # public, internal, confidential
    retention_period_days = Column(Integer, default=2555)  # 7 years
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)  # For automatic cleanup


class UsageMetrics(Base):
    """Persistent storage for usage metrics and costs"""
    __tablename__ = 'usage_metrics'
    
    id = Column(Integer, primary_key=True)
    date = Column(String(10))  # YYYY-MM-DD
    service = Column(String(50))  # mongodb, claude, system
    metric_type = Column(String(50))  # api_calls, storage_mb, tokens, operations
    
    # Quantities
    quantity = Column(Float)
    cost_usd = Column(Float)
    
    # Breakdown
    breakdown = Column(JSON)  # Detailed breakdown by operation/model/etc
    
    # Aggregation level
    aggregation_level = Column(String(20))  # daily, monthly, yearly
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ReportVersion(Base):
    """Version control for reports to track changes"""
    __tablename__ = 'report_versions'
    
    id = Column(Integer, primary_key=True)
    report_id = Column(Integer, ForeignKey('reports.id'))
    version_number = Column(Integer)
    
    # Content snapshot
    title = Column(String(500))
    full_content = Column(Text)
    sections = Column(JSON)
    key_findings = Column(JSON)
    
    # Version metadata
    change_description = Column(Text)
    changed_by = Column(String(100))
    change_type = Column(String(50))  # major, minor, patch
    is_current = Column(Boolean, default=False)
    
    # File attachments
    file_attachments = Column(JSON)  # URLs to exported files
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    report = relationship("Report", backref="versions")