"""
OrbitScope Intelligence Suite - Integrated Database Models
Merging MoneyMan ML and Research Synthesis Engine schemas
"""

from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Optional, Dict, Any, List

Base = declarative_base()

# ============================================================================
# FINANCIAL INTELLIGENCE MODELS (from MoneyMan ML)
# ============================================================================

class FinancialAccount(Base):
    """Financial accounts from Plaid integration"""
    __tablename__ = 'financial_accounts'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), default='user_001')
    account_id = Column(String(100), unique=True)
    institution_name = Column(String(200))
    account_name = Column(String(200))
    account_type = Column(String(50))
    account_subtype = Column(String(50))
    balance_available = Column(Float)
    balance_current = Column(Float)
    balance_limit = Column(Float)
    iso_currency_code = Column(String(10))
    plaid_access_token = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Transaction(Base):
    """Financial transactions from Plaid"""
    __tablename__ = 'transactions'
    
    id = Column(Integer, primary_key=True)
    account_id = Column(String(100), ForeignKey('financial_accounts.account_id'))
    transaction_id = Column(String(100), unique=True)
    amount = Column(Float)
    iso_currency_code = Column(String(10))
    date = Column(DateTime)
    name = Column(String(500))
    merchant_name = Column(String(200))
    category = Column(JSON)  # List of categories
    account_owner = Column(String(100))
    pending = Column(Boolean, default=False)
    transaction_type = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class BusinessMetric(Base):
    """Business analytics and KPI tracking"""
    __tablename__ = 'business_metrics'
    
    id = Column(Integer, primary_key=True)
    metric_name = Column(String(100))
    metric_value = Column(Float)
    metric_type = Column(String(50))  # revenue, expense, profit, growth
    period_start = Column(DateTime)
    period_end = Column(DateTime)
    currency = Column(String(10), default='USD')
    metric_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

# ============================================================================
# RESEARCH INTELLIGENCE MODELS (from Research Synthesis Engine)
# ============================================================================

class Source(Base):
    """Data sources for web scraping"""
    __tablename__ = 'sources'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), unique=True)
    base_url = Column(String(500))
    source_type = Column(String(50))  # news, academic, government, company
    industry_focus = Column(String(100))  # space, finance, manufacturing, etc.
    credibility_score = Column(Float, default=0.8)
    is_active = Column(Boolean, default=True)
    scraping_config = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Article(Base):
    """Scraped articles and content"""
    __tablename__ = 'articles'
    
    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey('sources.id'))
    url = Column(String(1000), unique=True)
    title = Column(String(500))
    author = Column(String(200))
    published_date = Column(DateTime)
    content = Column(Text)
    summary = Column(Text)
    content_type = Column(String(50), default='article')
    language = Column(String(10), default='en')
    
    # AI Analysis
    sentiment_score = Column(Float)
    relevance_score = Column(Float)
    quality_score = Column(Float)
    extracted_facts = Column(JSON)
    extracted_dates = Column(JSON)
    extracted_numbers = Column(JSON)
    citations = Column(JSON)
    
    # Metadata
    word_count = Column(Integer)
    reading_time = Column(Integer)  # minutes
    tags = Column(JSON)
    categories = Column(JSON)
    title_embedding = Column(JSON)  # Vector embedding
    content_embedding = Column(JSON)  # Vector embedding
    
    scraped_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Entity(Base):
    """Knowledge graph entities"""
    __tablename__ = 'entities'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), unique=True)
    entity_type = Column(String(50))  # company, person, technology, location
    description = Column(Text)
    aliases = Column(JSON)  # Alternative names
    is_space_related = Column(Boolean, default=False)
    is_finance_related = Column(Boolean, default=False)
    space_sector = Column(String(100))  # launch, satellite, manufacturing
    financial_sector = Column(String(100))  # banking, investment, fintech
    
    # External Links
    wikipedia_url = Column(String(500))
    official_website = Column(String(500))
    social_media = Column(JSON)
    
    # Company-specific data
    founded_date = Column(DateTime)
    headquarters = Column(String(200))
    
    # Analytics
    mention_count = Column(Integer, default=0)
    sentiment_average = Column(Float)
    last_mentioned = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Report(Base):
    """Generated intelligence reports"""
    __tablename__ = 'reports'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(500))
    report_type = Column(String(50))  # financial, research, intelligence, orbital
    executive_summary = Column(Text)
    full_content = Column(Text)
    word_count = Column(Integer)
    reading_time = Column(Integer)  # minutes
    confidence_score = Column(Float)
    
    # Structure
    sections = Column(JSON)
    key_findings = Column(JSON)
    recommendations = Column(JSON)
    risks = Column(JSON)
    opportunities = Column(JSON)
    
    # Visualizations
    charts = Column(JSON)
    tables = Column(JSON)
    
    # Generation metadata
    generation_params = Column(JSON)
    generation_time = Column(Float)  # seconds
    model_used = Column(String(50))
    
    # Export options
    exported_formats = Column(JSON)  # pdf, docx, html
    export_urls = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# ============================================================================
# ORBITAL INTELLIGENCE MODELS (legacy OrbitScope ML)
# ============================================================================

class ISS_Position(Base):
    """ISS position tracking (legacy)"""
    __tablename__ = 'iss_positions'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)
    timestamp_unix = Column(Integer)
    latitude = Column(Float)
    longitude = Column(Float)
    altitude_km = Column(Float)
    velocity_kmh = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class TLE_Data(Base):
    """Two-Line Element sets for orbital objects"""
    __tablename__ = 'tle_data'
    
    id = Column(Integer, primary_key=True)
    satellite_name = Column(String(200))
    norad_id = Column(Integer)
    tle_line1 = Column(String(100))
    tle_line2 = Column(String(100))
    epoch = Column(DateTime)
    source = Column(String(100))  # space-track.org, celestrak
    created_at = Column(DateTime, default=datetime.utcnow)

# ============================================================================
# INTELLIGENCE SUITE INTEGRATED MODELS
# ============================================================================

class IntelligenceInsight(Base):
    """Cross-domain intelligence insights"""
    __tablename__ = 'intelligence_insights'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(500))
    insight_type = Column(String(50))  # correlation, prediction, anomaly, opportunity
    domains = Column(JSON)  # ['financial', 'research', 'orbital']
    
    # Source data references
    financial_data_refs = Column(JSON)
    research_data_refs = Column(JSON)
    orbital_data_refs = Column(JSON)
    
    # Insight content
    description = Column(Text)
    confidence_score = Column(Float)
    impact_score = Column(Float)  # business impact 1-10
    urgency = Column(String(20))  # low, medium, high, critical
    
    # Actionability
    recommended_actions = Column(JSON)
    estimated_value = Column(Float)  # USD
    implementation_complexity = Column(String(20))  # low, medium, high
    
    # Metadata
    ai_model_used = Column(String(50))
    processing_time = Column(Float)  # seconds
    data_points_analyzed = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SystemIntegration(Base):
    """System integration and health monitoring"""
    __tablename__ = 'system_integrations'
    
    id = Column(Integer, primary_key=True)
    service_name = Column(String(100))  # moneyman_ml, research_engine, plaid, etc.
    service_type = Column(String(50))  # api, database, ml_service, external
    status = Column(String(20))  # healthy, degraded, offline
    
    # Connection details
    endpoint_url = Column(String(500))
    last_health_check = Column(DateTime)
    response_time_ms = Column(Float)
    
    # Metrics
    requests_today = Column(Integer, default=0)
    errors_today = Column(Integer, default=0)
    uptime_percentage = Column(Float, default=100.0)
    
    # Configuration
    config_hash = Column(String(100))  # For detecting configuration changes
    integration_settings = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# ============================================================================
# AUDIT AND COMPLIANCE MODELS (Enhanced)
# ============================================================================

class AuditLog(Base):
    """Comprehensive audit log for all system activities"""
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), default='system')
    session_id = Column(String(100))
    action = Column(String(100))  # create, read, update, delete, api_call, report_generate
    resource_type = Column(String(100))  # report, article, entity, transaction, insight
    resource_id = Column(String(100))
    
    # Integration tracking
    source_system = Column(String(50))  # moneyman_ml, research_engine, orbital_ml
    target_system = Column(String(50))
    integration_point = Column(String(100))
    
    # Request/Response data
    request_data = Column(JSON)
    response_data = Column(JSON)
    http_method = Column(String(10))
    endpoint = Column(String(200))
    status_code = Column(Integer)
    
    # Performance metrics
    execution_time_ms = Column(Float)
    memory_usage_mb = Column(Float)
    
    # Cost tracking
    operation_cost_usd = Column(Float, default=0.0)
    tokens_used = Column(Integer, default=0)
    
    # Compliance tracking
    compliance_checked = Column(Boolean, default=False)
    data_classification = Column(String(50))  # public, internal, confidential
    retention_period_days = Column(Integer, default=2555)  # 7 years
    
    # Geographic and network info
    ip_address = Column(String(50))
    user_agent = Column(String(500))
    geographic_location = Column(String(100))
    
    timestamp = Column(DateTime, default=datetime.utcnow)

class UsageMetrics(Base):
    """System usage and cost tracking"""
    __tablename__ = 'usage_metrics'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=datetime.utcnow)
    metric_type = Column(String(50))  # api_calls, claude_usage, database_operations
    service = Column(String(100))  # moneyman_ml, research_engine, plaid, mongodb
    
    # Volume metrics
    total_requests = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)
    
    # Performance metrics
    avg_response_time_ms = Column(Float)
    total_processing_time_ms = Column(Float)
    
    # Resource utilization
    cpu_usage_percent = Column(Float)
    memory_usage_mb = Column(Float)
    disk_usage_mb = Column(Float)
    network_io_mb = Column(Float)
    
    # Cost metrics
    cost_usd = Column(Float, default=0.0)
    tokens_consumed = Column(Integer, default=0)
    data_transfer_gb = Column(Float, default=0.0)
    
    # Business metrics
    reports_generated = Column(Integer, default=0)
    insights_created = Column(Integer, default=0)
    financial_transactions_processed = Column(Integer, default=0)
    articles_scraped = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)