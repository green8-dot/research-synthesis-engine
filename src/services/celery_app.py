"""
Celery Configuration for Research Synthesis Engine
"""
from celery import Celery
from research_synthesis.config.settings import settings

# Create Celery instance
celery_app = Celery(
    'research_synthesis',
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        'research_synthesis.services.scraping_tasks',
        'research_synthesis.services.analysis_tasks',
        'research_synthesis.services.report_tasks'
    ]
)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
    task_routes={
        'research_synthesis.services.scraping_tasks.*': {'queue': 'scraping'},
        'research_synthesis.services.analysis_tasks.*': {'queue': 'analysis'},
        'research_synthesis.services.report_tasks.*': {'queue': 'reports'}
    },
    task_default_queue='default',
    task_default_exchange='default',
    task_default_routing_key='default',
    beat_schedule={
        'daily-scrape': {
            'task': 'research_synthesis.services.scraping_tasks.daily_scrape',
            'schedule': 86400.0,  # Daily
        },
        'hourly-analysis': {
            'task': 'research_synthesis.services.analysis_tasks.analyze_new_content',
            'schedule': 3600.0,  # Hourly
        },
    }
)

# Auto-discover tasks
celery_app.autodiscover_tasks()

if __name__ == '__main__':
    celery_app.start()