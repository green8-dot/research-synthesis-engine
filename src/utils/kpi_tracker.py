"""
KPI Tracking System with Historical Data and Percentage Changes
Tracks key metrics over time and calculates meaningful trends
"""
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

class KPITracker:
    def __init__(self, db_path: str = "kpi_history.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize KPI tracking database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create KPI history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS kpi_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value INTEGER NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create daily summaries table for better performance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_summaries (
                date TEXT PRIMARY KEY,
                total_articles INTEGER DEFAULT 0,
                total_reports INTEGER DEFAULT 0,
                active_sources INTEGER DEFAULT 0,
                total_entities INTEGER DEFAULT 0,
                new_articles INTEGER DEFAULT 0,
                new_reports INTEGER DEFAULT 0,
                new_entities INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_metrics(self, metrics: Dict[str, int]):
        """Record current metrics snapshot"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        for metric_name, value in metrics.items():
            cursor.execute('''
                INSERT INTO kpi_history (timestamp, metric_name, metric_value)
                VALUES (?, ?, ?)
            ''', (timestamp, metric_name, value))
        
        # Update daily summary
        today = datetime.now().date().isoformat()
        
        # Get or create today's summary
        cursor.execute('SELECT * FROM daily_summaries WHERE date = ?', (today,))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing summary with latest values
            cursor.execute('''
                UPDATE daily_summaries 
                SET total_articles = ?, total_reports = ?, active_sources = ?, total_entities = ?
                WHERE date = ?
            ''', (
                metrics.get('total_articles', 0),
                metrics.get('total_reports', 0), 
                metrics.get('active_sources', 0),
                metrics.get('total_entities', 0),
                today
            ))
        else:
            # Create new daily summary
            cursor.execute('''
                INSERT INTO daily_summaries 
                (date, total_articles, total_reports, active_sources, total_entities)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                today,
                metrics.get('total_articles', 0),
                metrics.get('total_reports', 0),
                metrics.get('active_sources', 0),
                metrics.get('total_entities', 0)
            ))
        
        conn.commit()
        conn.close()
    
    async def update_all_kpis(self):
        """Force update all KPIs"""
        try:
            # Clear cache to force refresh
            self.kpi_cache.clear()
            self.last_update = datetime.now() - timedelta(hours=1)  # Force update
            
            # Trigger recalculation
            metrics = await self.get_dashboard_kpis()
            logger.info(f"KPIs updated: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error updating KPIs: {e}")
            return {}
    
    def get_metrics_with_changes(self) -> Dict[str, Any]:
        """Get current metrics with percentage changes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current metrics
        current_metrics = self._get_current_metrics()
        
        # Get yesterday's metrics for comparison
        yesterday = (datetime.now() - timedelta(days=1)).date().isoformat()
        cursor.execute('SELECT * FROM daily_summaries WHERE date = ?', (yesterday,))
        yesterday_data = cursor.fetchone()
        
        # Get week ago metrics
        week_ago = (datetime.now() - timedelta(days=7)).date().isoformat()
        cursor.execute('SELECT * FROM daily_summaries WHERE date <= ? ORDER BY date DESC LIMIT 1', (week_ago,))
        week_ago_data = cursor.fetchone()
        
        # Calculate percentage changes
        result = {}
        
        for metric_name, current_value in current_metrics.items():
            # Yesterday comparison
            yesterday_value = 0
            if yesterday_data:
                column_map = {
                    'total_articles': 1,
                    'total_reports': 2, 
                    'active_sources': 3,
                    'total_entities': 4
                }
                if metric_name in column_map:
                    yesterday_value = yesterday_data[column_map[metric_name]] or 0
            
            # Week ago comparison
            week_ago_value = 0
            if week_ago_data:
                if metric_name in column_map:
                    week_ago_value = week_ago_data[column_map[metric_name]] or 0
            
            # Calculate changes
            daily_change = current_value - yesterday_value
            daily_change_pct = self._calculate_percentage_change(yesterday_value, current_value)
            
            weekly_change = current_value - week_ago_value
            weekly_change_pct = self._calculate_percentage_change(week_ago_value, current_value)
            
            result[metric_name] = {
                'current': current_value,
                'daily_change': daily_change,
                'daily_change_pct': daily_change_pct,
                'weekly_change': weekly_change, 
                'weekly_change_pct': weekly_change_pct,
                'trend': self._determine_trend(daily_change_pct, weekly_change_pct)
            }
        
        conn.close()
        return result
    
    def _get_current_metrics(self) -> Dict[str, int]:
        """Get current system metrics directly from database"""
        try:
            from research_synthesis.database.connection import get_postgres_session
            from research_synthesis.database.models import Article, Report, Entity, Source
            from sqlalchemy import select, func
            import asyncio
            
            async def get_all_metrics():
                try:
                    async for db_session in get_postgres_session():
                        # Get article count
                        articles_result = await db_session.execute(select(func.count(Article.id)))
                        total_articles = articles_result.scalar() or 0
                        
                        # Get reports count  
                        reports_result = await db_session.execute(select(func.count(Report.id)))
                        total_reports = reports_result.scalar() or 0
                        
                        # Get entities count
                        entities_result = await db_session.execute(select(func.count(Entity.id)))
                        total_entities = entities_result.scalar() or 0
                        
                        # Get active sources count - sources that have scraped recently
                        sources_result = await db_session.execute(
                            select(func.count(Source.id)).where(Source.is_active == True)
                        )
                        active_sources = sources_result.scalar() or 0
                        
                        return {
                            'total_articles': total_articles,
                            'total_reports': total_reports,
                            'active_sources': active_sources,
                            'total_entities': total_entities
                        }
                except Exception as e:
                    print(f"Database error getting metrics: {e}")
                    return {
                        'total_articles': 0,
                        'total_reports': 0,
                        'active_sources': 0,
                        'total_entities': 0
                    }
            
            # Run async function safely
            try:
                # Try to get existing loop
                loop = asyncio.get_running_loop()
                # If we're in an async context, create a task
                import asyncio
                import threading
                
                # Run in a separate thread to avoid event loop conflicts
                result = {}
                def run_in_thread():
                    nonlocal result
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(get_all_metrics())
                    finally:
                        new_loop.close()
                
                thread = threading.Thread(target=run_in_thread)
                thread.start()
                thread.join()
                return result
                
            except RuntimeError:
                # No event loop running, safe to create one
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(get_all_metrics())
                finally:
                    loop.close()
                return result
        except Exception as e:
            print(f"Error getting current metrics: {e}")
            # Fallback - try to get some basic stats
            return {
                'total_articles': 0,
                'total_reports': 0,
                'active_sources': 5,  # Default fallback
                'total_entities': 0
            }
    
    def _calculate_percentage_change(self, old_value: int, new_value: int) -> float:
        """Calculate percentage change between two values"""
        if old_value == 0:
            return 100.0 if new_value > 0 else 0.0
        
        change = ((new_value - old_value) / old_value) * 100
        return round(change, 1)
    
    def _determine_trend(self, daily_change_pct: float, weekly_change_pct: float) -> str:
        """Determine overall trend direction"""
        # Strong positive trend
        if daily_change_pct > 5 and weekly_change_pct > 10:
            return "strong_up"
        # Positive trend
        elif daily_change_pct > 0 or weekly_change_pct > 0:
            return "up"
        # Strong negative trend
        elif daily_change_pct < -5 and weekly_change_pct < -10:
            return "strong_down"
        # Negative trend
        elif daily_change_pct < 0 or weekly_change_pct < 0:
            return "down"
        # Stable
        else:
            return "stable"
    
    def get_historical_data(self, metric_name: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get historical data for a specific metric"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        column_map = {
            'total_articles': 'total_articles',
            'total_reports': 'total_reports',
            'active_sources': 'active_sources', 
            'total_entities': 'total_entities'
        }
        
        if metric_name not in column_map:
            return []
        
        cursor.execute(f'''
            SELECT date, {column_map[metric_name]} 
            FROM daily_summaries 
            WHERE date >= ? AND date <= ?
            ORDER BY date ASC
        ''', (start_date.isoformat(), end_date.isoformat()))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'date': row[0],
                'value': row[1] or 0
            })
        
        conn.close()
        return results
    
    def get_growth_rate(self, metric_name: str, period_days: int = 7) -> Dict[str, float]:
        """Calculate growth rate over a period"""
        historical = self.get_historical_data(metric_name, period_days + 5)
        
        if len(historical) < 2:
            return {'growth_rate': 0.0, 'confidence': 'low'}
        
        # Get start and end values
        start_value = historical[0]['value']
        end_value = historical[-1]['value']
        
        # Calculate compound daily growth rate
        days = len(historical) - 1
        if start_value > 0 and days > 0:
            growth_rate = (pow(end_value / start_value, 1/days) - 1) * 100
        else:
            growth_rate = 0.0
        
        # Determine confidence based on data points
        confidence = 'high' if len(historical) >= period_days else 'medium' if len(historical) >= 3 else 'low'
        
        return {
            'growth_rate': round(growth_rate, 2),
            'confidence': confidence,
            'data_points': len(historical),
            'period_days': days
        }

# Global instance
kpi_tracker = KPITracker()