"""
Comprehensive Audit Service
Tracks all system operations for compliance, security, and performance analysis
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import json
from functools import wraps
import inspect
from pathlib import Path

from loguru import logger
from sqlalchemy import select, func, and_, or_
from research_synthesis.database.connection import get_postgres_session
from research_synthesis.database.models import AuditLog, UsageMetrics, ReportVersion, Report
from research_synthesis.services.usage_tracking_service import usage_tracker


class AuditService:
    """Comprehensive audit and compliance service"""
    
    def __init__(self):
        self.session_cache = {}
        
    async def log_activity(
        self,
        action: str,
        resource_type: str,
        resource_id: str = None,
        user_id: str = 'system',
        session_id: str = None,
        activity_description: str = None,
        old_values: Dict = None,
        new_values: Dict = None,
        operation_cost_usd: float = 0.0,
        tokens_used: int = 0,
        status: str = 'success',
        error_message: str = None,
        response_time_ms: float = 0.0,
        ip_address: str = None,
        user_agent: str = None,
        endpoint: str = None,
        method: str = None,
        data_classification: str = 'internal'
    ):
        """Log system activity with comprehensive metadata"""
        try:
            async for session in get_postgres_session():
                audit_entry = AuditLog(
                    user_id=user_id,
                    session_id=session_id,
                    action=action,
                    resource_type=resource_type,
                    resource_id=str(resource_id) if resource_id else None,
                    activity_description=activity_description,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    endpoint=endpoint,
                    method=method,
                    old_values=old_values,
                    new_values=new_values,
                    operation_cost_usd=operation_cost_usd,
                    tokens_used=tokens_used,
                    status=status,
                    error_message=error_message,
                    response_time_ms=response_time_ms,
                    compliance_checked=True,
                    data_classification=data_classification,
                    expires_at=datetime.utcnow() + timedelta(days=2555)  # 7 years retention
                )
                
                session.add(audit_entry)
                await session.commit()
                break
                
        except Exception as e:
            logger.error(f"Failed to log audit entry: {e}")
            # Fallback to file logging
            try:
                audit_file = Path("data/audit_fallback.log")
                audit_file.parent.mkdir(exist_ok=True)
                
                with open(audit_file, 'a') as f:
                    f.write(f"{datetime.utcnow().isoformat()} | {action} | {resource_type} | {status} | {error_message or 'OK'}/n")
            except:
                pass  # Silent fallback
    
    async def create_report_version(
        self,
        report_id: int,
        title: str,
        full_content: str,
        sections: Dict,
        key_findings: List,
        change_description: str = None,
        changed_by: str = 'system',
        change_type: str = 'minor'
    ):
        """Create a versioned snapshot of a report"""
        try:
            async for session in get_postgres_session():
                # Get current version number
                result = await session.execute(
                    select(func.max(ReportVersion.version_number)).where(
                        ReportVersion.report_id == report_id
                    )
                )
                current_version = result.scalar() or 0
                new_version = current_version + 1
                
                # Mark all previous versions as not current
                await session.execute(
                    ReportVersion.__table__.update().where(
                        ReportVersion.report_id == report_id
                    ).values(is_current=False)
                )
                
                # Create new version
                version = ReportVersion(
                    report_id=report_id,
                    version_number=new_version,
                    title=title,
                    full_content=full_content,
                    sections=sections,
                    key_findings=key_findings,
                    change_description=change_description,
                    changed_by=changed_by,
                    change_type=change_type,
                    is_current=True
                )
                
                session.add(version)
                await session.commit()
                
                # Log the versioning activity
                await self.log_activity(
                    action='version_create',
                    resource_type='report',
                    resource_id=str(report_id),
                    activity_description=f"Created version {new_version}: {change_description}",
                    new_values={'version': new_version, 'change_type': change_type},
                    user_id=changed_by
                )
                
                return new_version
                
        except Exception as e:
            logger.error(f"Failed to create report version: {e}")
            await self.log_activity(
                action='version_create',
                resource_type='report',
                resource_id=str(report_id),
                status='error',
                error_message=str(e)
            )
            raise
    
    async def get_audit_trail(
        self,
        resource_type: str = None,
        resource_id: str = None,
        user_id: str = None,
        action: str = None,
        days: int = 30,
        limit: int = 100
    ) -> List[Dict]:
        """Get audit trail with filtering"""
        try:
            async for session in get_postgres_session():
                query = select(AuditLog).order_by(AuditLog.created_at.desc())
                
                # Apply filters
                conditions = []
                if resource_type:
                    conditions.append(AuditLog.resource_type == resource_type)
                if resource_id:
                    conditions.append(AuditLog.resource_id == str(resource_id))
                if user_id:
                    conditions.append(AuditLog.user_id == user_id)
                if action:
                    conditions.append(AuditLog.action == action)
                
                # Date filter
                since_date = datetime.utcnow() - timedelta(days=days)
                conditions.append(AuditLog.created_at >= since_date)
                
                if conditions:
                    query = query.where(and_(*conditions))
                
                query = query.limit(limit)
                result = await session.execute(query)
                entries = result.scalars().all()
                
                return [
                    {
                        'id': entry.id,
                        'timestamp': entry.created_at.isoformat(),
                        'user_id': entry.user_id,
                        'action': entry.action,
                        'resource_type': entry.resource_type,
                        'resource_id': entry.resource_id,
                        'description': entry.activity_description,
                        'status': entry.status,
                        'cost_usd': entry.operation_cost_usd,
                        'tokens_used': entry.tokens_used,
                        'response_time_ms': entry.response_time_ms,
                        'error_message': entry.error_message,
                        'endpoint': entry.endpoint,
                        'method': entry.method
                    }
                    for entry in entries
                ]
                
        except Exception as e:
            logger.error(f"Failed to get audit trail: {e}")
            return []
    
    async def get_compliance_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate compliance report"""
        try:
            async for session in get_postgres_session():
                since_date = datetime.utcnow() - timedelta(days=days)
                
                # Total activities
                total_result = await session.execute(
                    select(func.count(AuditLog.id)).where(
                        AuditLog.created_at >= since_date
                    )
                )
                total_activities = total_result.scalar()
                
                # Activities by status
                status_result = await session.execute(
                    select(
                        AuditLog.status,
                        func.count(AuditLog.id)
                    ).where(
                        AuditLog.created_at >= since_date
                    ).group_by(AuditLog.status)
                )
                status_breakdown = {row[0]: row[1] for row in status_result}
                
                # Activities by resource type
                resource_result = await session.execute(
                    select(
                        AuditLog.resource_type,
                        func.count(AuditLog.id)
                    ).where(
                        AuditLog.created_at >= since_date
                    ).group_by(AuditLog.resource_type)
                )
                resource_breakdown = {row[0]: row[1] for row in resource_result}
                
                # Cost analysis
                cost_result = await session.execute(
                    select(func.sum(AuditLog.operation_cost_usd)).where(
                        AuditLog.created_at >= since_date
                    )
                )
                total_cost = cost_result.scalar() or 0.0
                
                # Error analysis
                error_result = await session.execute(
                    select(AuditLog.error_message, func.count(AuditLog.id)).where(
                        and_(
                            AuditLog.created_at >= since_date,
                            AuditLog.status == 'error',
                            AuditLog.error_message.isnot(None)
                        )
                    ).group_by(AuditLog.error_message).limit(10)
                )
                common_errors = [{'error': row[0], 'count': row[1]} for row in error_result]
                
                return {
                    'period': f'{since_date.date()} to {datetime.utcnow().date()}',
                    'total_activities': total_activities,
                    'status_breakdown': status_breakdown,
                    'resource_breakdown': resource_breakdown,
                    'total_cost_usd': round(total_cost, 6),
                    'error_rate': round(status_breakdown.get('error', 0) / max(total_activities, 1) * 100, 2),
                    'common_errors': common_errors,
                    'compliance_status': 'compliant' if status_breakdown.get('error', 0) / max(total_activities, 1) < 0.05 else 'needs_attention'
                }
                
        except Exception as e:
            logger.error(f"Failed to generate compliance report: {e}")
            return {'error': str(e)}
    
    async def cleanup_expired_logs(self):
        """Clean up expired audit logs"""
        try:
            async for session in get_postgres_session():
                result = await session.execute(
                    AuditLog.__table__.delete().where(
                        AuditLog.expires_at < datetime.utcnow()
                    )
                )
                deleted_count = result.rowcount
                await session.commit()
                
                if deleted_count > 0:
                    await self.log_activity(
                        action='cleanup',
                        resource_type='audit_log',
                        activity_description=f'Cleaned up {deleted_count} expired audit logs',
                        status='success'
                    )
                
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired logs: {e}")
            return 0
    
    async def aggregate_usage_metrics(self, date: str = None):
        """Aggregate daily usage metrics from audit logs"""
        if not date:
            date = datetime.utcnow().strftime('%Y-%m-%d')
            
        try:
            async for session in get_postgres_session():
                # Aggregate metrics by service
                start_date = datetime.strptime(date, '%Y-%m-%d')
                end_date = start_date + timedelta(days=1)
                
                # MongoDB operations
                mongodb_result = await session.execute(
                    select(
                        func.count(AuditLog.id),
                        func.sum(AuditLog.operation_cost_usd)
                    ).where(
                        and_(
                            AuditLog.created_at >= start_date,
                            AuditLog.created_at < end_date,
                            AuditLog.resource_type == 'mongodb'
                        )
                    )
                )
                mongodb_ops, mongodb_cost = mongodb_result.first()
                
                # Claude API operations
                claude_result = await session.execute(
                    select(
                        func.count(AuditLog.id),
                        func.sum(AuditLog.operation_cost_usd),
                        func.sum(AuditLog.tokens_used)
                    ).where(
                        and_(
                            AuditLog.created_at >= start_date,
                            AuditLog.created_at < end_date,
                            AuditLog.resource_type == 'claude'
                        )
                    )
                )
                claude_ops, claude_cost, claude_tokens = claude_result.first()
                
                # Store aggregated metrics
                metrics = [
                    {
                        'service': 'mongodb',
                        'metric_type': 'operations',
                        'quantity': mongodb_ops or 0,
                        'cost_usd': mongodb_cost or 0.0
                    },
                    {
                        'service': 'claude',
                        'metric_type': 'api_calls',
                        'quantity': claude_ops or 0,
                        'cost_usd': claude_cost or 0.0
                    },
                    {
                        'service': 'claude',
                        'metric_type': 'tokens',
                        'quantity': claude_tokens or 0,
                        'cost_usd': claude_cost or 0.0
                    }
                ]
                
                for metric in metrics:
                    # Check if metric already exists
                    existing = await session.execute(
                        select(UsageMetrics).where(
                            and_(
                                UsageMetrics.date == date,
                                UsageMetrics.service == metric['service'],
                                UsageMetrics.metric_type == metric['metric_type']
                            )
                        )
                    )
                    
                    if existing.first():
                        # Update existing
                        await session.execute(
                            UsageMetrics.__table__.update().where(
                                and_(
                                    UsageMetrics.date == date,
                                    UsageMetrics.service == metric['service'],
                                    UsageMetrics.metric_type == metric['metric_type']
                                )
                            ).values(
                                quantity=metric['quantity'],
                                cost_usd=metric['cost_usd'],
                                updated_at=datetime.utcnow()
                            )
                        )
                    else:
                        # Create new
                        usage_metric = UsageMetrics(
                            date=date,
                            service=metric['service'],
                            metric_type=metric['metric_type'],
                            quantity=metric['quantity'],
                            cost_usd=metric['cost_usd'],
                            aggregation_level='daily'
                        )
                        session.add(usage_metric)
                
                await session.commit()
                
                await self.log_activity(
                    action='aggregate_metrics',
                    resource_type='usage_metrics',
                    activity_description=f'Aggregated usage metrics for {date}',
                    status='success'
                )
                
        except Exception as e:
            logger.error(f"Failed to aggregate usage metrics: {e}")
            await self.log_activity(
                action='aggregate_metrics',
                resource_type='usage_metrics',
                status='error',
                error_message=str(e)
            )


# Global audit service instance
audit_service = AuditService()


def audit_operation(
    action: str,
    resource_type: str,
    track_cost: bool = True,
    data_classification: str = 'internal'
):
    """Decorator to automatically audit operations"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            status = 'success'
            error_message = None
            cost = 0.0
            tokens = 0
            
            try:
                result = await func(*args, **kwargs)
                
                # Extract cost and token info if available
                if track_cost and hasattr(result, 'cost_usd'):
                    cost = getattr(result, 'cost_usd', 0.0)
                if hasattr(result, 'tokens_used'):
                    tokens = getattr(result, 'tokens_used', 0)
                
                return result
                
            except Exception as e:
                status = 'error'
                error_message = str(e)
                raise
            
            finally:
                response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                # Get resource ID from arguments or result
                resource_id = None
                if args and hasattr(args[0], 'id'):
                    resource_id = getattr(args[0], 'id')
                elif 'id' in kwargs:
                    resource_id = kwargs['id']
                
                # Log the operation
                try:
                    await audit_service.log_activity(
                        action=action,
                        resource_type=resource_type,
                        resource_id=str(resource_id) if resource_id else None,
                        activity_description=f"{func.__name__} operation",
                        operation_cost_usd=cost,
                        tokens_used=tokens,
                        status=status,
                        error_message=error_message,
                        response_time_ms=response_time,
                        data_classification=data_classification
                    )
                except Exception as audit_error:
                    logger.warning(f"Failed to audit operation: {audit_error}")
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, create a task to handle auditing
            start_time = datetime.utcnow()
            status = 'success'
            error_message = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = 'error'
                error_message = str(e)
                raise
            finally:
                response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                # Schedule audit logging
                asyncio.create_task(
                    audit_service.log_activity(
                        action=action,
                        resource_type=resource_type,
                        activity_description=f"{func.__name__} operation",
                        status=status,
                        error_message=error_message,
                        response_time_ms=response_time,
                        data_classification=data_classification
                    )
                )
        
        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
    
    return decorator