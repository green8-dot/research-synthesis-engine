"""
Data Administration API Router
Provides endpoints for data auditing, troubleshooting, and recovery
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel
import json

router = APIRouter()

# Import data admin system
try:
    from research_synthesis.utils.data_admin import data_admin
    ADMIN_AVAILABLE = True
except ImportError as e:
    print(f"Data admin not available: {e}")
    ADMIN_AVAILABLE = False
    data_admin = None

# Import data sources for monitoring
try:
    from research_synthesis.api.scraping_router import sources_data, active_jobs
except ImportError:
    sources_data = {}
    active_jobs = []

# Import database components for report management
try:
    from research_synthesis.database.connection import get_postgres_session
    from research_synthesis.database.models import Report
    from sqlalchemy import select, func, desc, delete
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

class SnapshotRequest(BaseModel):
    data_type: str
    reason: Optional[str] = "manual"

class RecoveryRequest(BaseModel):
    data_type: str
    snapshot_id: Optional[int] = None

class PersistRequest(BaseModel):
    key: str
    data_type: str
    data: Any
    expiry_hours: Optional[int] = None

@router.get("/")
async def get_admin_status():
    """Get data administration system status"""
    if not ADMIN_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "Data admin system not initialized"
        }
    
    return {
        "status": "Data Admin System Active",
        "service": "ready",
        "capabilities": [
            "audit_tracking",
            "data_snapshots",
            "integrity_checks",
            "data_recovery",
            "persistent_storage",
            "health_monitoring"
        ],
        "monitored_types": [
            "articles", "reports", "sources", "entities", "scraping_jobs"
        ]
    }

@router.get("/health")
async def get_data_health():
    """Get comprehensive persistence health status for all sections"""
    if not ADMIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data admin system unavailable")
    
    try:
        health_status = data_admin.monitor_data_health()
        
        # Check persistence for all major data sections
        persistence_status = {}
        
        # 1. Reports - Database persistence
        reports_count = 0
        reports_persistence = "unknown"
        if DATABASE_AVAILABLE:
            try:
                async for db_session in get_postgres_session():
                    result = await db_session.execute(select(func.count(Report.id)))
                    reports_count = result.scalar() or 0
                    reports_persistence = "database_persistent" if reports_count > 0 else "database_ready"
                    break
            except Exception as e:
                reports_persistence = "database_error"
                print(f"Error counting reports: {e}")
        else:
            reports_persistence = "database_unavailable"
        
        # 2. Scraping Jobs - Database persistence  
        jobs_count = 0
        jobs_persistence = "unknown"
        if DATABASE_AVAILABLE:
            try:
                from research_synthesis.database.models import ScrapingJob
                async for db_session in get_postgres_session():
                    result = await db_session.execute(select(func.count(ScrapingJob.id)))
                    jobs_count = result.scalar() or 0
                    jobs_persistence = "database_persistent" if jobs_count > 0 else "database_ready"
                    break
            except Exception as e:
                jobs_persistence = "database_error"
                print(f"Error counting scraping jobs: {e}")
        else:
            jobs_persistence = "database_unavailable"
        
        # 3. Articles - Database persistence
        articles_count = 0
        articles_persistence = "unknown"
        if DATABASE_AVAILABLE:
            try:
                from research_synthesis.database.models import Article
                async for db_session in get_postgres_session():
                    result = await db_session.execute(select(func.count(Article.id)))
                    articles_count = result.scalar() or 0
                    articles_persistence = "database_persistent" if articles_count > 0 else "database_ready"
                    break
            except Exception as e:
                articles_persistence = "database_error"
                print(f"Error counting articles: {e}")
        else:
            articles_persistence = "database_unavailable"
        
        # 4. Entities - Database persistence
        entities_count = 0
        entities_persistence = "unknown"
        if DATABASE_AVAILABLE:
            try:
                from research_synthesis.database.models import Entity
                async for db_session in get_postgres_session():
                    result = await db_session.execute(select(func.count(Entity.id)))
                    entities_count = result.scalar() or 0
                    entities_persistence = "database_persistent" if entities_count > 0 else "database_ready"
                    break
            except Exception as e:
                entities_persistence = "database_error"
                print(f"Error counting entities: {e}")
        else:
            entities_persistence = "database_unavailable"
        
        # 5. Sources - In-memory with file backup
        sources_count = len(sources_data)
        sources_persistence = "in_memory_with_backup" if sources_count > 0 else "in_memory_ready"
        
        # Check for file backups
        import os
        backup_dir = "D:/orbitscope_ml/data_backups"
        has_file_backups = os.path.exists(backup_dir) and len(os.listdir(backup_dir)) > 0
        
        # Overall persistence assessment
        persistent_sections = []
        non_persistent_sections = []
        
        section_status = {
            "reports": {"count": reports_count, "persistence": reports_persistence},
            "scraping_jobs": {"count": jobs_count, "persistence": jobs_persistence},
            "articles": {"count": articles_count, "persistence": articles_persistence},
            "entities": {"count": entities_count, "persistence": entities_persistence},
            "sources": {"count": sources_count, "persistence": sources_persistence}
        }
        
        for section, info in section_status.items():
            if "persistent" in info["persistence"] or "backup" in info["persistence"]:
                persistent_sections.append(section)
            else:
                non_persistent_sections.append(section)
        
        health_status["persistence_analysis"] = {
            "sections": section_status,
            "persistent_sections": persistent_sections,
            "non_persistent_sections": non_persistent_sections,
            "file_backups_available": has_file_backups,
            "overall_persistence_health": "good" if len(persistent_sections) >= 3 else ("partial" if len(persistent_sections) > 0 else "poor")
        }
        
        health_status["current_data"] = {
            "sources": sources_count,
            "active_jobs": len(active_jobs),
            "reports": reports_count,
            "articles": articles_count,
            "entities": entities_count,
            "scraping_jobs_db": jobs_count
        }
        
        return health_status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.post("/audit")
async def log_audit_entry(
    operation_type: str,
    data_type: str,
    data_id: Optional[str] = None,
    success: bool = True,
    error: Optional[str] = None,
    metadata: Optional[Dict] = None
):
    """Log a data operation for audit trail"""
    if not ADMIN_AVAILABLE:
        return {"message": "Audit logging unavailable"}
    
    try:
        data_admin.audit_operation(
            operation_type=operation_type,
            data_type=data_type,
            data_id=data_id,
            success=success,
            error=error,
            metadata=metadata
        )
        
        return {
            "message": "Audit entry logged",
            "operation": operation_type,
            "data_type": data_type,
            "success": success
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audit logging failed: {str(e)}")

@router.post("/snapshot", response_model=Dict[str, Any])
async def create_data_snapshot(request: SnapshotRequest):
    """Create a snapshot of specified data type"""
    if not ADMIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data admin system unavailable")
    
    try:
        # Get current data based on type
        data_to_snapshot = []
        
        if request.data_type == "sources":
            data_to_snapshot = [
                {"name": k, **v} for k, v in sources_data.items()
            ]
        elif request.data_type == "jobs":
            data_to_snapshot = active_jobs
        elif request.data_type == "reports":
            data_to_snapshot = list(stored_reports.values())
        else:
            # For other types, try to get from database
            from research_synthesis.database.connection import get_postgres_session
            from sqlalchemy import select
            import asyncio
            
            # This would need proper async handling
            # For now, return empty for unknown types
            data_to_snapshot = []
        
        snapshot_id = data_admin.create_snapshot(
            data_type=request.data_type,
            data=data_to_snapshot,
            reason=request.reason
        )
        
        return {
            "message": "Snapshot created successfully",
            "snapshot_id": snapshot_id,
            "data_type": request.data_type,
            "record_count": len(data_to_snapshot),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Snapshot creation failed: {str(e)}")

@router.get("/integrity/{data_type}")
async def check_data_integrity(data_type: str):
    """Check integrity of specified data type"""
    if not ADMIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data admin system unavailable")
    
    try:
        # Get current data
        current_data = []
        
        if data_type == "sources":
            current_data = [{"name": k, **v} for k, v in sources_data.items()]
        elif data_type == "jobs":
            current_data = active_jobs
        elif data_type == "reports":
            current_data = list(stored_reports.values())
        
        integrity_result = data_admin.check_data_integrity(data_type, current_data)
        
        return integrity_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Integrity check failed: {str(e)}")

@router.post("/recover", response_model=Dict[str, Any])
async def recover_data(request: RecoveryRequest):
    """Recover lost data from snapshots"""
    if not ADMIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data admin system unavailable")
    
    try:
        recovery_result = data_admin.recover_lost_data(
            data_type=request.data_type,
            snapshot_id=request.snapshot_id
        )
        
        if recovery_result["success"]:
            # Apply recovered data based on type
            if request.data_type == "sources" and recovery_result["data"]:
                # Restore sources
                for source in recovery_result["data"]:
                    name = source.pop("name", None)
                    if name:
                        sources_data[name] = source
            
            elif request.data_type == "reports" and recovery_result["data"]:
                # Restore reports to database
                if DATABASE_AVAILABLE:
                    async for db_session in get_postgres_session():
                        try:
                            for report_data in recovery_result["data"]:
                                db_report = Report(
                                    title=report_data.get("title", "Recovered Report"),
                                    report_type=report_data.get("type", "recovered"),
                                    full_content=json.dumps(report_data.get("content", {})),
                                    word_count=report_data.get("word_count", 0),
                                    confidence_score=report_data.get("confidence_score", 0.5)
                                )
                                db_session.add(db_report)
                            await db_session.commit()
                        except Exception as e:
                            await db_session.rollback()
                            print(f"Report recovery failed: {e}")
                        break
        
        return recovery_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data recovery failed: {str(e)}")

@router.post("/persist")
async def persist_transient_data(request: PersistRequest):
    """Persist transient data to survive restarts"""
    if not ADMIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data admin system unavailable")
    
    try:
        data_admin.persist_transient_data(
            key=request.key,
            data_type=request.data_type,
            data=request.data,
            expiry_hours=request.expiry_hours
        )
        
        return {
            "message": "Data persisted successfully",
            "key": request.key,
            "data_type": request.data_type,
            "expiry_hours": request.expiry_hours
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data persistence failed: {str(e)}")

@router.get("/retrieve/{key}")
async def retrieve_persistent_data(key: str):
    """Retrieve persisted data by key"""
    if not ADMIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data admin system unavailable")
    
    try:
        data = data_admin.retrieve_persistent_data(key)
        
        if data is None:
            raise HTTPException(status_code=404, detail="Data not found or expired")
        
        return {
            "key": key,
            "data": data,
            "retrieved_at": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data retrieval failed: {str(e)}")

@router.get("/report")
async def get_data_loss_report(days: int = 7):
    """Generate comprehensive data loss report"""
    if not ADMIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data admin system unavailable")
    
    try:
        report = data_admin.get_data_loss_report(days=days)
        
        # Add current system state
        report["current_state"] = {
            "sources_count": len(sources_data),
            "active_jobs": len(active_jobs),
            "reports_count": len(stored_reports),
            "timestamp": datetime.now().isoformat()
        }
        
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@router.post("/backup")
async def create_system_backup(backup_type: str = "full"):
    """Create comprehensive system backup"""
    if not ADMIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data admin system unavailable")
    
    try:
        # Create snapshots of all in-memory data first
        snapshots_created = []
        
        # Snapshot sources
        if sources_data:
            snapshot_id = data_admin.create_snapshot(
                "sources",
                [{"name": k, **v} for k, v in sources_data.items()],
                "backup"
            )
            snapshots_created.append(f"sources_{snapshot_id}")
        
        # Snapshot jobs
        if active_jobs:
            snapshot_id = data_admin.create_snapshot(
                "jobs",
                active_jobs,
                "backup"
            )
            snapshots_created.append(f"jobs_{snapshot_id}")
        
        # Snapshot reports
        if stored_reports:
            snapshot_id = data_admin.create_snapshot(
                "reports",
                list(stored_reports.values()),
                "backup"
            )
            snapshots_created.append(f"reports_{snapshot_id}")
        
        # Create file backup
        backup_path = data_admin.create_backup(backup_type)
        
        return {
            "message": "Backup created successfully",
            "backup_type": backup_type,
            "backup_path": backup_path,
            "snapshots_created": snapshots_created,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backup creation failed: {str(e)}")

@router.get("/audit-log")
async def get_audit_log(data_type: Optional[str] = None, limit: int = 100):
    """Get recent audit log entries"""
    if not ADMIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data admin system unavailable")
    
    try:
        import sqlite3
        conn = sqlite3.connect(data_admin.db_path)
        cursor = conn.cursor()
        
        if data_type:
            cursor.execute('''
                SELECT timestamp, operation_type, data_type, data_id, success, error_message
                FROM data_audit_log
                WHERE data_type = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (data_type, limit))
        else:
            cursor.execute('''
                SELECT timestamp, operation_type, data_type, data_id, success, error_message
                FROM data_audit_log
                ORDER BY created_at DESC
                LIMIT ?
            ''', (limit,))
        
        entries = cursor.fetchall()
        conn.close()
        
        return {
            "audit_entries": [
                {
                    "timestamp": row[0],
                    "operation": row[1],
                    "data_type": row[2],
                    "data_id": row[3],
                    "success": bool(row[4]),
                    "error": row[5]
                } for row in entries
            ],
            "count": len(entries),
            "limit": limit
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audit log retrieval failed: {str(e)}")

@router.post("/auto-persist")
async def enable_auto_persistence(background_tasks: BackgroundTasks):
    """Enable automatic persistence of transient data"""
    if not ADMIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data admin system unavailable")
    
    try:
        # Persist current state
        data_admin.persist_transient_data(
            "sources_backup",
            "sources",
            dict(sources_data),
            expiry_hours=24
        )
        
        data_admin.persist_transient_data(
            "jobs_backup", 
            "jobs",
            active_jobs,
            expiry_hours=24
        )
        
        data_admin.persist_transient_data(
            "reports_backup",
            "reports",
            dict(stored_reports),
            expiry_hours=168  # 1 week for reports
        )
        
        return {
            "message": "Auto-persistence enabled",
            "persisted": ["sources", "jobs", "reports"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto-persistence failed: {str(e)}")

@router.post("/restore-on-startup")
async def restore_persisted_data():
    """Restore persisted data on system startup"""
    if not ADMIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data admin system unavailable")
    
    restored = []
    
    try:
        # Restore sources
        sources_backup = data_admin.retrieve_persistent_data("sources_backup")
        if sources_backup:
            sources_data.update(sources_backup)
            restored.append(f"sources ({len(sources_backup)} items)")
        
        # Restore jobs
        jobs_backup = data_admin.retrieve_persistent_data("jobs_backup")
        if jobs_backup:
            active_jobs.extend(jobs_backup)
            restored.append(f"jobs ({len(jobs_backup)} items)")
        
        # Restore reports
        reports_backup = data_admin.retrieve_persistent_data("reports_backup")
        if reports_backup:
            stored_reports.update(reports_backup)
            restored.append(f"reports ({len(reports_backup)} items)")
        
        return {
            "message": "Data restored from persistent storage",
            "restored": restored,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data restoration failed: {str(e)}")

# Report Management Endpoints
class ReportAdminRequest(BaseModel):
    action: str  # archive, delete, export
    report_ids: Optional[List[int]] = None
    days_old: Optional[int] = None

@router.get("/reports/stats")
async def get_reports_statistics():
    """Get comprehensive reports statistics"""
    if not DATABASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        async for db_session in get_postgres_session():
            # Total reports
            total_result = await db_session.execute(select(func.count(Report.id)))
            total_reports = total_result.scalar() or 0
            
            # Reports by type
            type_result = await db_session.execute(
                select(Report.report_type, func.count(Report.id).label('count'))
                .group_by(Report.report_type)
            )
            reports_by_type = {row[0]: row[1] for row in type_result.fetchall()}
            
            # Storage usage (estimate)
            size_result = await db_session.execute(
                select(func.sum(func.length(Report.full_content)))
            )
            total_size_chars = size_result.scalar() or 0
            estimated_size_mb = (total_size_chars * 2) / (1024 * 1024)  # Rough estimate
            
            # Recent activity
            recent_result = await db_session.execute(
                select(func.count(Report.id))
                .filter(Report.created_at >= datetime.now().replace(day=datetime.now().day-7))
            )
            reports_last_7_days = recent_result.scalar() or 0
            
            break
        
        return {
            "total_reports": total_reports,
            "reports_by_type": reports_by_type,
            "estimated_storage_mb": round(estimated_size_mb, 2),
            "reports_last_7_days": reports_last_7_days,
            "database_status": "connected",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistics retrieval failed: {str(e)}")

@router.post("/reports/cleanup")
async def cleanup_old_reports(request: ReportAdminRequest):
    """Clean up old or selected reports"""
    if not DATABASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        deleted_count = 0
        async for db_session in get_postgres_session():
            if request.report_ids:
                # Delete specific reports
                result = await db_session.execute(
                    delete(Report).filter(Report.id.in_(request.report_ids))
                )
                deleted_count = result.rowcount
                
            elif request.days_old:
                # Delete reports older than specified days
                cutoff_date = datetime.now().replace(day=datetime.now().day - request.days_old)
                result = await db_session.execute(
                    delete(Report).filter(Report.created_at < cutoff_date)
                )
                deleted_count = result.rowcount
            
            await db_session.commit()
            break
        
        return {
            "action": "cleanup_completed",
            "deleted_reports": deleted_count,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@router.get("/reports/export")
async def export_reports_data():
    """Export all reports data for backup"""
    if not DATABASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        reports_data = []
        async for db_session in get_postgres_session():
            result = await db_session.execute(
                select(Report).order_by(desc(Report.created_at))
            )
            reports = result.scalars().all()
            
            for report in reports:
                reports_data.append({
                    "id": report.id,
                    "title": report.title,
                    "report_type": report.report_type,
                    "word_count": report.word_count,
                    "confidence_score": report.confidence_score,
                    "created_at": report.created_at.isoformat() if report.created_at else None,
                    "executive_summary": json.loads(report.executive_summary) if report.executive_summary else {},
                    "full_content": json.loads(report.full_content) if report.full_content else {},
                    "generation_params": json.loads(report.generation_params) if report.generation_params else {}
                })
            break
        
        return {
            "export_type": "reports",
            "total_reports": len(reports_data),
            "reports_data": reports_data,
            "exported_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.post("/reports/validate")
async def validate_reports_integrity():
    """Validate reports data integrity"""
    if not DATABASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        validation_issues = []
        total_checked = 0
        
        async for db_session in get_postgres_session():
            result = await db_session.execute(select(Report))
            reports = result.scalars().all()
            
            for report in reports:
                total_checked += 1
                issues = []
                
                # Check for required fields
                if not report.title:
                    issues.append("Missing title")
                if not report.report_type:
                    issues.append("Missing report type")
                if not report.full_content:
                    issues.append("Missing content")
                    
                # Validate JSON fields
                for field_name, field_value in [
                    ("executive_summary", report.executive_summary),
                    ("full_content", report.full_content),
                    ("generation_params", report.generation_params)
                ]:
                    if field_value:
                        try:
                            json.loads(field_value)
                        except json.JSONDecodeError:
                            issues.append(f"Invalid JSON in {field_name}")
                
                if issues:
                    validation_issues.append({
                        "report_id": report.id,
                        "title": report.title,
                        "issues": issues
                    })
            
            break
        
        return {
            "validation_complete": True,
            "total_reports_checked": total_checked,
            "issues_found": len(validation_issues),
            "validation_issues": validation_issues,
            "health_status": "healthy" if not validation_issues else "issues_found",
            "validated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.get("/ui-issues")
async def get_ui_issue_reports(limit: int = 100, category: Optional[str] = None, severity: Optional[str] = None):
    """Get UI issue reports from data admin log"""
    if not ADMIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data admin system unavailable")
    
    try:
        import sqlite3
        import json
        conn = sqlite3.connect(data_admin.db_path)
        cursor = conn.cursor()
        
        # Build query based on filters
        base_query = '''
            SELECT timestamp, data_id, metadata, created_at
            FROM data_audit_log
            WHERE data_type = 'ui_issue_report' AND operation_type = 'CREATE'
        '''
        params = []
        
        # Add category filter if provided
        if category:
            base_query += ' AND metadata LIKE ?'
            params.append(f'%"category":"{category}"%')
        
        # Add severity filter if provided  
        if severity:
            base_query += ' AND metadata LIKE ?'
            params.append(f'%"severity":"{severity}"%')
        
        base_query += ' ORDER BY created_at DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(base_query, params)
        entries = cursor.fetchall()
        conn.close()
        
        # Parse and structure the UI issue reports
        ui_issues = []
        for row in entries:
            try:
                metadata = json.loads(row[2]) if row[2] else {}
                ui_issues.append({
                    "timestamp": row[0],
                    "issue_id": row[1],
                    "page": metadata.get("page", "unknown"),
                    "category": metadata.get("category", "unknown"),
                    "severity": metadata.get("severity", "medium"),
                    "element": metadata.get("element", "unknown"),
                    "description": metadata.get("description", ""),
                    "suggested_fix": metadata.get("suggested_fix", ""),
                    "created_at": row[3]
                })
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error parsing UI issue metadata: {e}")
                continue
        
        # Group by category and severity for statistics
        category_stats = {}
        severity_stats = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        for issue in ui_issues:
            cat = issue["category"]
            sev = issue["severity"]
            
            category_stats[cat] = category_stats.get(cat, 0) + 1
            if sev in severity_stats:
                severity_stats[sev] += 1
        
        return {
            "total_ui_issues": len(ui_issues),
            "ui_issues": ui_issues,
            "category_breakdown": category_stats,
            "severity_breakdown": severity_stats,
            "filters_applied": {
                "category": category,
                "severity": severity,
                "limit": limit
            },
            "retrieved_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve UI issues: {str(e)}")

@router.post("/ui-issues/export")
async def export_ui_issue_reports():
    """Export all UI issue reports as structured data"""
    if not ADMIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data admin system unavailable")
    
    try:
        import sqlite3
        import json
        conn = sqlite3.connect(data_admin.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, data_id, metadata, created_at
            FROM data_audit_log
            WHERE data_type = 'ui_issue_report' AND operation_type = 'CREATE'
            ORDER BY created_at DESC
        ''')
        
        entries = cursor.fetchall()
        conn.close()
        
        export_data = []
        for row in entries:
            try:
                metadata = json.loads(row[2]) if row[2] else {}
                export_data.append({
                    "timestamp": row[0],
                    "issue_id": row[1],
                    "page": metadata.get("page", "unknown"),
                    "category": metadata.get("category", "unknown"), 
                    "severity": metadata.get("severity", "medium"),
                    "element": metadata.get("element", "unknown"),
                    "description": metadata.get("description", ""),
                    "suggested_fix": metadata.get("suggested_fix", ""),
                    "user_agent": metadata.get("user_agent", ""),
                    "timestamp_iso": metadata.get("timestamp", ""),
                    "created_at": row[3]
                })
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error parsing UI issue metadata for export: {e}")
                continue
        
        return {
            "export_type": "ui_issue_reports",
            "total_issues": len(export_data),
            "ui_issues_data": export_data,
            "exported_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")