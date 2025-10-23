"""
Version Management API Router

Comprehensive version control with:
- Version history tracking
- System snapshots for rollback
- Release management
- Backup and restore capabilities
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, Optional, List
from loguru import logger
from pydantic import BaseModel

from research_synthesis.utils.version_manager import version_manager


class VersionReleaseRequest(BaseModel):
    increment_type: str = "minor"  # major, minor, patch
    description: Optional[str] = None
    create_snapshot: bool = True


class SnapshotRestoreRequest(BaseModel):
    snapshot_id: str
    confirm: bool = False


router = APIRouter()


@router.get("/current")
async def get_current_version():
    """Get current version information"""
    try:
        version_info = version_manager.get_current_version_info()
        return {
            "status": "success",
            "version_info": version_info
        }
    except Exception as e:
        logger.error(f"Failed to get current version: {e}")
        raise HTTPException(status_code=500, detail=f"Version retrieval failed: {str(e)}")


@router.get("/history")
async def get_version_history(limit: Optional[int] = 10):
    """Get version release history"""
    try:
        history = version_manager.get_version_history(limit)
        return {
            "status": "success",
            "version_history": history,
            "total_releases": len(version_manager.version_history)
        }
    except Exception as e:
        logger.error(f"Failed to get version history: {e}")
        raise HTTPException(status_code=500, detail=f"History retrieval failed: {str(e)}")


@router.post("/release")
async def create_version_release(request: VersionReleaseRequest, background_tasks: BackgroundTasks):
    """Create a new version release with optional snapshot"""
    try:
        if request.increment_type not in ["major", "minor", "patch"]:
            raise HTTPException(status_code=400, detail="increment_type must be 'major', 'minor', or 'patch'")
        
        logger.info(f"Creating {request.increment_type} version release")
        
        # Create release (this may take time if creating snapshot)
        if request.create_snapshot:
            # Run in background for large snapshots
            background_tasks.add_task(
                _create_release_with_snapshot, 
                request.increment_type, 
                request.description, 
                request.create_snapshot
            )
            
            return {
                "status": "accepted",
                "message": "Version release creation started in background",
                "increment_type": request.increment_type,
                "snapshot_creation": "in_progress"
            }
        else:
            # Create release immediately without snapshot
            release_info = version_manager.create_version_release(
                request.increment_type,
                request.description,
                False
            )
            
            return {
                "status": "success",
                "release_info": release_info,
                "new_version": release_info["version"]["version_string"]
            }
        
    except Exception as e:
        logger.error(f"Failed to create version release: {e}")
        raise HTTPException(status_code=500, detail=f"Release creation failed: {str(e)}")


async def _create_release_with_snapshot(increment_type: str, description: Optional[str], create_snapshot: bool):
    """Background task for creating release with snapshot"""
    try:
        release_info = version_manager.create_version_release(increment_type, description, create_snapshot)
        logger.info(f"Version release completed: {release_info['version']['version_string']}")
    except Exception as e:
        logger.error(f"Background release creation failed: {e}")


@router.get("/snapshots")
async def get_available_snapshots():
    """Get list of available system snapshots"""
    try:
        snapshots = version_manager.get_available_snapshots()
        return {
            "status": "success",
            "snapshots": snapshots,
            "total_snapshots": len(snapshots)
        }
    except Exception as e:
        logger.error(f"Failed to get snapshots: {e}")
        raise HTTPException(status_code=500, detail=f"Snapshots retrieval failed: {str(e)}")


@router.post("/snapshots/create")
async def create_manual_snapshot(
    description: str = "Manual snapshot", 
    background_tasks: BackgroundTasks = None
):
    """Create a manual system snapshot"""
    try:
        current_version = version_manager.current_version
        
        # Create snapshot in background
        if background_tasks:
            background_tasks.add_task(_create_manual_snapshot, current_version, description)
            return {
                "status": "accepted",
                "message": "Snapshot creation started in background",
                "version": current_version.version_string
            }
        else:
            snapshot_info = version_manager.create_system_snapshot(current_version, description)
            return {
                "status": "success",
                "snapshot": snapshot_info
            }
        
    except Exception as e:
        logger.error(f"Failed to create manual snapshot: {e}")
        raise HTTPException(status_code=500, detail=f"Snapshot creation failed: {str(e)}")


async def _create_manual_snapshot(version, description: str):
    """Background task for manual snapshot creation"""
    try:
        snapshot_info = version_manager.create_system_snapshot(version, description)
        logger.info(f"Manual snapshot created: {snapshot_info['snapshot_id']}")
    except Exception as e:
        logger.error(f"Background snapshot creation failed: {e}")


@router.post("/snapshots/restore")
async def restore_from_snapshot(request: SnapshotRestoreRequest):
    """Restore system from a snapshot"""
    try:
        if not request.confirm:
            # Return confirmation requirement
            return {
                "status": "confirmation_required",
                "warning": "This will overwrite the current system state with the snapshot data.",
                "snapshot_id": request.snapshot_id,
                "instructions": "Set 'confirm: true' in the request to proceed with restoration."
            }
        
        logger.warning(f"Restoring system from snapshot: {request.snapshot_id}")
        
        restore_result = version_manager.restore_from_snapshot(request.snapshot_id, True)
        
        if restore_result["status"] == "success":
            logger.info(f"System successfully restored from {request.snapshot_id}")
        
        return restore_result
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Snapshot not found: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to restore from snapshot: {e}")
        raise HTTPException(status_code=500, detail=f"Restoration failed: {str(e)}")


@router.get("/check-updates")
async def check_for_updates():
    """Check if system should be updated to new version"""
    try:
        check_result = await version_manager.auto_version_check()
        return {
            "status": "success",
            "update_check": check_result
        }
    except Exception as e:
        logger.error(f"Failed to check for updates: {e}")
        raise HTTPException(status_code=500, detail=f"Update check failed: {str(e)}")


@router.get("/system-info")
async def get_system_version_info():
    """Get comprehensive system version information"""
    try:
        current_info = version_manager.get_current_version_info()
        snapshots = version_manager.get_available_snapshots()
        
        # Calculate storage usage
        total_snapshot_size = sum(s.get("size_mb", 0) for s in snapshots)
        
        return {
            "status": "success",
            "system_info": {
                **current_info,
                "snapshot_storage_mb": total_snapshot_size,
                "version_manager_status": "operational",
                "features": {
                    "version_tracking": True,
                    "system_snapshots": True,
                    "rollback_capability": True,
                    "automatic_backups": True
                }
            }
        }
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        raise HTTPException(status_code=500, detail=f"System info retrieval failed: {str(e)}")


@router.post("/increment/{increment_type}")
async def quick_version_increment(increment_type: str):
    """Quick version increment without snapshot"""
    try:
        if increment_type not in ["major", "minor", "patch"]:
            raise HTTPException(status_code=400, detail="increment_type must be 'major', 'minor', or 'patch'")
        
        new_version = version_manager.increment_version(increment_type)
        
        return {
            "status": "success",
            "previous_version": version_manager.version_history[-1]["version"]["version_string"] if version_manager.version_history else "unknown",
            "new_version": new_version.version_string,
            "increment_type": increment_type
        }
        
    except Exception as e:
        logger.error(f"Failed to increment version: {e}")
        raise HTTPException(status_code=500, detail=f"Version increment failed: {str(e)}")


@router.delete("/snapshots/{snapshot_id}")
async def delete_snapshot(snapshot_id: str):
    """Delete a specific snapshot"""
    try:
        snapshot_file = version_manager.snapshots_dir / f"{snapshot_id}.zip"
        metadata_file = version_manager.snapshots_dir / f"{snapshot_id}_metadata.json"
        
        if not snapshot_file.exists():
            raise HTTPException(status_code=404, detail="Snapshot not found")
        
        # Get size before deletion
        size_mb = round(snapshot_file.stat().st_size / (1024 * 1024), 2)
        
        # Delete files
        snapshot_file.unlink()
        if metadata_file.exists():
            metadata_file.unlink()
        
        logger.info(f"Deleted snapshot: {snapshot_id}")
        
        return {
            "status": "success",
            "deleted_snapshot": snapshot_id,
            "freed_space_mb": size_mb
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete snapshot {snapshot_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Snapshot deletion failed: {str(e)}")


@router.get("/maintenance")
async def get_maintenance_info():
    """Get version management maintenance information"""
    try:
        snapshots = version_manager.get_available_snapshots()
        total_size = sum(s.get("size_mb", 0) for s in snapshots)
        
        # Identify old snapshots (older than 30 days)
        from datetime import datetime, timedelta
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        old_snapshots = [
            s for s in snapshots 
            if datetime.fromisoformat(s["created_at"].replace("Z", "+00:00")) < cutoff_date
        ]
        
        return {
            "status": "success",
            "maintenance_info": {
                "total_snapshots": len(snapshots),
                "total_storage_mb": total_size,
                "old_snapshots_count": len(old_snapshots),
                "old_snapshots_size_mb": sum(s.get("size_mb", 0) for s in old_snapshots),
                "recommended_actions": [
                    f"Delete {len(old_snapshots)} old snapshots to free {sum(s.get('size_mb', 0) for s in old_snapshots):.1f} MB"
                ] if old_snapshots else ["No maintenance required"]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get maintenance info: {e}")
        raise HTTPException(status_code=500, detail=f"Maintenance info failed: {str(e)}")