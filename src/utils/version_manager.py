"""
Version Management System

Comprehensive version control with:
- Automatic version tracking
- System snapshot creation
- Rollback capabilities
- Version history management
- Critical system state preservation
"""

import json
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib
import os
import asyncio

from loguru import logger
from research_synthesis.config.settings import settings


class VersionInfo:
    """Version information structure"""
    def __init__(self, major: int, minor: int, patch: int, build: str = ""):
        self.major = major
        self.minor = minor
        self.patch = patch
        self.build = build
        self.timestamp = datetime.utcnow()
    
    @property
    def version_string(self) -> str:
        """Get formatted version string"""
        base = f"{self.major}.{self.minor}.{self.patch}"
        return f"{base}-{self.build}" if self.build else base
    
    @property
    def semantic_version(self) -> str:
        """Get semantic version for comparison"""
        return f"{self.major:03d}.{self.minor:03d}.{self.patch:03d}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch,
            "build": self.build,
            "version_string": self.version_string,
            "timestamp": self.timestamp.isoformat(),
            "semantic_version": self.semantic_version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VersionInfo':
        """Create from dictionary"""
        version = cls(data["major"], data["minor"], data["patch"], data.get("build", ""))
        if "timestamp" in data:
            version.timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        return version


class SystemSnapshot:
    """System snapshot for version rollback"""
    def __init__(self, version: VersionInfo, description: str):
        self.version = version
        self.description = description
        self.created_at = datetime.utcnow()
        self.snapshot_id = self._generate_snapshot_id()
        self.file_hash = None
        self.size_mb = 0
        
    def _generate_snapshot_id(self) -> str:
        """Generate unique snapshot ID"""
        timestamp = self.created_at.strftime("%Y%m%d_%H%M%S")
        return f"snapshot_{self.version.version_string}_{timestamp}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "snapshot_id": self.snapshot_id,
            "version": self.version.to_dict(),
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "file_hash": self.file_hash,
            "size_mb": self.size_mb
        }


class VersionManager:
    """Comprehensive version management system"""
    
    def __init__(self):
        self.project_root = Path("D:/orbitscope_ml")
        self.versions_dir = self.project_root / "versions"
        self.snapshots_dir = self.versions_dir / "snapshots"
        self.version_file = self.versions_dir / "version_history.json"
        self.current_version_file = self.project_root / "VERSION"
        
        # Current version tracking
        self.current_version = self._load_current_version()
        
        # Ensure directories exist
        self.versions_dir.mkdir(exist_ok=True)
        self.snapshots_dir.mkdir(exist_ok=True)
        
        # Version history
        self.version_history = self._load_version_history()
        
    def _load_current_version(self) -> VersionInfo:
        """Load current version from file"""
        if self.current_version_file.exists():
            try:
                with open(self.current_version_file, 'r') as f:
                    data = json.load(f)
                return VersionInfo.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load current version: {e}")
        
        # Default to version 1.0.0 if no version file exists
        return VersionInfo(1, 0, 0, "initial")
    
    def _save_current_version(self):
        """Save current version to file"""
        try:
            with open(self.current_version_file, 'w') as f:
                json.dump(self.current_version.to_dict(), f, indent=2)
            logger.info(f"Version updated to {self.current_version.version_string}")
        except Exception as e:
            logger.error(f"Failed to save current version: {e}")
    
    def _load_version_history(self) -> List[Dict[str, Any]]:
        """Load version history"""
        if self.version_file.exists():
            try:
                with open(self.version_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load version history: {e}")
        return []
    
    def _save_version_history(self):
        """Save version history"""
        try:
            with open(self.version_file, 'w') as f:
                json.dump(self.version_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save version history: {e}")
    
    def increment_version(self, increment_type: str = "patch", build_info: str = "") -> VersionInfo:
        """Increment version number"""
        if increment_type == "major":
            self.current_version = VersionInfo(
                self.current_version.major + 1, 0, 0, build_info
            )
        elif increment_type == "minor":
            self.current_version = VersionInfo(
                self.current_version.major, self.current_version.minor + 1, 0, build_info
            )
        elif increment_type == "patch":
            self.current_version = VersionInfo(
                self.current_version.major, self.current_version.minor, 
                self.current_version.patch + 1, build_info
            )
        
        self._save_current_version()
        return self.current_version
    
    def create_version_release(self, 
                             increment_type: str = "minor", 
                             description: str = "",
                             create_snapshot: bool = True) -> Dict[str, Any]:
        """Create a new version release with optional snapshot"""
        
        # Record current state before increment
        pre_version = self.current_version.to_dict()
        
        # Generate build info based on current features
        build_info = self._generate_build_info()
        
        # Increment version
        new_version = self.increment_version(increment_type, build_info)
        
        # Create release record
        release_record = {
            "version": new_version.to_dict(),
            "previous_version": pre_version,
            "release_type": increment_type,
            "description": description or self._generate_release_description(increment_type),
            "features_added": self._get_recent_features(),
            "system_health": self._get_system_health_at_release(),
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Create system snapshot if requested
        snapshot_info = None
        if create_snapshot:
            try:
                snapshot_info = self.create_system_snapshot(new_version, description)
                release_record["snapshot"] = snapshot_info
                logger.info(f"System snapshot created: {snapshot_info['snapshot_id']}")
            except Exception as e:
                logger.error(f"Failed to create snapshot: {e}")
                release_record["snapshot_error"] = str(e)
        
        # Add to version history
        self.version_history.append(release_record)
        self._save_version_history()
        
        logger.info(f"Version release created: {new_version.version_string}")
        
        return release_record
    
    def create_system_snapshot(self, version: VersionInfo, description: str) -> Dict[str, Any]:
        """Create a complete system snapshot"""
        snapshot = SystemSnapshot(version, description)
        snapshot_file = self.snapshots_dir / f"{snapshot.snapshot_id}.zip"
        
        # Files and directories to include in snapshot
        include_patterns = [
            "research_synthesis/**/*.py",
            "core/**/*.py", 
            "scripts/**/*.py",
            "research_synthesis/web/**/*",
            "research_synthesis/database/models.py",
            "research_synthesis/config/settings.py",
            "requirements.txt",
            "research_synthesis_requirements.txt",
            "VERSION",
            "CLAUDE.md",
            "INTEGRATION_ARCHITECTURE.md",
            "*.md"
        ]
        
        # Create zip archive
        try:
            with zipfile.ZipFile(snapshot_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for pattern in include_patterns:
                    for file_path in self.project_root.glob(pattern):
                        if file_path.is_file():
                            # Add file to zip with relative path
                            arcname = file_path.relative_to(self.project_root)
                            zipf.write(file_path, arcname)
                
                # Add version metadata
                metadata = {
                    "snapshot": snapshot.to_dict(),
                    "created_files": len(zipf.namelist()),
                    "creation_timestamp": datetime.utcnow().isoformat()
                }
                zipf.writestr("snapshot_metadata.json", json.dumps(metadata, indent=2))
            
            # Calculate file hash and size
            snapshot.file_hash = self._calculate_file_hash(snapshot_file)
            snapshot.size_mb = round(snapshot_file.stat().st_size / (1024 * 1024), 2)
            
            # Save snapshot metadata
            metadata_file = self.snapshots_dir / f"{snapshot.snapshot_id}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(snapshot.to_dict(), f, indent=2)
            
            logger.info(f"System snapshot created: {snapshot.snapshot_id} ({snapshot.size_mb} MB)")
            
            return snapshot.to_dict()
            
        except Exception as e:
            logger.error(f"Failed to create system snapshot: {e}")
            if snapshot_file.exists():
                snapshot_file.unlink()  # Clean up partial file
            raise
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _generate_build_info(self) -> str:
        """Generate build information"""
        components = []
        
        # Check for major features
        if (self.project_root / "research_synthesis" / "api" / "orbital_integration_router.py").exists():
            components.append("orbital")
        if (self.project_root / "research_synthesis" / "services" / "comprehensive_job_manager.py").exists():
            components.append("jobmgr")
        if (self.project_root / "research_synthesis" / "services" / "theorycrafting_service.py").exists():
            components.append("ml")
        
        return "+".join(components) if components else "base"
    
    def _generate_release_description(self, increment_type: str) -> str:
        """Generate automatic release description"""
        descriptions = {
            "major": "Major release with significant architectural changes and new features",
            "minor": "Minor release with new features and enhancements",
            "patch": "Patch release with bug fixes and minor improvements"
        }
        return descriptions.get(increment_type, "Version update")
    
    def _get_recent_features(self) -> List[str]:
        """Get list of recently added features"""
        features = []
        
        # Check for new API routers
        api_dir = self.project_root / "research_synthesis" / "api"
        if api_dir.exists():
            for router_file in api_dir.glob("*_router.py"):
                if "orbital" in router_file.name:
                    features.append("Orbital Integration API")
                elif "comprehensive_job" in router_file.name:
                    features.append("Comprehensive Job Management")
                elif "theorycrafting" in router_file.name:
                    features.append("ML-Powered Theorycrafting")
        
        # Check for services
        services_dir = self.project_root / "research_synthesis" / "services"
        if services_dir.exists():
            if (services_dir / "orbital_integration_service.py").exists():
                features.append("Orbital Mechanics Integration")
            if (services_dir / "comprehensive_job_manager.py").exists():
                features.append("Advanced Job Management System")
        
        return features
    
    def _get_system_health_at_release(self) -> Dict[str, Any]:
        """Get current system health metrics"""
        return {
            "components_active": self._count_active_components(),
            "api_endpoints": self._count_api_endpoints(),
            "database_status": "operational",
            "performance_score": 100  # Would integrate with actual health monitoring
        }
    
    def _count_active_components(self) -> int:
        """Count active system components"""
        components = 0
        api_dir = self.project_root / "research_synthesis" / "api"
        if api_dir.exists():
            components = len(list(api_dir.glob("*_router.py")))
        return components
    
    def _count_api_endpoints(self) -> int:
        """Count total API endpoints (estimated)"""
        return self._count_active_components() * 5  # Rough estimate
    
    def get_version_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get version history"""
        history = sorted(self.version_history, 
                        key=lambda x: x["version"]["timestamp"], 
                        reverse=True)
        return history[:limit] if limit else history
    
    def get_available_snapshots(self) -> List[Dict[str, Any]]:
        """Get list of available snapshots"""
        snapshots = []
        for metadata_file in self.snapshots_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    snapshot_data = json.load(f)
                snapshots.append(snapshot_data)
            except Exception as e:
                logger.warning(f"Failed to read snapshot metadata {metadata_file}: {e}")
        
        return sorted(snapshots, key=lambda x: x["created_at"], reverse=True)
    
    def restore_from_snapshot(self, snapshot_id: str, confirm: bool = False) -> Dict[str, Any]:
        """Restore system from snapshot"""
        if not confirm:
            return {
                "status": "confirmation_required",
                "warning": "This will overwrite current system state. Set confirm=True to proceed.",
                "snapshot_id": snapshot_id
            }
        
        snapshot_file = self.snapshots_dir / f"{snapshot_id}.zip"
        if not snapshot_file.exists():
            raise FileNotFoundError(f"Snapshot {snapshot_id} not found")
        
        # Create backup of current state before restore
        backup_result = self.create_system_snapshot(
            VersionInfo(99, 99, 99, "pre-restore-backup"),
            f"Automatic backup before restoring {snapshot_id}"
        )
        
        try:
            # Extract snapshot
            with zipfile.ZipFile(snapshot_file, 'r') as zipf:
                zipf.extractall(self.project_root)
            
            # Reload version info
            self.current_version = self._load_current_version()
            
            logger.info(f"System restored from snapshot {snapshot_id}")
            
            return {
                "status": "success",
                "snapshot_id": snapshot_id,
                "backup_created": backup_result["snapshot_id"],
                "restored_version": self.current_version.version_string
            }
            
        except Exception as e:
            logger.error(f"Failed to restore from snapshot {snapshot_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "backup_available": backup_result["snapshot_id"]
            }
    
    def get_current_version_info(self) -> Dict[str, Any]:
        """Get current version information"""
        return {
            "current_version": self.current_version.to_dict(),
            "version_string": self.current_version.version_string,
            "total_releases": len(self.version_history),
            "snapshots_available": len(self.get_available_snapshots()),
            "last_release": self.version_history[-1] if self.version_history else None
        }
    
    async def auto_version_check(self) -> Dict[str, Any]:
        """Check if version should be automatically incremented"""
        # Check for significant changes that warrant version increment
        significant_changes = []
        
        # Check if new routers were added since last version
        api_dir = self.project_root / "research_synthesis" / "api"
        if api_dir.exists():
            router_count = len(list(api_dir.glob("*_router.py")))
            last_release = self.version_history[-1] if self.version_history else None
            
            if last_release:
                last_components = last_release.get("system_health", {}).get("components_active", 0)
                if router_count > last_components:
                    significant_changes.append(f"Added {router_count - last_components} new API components")
        
        recommendation = "none"
        if len(significant_changes) >= 3:
            recommendation = "major"
        elif len(significant_changes) >= 1:
            recommendation = "minor"
        
        return {
            "current_version": self.current_version.version_string,
            "changes_detected": significant_changes,
            "recommended_increment": recommendation,
            "auto_increment_suggested": len(significant_changes) > 0
        }


# Global version manager instance
version_manager = VersionManager()