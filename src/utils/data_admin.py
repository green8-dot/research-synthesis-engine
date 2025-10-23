"""
Data Administration System for Audit, Troubleshooting, and Debugging Data Loss
Provides comprehensive tracking and recovery mechanisms for all data operations
"""
import json
import sqlite3
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
import shutil
import pickle

class DataAdminSystem:
    def __init__(self, db_path: str = "data_admin.db", backup_dir: str = "data_backups"):
        self.db_path = db_path
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.init_audit_database()
        
    def init_audit_database(self):
        """Initialize audit tracking database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Data audit log - tracks all data operations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                operation_type TEXT NOT NULL,  -- CREATE, UPDATE, DELETE, ACCESS
                data_type TEXT NOT NULL,       -- article, report, entity, source, etc.
                data_id TEXT,
                data_hash TEXT,                -- Hash of data for integrity check
                user_session TEXT,
                ip_address TEXT,
                data_size INTEGER,
                success BOOLEAN,
                error_message TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Data snapshots - periodic snapshots of critical data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_time TEXT NOT NULL,
                data_type TEXT NOT NULL,
                record_count INTEGER,
                data_content TEXT,              -- Serialized data
                data_hash TEXT,
                snapshot_reason TEXT,           -- manual, scheduled, pre_operation
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Data integrity checks
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS integrity_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                check_time TEXT NOT NULL,
                check_type TEXT NOT NULL,       -- consistency, completeness, accuracy
                data_type TEXT NOT NULL,
                expected_count INTEGER,
                actual_count INTEGER,
                missing_records TEXT,
                corrupt_records TEXT,
                check_passed BOOLEAN,
                remediation_action TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Data loss incidents
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_loss_incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_time TEXT NOT NULL,
                data_type TEXT NOT NULL,
                records_lost INTEGER,
                root_cause TEXT,
                recovery_attempted BOOLEAN,
                recovery_successful BOOLEAN,
                recovery_method TEXT,
                records_recovered INTEGER,
                incident_details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Persistent storage for transient data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persistent_storage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                storage_key TEXT UNIQUE NOT NULL,
                data_type TEXT NOT NULL,
                data_value TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                expiry_time TEXT,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def audit_operation(self, operation_type: str, data_type: str, data_id: str = None,
                       data: Any = None, success: bool = True, error: str = None,
                       metadata: Dict = None):
        """Log all data operations for audit trail"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        data_hash = None
        data_size = 0
        
        if data:
            # Calculate hash for integrity verification
            data_str = json.dumps(data, sort_keys=True) if not isinstance(data, str) else data
            data_hash = hashlib.sha256(data_str.encode()).hexdigest()
            data_size = len(data_str)
        
        cursor.execute('''
            INSERT INTO data_audit_log 
            (timestamp, operation_type, data_type, data_id, data_hash, data_size, 
             success, error_message, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            operation_type,
            data_type,
            data_id,
            data_hash,
            data_size,
            success,
            error,
            json.dumps(metadata) if metadata else None
        ))
        
        conn.commit()
        conn.close()
        
        # Log critical failures
        if not success:
            logger.error(f"Data operation failed: {operation_type} on {data_type}/{data_id} - {error}")
    
    def create_snapshot(self, data_type: str, data: List[Dict], reason: str = "scheduled"):
        """Create a snapshot of critical data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        data_str = json.dumps(data, sort_keys=True)
        data_hash = hashlib.sha256(data_str.encode()).hexdigest()
        
        cursor.execute('''
            INSERT INTO data_snapshots 
            (snapshot_time, data_type, record_count, data_content, data_hash, snapshot_reason)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            data_type,
            len(data),
            data_str,
            data_hash,
            reason
        ))
        
        snapshot_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Also create file backup
        backup_file = self.backup_dir / f"{data_type}_{snapshot_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(backup_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Created snapshot #{snapshot_id} for {data_type} with {len(data)} records")
        return snapshot_id
    
    def check_data_integrity(self, data_type: str, current_data: List[Dict]) -> Dict[str, Any]:
        """Check data integrity and detect potential loss"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get last snapshot for comparison
        cursor.execute('''
            SELECT record_count, data_content, data_hash 
            FROM data_snapshots 
            WHERE data_type = ? 
            ORDER BY created_at DESC 
            LIMIT 1
        ''', (data_type,))
        
        last_snapshot = cursor.fetchone()
        
        integrity_result = {
            "check_time": datetime.now().isoformat(),
            "data_type": data_type,
            "current_count": len(current_data),
            "issues_found": []
        }
        
        if last_snapshot:
            expected_count = last_snapshot[0]
            last_data = json.loads(last_snapshot[1])
            
            # Check for data loss
            if len(current_data) < expected_count:
                integrity_result["issues_found"].append({
                    "type": "potential_data_loss",
                    "expected": expected_count,
                    "actual": len(current_data),
                    "missing": expected_count - len(current_data)
                })
                
                # Identify missing records
                if data_type in ["sources", "reports", "articles"]:
                    last_ids = {item.get("id", item.get("name")) for item in last_data}
                    current_ids = {item.get("id", item.get("name")) for item in current_data}
                    missing_ids = last_ids - current_ids
                    
                    if missing_ids:
                        integrity_result["missing_records"] = list(missing_ids)
            
            # Check for data corruption
            for item in current_data:
                if self._is_corrupt(item, data_type):
                    integrity_result["issues_found"].append({
                        "type": "corrupt_record",
                        "record": item.get("id", item.get("name", "unknown"))
                    })
        
        # Log integrity check
        cursor.execute('''
            INSERT INTO integrity_checks 
            (check_time, check_type, data_type, expected_count, actual_count, 
             missing_records, check_passed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            "comprehensive",
            data_type,
            last_snapshot[0] if last_snapshot else 0,
            len(current_data),
            json.dumps(integrity_result.get("missing_records", [])),
            len(integrity_result["issues_found"]) == 0
        ))
        
        conn.commit()
        conn.close()
        
        return integrity_result
    
    def _is_corrupt(self, record: Dict, data_type: str) -> bool:
        """Check if a record is corrupted based on data type"""
        if data_type == "articles":
            return not record.get("title") or not record.get("url")
        elif data_type == "reports":
            return not record.get("title") or not record.get("content")
        elif data_type == "sources":
            return not record.get("name") or not record.get("url")
        return False
    
    def recover_lost_data(self, data_type: str, snapshot_id: Optional[int] = None) -> Dict[str, Any]:
        """Attempt to recover lost data from snapshots"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if snapshot_id:
            # Recover from specific snapshot
            cursor.execute('''
                SELECT data_content, record_count 
                FROM data_snapshots 
                WHERE id = ? AND data_type = ?
            ''', (snapshot_id, data_type))
        else:
            # Recover from most recent snapshot
            cursor.execute('''
                SELECT data_content, record_count 
                FROM data_snapshots 
                WHERE data_type = ? 
                ORDER BY created_at DESC 
                LIMIT 1
            ''', (data_type,))
        
        snapshot = cursor.fetchone()
        
        if snapshot:
            recovered_data = json.loads(snapshot[0])
            
            # Log recovery attempt
            cursor.execute('''
                INSERT INTO data_loss_incidents 
                (incident_time, data_type, recovery_attempted, recovery_successful, 
                 recovery_method, records_recovered)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                data_type,
                True,
                True,
                f"snapshot_recovery_{snapshot_id or 'latest'}",
                len(recovered_data)
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Successfully recovered {len(recovered_data)} {data_type} records")
            
            return {
                "success": True,
                "records_recovered": len(recovered_data),
                "data": recovered_data
            }
        
        conn.close()
        return {
            "success": False,
            "error": "No snapshot available for recovery"
        }
    
    def persist_transient_data(self, key: str, data_type: str, data: Any, expiry_hours: int = None):
        """Persist transient data that would normally be lost on restart"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        data_str = json.dumps(data) if not isinstance(data, str) else data
        expiry = None
        if expiry_hours:
            expiry = (datetime.now() + timedelta(hours=expiry_hours)).isoformat()
        
        cursor.execute('''
            INSERT OR REPLACE INTO persistent_storage 
            (storage_key, data_type, data_value, last_updated, expiry_time)
            VALUES (?, ?, ?, ?, ?)
        ''', (key, data_type, data_str, datetime.now().isoformat(), expiry))
        
        conn.commit()
        conn.close()
    
    def retrieve_persistent_data(self, key: str) -> Optional[Any]:
        """Retrieve persisted transient data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT data_value, expiry_time 
            FROM persistent_storage 
            WHERE storage_key = ? AND is_active = 1
        ''', (key,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            # Check expiry
            if result[1] and datetime.fromisoformat(result[1]) < datetime.now():
                return None
            
            try:
                return json.loads(result[0])
            except:
                return result[0]
        
        return None
    
    def get_data_loss_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive data loss report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Get audit summary
        cursor.execute('''
            SELECT operation_type, data_type, COUNT(*), SUM(success = 0)
            FROM data_audit_log
            WHERE timestamp >= ?
            GROUP BY operation_type, data_type
        ''', (since,))
        
        audit_summary = cursor.fetchall()
        
        # Get integrity check failures
        cursor.execute('''
            SELECT data_type, COUNT(*), SUM(check_passed = 0)
            FROM integrity_checks
            WHERE check_time >= ?
            GROUP BY data_type
        ''', (since,))
        
        integrity_summary = cursor.fetchall()
        
        # Get data loss incidents
        cursor.execute('''
            SELECT data_type, COUNT(*), SUM(records_lost), SUM(records_recovered)
            FROM data_loss_incidents
            WHERE incident_time >= ?
            GROUP BY data_type
        ''', (since,))
        
        loss_incidents = cursor.fetchall()
        
        conn.close()
        
        return {
            "report_period": f"Last {days} days",
            "generated_at": datetime.now().isoformat(),
            "audit_summary": [
                {
                    "operation": row[0],
                    "data_type": row[1],
                    "total_operations": row[2],
                    "failures": row[3] or 0
                } for row in audit_summary
            ],
            "integrity_checks": [
                {
                    "data_type": row[0],
                    "total_checks": row[1],
                    "failures": row[2] or 0
                } for row in integrity_summary
            ],
            "data_loss_incidents": [
                {
                    "data_type": row[0],
                    "incident_count": row[1],
                    "total_records_lost": row[2] or 0,
                    "total_records_recovered": row[3] or 0
                } for row in loss_incidents
            ]
        }
    
    def monitor_data_health(self) -> Dict[str, Any]:
        """Real-time data health monitoring"""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": "healthy",
            "components": {}
        }
        
        # Check each data component
        data_types = ["articles", "reports", "sources", "entities", "scraping_jobs"]
        
        for data_type in data_types:
            component_health = self._check_component_health(data_type)
            health_status["components"][data_type] = component_health
            
            if component_health["status"] != "healthy":
                health_status["overall_health"] = "degraded"
        
        return health_status
    
    def _check_component_health(self, data_type: str) -> Dict[str, Any]:
        """Check health of a specific data component"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check recent failures
        cursor.execute('''
            SELECT COUNT(*) 
            FROM data_audit_log
            WHERE data_type = ? AND success = 0 
            AND timestamp >= datetime('now', '-1 hour')
        ''', (data_type,))
        
        recent_failures = cursor.fetchone()[0]
        
        # Check last integrity check
        cursor.execute('''
            SELECT check_passed, check_time
            FROM integrity_checks
            WHERE data_type = ?
            ORDER BY created_at DESC
            LIMIT 1
        ''', (data_type,))
        
        last_check = cursor.fetchone()
        
        conn.close()
        
        status = "healthy"
        issues = []
        
        if recent_failures > 5:
            status = "degraded"
            issues.append(f"{recent_failures} recent operation failures")
        
        if last_check and not last_check[0]:
            status = "unhealthy"
            issues.append("Failed last integrity check")
        
        return {
            "status": status,
            "recent_failures": recent_failures,
            "last_integrity_check": last_check[1] if last_check else None,
            "issues": issues
        }
    
    def create_backup(self, backup_type: str = "full") -> str:
        """Create comprehensive backup of all data"""
        backup_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.backup_dir / f"backup_{backup_type}_{backup_id}"
        backup_path.mkdir(exist_ok=True)
        
        # Backup SQLite databases
        db_files = [
            "research_synthesis.db",
            "kpi_history.db", 
            self.db_path
        ]
        
        for db_file in db_files:
            if Path(db_file).exists():
                shutil.copy2(db_file, backup_path / db_file)
        
        # Backup in-memory data by requesting from API
        # This would need to be integrated with the actual API
        
        logger.info(f"Created {backup_type} backup: {backup_path}")
        return str(backup_path)

# Global instance
data_admin = DataAdminSystem()