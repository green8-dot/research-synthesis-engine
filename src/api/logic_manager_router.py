"""
Logic Manager API Router

Provides API endpoints for managing business logic rules and decision trees.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()

# Import logic manager
try:
    from research_synthesis.services.logic_manager import logic_manager, LogicRule, LogicType, LogicPriority
    LOGIC_MANAGER_AVAILABLE = True
except ImportError:
    LOGIC_MANAGER_AVAILABLE = False
    logic_manager = None


class LogicRuleRequest(BaseModel):
    name: str
    description: str
    logic_type: str
    priority: str
    conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    is_active: bool = True
    metadata: Dict[str, Any] = {}


@router.get("/")
async def get_logic_manager_overview():
    """Get logic manager overview"""
    if not LOGIC_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Logic manager not available")
    
    stats = logic_manager.get_statistics()
    
    return {
        "status": "Logic Manager active",
        "service": "ready",
        "available_endpoints": [
            "/rules - List all logic rules",
            "/rules/{rule_id} - Get specific rule details",
            "/execute - Execute logic processing cycle",
            "/statistics - Get detailed statistics",
            "/export - Export all rules",
            "/import - Import rules from file"
        ],
        "statistics": stats,
        "last_checked": datetime.now().isoformat()
    }


@router.get("/rules")
async def get_logic_rules(logic_type: Optional[str] = None, priority: Optional[str] = None):
    """Get all logic rules with optional filtering"""
    if not LOGIC_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Logic manager not available")
    
    try:
        filter_type = LogicType(logic_type) if logic_type else None
        filter_priority = LogicPriority[priority] if priority else None
        
        rules = logic_manager.list_rules(filter_type, filter_priority)
        
        return {
            "rules": [
                {
                    "id": rule.id,
                    "name": rule.name,
                    "description": rule.description,
                    "logic_type": rule.logic_type.value,
                    "priority": rule.priority.name,
                    "is_active": rule.is_active,
                    "execution_count": rule.execution_count,
                    "success_count": rule.success_count,
                    "failure_count": rule.failure_count,
                    "last_executed": rule.last_executed.isoformat() if rule.last_executed else None,
                    "conditions": rule.conditions,
                    "actions": rule.actions
                }
                for rule in rules
            ],
            "total_rules": len(rules),
            "filters_applied": {
                "logic_type": logic_type,
                "priority": priority
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get rules: {str(e)}")


@router.get("/rules/{rule_id}")
async def get_logic_rule(rule_id: str):
    """Get specific logic rule details"""
    if not LOGIC_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Logic manager not available")
    
    rule = logic_manager.get_rule(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    
    return {
        "id": rule.id,
        "name": rule.name,
        "description": rule.description,
        "logic_type": rule.logic_type.value,
        "priority": rule.priority.name,
        "is_active": rule.is_active,
        "execution_count": rule.execution_count,
        "success_count": rule.success_count,
        "failure_count": rule.failure_count,
        "success_rate": (rule.success_count / rule.execution_count * 100) if rule.execution_count > 0 else 0,
        "last_executed": rule.last_executed.isoformat() if rule.last_executed else None,
        "conditions": rule.conditions,
        "actions": rule.actions,
        "metadata": rule.metadata
    }


@router.post("/rules")
async def create_logic_rule(rule_request: LogicRuleRequest):
    """Create a new logic rule"""
    if not LOGIC_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Logic manager not available")
    
    try:
        # Generate unique ID
        rule_id = f"custom_{rule_request.name.lower().replace(' ', '_')}_{int(datetime.now().timestamp())}"
        
        # Create rule
        rule = LogicRule(
            id=rule_id,
            name=rule_request.name,
            description=rule_request.description,
            logic_type=LogicType(rule_request.logic_type),
            priority=LogicPriority[rule_request.priority],
            conditions=rule_request.conditions,
            actions=rule_request.actions,
            is_active=rule_request.is_active,
            metadata=rule_request.metadata
        )
        
        logic_manager.add_rule(rule)
        
        return {
            "message": "Rule created successfully",
            "rule_id": rule_id,
            "rule": {
                "id": rule.id,
                "name": rule.name,
                "description": rule.description,
                "logic_type": rule.logic_type.value,
                "priority": rule.priority.name,
                "is_active": rule.is_active
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to create rule: {str(e)}")


@router.delete("/rules/{rule_id}")
async def delete_logic_rule(rule_id: str):
    """Delete a logic rule"""
    if not LOGIC_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Logic manager not available")
    
    if logic_manager.remove_rule(rule_id):
        return {"message": f"Rule '{rule_id}' deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Rule not found")


@router.post("/rules/{rule_id}/toggle")
async def toggle_logic_rule(rule_id: str):
    """Toggle rule active status"""
    if not LOGIC_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Logic manager not available")
    
    rule = logic_manager.get_rule(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    
    rule.is_active = not rule.is_active
    
    return {
        "message": f"Rule '{rule_id}' {'activated' if rule.is_active else 'deactivated'}",
        "rule_id": rule_id,
        "is_active": rule.is_active
    }


@router.post("/execute")
async def execute_logic_cycle():
    """Execute a complete logic processing cycle"""
    if not LOGIC_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Logic manager not available")
    
    try:
        results = await logic_manager.run_logic_cycle()
        
        return {
            "message": "Logic cycle completed successfully",
            "execution_results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Logic execution failed: {str(e)}")


@router.get("/statistics")
async def get_logic_statistics():
    """Get detailed logic manager statistics"""
    if not LOGIC_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Logic manager not available")
    
    stats = logic_manager.get_statistics()
    
    # Add recent execution history
    recent_history = []
    for execution in logic_manager.execution_history[-10:]:  # Last 10 executions
        recent_history.append({
            "timestamp": execution["timestamp"].isoformat(),
            "processed_rules": execution["results"]["processed_rules"],
            "executed_rules": execution["results"]["executed_rules"],
            "successful_rules": execution["results"]["successful_rules"],
            "total_actions": execution["results"]["total_actions"],
            "successful_actions": execution["results"]["successful_actions"],
            "error_count": len(execution["results"]["errors"])
        })
    
    return {
        "statistics": stats,
        "recent_executions": recent_history,
        "system_status": {
            "logic_manager_active": True,
            "total_rules_configured": len(logic_manager.rules),
            "default_rules_loaded": True
        }
    }


@router.get("/export")
async def export_logic_rules():
    """Export all logic rules"""
    if not LOGIC_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Logic manager not available")
    
    try:
        export_data = logic_manager.export_rules()
        return export_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.post("/import")
async def import_logic_rules(rules_data: Dict[str, Any]):
    """Import logic rules from data"""
    if not LOGIC_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Logic manager not available")
    
    try:
        result = logic_manager.import_rules(rules_data)
        
        if result["errors"]:
            return {
                "message": "Import completed with errors",
                "imported_rules": result["imported_rules"],
                "total_rules": result["total_rules"],
                "errors": result["errors"]
            }
        else:
            return {
                "message": "Import completed successfully",
                "imported_rules": result["imported_rules"],
                "total_rules": result["total_rules"]
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


@router.get("/health")
async def get_logic_health():
    """Get logic manager health status"""
    if not LOGIC_MANAGER_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "Logic manager not installed or configured"
        }
    
    stats = logic_manager.get_statistics()
    recent_executions = len(logic_manager.execution_history)
    
    # Determine health based on statistics
    health_score = 100
    health_issues = []
    
    # Check for execution failures
    if stats["execution_stats"]["failed_executions"] > 0:
        failure_rate = (stats["execution_stats"]["failed_executions"] / 
                       stats["execution_stats"]["total_executions"] * 100)
        if failure_rate > 10:
            health_score -= 20
            health_issues.append(f"High failure rate: {failure_rate:.1f}%")
    
    # Check for inactive rules
    if stats["inactive_rules"] > stats["active_rules"]:
        health_score -= 10
        health_issues.append("More inactive rules than active rules")
    
    # Check for recent activity
    if recent_executions == 0:
        health_score -= 5
        health_issues.append("No recent execution history")
    
    health_status = "excellent" if health_score >= 90 else "good" if health_score >= 70 else "needs_attention"
    
    return {
        "status": health_status,
        "health_score": health_score,
        "statistics": stats,
        "recent_executions": recent_executions,
        "issues": health_issues,
        "recommendations": [
            "Monitor rule execution success rates",
            "Review and activate unused rules",
            "Run logic cycles regularly for optimal performance"
        ],
        "timestamp": datetime.now().isoformat()
    }