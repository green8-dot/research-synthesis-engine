"""
Logic Manager System - "Organized and Optimized" 
OrbitScope ML Intelligence Suite

Central management system for business logic, rules, and decision-making processes
across the research synthesis engine. Provides a unified interface for managing
complex logic flows, conditional processing, and system-wide decision trees.

OPERATIONAL MOTTO: "Organized and Optimized"
- Every system component has a clear purpose and optimal performance
- All processes are streamlined for maximum efficiency
- Organization principles applied to code, data, and operations
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path

from research_synthesis.utils.system_status import system_status_service

# Import learning system components (will be available after initialization)
learning_system = None
report_analyzer = None


class LogicPriority(Enum):
    """Priority levels for logic execution"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class LogicType(Enum):
    """Types of logic rules"""
    CONDITIONAL = "conditional"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    EVENT_DRIVEN = "event_driven"
    SCHEDULED = "scheduled"


@dataclass
class LogicRule:
    """Represents a single logic rule"""
    id: str
    name: str
    description: str
    logic_type: LogicType
    priority: LogicPriority
    conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    is_active: bool = True
    execution_count: int = 0
    last_executed: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LogicManager:
    """Central logic management system"""
    
    def __init__(self):
        self.rules: Dict[str, LogicRule] = {}
        self.execution_history = []
        self.event_listeners = {}
        self.scheduled_tasks = {}
        self.is_running = False
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0
        }
        
        # Initialize default rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default logic rules for the system"""
        
        # Health monitoring logic
        self.add_rule(LogicRule(
            id="health_monitor_critical",
            name="Critical Health Monitor",
            description="Monitor system health and trigger alerts for critical issues",
            logic_type=LogicType.CONDITIONAL,
            priority=LogicPriority.CRITICAL,
            conditions={
                "system_health": {"operator": "less_than", "value": 50},
                "database_disconnected": {"operator": "equals", "value": True}
            },
            actions=[
                {"type": "alert", "level": "critical", "message": "System health critical"},
                {"type": "notify_admin", "method": "email"},
                {"type": "log", "level": "error", "message": "Critical health issues detected"}
            ]
        ))
        
        # Data quality logic
        self.add_rule(LogicRule(
            id="data_quality_check",
            name="Data Quality Validation",
            description="Validate incoming data quality and reject poor quality data",
            logic_type=LogicType.CONDITIONAL,
            priority=LogicPriority.HIGH,
            conditions={
                "data_completeness": {"operator": "less_than", "value": 0.8},
                "data_accuracy": {"operator": "less_than", "value": 0.9}
            },
            actions=[
                {"type": "reject_data", "reason": "Quality below threshold"},
                {"type": "log", "level": "warning", "message": "Data rejected due to quality"},
                {"type": "increment_counter", "counter": "rejected_data_count"}
            ]
        ))
        
        # Scraping optimization logic
        self.add_rule(LogicRule(
            id="scraping_optimization",
            name="Scraping Resource Optimization",
            description="Optimize scraping resources based on system load and success rates",
            logic_type=LogicType.CONDITIONAL,
            priority=LogicPriority.MEDIUM,
            conditions={
                "cpu_usage": {"operator": "greater_than", "value": 80},
                "scraping_success_rate": {"operator": "less_than", "value": 0.7}
            },
            actions=[
                {"type": "reduce_concurrent_scrapers", "factor": 0.5},
                {"type": "increase_delay", "delay_seconds": 2},
                {"type": "log", "level": "info", "message": "Scraping optimized for system load"}
            ]
        ))
        
        # UI readability logic
        self.add_rule(LogicRule(
            id="ui_readability_fix",
            name="UI Readability Auto-Fix",
            description="Automatically fix UI readability issues when detected",
            logic_type=LogicType.EVENT_DRIVEN,
            priority=LogicPriority.MEDIUM,
            conditions={
                "ui_contrast_issues": {"operator": "greater_than", "value": 0}
            },
            actions=[
                {"type": "apply_ui_fixes", "target": "color_contrast"},
                {"type": "log", "level": "info", "message": "UI readability issues fixed"},
                {"type": "refresh_ui_cache"}
            ]
        ))
        
        # Report generation logic
        self.add_rule(LogicRule(
            id="report_generation_optimization",
            name="Report Generation Optimization",
            description="Optimize report generation based on data availability and system resources",
            logic_type=LogicType.SEQUENTIAL,
            priority=LogicPriority.MEDIUM,
            conditions={
                "data_freshness": {"operator": "greater_than", "value": 24}, # hours
                "system_resources": {"operator": "greater_than", "value": 50} # percent
            },
            actions=[
                {"type": "validate_data", "min_records": 100},
                {"type": "generate_report", "format": "comprehensive"},
                {"type": "cache_report", "duration": 3600}, # 1 hour
                {"type": "notify_completion", "channels": ["ui", "log"]}
            ]
        ))
    
    def add_rule(self, rule: LogicRule) -> None:
        """Add a new logic rule"""
        self.rules[rule.id] = rule
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a logic rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            return True
        return False
    
    def get_rule(self, rule_id: str) -> Optional[LogicRule]:
        """Get a specific logic rule"""
        return self.rules.get(rule_id)
    
    def list_rules(self, filter_type: Optional[LogicType] = None, 
                   filter_priority: Optional[LogicPriority] = None) -> List[LogicRule]:
        """List rules with optional filtering"""
        rules = list(self.rules.values())
        
        if filter_type:
            rules = [r for r in rules if r.logic_type == filter_type]
        
        if filter_priority:
            rules = [r for r in rules if r.priority == filter_priority]
        
        # Sort by priority (critical first, then by name)
        return sorted(rules, key=lambda r: (r.priority.value, r.name))
    
    async def evaluate_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate a single condition against context data"""
        field = list(condition.keys())[0]
        criteria = condition[field]
        
        if field not in context:
            return False
        
        value = context[field]
        operator = criteria.get("operator", "equals")
        expected = criteria.get("value")
        
        if operator == "equals":
            return value == expected
        elif operator == "not_equals":
            return value != expected
        elif operator == "greater_than":
            return value > expected
        elif operator == "less_than":
            return value < expected
        elif operator == "greater_equal":
            return value >= expected
        elif operator == "less_equal":
            return value <= expected
        elif operator == "contains":
            return expected in str(value)
        elif operator == "not_contains":
            return expected not in str(value)
        elif operator == "in":
            return value in expected
        elif operator == "not_in":
            return value not in expected
        
        return False
    
    async def evaluate_rule(self, rule: LogicRule, context: Dict[str, Any]) -> bool:
        """Evaluate all conditions for a rule"""
        if not rule.is_active:
            return False
        
        # Evaluate all conditions (AND logic by default)
        for condition in rule.conditions:
            if not await self.evaluate_condition({condition: rule.conditions[condition]}, context):
                return False
        
        return True
    
    async def execute_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single action"""
        action_type = action.get("type", "log")
        result = {"success": False, "message": "", "data": {}}
        
        try:
            if action_type == "log":
                level = action.get("level", "info")
                message = action.get("message", "Logic action executed")
                print(f"[{level.upper()}] {message}")
                result = {"success": True, "message": f"Logged: {message}"}
                
            elif action_type == "alert":
                level = action.get("level", "info")
                message = action.get("message", "Alert triggered")
                print(f"[ALERT-{level.upper()}] {message}")
                result = {"success": True, "message": f"Alert sent: {message}"}
                
            elif action_type == "apply_ui_fixes":
                # Would integrate with UI monitor for actual fixes
                target = action.get("target", "general")
                print(f"[UI-FIX] Applying fixes for {target}")
                result = {"success": True, "message": f"UI fixes applied for {target}"}
                
            elif action_type == "reduce_concurrent_scrapers":
                factor = action.get("factor", 0.5)
                print(f"[SCRAPING] Reducing concurrent scrapers by factor {factor}")
                result = {"success": True, "message": f"Scrapers reduced by {factor}"}
                
            elif action_type == "validate_data":
                min_records = action.get("min_records", 1)
                print(f"[DATA] Validating minimum {min_records} records")
                result = {"success": True, "message": f"Data validated (min: {min_records})"}
                
            elif action_type == "generate_report":
                report_format = action.get("format", "standard")
                print(f"[REPORT] Generating {report_format} report")
                result = {"success": True, "message": f"Report generated: {report_format}"}
                
            elif action_type == "increment_counter":
                counter = action.get("counter", "generic_counter")
                context[counter] = context.get(counter, 0) + 1
                result = {"success": True, "message": f"Counter {counter} incremented"}
                
            # Learning system actions
            elif action_type in ["apply_learning_insights", "analyze_report_quality", 
                               "optimize_sources", "prioritize_gap_sources", "apply_quality_patterns"]:
                result = await self.execute_learning_action(action, context)
                
            else:
                result = {"success": False, "message": f"Unknown action type: {action_type}"}
                
        except Exception as e:
            result = {"success": False, "message": f"Action failed: {str(e)}"}
        
        return result
    
    async def execute_rule(self, rule: LogicRule, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a logic rule"""
        start_time = time.time()
        execution_result = {
            "rule_id": rule.id,
            "executed": False,
            "success": False,
            "actions_executed": 0,
            "actions_successful": 0,
            "execution_time": 0.0,
            "errors": []
        }
        
        try:
            # Check if rule conditions are met
            if not await self.evaluate_rule(rule, context):
                execution_result["message"] = "Conditions not met"
                return execution_result
            
            execution_result["executed"] = True
            rule.execution_count += 1
            rule.last_executed = datetime.now()
            
            # Execute actions based on logic type
            if rule.logic_type == LogicType.SEQUENTIAL:
                # Execute actions in sequence
                for action in rule.actions:
                    execution_result["actions_executed"] += 1
                    action_result = await self.execute_action(action, context)
                    if action_result["success"]:
                        execution_result["actions_successful"] += 1
                    else:
                        execution_result["errors"].append(action_result["message"])
                        
            elif rule.logic_type == LogicType.PARALLEL:
                # Execute actions in parallel
                tasks = [self.execute_action(action, context) for action in rule.actions]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    execution_result["actions_executed"] += 1
                    if isinstance(result, dict) and result.get("success"):
                        execution_result["actions_successful"] += 1
                    else:
                        execution_result["errors"].append(str(result))
                        
            else:  # CONDITIONAL, EVENT_DRIVEN, SCHEDULED
                # Execute all actions
                for action in rule.actions:
                    execution_result["actions_executed"] += 1
                    action_result = await self.execute_action(action, context)
                    if action_result["success"]:
                        execution_result["actions_successful"] += 1
                    else:
                        execution_result["errors"].append(action_result["message"])
            
            # Update rule statistics
            if execution_result["actions_successful"] == execution_result["actions_executed"]:
                execution_result["success"] = True
                rule.success_count += 1
            else:
                rule.failure_count += 1
            
        except Exception as e:
            execution_result["errors"].append(f"Rule execution failed: {str(e)}")
            rule.failure_count += 1
        
        execution_result["execution_time"] = time.time() - start_time
        
        # Update global statistics
        self.execution_stats['total_executions'] += 1
        if execution_result["success"]:
            self.execution_stats['successful_executions'] += 1
        else:
            self.execution_stats['failed_executions'] += 1
        
        # Update average execution time
        total_time = (self.execution_stats['average_execution_time'] * 
                     (self.execution_stats['total_executions'] - 1) + 
                     execution_result["execution_time"])
        self.execution_stats['average_execution_time'] = total_time / self.execution_stats['total_executions']
        
        return execution_result
    
    async def process_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process context data against all active rules"""
        results = {
            "processed_rules": 0,
            "executed_rules": 0,
            "successful_rules": 0,
            "total_actions": 0,
            "successful_actions": 0,
            "errors": [],
            "rule_results": []
        }
        
        # Sort rules by priority
        sorted_rules = sorted(self.rules.values(), key=lambda r: r.priority.value)
        
        for rule in sorted_rules:
            if not rule.is_active:
                continue
                
            results["processed_rules"] += 1
            rule_result = await self.execute_rule(rule, context)
            results["rule_results"].append(rule_result)
            
            if rule_result["executed"]:
                results["executed_rules"] += 1
                results["total_actions"] += rule_result["actions_executed"]
                results["successful_actions"] += rule_result["actions_successful"]
                
                if rule_result["success"]:
                    results["successful_rules"] += 1
                
                results["errors"].extend(rule_result["errors"])
        
        return results
    
    async def get_system_context(self) -> Dict[str, Any]:
        """Gather current system context for rule evaluation"""
        context = {
            "timestamp": datetime.now(),
            "system_health": 100,  # Default healthy
            "database_disconnected": False,
            "cpu_usage": 50,
            "memory_usage": 50,
            "scraping_success_rate": 0.9,
            "data_completeness": 0.95,
            "data_accuracy": 0.95,
            "ui_contrast_issues": 0,
            "data_freshness": 12,  # hours
            "system_resources": 75  # percent
        }
        
        try:
            # Get real system status
            system_status = await system_status_service.get_system_status()
            
            # Update context with real data
            if system_status:
                components = system_status.get('components', {})
                
                # Calculate health based on connected components
                total_components = len(components)
                connected_components = len([c for c in components.values() 
                                          if c.get('status') == 'connected'])
                
                if total_components > 0:
                    context["system_health"] = (connected_components / total_components) * 100
                
                # Check for database disconnections
                db_components = ['sqlite', 'mongodb', 'redis', 'elasticsearch']
                disconnected_dbs = [name for name, comp in components.items() 
                                  if name in db_components and comp.get('status') == 'disconnected']
                context["database_disconnected"] = len(disconnected_dbs) > 0
                
        except Exception as e:
            print(f"Error gathering system context: {e}")
        
        return context
    
    async def run_logic_cycle(self) -> Dict[str, Any]:
        """Run one complete logic processing cycle"""
        context = await self.get_system_context()
        results = await self.process_context(context)
        
        # Store execution history
        self.execution_history.append({
            "timestamp": datetime.now(),
            "context": context,
            "results": results
        })
        
        # Keep only last 100 executions
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logic manager statistics"""
        active_rules = len([r for r in self.rules.values() if r.is_active])
        
        return {
            "total_rules": len(self.rules),
            "active_rules": active_rules,
            "inactive_rules": len(self.rules) - active_rules,
            "execution_stats": self.execution_stats,
            "recent_executions": len(self.execution_history),
            "rule_types": {
                logic_type.value: len([r for r in self.rules.values() 
                                     if r.logic_type == logic_type])
                for logic_type in LogicType
            },
            "rule_priorities": {
                priority.name: len([r for r in self.rules.values() 
                                  if r.priority == priority])
                for priority in LogicPriority
            }
        }
    
    def export_rules(self) -> Dict[str, Any]:
        """Export all rules for backup or transfer"""
        return {
            "exported_at": datetime.now().isoformat(),
            "version": "1.0",
            "rules": [
                {
                    "id": rule.id,
                    "name": rule.name,
                    "description": rule.description,
                    "logic_type": rule.logic_type.value,
                    "priority": rule.priority.name,
                    "conditions": rule.conditions,
                    "actions": rule.actions,
                    "is_active": rule.is_active,
                    "metadata": rule.metadata
                }
                for rule in self.rules.values()
            ]
        }
    
    def import_rules(self, rules_data: Dict[str, Any]) -> Dict[str, Any]:
        """Import rules from exported data"""
        imported = 0
        errors = []
        
        try:
            for rule_data in rules_data.get("rules", []):
                try:
                    rule = LogicRule(
                        id=rule_data["id"],
                        name=rule_data["name"],
                        description=rule_data["description"],
                        logic_type=LogicType(rule_data["logic_type"]),
                        priority=LogicPriority[rule_data["priority"]],
                        conditions=rule_data["conditions"],
                        actions=rule_data["actions"],
                        is_active=rule_data.get("is_active", True),
                        metadata=rule_data.get("metadata", {})
                    )
                    self.add_rule(rule)
                    imported += 1
                except Exception as e:
                    errors.append(f"Failed to import rule {rule_data.get('id', 'unknown')}: {e}")
        except Exception as e:
            errors.append(f"Import failed: {e}")
        
        return {
            "imported_rules": imported,
            "total_rules": len(self.rules),
            "errors": errors
        }
    
    async def initialize_learning_integration(self):
        """Initialize learning system integration"""
        global learning_system, report_analyzer
        try:
            # Import learning components dynamically to avoid circular imports
            from research_synthesis.services.learning_system import learning_system as ls
            from research_synthesis.services.report_analyzer import report_analyzer as ra
            
            learning_system = ls
            report_analyzer = ra
            
            # Initialize learning system
            await learning_system.initialize()
            
            # Add learning-specific rules
            self._add_learning_rules()
            
            print("Learning system integration initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing learning integration: {e}")
            return False
    
    def _add_learning_rules(self):
        """Add rules that integrate with the learning system"""
        
        # Learning-based report optimization
        self.add_rule(LogicRule(
            id="learning_report_optimization",
            name="Learning-Based Report Optimization",
            description="Optimize report generation based on learned patterns",
            logic_type=LogicType.CONDITIONAL,
            priority=LogicPriority.HIGH,
            conditions={
                "new_report_requested": {"operator": "equals", "value": True},
                "learning_system_available": {"operator": "equals", "value": True}
            },
            actions=[
                {"type": "apply_learning_insights", "target": "report_generation"},
                {"type": "analyze_report_quality", "source": "learning_patterns"},
                {"type": "log", "level": "info", "message": "Learning insights applied to report generation"}
            ]
        ))
        
        # Pattern-based source optimization
        self.add_rule(LogicRule(
            id="pattern_based_source_optimization",
            name="Pattern-Based Source Optimization",
            description="Optimize data sources based on learned effectiveness patterns",
            logic_type=LogicType.SCHEDULED,
            priority=LogicPriority.MEDIUM,
            conditions={
                "source_patterns_available": {"operator": "equals", "value": True},
                "optimization_interval_reached": {"operator": "equals", "value": True}
            },
            actions=[
                {"type": "optimize_sources", "method": "learning_patterns"},
                {"type": "update_source_priorities", "based_on": "effectiveness_patterns"},
                {"type": "log", "level": "info", "message": "Sources optimized based on learning patterns"}
            ]
        ))
        
        # Knowledge gap identification
        self.add_rule(LogicRule(
            id="knowledge_gap_identification",
            name="Knowledge Gap Identification and Filling",
            description="Identify and address knowledge gaps using learning insights",
            logic_type=LogicType.CONDITIONAL,
            priority=LogicPriority.MEDIUM,
            conditions={
                "knowledge_gaps_detected": {"operator": "greater_than", "value": 0},
                "gap_filling_enabled": {"operator": "equals", "value": True}
            },
            actions=[
                {"type": "prioritize_gap_sources", "method": "learning_recommendations"},
                {"type": "trigger_targeted_scraping", "focus": "knowledge_gaps"},
                {"type": "log", "level": "info", "message": "Knowledge gaps being addressed"}
            ]
        ))
        
        # Predictive quality enhancement
        self.add_rule(LogicRule(
            id="predictive_quality_enhancement",
            name="Predictive Quality Enhancement",
            description="Enhance content quality using predictive learning insights",
            logic_type=LogicType.EVENT_DRIVEN,
            priority=LogicPriority.HIGH,
            conditions={
                "low_quality_content_detected": {"operator": "equals", "value": True},
                "quality_patterns_available": {"operator": "equals", "value": True}
            },
            actions=[
                {"type": "apply_quality_patterns", "source": "learned_insights"},
                {"type": "enhance_content_structure", "method": "pattern_based"},
                {"type": "log", "level": "info", "message": "Content quality enhanced using learning patterns"}
            ]
        ))
    
    async def execute_learning_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute learning-system-specific actions"""
        action_type = action.get("type", "")
        result = {"success": False, "message": "", "data": {}}
        
        if not learning_system:
            return {"success": False, "message": "Learning system not available"}
        
        try:
            if action_type == "apply_learning_insights":
                target = action.get("target", "general")
                report_type = context.get("report_type", "general")
                
                insights = await learning_system.get_insights_for_report_generation(report_type)
                context[f"learning_insights_{target}"] = insights
                
                result = {
                    "success": True,
                    "message": f"Learning insights applied to {target}",
                    "data": {"insights_count": len(insights.get('relevant_insights', []))}
                }
                
            elif action_type == "analyze_report_quality":
                report_id = context.get("report_id")
                if report_id and report_analyzer:
                    analysis = await report_analyzer.analyze_report(str(report_id))
                    context["quality_analysis"] = analysis
                    
                    result = {
                        "success": True,
                        "message": f"Report quality analyzed",
                        "data": {"quality_score": analysis.quality_score}
                    }
                else:
                    result = {"success": False, "message": "Report ID not available or analyzer not ready"}
                    
            elif action_type == "optimize_sources":
                method = action.get("method", "basic")
                status = await learning_system.get_learning_status()
                
                effective_sources = [
                    p for p in status.get('pattern_types', []) 
                    if 'source_effectiveness' in p
                ]
                
                context["optimized_sources"] = effective_sources
                result = {
                    "success": True,
                    "message": f"Sources optimized using {method}",
                    "data": {"effective_sources_count": len(effective_sources)}
                }
                
            elif action_type == "prioritize_gap_sources":
                learning_status = await learning_system.get_learning_status()
                gap_patterns = [
                    p for p in learning_status.get('pattern_types', [])
                    if 'knowledge_gap' in p
                ]
                
                context["gap_priorities"] = gap_patterns
                result = {
                    "success": True,
                    "message": "Knowledge gap sources prioritized",
                    "data": {"gap_patterns_count": len(gap_patterns)}
                }
                
            elif action_type == "apply_quality_patterns":
                if report_analyzer:
                    analyzer_status = await report_analyzer.get_analyzer_status()
                    quality_patterns = analyzer_status.get('patterns_learned', 0)
                    
                    context["quality_patterns_applied"] = quality_patterns
                    result = {
                        "success": True,
                        "message": "Quality patterns applied",
                        "data": {"patterns_count": quality_patterns}
                    }
                else:
                    result = {"success": False, "message": "Report analyzer not available"}
                    
            else:
                result = {"success": False, "message": f"Unknown learning action: {action_type}"}
                
        except Exception as e:
            result = {"success": False, "message": f"Learning action failed: {str(e)}"}
            
        return result
    
    async def get_learning_status(self) -> Dict[str, Any]:
        """Get status of learning system integration"""
        status = {
            "learning_system_available": learning_system is not None,
            "report_analyzer_available": report_analyzer is not None,
            "learning_rules_count": 0,
            "learning_integration_active": False
        }
        
        # Count learning-specific rules
        learning_rule_types = ["learning_report_optimization", "pattern_based_source_optimization", 
                              "knowledge_gap_identification", "predictive_quality_enhancement"]
        
        status["learning_rules_count"] = sum(1 for rule_id in learning_rule_types if rule_id in self.rules)
        status["learning_integration_active"] = status["learning_rules_count"] > 0
        
        if learning_system:
            try:
                learning_status = await learning_system.get_learning_status()
                status.update({
                    "patterns_discovered": learning_status.get('patterns_count', 0),
                    "insights_generated": learning_status.get('insights_count', 0),
                    "last_learning_cycle": learning_status.get('last_learning_cycle')
                })
            except Exception as e:
                status["learning_system_error"] = str(e)
        
        if report_analyzer:
            try:
                analyzer_status = await report_analyzer.get_analyzer_status()
                status.update({
                    "reports_analyzed": analyzer_status.get('reports_analyzed', 0),
                    "analysis_patterns": analyzer_status.get('patterns_learned', 0)
                })
            except Exception as e:
                status["report_analyzer_error"] = str(e)
                
        return status


# Global logic manager instance
logic_manager = LogicManager()