"""
System Integration Framework
Handles incorporation of new systems, functions, and operations across the platform
"""
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger
import json
import importlib
import inspect

@dataclass
class SystemComponent:
    """Represents a system component that can be integrated"""
    name: str
    component_type: str  # 'service', 'api', 'utility', 'monitoring', 'data_processor'
    description: str
    module_path: str
    router_path: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    required_config: Dict[str, Any] = field(default_factory=dict)
    endpoints: List[str] = field(default_factory=list)
    data_types: List[str] = field(default_factory=list)
    monitoring_enabled: bool = True
    admin_integration: bool = True
    auto_register: bool = True
    priority: int = 5  # 1-10, higher = more important
    status: str = "inactive"  # inactive, active, error, deprecated
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class IntegrationRule:
    """Rules for integrating systems together"""
    name: str
    source_component: str
    target_component: str
    integration_type: str  # 'data_flow', 'api_call', 'event_trigger', 'logging'
    configuration: Dict[str, Any] = field(default_factory=dict)
    conditions: List[str] = field(default_factory=list)
    active: bool = True

class SystemIntegrationManager:
    """Manages system integration and component registration"""
    
    def __init__(self, config_path: str = "system_integration.json"):
        self.config_path = Path(config_path)
        self.components: Dict[str, SystemComponent] = {}
        self.integration_rules: Dict[str, IntegrationRule] = {}
        self.load_configuration()
        
        # Initialize data admin integration
        try:
            from research_synthesis.utils.data_admin import data_admin
            self.data_admin = data_admin
            self.admin_available = True
        except ImportError:
            self.data_admin = None
            self.admin_available = False
            logger.warning("Data admin not available for system integration logging")
    
    def load_configuration(self):
        """Load system configuration from file"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                # Load components
                for comp_data in config.get('components', []):
                    component = SystemComponent(**comp_data)
                    # Convert string datetime back to datetime object
                    if isinstance(component.created_at, str):
                        component.created_at = datetime.fromisoformat(component.created_at)
                    self.components[component.name] = component
                
                # Load integration rules
                for rule_data in config.get('integration_rules', []):
                    rule = IntegrationRule(**rule_data)
                    self.integration_rules[rule.name] = rule
                    
                logger.info(f"Loaded {len(self.components)} components and {len(self.integration_rules)} integration rules")
            except Exception as e:
                logger.error(f"Failed to load system configuration: {e}")
                self._initialize_default_config()
        else:
            self._initialize_default_config()
    
    def _initialize_default_config(self):
        """Initialize with default system components"""
        logger.info("Initializing default system configuration")
        
        # Core system components
        default_components = [
            SystemComponent(
                name="ui_monitoring",
                component_type="monitoring",
                description="UI monitoring and issue reporting system",
                module_path="research_synthesis.services.ui_monitor",
                router_path="research_synthesis.api.ui_monitoring_router",
                endpoints=["/status", "/scan", "/issues", "/fix", "/report-issue"],
                data_types=["ui_issue_report"],
                admin_integration=True,
                priority=8
            ),
            SystemComponent(
                name="automation_ideas",
                component_type="service",
                description="Automation ideas discovery and analysis",
                module_path="research_synthesis.services.automation_ideas",
                router_path="research_synthesis.api.automation_ideas_router",
                endpoints=["/discover", "/analyze", "/categorize"],
                data_types=["automation_idea", "discovery_session"],
                priority=7
            ),
            SystemComponent(
                name="data_admin",
                component_type="utility",
                description="Data administration and audit system",
                module_path="research_synthesis.utils.data_admin",
                router_path="research_synthesis.api.data_admin_router",
                endpoints=["/status", "/audit", "/snapshot", "/backup", "/ui-issues"],
                data_types=["audit_log", "data_snapshot", "ui_issue_report"],
                priority=10
            ),
            SystemComponent(
                name="system_status",
                component_type="monitoring",
                description="System health and status monitoring",
                module_path="research_synthesis.utils.system_status",
                router_path="research_synthesis.api.system_status_router",
                endpoints=["/health", "/components", "/metrics"],
                data_types=["system_health", "component_status"],
                priority=9
            )
        ]
        
        for component in default_components:
            self.register_component(component)
            
        # Default integration rules
        default_rules = [
            IntegrationRule(
                name="ui_issues_to_admin",
                source_component="ui_monitoring",
                target_component="data_admin",
                integration_type="logging",
                configuration={"log_level": "info", "include_metadata": True},
                conditions=["issue_created", "issue_resolved"]
            ),
            IntegrationRule(
                name="automation_discovery_to_admin",
                source_component="automation_ideas",
                target_component="data_admin",
                integration_type="logging",
                configuration={"log_level": "debug", "include_stats": True},
                conditions=["discovery_started", "discovery_completed"]
            )
        ]
        
        for rule in default_rules:
            self.integration_rules[rule.name] = rule
        
        self.save_configuration()
    
    def save_configuration(self):
        """Save current configuration to file"""
        try:
            config = {
                'components': [
                    {
                        **component.__dict__,
                        'created_at': component.created_at.isoformat()
                    }
                    for component in self.components.values()
                ],
                'integration_rules': [
                    rule.__dict__ for rule in self.integration_rules.values()
                ]
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            logger.debug("System configuration saved")
        except Exception as e:
            logger.error(f"Failed to save system configuration: {e}")
    
    def register_component(self, component: SystemComponent) -> bool:
        """Register a new system component"""
        try:
            # Validate component
            if not self._validate_component(component):
                return False
            
            # Check dependencies
            missing_deps = self._check_dependencies(component)
            if missing_deps:
                logger.warning(f"Component {component.name} has missing dependencies: {missing_deps}")
            
            # Register component
            self.components[component.name] = component
            component.status = "active"
            
            # Log registration
            if self.admin_available:
                self.data_admin.log_operation(
                    operation_type="CREATE",
                    data_type="system_component",
                    data_id=component.name,
                    metadata={
                        "component_type": component.component_type,
                        "module_path": component.module_path,
                        "endpoints": component.endpoints,
                        "priority": component.priority
                    },
                    success=True
                )
            
            # Auto-setup integrations if enabled
            if component.auto_register:
                self._setup_component_integrations(component)
            
            self.save_configuration()
            logger.info(f"Successfully registered component: {component.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register component {component.name}: {e}")
            if self.admin_available:
                self.data_admin.log_operation(
                    operation_type="CREATE",
                    data_type="system_component",
                    data_id=component.name,
                    success=False,
                    error_message=str(e)
                )
            return False
    
    def _validate_component(self, component: SystemComponent) -> bool:
        """Validate component configuration"""
        if not component.name or not component.module_path:
            logger.error("Component must have name and module_path")
            return False
        
        # Check if module exists
        try:
            importlib.import_module(component.module_path)
        except ImportError as e:
            logger.error(f"Component module {component.module_path} not found: {e}")
            return False
        
        return True
    
    def _check_dependencies(self, component: SystemComponent) -> List[str]:
        """Check if component dependencies are available"""
        missing_deps = []
        
        for dep in component.dependencies:
            if dep not in self.components:
                missing_deps.append(dep)
            elif self.components[dep].status != "active":
                missing_deps.append(f"{dep} (inactive)")
        
        return missing_deps
    
    def _setup_component_integrations(self, component: SystemComponent):
        """Set up automatic integrations for a new component"""
        # Set up admin logging if component supports it
        if component.admin_integration and self.admin_available:
            rule_name = f"{component.name}_to_admin"
            if rule_name not in self.integration_rules:
                admin_rule = IntegrationRule(
                    name=rule_name,
                    source_component=component.name,
                    target_component="data_admin",
                    integration_type="logging",
                    configuration={"log_level": "info", "auto_generated": True}
                )
                self.integration_rules[rule_name] = admin_rule
                logger.info(f"Auto-created admin integration rule: {rule_name}")
        
        # Set up monitoring integration
        if component.monitoring_enabled and "system_status" in self.components:
            rule_name = f"{component.name}_monitoring"
            if rule_name not in self.integration_rules:
                monitoring_rule = IntegrationRule(
                    name=rule_name,
                    source_component=component.name,
                    target_component="system_status",
                    integration_type="event_trigger",
                    configuration={"health_check": True, "auto_generated": True}
                )
                self.integration_rules[rule_name] = monitoring_rule
                logger.info(f"Auto-created monitoring integration rule: {rule_name}")
    
    def get_active_components(self, component_type: Optional[str] = None) -> List[SystemComponent]:
        """Get all active components, optionally filtered by type"""
        components = [
            comp for comp in self.components.values()
            if comp.status == "active"
        ]
        
        if component_type:
            components = [comp for comp in components if comp.component_type == component_type]
        
        # Sort by priority (highest first)
        return sorted(components, key=lambda x: x.priority, reverse=True)
    
    def get_integration_map(self) -> Dict[str, List[str]]:
        """Get map of component integrations"""
        integration_map = {}
        
        for rule in self.integration_rules.values():
            if rule.active:
                if rule.source_component not in integration_map:
                    integration_map[rule.source_component] = []
                integration_map[rule.source_component].append(rule.target_component)
        
        return integration_map
    
    def apply_integration_rule(self, rule_name: str, data: Dict[str, Any]) -> bool:
        """Apply an integration rule with given data"""
        if rule_name not in self.integration_rules:
            logger.error(f"Integration rule {rule_name} not found")
            return False
        
        rule = self.integration_rules[rule_name]
        if not rule.active:
            return True  # Skip inactive rules
        
        try:
            # Check conditions
            for condition in rule.conditions:
                if condition not in data.get('triggers', []):
                    return True  # Condition not met, skip
            
            # Apply integration based on type
            if rule.integration_type == "logging" and self.admin_available:
                self.data_admin.log_operation(
                    operation_type="INTEGRATION",
                    data_type=f"{rule.source_component}_to_{rule.target_component}",
                    data_id=data.get('id', 'unknown'),
                    metadata={
                        "rule_name": rule_name,
                        "source_data": data,
                        "configuration": rule.configuration
                    },
                    success=True
                )
            
            elif rule.integration_type == "api_call":
                # Implement API call integration
                logger.debug(f"API call integration {rule_name} triggered")
            
            elif rule.integration_type == "event_trigger":
                # Implement event trigger integration  
                logger.debug(f"Event trigger integration {rule_name} triggered")
            
            elif rule.integration_type == "data_flow":
                # Implement data flow integration
                logger.debug(f"Data flow integration {rule_name} triggered")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply integration rule {rule_name}: {e}")
            return False
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview"""
        active_components = self.get_active_components()
        
        return {
            "total_components": len(self.components),
            "active_components": len(active_components),
            "component_types": {
                comp_type: len([c for c in active_components if c.component_type == comp_type])
                for comp_type in set(c.component_type for c in active_components)
            },
            "integration_rules": len([r for r in self.integration_rules.values() if r.active]),
            "data_admin_available": self.admin_available,
            "components": [
                {
                    "name": comp.name,
                    "type": comp.component_type,
                    "status": comp.status,
                    "priority": comp.priority,
                    "endpoints": len(comp.endpoints),
                    "data_types": len(comp.data_types)
                }
                for comp in active_components
            ],
            "overview_generated": datetime.now().isoformat()
        }

# Global system integration manager instance
system_integration_manager = SystemIntegrationManager()

def register_new_component(
    name: str,
    component_type: str,
    description: str,
    module_path: str,
    **kwargs
) -> bool:
    """Helper function to register a new system component"""
    component = SystemComponent(
        name=name,
        component_type=component_type,
        description=description,
        module_path=module_path,
        **kwargs
    )
    
    return system_integration_manager.register_component(component)

def log_system_operation(component_name: str, operation_data: Dict[str, Any]):
    """Helper function to log system operations and trigger integrations"""
    # Apply relevant integration rules
    for rule_name, rule in system_integration_manager.integration_rules.items():
        if rule.source_component == component_name:
            system_integration_manager.apply_integration_rule(rule_name, operation_data)