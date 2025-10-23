"""
System Integration API Router
Provides endpoints for managing system components and integrations
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel

router = APIRouter()

# Import system integration manager
try:
    from research_synthesis.utils.system_integration import (
        system_integration_manager,
        SystemComponent,
        IntegrationRule,
        register_new_component,
        log_system_operation
    )
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"System integration not available: {e}")
    INTEGRATION_AVAILABLE = False
    system_integration_manager = None

class ComponentRegistrationRequest(BaseModel):
    name: str
    component_type: str
    description: str
    module_path: str
    router_path: Optional[str] = None
    dependencies: List[str] = []
    required_config: Dict[str, Any] = {}
    endpoints: List[str] = []
    data_types: List[str] = []
    monitoring_enabled: bool = True
    admin_integration: bool = True
    auto_register: bool = True
    priority: int = 5

class IntegrationRuleRequest(BaseModel):
    name: str
    source_component: str
    target_component: str
    integration_type: str
    configuration: Dict[str, Any] = {}
    conditions: List[str] = []
    active: bool = True

class SystemOperationLog(BaseModel):
    component_name: str
    operation_type: str
    data: Dict[str, Any]
    triggers: List[str] = []

@router.get("/status")
async def get_integration_status():
    """Get system integration manager status"""
    if not INTEGRATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="System integration unavailable")
    
    try:
        overview = system_integration_manager.get_system_overview()
        integration_map = system_integration_manager.get_integration_map()
        
        return {
            "status": "operational",
            "integration_manager_available": True,
            "system_overview": overview,
            "integration_map": integration_map,
            "checked_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.get("/components")
async def get_system_components(
    component_type: Optional[str] = None,
    status: Optional[str] = None,
    active_only: bool = True
):
    """Get list of system components"""
    if not INTEGRATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="System integration unavailable")
    
    try:
        if active_only:
            components = system_integration_manager.get_active_components(component_type)
        else:
            components = list(system_integration_manager.components.values())
            if component_type:
                components = [c for c in components if c.component_type == component_type]
            if status:
                components = [c for c in components if c.status == status]
        
        return {
            "total_components": len(components),
            "filters_applied": {
                "component_type": component_type,
                "status": status,
                "active_only": active_only
            },
            "components": [
                {
                    "name": comp.name,
                    "component_type": comp.component_type,
                    "description": comp.description,
                    "module_path": comp.module_path,
                    "router_path": comp.router_path,
                    "dependencies": comp.dependencies,
                    "endpoints": comp.endpoints,
                    "data_types": comp.data_types,
                    "monitoring_enabled": comp.monitoring_enabled,
                    "admin_integration": comp.admin_integration,
                    "priority": comp.priority,
                    "status": comp.status,
                    "version": comp.version,
                    "created_at": comp.created_at.isoformat()
                }
                for comp in components
            ],
            "retrieved_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve components: {str(e)}")

@router.post("/components/register")
async def register_component(request: ComponentRegistrationRequest):
    """Register a new system component"""
    if not INTEGRATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="System integration unavailable")
    
    try:
        # Check if component already exists
        if request.name in system_integration_manager.components:
            existing_comp = system_integration_manager.components[request.name]
            if existing_comp.status == "active":
                raise HTTPException(status_code=409, detail=f"Component {request.name} already exists and is active")
        
        # Create component from request
        component = SystemComponent(
            name=request.name,
            component_type=request.component_type,
            description=request.description,
            module_path=request.module_path,
            router_path=request.router_path,
            dependencies=request.dependencies,
            required_config=request.required_config,
            endpoints=request.endpoints,
            data_types=request.data_types,
            monitoring_enabled=request.monitoring_enabled,
            admin_integration=request.admin_integration,
            auto_register=request.auto_register,
            priority=request.priority
        )
        
        # Register the component
        success = system_integration_manager.register_component(component)
        
        if success:
            return {
                "message": "Component registered successfully",
                "component_name": request.name,
                "status": "active",
                "auto_integrations_created": component.auto_register,
                "admin_integration_enabled": component.admin_integration,
                "monitoring_enabled": component.monitoring_enabled,
                "registered_at": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Component registration failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@router.get("/integration-rules")
async def get_integration_rules(
    source_component: Optional[str] = None,
    target_component: Optional[str] = None,
    integration_type: Optional[str] = None,
    active_only: bool = True
):
    """Get integration rules"""
    if not INTEGRATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="System integration unavailable")
    
    try:
        rules = list(system_integration_manager.integration_rules.values())
        
        # Apply filters
        if active_only:
            rules = [r for r in rules if r.active]
        if source_component:
            rules = [r for r in rules if r.source_component == source_component]
        if target_component:
            rules = [r for r in rules if r.target_component == target_component]
        if integration_type:
            rules = [r for r in rules if r.integration_type == integration_type]
        
        return {
            "total_rules": len(rules),
            "filters_applied": {
                "source_component": source_component,
                "target_component": target_component,
                "integration_type": integration_type,
                "active_only": active_only
            },
            "integration_rules": [
                {
                    "name": rule.name,
                    "source_component": rule.source_component,
                    "target_component": rule.target_component,
                    "integration_type": rule.integration_type,
                    "configuration": rule.configuration,
                    "conditions": rule.conditions,
                    "active": rule.active
                }
                for rule in rules
            ],
            "retrieved_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve rules: {str(e)}")

@router.post("/integration-rules/create")
async def create_integration_rule(request: IntegrationRuleRequest):
    """Create a new integration rule"""
    if not INTEGRATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="System integration unavailable")
    
    try:
        # Check if rule already exists
        if request.name in system_integration_manager.integration_rules:
            raise HTTPException(status_code=409, detail=f"Integration rule {request.name} already exists")
        
        # Validate source and target components exist
        if request.source_component not in system_integration_manager.components:
            raise HTTPException(status_code=400, detail=f"Source component {request.source_component} not found")
        
        if request.target_component not in system_integration_manager.components:
            raise HTTPException(status_code=400, detail=f"Target component {request.target_component} not found")
        
        # Create integration rule
        rule = IntegrationRule(
            name=request.name,
            source_component=request.source_component,
            target_component=request.target_component,
            integration_type=request.integration_type,
            configuration=request.configuration,
            conditions=request.conditions,
            active=request.active
        )
        
        # Add to manager
        system_integration_manager.integration_rules[request.name] = rule
        system_integration_manager.save_configuration()
        
        return {
            "message": "Integration rule created successfully",
            "rule_name": request.name,
            "source_component": request.source_component,
            "target_component": request.target_component,
            "integration_type": request.integration_type,
            "active": request.active,
            "created_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rule creation failed: {str(e)}")

@router.post("/log-operation")
async def log_system_operation(request: SystemOperationLog):
    """Log a system operation and trigger integrations"""
    if not INTEGRATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="System integration unavailable")
    
    try:
        # Validate component exists
        if request.component_name not in system_integration_manager.components:
            raise HTTPException(status_code=400, detail=f"Component {request.component_name} not found")
        
        # Prepare operation data
        operation_data = {
            "id": request.data.get("id", f"{request.component_name}_{datetime.now().timestamp()}"),
            "operation_type": request.operation_type,
            "triggers": request.triggers,
            **request.data
        }
        
        # Log the operation and trigger integrations
        log_system_operation(request.component_name, operation_data)
        
        # Count how many integration rules were triggered
        triggered_rules = []
        for rule_name, rule in system_integration_manager.integration_rules.items():
            if (rule.source_component == request.component_name and 
                rule.active and 
                any(condition in request.triggers for condition in rule.conditions if rule.conditions)):
                triggered_rules.append(rule_name)
        
        return {
            "message": "System operation logged successfully",
            "component_name": request.component_name,
            "operation_type": request.operation_type,
            "triggered_integrations": len(triggered_rules),
            "integration_rules_triggered": triggered_rules,
            "logged_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Operation logging failed: {str(e)}")

@router.get("/overview")
async def get_system_overview():
    """Get comprehensive system integration overview"""
    if not INTEGRATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="System integration unavailable")
    
    try:
        overview = system_integration_manager.get_system_overview()
        integration_map = system_integration_manager.get_integration_map()
        
        # Add additional metrics
        active_rules = [r for r in system_integration_manager.integration_rules.values() if r.active]
        integration_types = {}
        for rule in active_rules:
            int_type = rule.integration_type
            integration_types[int_type] = integration_types.get(int_type, 0) + 1
        
        return {
            **overview,
            "integration_details": {
                "total_rules": len(system_integration_manager.integration_rules),
                "active_rules": len(active_rules),
                "integration_types": integration_types,
                "integration_map": integration_map
            },
            "system_health": {
                "components_operational": len([c for c in system_integration_manager.components.values() if c.status == "active"]),
                "admin_integration_available": system_integration_manager.admin_available,
                "configuration_file_exists": system_integration_manager.config_path.exists()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Overview generation failed: {str(e)}")

@router.post("/components/{component_name}/activate")
async def activate_component(component_name: str):
    """Activate a system component"""
    if not INTEGRATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="System integration unavailable")
    
    try:
        if component_name not in system_integration_manager.components:
            raise HTTPException(status_code=404, detail=f"Component {component_name} not found")
        
        component = system_integration_manager.components[component_name]
        
        # Check dependencies
        missing_deps = system_integration_manager._check_dependencies(component)
        if missing_deps:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot activate component due to missing dependencies: {missing_deps}"
            )
        
        # Activate component
        component.status = "active"
        system_integration_manager.save_configuration()
        
        # Set up integrations
        if component.auto_register:
            system_integration_manager._setup_component_integrations(component)
        
        return {
            "message": "Component activated successfully",
            "component_name": component_name,
            "status": "active",
            "integrations_setup": component.auto_register,
            "activated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Component activation failed: {str(e)}")

@router.post("/components/{component_name}/deactivate")
async def deactivate_component(component_name: str):
    """Deactivate a system component"""
    if not INTEGRATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="System integration unavailable")
    
    try:
        if component_name not in system_integration_manager.components:
            raise HTTPException(status_code=404, detail=f"Component {component_name} not found")
        
        component = system_integration_manager.components[component_name]
        component.status = "inactive"
        system_integration_manager.save_configuration()
        
        # Disable related integration rules
        disabled_rules = []
        for rule_name, rule in system_integration_manager.integration_rules.items():
            if rule.source_component == component_name or rule.target_component == component_name:
                rule.active = False
                disabled_rules.append(rule_name)
        
        return {
            "message": "Component deactivated successfully",
            "component_name": component_name,
            "status": "inactive",
            "disabled_integration_rules": len(disabled_rules),
            "deactivated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Component deactivation failed: {str(e)}")