"""
Configuration Management API Router
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, Optional, List
from loguru import logger
from pydantic import BaseModel

from research_synthesis.services.configuration_manager import config_manager

router = APIRouter()


class ConfigurationUpdateRequest(BaseModel):
    category: str
    settings: Dict[str, Any]


class ProfileApplicationRequest(BaseModel):
    profile_name: str
    apply_immediately: bool = True


@router.get("/")
async def get_configuration_status():
    """Get configuration management status"""
    status = config_manager.get_configuration_status()
    return {
        "service": "Configuration Management",
        "status": "active",
        "description": "Comprehensive configuration validation, optimization, and auto-fixing",
        "endpoints": [
            "/scan - Scan all configurations for issues",
            "/issues - Get detected configuration issues",
            "/fix - Auto-fix configuration issues", 
            "/profiles - Get available configuration profiles",
            "/profiles/{profile_name} - Apply configuration profile",
            "/recommendations - Get configuration recommendations"
        ],
        "current_status": status
    }

@router.post("/scan")
async def scan_configurations(background_tasks: BackgroundTasks):
    """Scan all system configurations for issues"""
    try:
        logger.info("Starting comprehensive configuration scan...")
        scan_results = await config_manager.scan_all_configurations()
        
        return {
            "status": "success",
            "scan_results": scan_results
        }
        
    except Exception as e:
        logger.error(f"Configuration scan failed: {e}")
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")

@router.get("/issues")
async def get_configuration_issues(
    category: Optional[str] = None,
    severity: Optional[str] = None,
    auto_fixable: Optional[bool] = None
):
    """Get detected configuration issues with optional filtering"""
    try:
        issues = list(config_manager.config_issues.values())
        
        # Apply filters
        if category:
            issues = [issue for issue in issues if issue.category == category]
        if severity:
            issues = [issue for issue in issues if issue.severity == severity]
        if auto_fixable is not None:
            issues = [issue for issue in issues if issue.auto_fixable == auto_fixable]
            
        # Convert to dict format
        issues_data = []
        for issue in issues:
            issues_data.append({
                "id": issue.id,
                "category": issue.category,
                "severity": issue.severity,
                "title": issue.title,
                "description": issue.description,
                "current_value": issue.current_value,
                "recommended_value": issue.recommended_value,
                "auto_fixable": issue.auto_fixable,
                "impact": issue.impact,
                "location": issue.location,
                "detected_at": issue.detected_at.isoformat()
            })
            
        return {
            "status": "success",
            "total_issues": len(issues_data),
            "filters_applied": {
                "category": category,
                "severity": severity,
                "auto_fixable": auto_fixable
            },
            "issues": issues_data
        }
        
    except Exception as e:
        logger.error(f"Error getting configuration issues: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get issues: {str(e)}")

@router.post("/fix")
async def fix_configuration_issues(issue_ids: Optional[List[str]] = None):
    """Auto-fix configuration issues"""
    try:
        logger.info("Starting configuration auto-fix...")
        fix_results = await config_manager.auto_fix_issues(issue_ids)
        
        return {
            "status": "success",
            "fix_results": fix_results,
            "message": f"Fixed {fix_results['successful_fixes']} of {fix_results['total_attempts']} issues"
        }
        
    except Exception as e:
        logger.error(f"Configuration auto-fix failed: {e}")
        raise HTTPException(status_code=500, detail=f"Auto-fix failed: {str(e)}")

@router.get("/profiles")
async def get_configuration_profiles():
    """Get available configuration profiles"""
    try:
        profiles_info = {}
        for name, profile in config_manager.profiles.items():
            profiles_info[name] = {
                "name": profile.name,
                "description": profile.description,
                "target_scenario": profile.target_scenario,
                "priority": profile.priority,
                "categories": list(profile.configurations.keys())
            }
            
        return {
            "status": "success",
            "available_profiles": profiles_info,
            "total_profiles": len(profiles_info),
            "recommended_profile": "balanced"
        }
        
    except Exception as e:
        logger.error(f"Error getting configuration profiles: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get profiles: {str(e)}")

@router.get("/profiles/{profile_name}")
async def get_profile_details(profile_name: str):
    """Get detailed information about a specific configuration profile"""
    try:
        if profile_name not in config_manager.profiles:
            raise HTTPException(status_code=404, detail="Profile not found")
            
        profile = config_manager.profiles[profile_name]
        
        return {
            "status": "success",
            "profile": {
                "name": profile.name,
                "description": profile.description,
                "target_scenario": profile.target_scenario,
                "priority": profile.priority,
                "configurations": profile.configurations
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting profile details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get profile details: {str(e)}")

@router.post("/profiles/{profile_name}/apply")
async def apply_configuration_profile(profile_name: str, request: ProfileApplicationRequest):
    """Apply a configuration profile"""
    try:
        if profile_name not in config_manager.profiles:
            raise HTTPException(status_code=404, detail="Profile not found")
            
        result = config_manager.apply_configuration_profile(profile_name)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
            
        # If apply_immediately is True, also trigger the configuration updates
        if request.apply_immediately:
            logger.info(f"Applying configuration profile '{profile_name}' immediately...")
            
            # Here we would apply the actual configuration changes
            # For now, we'll prepare the response
            
        return {
            "status": "success",
            "profile_application": result,
            "immediate_application": request.apply_immediately,
            "message": f"Configuration profile '{profile_name}' has been applied"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error applying configuration profile: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to apply profile: {str(e)}")

@router.get("/recommendations")
async def get_configuration_recommendations():
    """Get configuration optimization recommendations"""
    try:
        # Get current status
        status = config_manager.get_configuration_status()
        
        # Generate recommendations based on current issues
        recommendations = []
        
        if status["total_issues"] == 0:
            recommendations.append("Configuration appears optimal - no issues detected")
        else:
            if status["issues_by_severity"]["critical"] > 0:
                recommendations.append(f"URGENT: {status['issues_by_severity']['critical']} critical configuration issues require immediate attention")
                
            if status["auto_fixable_count"] > 0:
                recommendations.append(f"Quick wins available: {status['auto_fixable_count']} issues can be automatically fixed")
                
            # Category-specific recommendations
            if status["issues_by_category"]["scraping"] > 0:
                recommendations.append("Consider applying 'high_performance' profile to optimize scraping configuration")
                
            if status["issues_by_category"]["reports"] > 0:
                recommendations.append("Reports configuration needs attention - consider topic-specific templates")
                
            if status["issues_by_category"]["system"] > 0:
                recommendations.append("System configuration issues detected - check resource monitoring and cleanup settings")
        
        # Profile recommendations based on current load
        profile_recommendations = {
            "current_recommended": "balanced",
            "alternatives": {
                "high_performance": "For production environments with high load",
                "conservative": "For development or resource-constrained environments"
            }
        }
        
        return {
            "status": "success",
            "recommendations": recommendations,
            "profile_recommendations": profile_recommendations,
            "configuration_status": status,
            "next_actions": _get_recommended_actions(status)
        }
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")

@router.post("/optimize")
async def optimize_configuration(background_tasks: BackgroundTasks):
    """Comprehensive configuration optimization"""
    try:
        logger.info("Starting comprehensive configuration optimization...")
        
        # Step 1: Scan for issues
        scan_results = await config_manager.scan_all_configurations()
        
        # Step 2: Auto-fix what we can
        fix_results = await config_manager.auto_fix_issues()
        
        # Step 3: Apply balanced profile if no specific profile is set
        profile_result = config_manager.apply_configuration_profile("balanced")
        
        optimization_summary = {
            "scan_results": scan_results,
            "fix_results": fix_results,
            "profile_applied": profile_result,
            "total_improvements": fix_results["successful_fixes"],
            "remaining_issues": len(config_manager.config_issues)
        }
        
        return {
            "status": "success",
            "optimization_summary": optimization_summary,
            "message": f"Configuration optimization complete: {optimization_summary['total_improvements']} improvements applied"
        }
        
    except Exception as e:
        logger.error(f"Configuration optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

def _get_recommended_actions(status: Dict[str, Any]) -> List[str]:
    """Generate recommended actions based on status"""
    actions = []
    
    if status["total_issues"] > 0:
        actions.append("Run configuration scan to identify specific issues")
        
        if status["auto_fixable_count"] > 0:
            actions.append("Apply auto-fixes for quick improvements")
            
        if status["issues_by_severity"]["critical"] > 0:
            actions.append("Address critical issues immediately")
            
        actions.append("Consider applying an appropriate configuration profile")
    else:
        actions.append("Configuration is optimal - schedule regular scans to maintain health")
        
    return actions