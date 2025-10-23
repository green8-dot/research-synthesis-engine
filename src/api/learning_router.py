"""
Learning System API Router
Provides endpoints for the learning system functionality
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()

# Import learning components dynamically to avoid circular imports
learning_system = None
report_analyzer = None  
logic_manager = None

def get_learning_components():
    """Get learning system components dynamically"""
    global learning_system, report_analyzer, logic_manager
    
    if learning_system is None:
        try:
            from research_synthesis.services.learning_system import learning_system as ls
            learning_system = ls
        except ImportError as e:
            print(f"Learning system import failed: {e}")
            
    if report_analyzer is None:
        try:
            from research_synthesis.services.report_analyzer import report_analyzer as ra
            report_analyzer = ra
        except ImportError as e:
            print(f"Report analyzer import failed: {e}")
            
    if logic_manager is None:
        try:
            from research_synthesis.services.logic_manager import logic_manager as lm
            logic_manager = lm
        except ImportError as e:
            print(f"Logic manager import failed: {e}")
            
    return learning_system, report_analyzer, logic_manager

class LearningStatusResponse(BaseModel):
    system_status: str
    patterns_count: int
    insights_count: int
    knowledge_graph_nodes: int
    predictive_insights_enabled: bool
    last_learning_cycle: Optional[str]

class ReportAnalysisResponse(BaseModel):
    report_id: str
    quality_score: float
    complexity_score: float
    topic_coverage: List[str]
    key_themes: List[str]
    readability_score: float
    insight_density: float
    recommendations: List[str]

@router.get("/")
async def get_learning_system_status():
    """Get overall status of learning systems"""
    try:
        # Get components dynamically
        learning_sys, report_anly, logic_mgr = get_learning_components()
        
        status = {
            "learning_system": "not_available",
            "report_analyzer": "not_available", 
            "logic_manager": "not_available",
            "integration_status": "inactive",
            "available_endpoints": [
                "/status",
                "/patterns",
                "/insights",
                "/analyze-report/{report_id}",
                "/learning-integration-status"
            ]
        }
        
        if learning_sys:
            learning_status = await learning_sys.get_learning_status()
            status["learning_system"] = learning_status.get("system_status", "unknown")
            
        if report_anly:
            analyzer_status = await report_anly.get_analyzer_status()
            status["report_analyzer"] = analyzer_status.get("system_status", "unknown")
            
        if logic_mgr:
            logic_status = await logic_mgr.get_learning_status()
            status["logic_manager"] = "active" if logic_status.get("learning_integration_active") else "inactive"
            status["integration_status"] = "active" if logic_status.get("learning_integration_active") else "inactive"
            
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting learning system status: {str(e)}")

@router.get("/status")
async def get_detailed_learning_status():
    """Get detailed learning system status"""
    learning_sys, _, _ = get_learning_components()
    
    if not learning_sys:
        raise HTTPException(status_code=503, detail="Learning system not available")
        
    try:
        return await learning_sys.get_learning_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting learning status: {str(e)}")

@router.get("/patterns")
async def get_learning_patterns():
    """Get discovered learning patterns"""
    if not learning_system:
        raise HTTPException(status_code=503, detail="Learning system not available")
        
    try:
        status = await learning_system.get_learning_status()
        return {
            "patterns_count": status.get("patterns_count", 0),
            "pattern_types": status.get("pattern_types", []),
            "performance_metrics": status.get("performance_metrics", {}),
            "discovery_active": status.get("system_status") == "active"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting learning patterns: {str(e)}")

@router.get("/insights")
async def get_learning_insights():
    """Get generated insights from learning system"""
    if not learning_system:
        raise HTTPException(status_code=503, detail="Learning system not available")
        
    try:
        status = await learning_system.get_learning_status()
        return {
            "insights_count": status.get("insights_count", 0),
            "insight_types": status.get("insight_types", []),
            "knowledge_graph_nodes": status.get("knowledge_graph_nodes", 0),
            "predictive_insights_enabled": status.get("predictive_insights_enabled", False)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting learning insights: {str(e)}")

@router.post("/insights/generate")
async def generate_insights():
    """Manually trigger insight generation"""
    if not learning_system:
        raise HTTPException(status_code=503, detail="Learning system not available")
        
    try:
        # Trigger manual insight generation
        insights = await learning_system.analyze_reports_for_insights()
        
        return {
            "message": "Insights generation completed",
            "insights_generated": len(insights),
            "insights": [
                {
                    "insight_id": insight.insight_id,
                    "type": insight.insight_type,
                    "summary": insight.summary,
                    "confidence_score": insight.confidence_score
                }
                for insight in insights[:10]  # Return first 10 for API response
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")

@router.post("/patterns/discover")
async def discover_patterns():
    """Manually trigger pattern discovery"""
    if not learning_system:
        raise HTTPException(status_code=503, detail="Learning system not available")
        
    try:
        # Trigger manual pattern discovery
        patterns = await learning_system.discover_patterns()
        
        return {
            "message": "Pattern discovery completed",
            "patterns_discovered": len(patterns),
            "patterns": [
                {
                    "pattern_id": pattern.pattern_id,
                    "type": pattern.pattern_type,
                    "confidence": pattern.confidence,
                    "frequency": pattern.frequency
                }
                for pattern in patterns[:10]  # Return first 10 for API response
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error discovering patterns: {str(e)}")

@router.get("/analyze-report/{report_id}")
async def analyze_report(report_id: str):
    """Analyze a specific report using the report analyzer"""
    if not report_analyzer:
        raise HTTPException(status_code=503, detail="Report analyzer not available")
        
    try:
        analysis = await report_analyzer.analyze_report(report_id)
        
        return {
            "report_id": analysis.report_id,
            "quality_score": analysis.quality_score,
            "complexity_score": analysis.complexity_score,
            "topic_coverage": analysis.topic_coverage,
            "entity_mentions": dict(list(analysis.entity_mentions.items())[:10]),  # Limit for response size
            "key_themes": analysis.key_themes,
            "readability_score": analysis.readability_score,
            "insight_density": analysis.insight_density,
            "source_diversity": analysis.source_diversity,
            "factual_accuracy": analysis.factual_accuracy,
            "recommendations_quality": analysis.recommendations_quality
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing report: {str(e)}")

@router.get("/analyze-report/{report_id}/suggestions")
async def get_report_improvement_suggestions(report_id: str):
    """Get improvement suggestions for a specific report"""
    if not report_analyzer:
        raise HTTPException(status_code=503, detail="Report analyzer not available")
        
    try:
        suggestions = await report_analyzer.get_improvement_suggestions(report_id)
        return suggestions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting improvement suggestions: {str(e)}")

@router.post("/analyze-reports")
async def analyze_batch_reports(report_ids: List[str]):
    """Analyze multiple reports in batch"""
    if not report_analyzer:
        raise HTTPException(status_code=503, detail="Report analyzer not available")
        
    if len(report_ids) > 20:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size limited to 20 reports")
        
    try:
        analyses = await report_analyzer.analyze_batch_reports(report_ids)
        
        return {
            "batch_size": len(report_ids),
            "analyzed_count": len(analyses),
            "analyses": [
                {
                    "report_id": analysis.report_id,
                    "quality_score": analysis.quality_score,
                    "complexity_score": analysis.complexity_score,
                    "topic_coverage_count": len(analysis.topic_coverage),
                    "key_themes": analysis.key_themes
                }
                for analysis in analyses.values()
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing batch reports: {str(e)}")

@router.get("/report-insights/{report_type}")
async def get_report_type_insights(report_type: str):
    """Get learning insights for improving a specific report type"""
    if not learning_system:
        raise HTTPException(status_code=503, detail="Learning system not available")
        
    try:
        insights = await learning_system.get_insights_for_report_generation(report_type)
        return insights
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting report insights: {str(e)}")

@router.get("/learning-integration-status")
async def get_learning_integration_status():
    """Get status of learning system integration with logic manager"""
    if not logic_manager:
        raise HTTPException(status_code=503, detail="Logic manager not available")
        
    try:
        return await logic_manager.get_learning_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting integration status: {str(e)}")

@router.post("/initialize-learning-integration")
async def initialize_learning_integration():
    """Initialize learning system integration"""
    _, _, logic_mgr = get_learning_components()
    
    if not logic_mgr:
        raise HTTPException(status_code=503, detail="Logic manager not available")
        
    try:
        success = await logic_mgr.initialize_learning_integration()
        
        if success:
            return {
                "message": "Learning integration initialized successfully",
                "status": "active",
                "integration_details": await logic_mgr.get_learning_status()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to initialize learning integration")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing learning integration: {str(e)}")

@router.get("/knowledge-graph/status")
async def get_knowledge_graph_status():
    """Get knowledge graph status"""
    if not learning_system:
        raise HTTPException(status_code=503, detail="Learning system not available")
        
    try:
        status = await learning_system.get_learning_status()
        return {
            "nodes_count": status.get("knowledge_graph_nodes", 0),
            "learning_active": status.get("system_status") == "active",
            "last_update": status.get("last_learning_cycle"),
            "graph_features": {
                "entity_relationships": "active",
                "topic_correlations": "active",
                "predictive_connections": status.get("predictive_insights_enabled", False)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting knowledge graph status: {str(e)}")

@router.get("/analyzer/status")
async def get_analyzer_status():
    """Get report analyzer detailed status"""
    if not report_analyzer:
        raise HTTPException(status_code=503, detail="Report analyzer not available")
        
    try:
        return await report_analyzer.get_analyzer_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting analyzer status: {str(e)}")

@router.post("/learning-cycle/trigger")
async def trigger_learning_cycle():
    """Manually trigger a learning cycle"""
    if not learning_system:
        raise HTTPException(status_code=503, detail="Learning system not available")
        
    try:
        # Trigger the learning components
        insights = await learning_system.analyze_reports_for_insights()
        patterns = await learning_system.discover_patterns()
        await learning_system.update_knowledge_graph()
        predictions = await learning_system.generate_predictive_insights()
        
        return {
            "message": "Learning cycle completed successfully",
            "results": {
                "insights_generated": len(insights),
                "patterns_discovered": len(patterns),
                "predictions_generated": len(predictions),
                "knowledge_graph_updated": True
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error triggering learning cycle: {str(e)}")

@router.get("/predictive-insights")
async def get_predictive_insights():
    """Get current predictive insights"""
    if not learning_system:
        raise HTTPException(status_code=503, detail="Learning system not available")
        
    try:
        predictions = await learning_system.generate_predictive_insights()
        return {
            "predictions_count": len(predictions),
            "predictions": predictions,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting predictive insights: {str(e)}")