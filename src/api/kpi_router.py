"""
KPI API Router with Historical Tracking and Percentage Changes
Provides detailed metrics with trends and growth analysis
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from datetime import datetime

router = APIRouter()

@router.get("/")
async def get_kpi_overview():
    """Get KPI system overview"""
    return {
        "status": "KPI Tracking API active",
        "service": "ready",
        "available_endpoints": [
            "/current - Current metrics with percentage changes",
            "/historical/{metric} - Historical data for specific metric",
            "/trends - Growth rates and trend analysis",
            "/record - Record current metrics snapshot (POST)",
            "/dashboard - Dashboard-ready KPI data"
        ],
        "tracked_metrics": [
            "total_articles",
            "total_reports", 
            "active_sources",
            "total_entities"
        ]
    }

@router.get("/current")
async def get_current_kpis():
    """Get current KPIs with percentage changes and trends"""
    try:
        from research_synthesis.utils.kpi_tracker import kpi_tracker
        return kpi_tracker.get_metrics_with_changes()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get KPIs: {str(e)}")

@router.get("/historical/{metric_name}")
async def get_historical_data(metric_name: str, days: int = 30):
    """Get historical data for a specific metric"""
    try:
        from research_synthesis.utils.kpi_tracker import kpi_tracker
        
        valid_metrics = ["total_articles", "total_reports", "active_sources", "total_entities"]
        if metric_name not in valid_metrics:
            raise HTTPException(status_code=400, detail=f"Invalid metric. Must be one of: {valid_metrics}")
        
        historical_data = kpi_tracker.get_historical_data(metric_name, days)
        
        return {
            "metric": metric_name,
            "period_days": days,
            "data_points": len(historical_data),
            "data": historical_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get historical data: {str(e)}")

@router.get("/trends")
async def get_trend_analysis():
    """Get growth rates and trend analysis for all metrics"""
    try:
        from research_synthesis.utils.kpi_tracker import kpi_tracker
        
        metrics = ["total_articles", "total_reports", "active_sources", "total_entities"]
        trends = {}
        
        for metric in metrics:
            # Get growth rates for different periods
            trends[metric] = {
                "weekly": kpi_tracker.get_growth_rate(metric, 7),
                "monthly": kpi_tracker.get_growth_rate(metric, 30),
                "recent_data": kpi_tracker.get_historical_data(metric, 7)
            }
        
        return {
            "analysis_date": datetime.now().isoformat(),
            "trends": trends,
            "summary": _generate_trend_summary(trends)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trends: {str(e)}")

@router.post("/record")
async def record_current_metrics():
    """Record current metrics snapshot for tracking"""
    try:
        from research_synthesis.utils.kpi_tracker import kpi_tracker
        
        # Get current metrics
        current_metrics = kpi_tracker._get_current_metrics()
        
        # Record them
        kpi_tracker.record_metrics(current_metrics)
        
        return {
            "message": "Metrics recorded successfully",
            "timestamp": datetime.now().isoformat(),
            "metrics": current_metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record metrics: {str(e)}")

@router.get("/dashboard")
async def get_dashboard_kpis():
    """Get KPI data formatted for dashboard display"""
    try:
        from research_synthesis.utils.kpi_tracker import kpi_tracker
        
        # Get metrics with changes
        metrics = kpi_tracker.get_metrics_with_changes()
        
        # Format for dashboard
        dashboard_data = {}
        
        for metric_name, data in metrics.items():
            # Determine display properties based on trend
            trend_icon = {
                'strong_up': 'fa-arrow-up text-success',
                'up': 'fa-arrow-up text-success',
                'stable': 'fa-minus text-muted',
                'down': 'fa-arrow-down text-warning',
                'strong_down': 'fa-arrow-down text-danger'
            }.get(data['trend'], 'fa-minus text-muted')
            
            trend_color = {
                'strong_up': 'success',
                'up': 'success', 
                'stable': 'secondary',
                'down': 'warning',
                'strong_down': 'danger'
            }.get(data['trend'], 'secondary')
            
            # Format display name
            display_name = {
                'total_articles': 'Articles Collected',
                'total_reports': 'Reports Generated',
                'active_sources': 'Active Sources',
                'total_entities': 'Knowledge Entities'
            }.get(metric_name, metric_name.replace('_', ' ').title())
            
            dashboard_data[metric_name] = {
                'display_name': display_name,
                'current_value': data['current'],
                'daily_change': data['daily_change'],
                'daily_change_pct': data['daily_change_pct'],
                'weekly_change': data['weekly_change'],
                'weekly_change_pct': data['weekly_change_pct'],
                'trend': data['trend'],
                'trend_icon': trend_icon,
                'trend_color': trend_color,
                'display_change': f"{'+' if data['daily_change'] >= 0 else ''}{data['daily_change']} ({'+' if data['daily_change_pct'] >= 0 else ''}{data['daily_change_pct']}%)"
            }
        
        return {
            "metrics": dashboard_data,
            "last_updated": datetime.now().isoformat(),
            "summary": _generate_dashboard_summary(dashboard_data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard KPIs: {str(e)}")

def _generate_trend_summary(trends: Dict[str, Any]) -> Dict[str, str]:
    """Generate human-readable trend summary"""
    summary = {}
    
    for metric, data in trends.items():
        weekly_growth = data['weekly']['growth_rate']
        confidence = data['weekly']['confidence']
        
        if weekly_growth > 5:
            summary[metric] = f"Strong growth ({weekly_growth:.1f}%/day)"
        elif weekly_growth > 1:
            summary[metric] = f"Growing ({weekly_growth:.1f}%/day)"
        elif weekly_growth > -1:
            summary[metric] = f"Stable ({weekly_growth:.1f}%/day)"
        elif weekly_growth > -5:
            summary[metric] = f"Declining ({weekly_growth:.1f}%/day)"
        else:
            summary[metric] = f"Sharp decline ({weekly_growth:.1f}%/day)"
        
        if confidence == 'low':
            summary[metric] += " (limited data)"
    
    return summary

def _generate_dashboard_summary(dashboard_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary statistics for dashboard"""
    trends = [data['trend'] for data in dashboard_data.values()]
    
    positive_trends = sum(1 for t in trends if t in ['up', 'strong_up'])
    negative_trends = sum(1 for t in trends if t in ['down', 'strong_down'])
    stable_trends = sum(1 for t in trends if t == 'stable')
    
    overall_status = "excellent" if positive_trends >= 3 else /
                    "good" if positive_trends >= negative_trends else /
                    "needs_attention" if negative_trends > positive_trends else /
                    "stable"
    
    return {
        "overall_status": overall_status,
        "positive_trends": positive_trends,
        "negative_trends": negative_trends,
        "stable_trends": stable_trends,
        "total_metrics": len(dashboard_data),
        "status_message": _get_status_message(overall_status, positive_trends, negative_trends)
    }

def _get_status_message(status: str, positive: int, negative: int) -> str:
    """Get human-readable status message"""
    if status == "excellent":
        return f"System performing excellently with {positive} growing metrics"
    elif status == "good":
        return f"System performing well with {positive} positive trends"
    elif status == "needs_attention":
        return f"System needs attention - {negative} declining metrics"
    else:
        return "System metrics are stable"