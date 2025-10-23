"""
Usage Analytics API Router
Provides endpoints for MongoDB and Claude API usage statistics with cost tracking
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from research_synthesis.services.usage_tracking_service import usage_tracker
from loguru import logger

router = APIRouter(prefix="/api/v1/usage", tags=["usage"])


@router.get("/stats")
async def get_usage_stats(
    days: int = Query(7, ge=1, le=365, description="Number of days to analyze")
) -> Dict[str, Any]:
    """Get comprehensive usage and cost statistics"""
    try:
        stats = await usage_tracker.get_usage_stats(days=days)
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        logger.error(f"Failed to get usage stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/costs/summary")
async def get_cost_summary() -> Dict[str, Any]:
    """Get cost summary for current month"""
    try:
        stats = await usage_tracker.get_usage_stats(days=30)
        current_month = datetime.now().strftime('%Y-%m')
        
        mongodb_monthly = usage_tracker.monthly_totals.get(f"{current_month}_mongodb", 0.0)
        claude_monthly = usage_tracker.monthly_totals.get(f"{current_month}_claude", 0.0)
        total_monthly = mongodb_monthly + claude_monthly
        
        # Calculate daily averages
        days_in_month = datetime.now().day
        daily_avg = total_monthly / days_in_month if days_in_month > 0 else 0
        
        return {
            "status": "success",
            "data": {
                "current_month": current_month,
                "total_cost_usd": round(total_monthly, 6),
                "mongodb_cost_usd": round(mongodb_monthly, 6),
                "claude_cost_usd": round(claude_monthly, 6),
                "daily_average_usd": round(daily_avg, 6),
                "projected_monthly_usd": round(daily_avg * 30, 4),
                "days_tracked": days_in_month
            }
        }
    except Exception as e:
        logger.error(f"Failed to get cost summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mongodb/stats")
async def get_mongodb_stats() -> Dict[str, Any]:
    """Get detailed MongoDB usage and storage statistics"""
    try:
        storage_stats = await usage_tracker._get_storage_stats()
        
        # Get recent MongoDB operations from usage history
        recent_ops = []
        today = datetime.now().strftime('%Y-%m-%d')
        
        for record in usage_tracker.daily_usage.get(today, []):
            if record.service == 'mongodb':
                recent_ops.append({
                    'timestamp': record.timestamp.isoformat(),
                    'operation': record.operation,
                    'collection': record.metadata.get('collection', 'unknown'),
                    'documents': record.quantity,
                    'cost_usd': record.cost_usd
                })
        
        return {
            "status": "success",
            "data": {
                "storage": storage_stats,
                "recent_operations": recent_ops[-10:],  # Last 10 operations
                "total_operations_today": len(recent_ops),
                "total_cost_today_usd": round(sum(op['cost_usd'] for op in recent_ops), 6)
            }
        }
    except Exception as e:
        logger.error(f"Failed to get MongoDB stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/claude/stats")
async def get_claude_stats() -> Dict[str, Any]:
    """Get detailed Claude API usage statistics"""
    try:
        # Get recent Claude API calls
        recent_calls = []
        today = datetime.now().strftime('%Y-%m-%d')
        
        total_tokens_today = 0
        total_cost_today = 0.0
        model_usage = {}
        
        for record in usage_tracker.daily_usage.get(today, []):
            if record.service == 'claude':
                model = record.metadata.get('model', 'unknown')
                input_tokens = record.metadata.get('input_tokens', 0)
                output_tokens = record.metadata.get('output_tokens', 0)
                
                recent_calls.append({
                    'timestamp': record.timestamp.isoformat(),
                    'model': model,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': record.quantity,
                    'cost_usd': record.cost_usd
                })
                
                total_tokens_today += record.quantity
                total_cost_today += record.cost_usd
                
                if model not in model_usage:
                    model_usage[model] = {
                        'calls': 0,
                        'tokens': 0,
                        'cost_usd': 0.0
                    }
                
                model_usage[model]['calls'] += 1
                model_usage[model]['tokens'] += record.quantity
                model_usage[model]['cost_usd'] += record.cost_usd
        
        return {
            "status": "success",
            "data": {
                "recent_api_calls": recent_calls[-10:],  # Last 10 calls
                "total_calls_today": len(recent_calls),
                "total_tokens_today": total_tokens_today,
                "total_cost_today_usd": round(total_cost_today, 6),
                "model_breakdown": {
                    model: {
                        'calls': stats['calls'],
                        'tokens': stats['tokens'],
                        'cost_usd': round(stats['cost_usd'], 6)
                    }
                    for model, stats in model_usage.items()
                },
                "pricing_info": usage_tracker.CLAUDE_PRICING
            }
        }
    except Exception as e:
        logger.error(f"Failed to get Claude stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/track/mongodb")
async def track_mongodb_operation(
    operation: str,
    collection: str,
    document_count: int = 1,
    data_size_kb: float = 0
) -> Dict[str, Any]:
    """Manually track a MongoDB operation"""
    try:
        await usage_tracker.track_mongodb_operation(
            operation=operation,
            collection=collection,
            document_count=document_count,
            data_size_kb=data_size_kb
        )
        
        return {
            "status": "success",
            "message": f"MongoDB {operation} operation tracked",
            "details": {
                "collection": collection,
                "documents": document_count,
                "data_size_kb": data_size_kb
            }
        }
    except Exception as e:
        logger.error(f"Failed to track MongoDB operation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/track/claude")
async def track_claude_usage_manual(
    model: str,
    input_tokens: int,
    output_tokens: int,
    operation: str = "api_call"
) -> Dict[str, Any]:
    """Manually track Claude API usage"""
    try:
        await usage_tracker.track_claude_api_usage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            operation=operation
        )
        
        # Calculate cost
        model_pricing = usage_tracker.CLAUDE_PRICING.get(model, usage_tracker.CLAUDE_PRICING['claude-3-sonnet'])
        input_cost = input_tokens * model_pricing['input_tokens']
        output_cost = output_tokens * model_pricing['output_tokens']
        total_cost = input_cost + output_cost
        
        return {
            "status": "success",
            "message": f"Claude {model} usage tracked",
            "details": {
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cost_usd": round(total_cost, 6)
            }
        }
    except Exception as e:
        logger.error(f"Failed to track Claude usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/csv")
async def export_usage_csv(
    days: int = Query(30, ge=1, le=365, description="Number of days to export"),
    service: Optional[str] = Query(None, description="Filter by service: mongodb, claude")
):
    """Export usage data as CSV"""
    try:
        from io import StringIO
        import csv
        from fastapi.responses import StreamingResponse
        
        # Collect data
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            'timestamp', 'service', 'operation', 'resource_type', 
            'quantity', 'cost_usd', 'metadata'
        ])
        
        # Data
        for days_back in range(days):
            date = end_date - timedelta(days=days_back)
            date_key = date.strftime('%Y-%m-%d')
            
            for record in usage_tracker.daily_usage.get(date_key, []):
                if service is None or record.service == service:
                    writer.writerow([
                        record.timestamp.isoformat(),
                        record.service,
                        record.operation,
                        record.resource_type,
                        record.quantity,
                        record.cost_usd,
                        str(record.metadata or {})
                    ])
        
        output.seek(0)
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=usage_export_{start_date}_{end_date}.csv"
            }
        )
    except Exception as e:
        logger.error(f"Failed to export CSV: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/reset")
async def reset_usage_data() -> Dict[str, Any]:
    """Reset all usage tracking data (use with caution)"""
    try:
        usage_tracker.daily_usage.clear()
        usage_tracker.monthly_totals.clear()
        usage_tracker._save_usage_history()
        
        return {
            "status": "success",
            "message": "All usage data has been reset"
        }
    except Exception as e:
        logger.error(f"Failed to reset usage data: {e}")
        raise HTTPException(status_code=500, detail=str(e))