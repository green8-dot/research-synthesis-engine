"""
Claude Model Selection API Router
Provides endpoints for intelligent Claude model selection with Pro subscription optimization
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
from pydantic import BaseModel

from research_synthesis.services.claude_model_selector import model_selector, ClaudeModel
from loguru import logger

router = APIRouter(prefix="/api/v1/claude-models", tags=["claude-models"])


class ModelSelectionRequest(BaseModel):
    prompt: str
    context: Optional[Dict[str, Any]] = None
    
    
class ModelSelectionResponse(BaseModel):
    selected_model: str
    reasoning: str
    estimated_cost: float
    complexity: str
    pro_status: Dict[str, Any]


@router.post("/select", response_model=ModelSelectionResponse)
async def select_optimal_model(request: ModelSelectionRequest):
    """Select the optimal Claude model for a given prompt"""
    try:
        # Get model selection
        model, reasoning = model_selector.select_model(request.prompt, request.context)
        
        # Get task requirements for additional info
        requirements = model_selector.analyze_task(request.prompt, request.context)
        
        # Estimate cost
        estimated_cost = model_selector._estimate_cost(model, requirements.input_length + requirements.output_length)
        
        # Get Pro subscription status
        pro_status = model_selector.get_opus_usage_status()
        
        return ModelSelectionResponse(
            selected_model=model.value,
            reasoning=reasoning,
            estimated_cost=estimated_cost,
            complexity=requirements.complexity,
            pro_status=pro_status
        )
        
    except Exception as e:
        logger.error(f"Model selection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/opus-status")
async def get_opus_usage_status():
    """Get current Opus usage status for Pro subscription"""
    try:
        return model_selector.get_opus_usage_status()
    except Exception as e:
        logger.error(f"Failed to get Opus status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/opus-usage/track")
async def track_opus_usage(tokens_used: int):
    """Track Opus token usage"""
    try:
        model_selector.track_opus_usage(tokens_used)
        return {
            "status": "success",
            "message": f"Tracked {tokens_used:,} Opus tokens",
            "current_status": model_selector.get_opus_usage_status()
        }
    except Exception as e:
        logger.error(f"Failed to track Opus usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/opus-usage/reset")
async def reset_opus_counter():
    """Reset daily Opus usage counter (when free tier refreshes)"""
    try:
        model_selector.reset_daily_opus_counter()
        return {
            "status": "success",
            "message": "Daily Opus counter reset",
            "current_status": model_selector.get_opus_usage_status()
        }
    except Exception as e:
        logger.error(f"Failed to reset Opus counter: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/usage-statistics")
async def get_model_usage_statistics():
    """Get comprehensive model usage statistics"""
    try:
        return model_selector.get_usage_statistics()
    except Exception as e:
        logger.error(f"Failed to get usage statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_available_models():
    """List all available Claude models with their capabilities"""
    try:
        models_info = {}
        for model in ClaudeModel:
            capabilities = model_selector.MODEL_CAPABILITIES[model]
            models_info[model.value] = {
                "name": model.value,
                "capabilities": capabilities,
                "recommended_for": model_selector._get_model_recommendations(model)
            }
        
        return {
            "models": models_info,
            "default_model": model_selector.default_model.value,
            "pro_subscription_enabled": model_selector.has_pro_subscription
        }
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-task")
async def analyze_task_requirements(prompt: str, context: Optional[Dict[str, Any]] = None):
    """Analyze task requirements without selecting a model"""
    try:
        requirements = model_selector.analyze_task(prompt, context)
        
        return {
            "task_analysis": {
                "complexity": requirements.complexity,
                "estimated_input_tokens": requirements.input_length,
                "estimated_output_tokens": requirements.output_length,
                "quality_needed": requirements.quality_needed,
                "domain_expertise_required": requirements.domain_expertise,
                "reasoning_depth": requirements.reasoning_depth,
                "creativity_needed": requirements.creativity_needed,
                "cost_sensitivity": requirements.cost_sensitivity
            },
            "model_scores": {
                model.value: model_selector._calculate_model_score(model, requirements)
                for model in ClaudeModel
            }
        }
        
    except Exception as e:
        logger.error(f"Task analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cost-estimates")
async def get_cost_estimates(
    input_tokens: int = Query(..., description="Number of input tokens"),
    output_tokens: int = Query(..., description="Number of output tokens")
):
    """Get cost estimates for all models"""
    try:
        estimates = {}
        total_tokens = input_tokens + output_tokens
        
        for model in ClaudeModel:
            cost = model_selector._estimate_cost(model, total_tokens)
            estimates[model.value] = {
                "estimated_cost_usd": cost,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            }
        
        # Add Pro subscription context for Opus
        if model_selector.has_pro_subscription:
            opus_status = model_selector.get_opus_usage_status()
            remaining_free = opus_status.get("remaining_free_tokens", 0)
            
            if total_tokens <= remaining_free:
                estimates[ClaudeModel.OPUS.value]["pro_note"] = "Within free tier - no charge"
            else:
                free_portion = max(0, remaining_free)
                paid_portion = total_tokens - free_portion
                paid_cost = model_selector._estimate_cost(ClaudeModel.OPUS, paid_portion)
                estimates[ClaudeModel.OPUS.value]["pro_note"] = f"Partial charge: ${paid_cost:.6f} (after {free_portion:,} free tokens)"
        
        return {
            "cost_estimates": estimates,
            "pro_subscription": model_selector.has_pro_subscription,
            "opus_status": model_selector.get_opus_usage_status() if model_selector.has_pro_subscription else None
        }
        
    except Exception as e:
        logger.error(f"Cost estimation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper method to add to model selector class
def _get_model_recommendations(self, model: ClaudeModel) -> Dict[str, Any]:
    """Get recommendations for when to use each model"""
    recommendations = {
        ClaudeModel.HAIKU: {
            "best_for": ["Simple questions", "Quick summaries", "Fast responses", "Basic analysis"],
            "avoid_for": ["Complex reasoning", "Creative writing", "Domain expertise tasks"],
            "use_when": "Speed and cost efficiency are prioritized over sophistication"
        },
        ClaudeModel.SONNET: {
            "best_for": ["Balanced tasks", "Code generation", "Research analysis", "Most general use"],
            "avoid_for": ["Extremely complex reasoning", "Basic simple queries"],
            "use_when": "Need good quality output without premium cost"
        },
        ClaudeModel.OPUS: {
            "best_for": ["Complex analysis", "Creative writing", "Advanced reasoning", "Domain expertise"],
            "avoid_for": ["Simple queries", "When cost efficiency is critical"],
            "use_when": "Maximum quality is required and within Pro free tier"
        }
    }
    
    return recommendations.get(model, {"best_for": [], "avoid_for": [], "use_when": ""})

# Add the method to the class
model_selector._get_model_recommendations = _get_model_recommendations.__get__(model_selector, type(model_selector))