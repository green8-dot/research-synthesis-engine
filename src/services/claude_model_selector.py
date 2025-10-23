"""
Intelligent Claude Model Selection Service
Automatically chooses the most cost-effective Claude model based on task requirements
"""
import re
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
from dataclasses import dataclass
from loguru import logger


class ClaudeModel(Enum):
    """Available Claude models with cost optimization"""
    HAIKU = "claude-3-haiku-20240307"      # Fastest, cheapest - for simple tasks
    SONNET = "claude-3-sonnet-20240229"    # Balanced - default for most tasks
    OPUS = "claude-3-opus-20240229"        # Most capable - for complex tasks only


@dataclass
class TaskRequirements:
    """Task requirements for model selection"""
    complexity: str  # simple, moderate, complex
    input_length: int  # estimated tokens
    output_length: int  # estimated tokens
    quality_needed: str  # basic, standard, premium
    domain_expertise: bool  # requires specialized knowledge
    reasoning_depth: str  # shallow, moderate, deep
    creativity_needed: bool  # requires creative thinking
    cost_sensitivity: str  # low, medium, high (high = prefer cheaper models)


class ClaudeModelSelector:
    """Intelligent model selection based on task requirements and cost optimization"""
    
    def __init__(self, default_model: ClaudeModel = ClaudeModel.SONNET, 
                 has_pro_subscription: bool = True):
        self.default_model = default_model
        self.usage_stats = {}  # Track model usage for optimization
        self.has_pro_subscription = has_pro_subscription
        self.opus_tokens_used_today = 0  # Track daily Opus usage
        self.opus_free_limit_per_period = 50000  # Estimated free tokens per refresh period
    
    # Model capabilities scoring (1-10 scale)
    MODEL_CAPABILITIES = {
        ClaudeModel.HAIKU: {
            'speed': 10,
            'cost_efficiency': 10,
            'reasoning': 6,
            'creativity': 5,
            'domain_knowledge': 6,
            'complex_analysis': 4,
            'code_generation': 7,
            'max_recommended_tokens': 50000
        },
        ClaudeModel.SONNET: {
            'speed': 7,
            'cost_efficiency': 7,
            'reasoning': 8,
            'creativity': 8,
            'domain_knowledge': 8,
            'complex_analysis': 8,
            'code_generation': 9,
            'max_recommended_tokens': 150000
        },
        ClaudeModel.OPUS: {
            'speed': 5,
            'cost_efficiency': 3,
            'reasoning': 10,
            'creativity': 10,
            'domain_knowledge': 10,
            'complex_analysis': 10,
            'code_generation': 10,
            'max_recommended_tokens': 200000
        }
    }
    
    # Task type patterns for automatic classification
    TASK_PATTERNS = {
        'simple': [
            r'/b(quick|simple|basic|brief|short)/b',
            r'/b(summarize|list|count|show|display)/b',
            r'/b(yes|no|true|false)/b',
            r'/b(what is|who is|when is|where is)/b'
        ],
        'complex': [
            r'/b(analyze|evaluate|compare|assess|critique)/b',
            r'/b(design|architect|plan|strategy)/b',
            r'/b(research|investigate|explore)/b',
            r'/b(complex|detailed|comprehensive|in-depth)/b',
            r'/b(optimization|algorithm|machine learning|ai)/b',
            r'/b(multi-step|multi-part|elaborate)/b'
        ],
        'creative': [
            r'/b(create|generate|write|compose|design)/b',
            r'/b(creative|innovative|original|unique)/b',
            r'/b(story|narrative|fiction|poem|article)/b',
            r'/b(brainstorm|ideate|imagine)/b'
        ],
        'code': [
            r'/b(code|program|script|function|class)/b',
            r'/b(python|javascript|java|c/+/+|sql|html|css)/b',
            r'/b(debug|fix|optimize|refactor)/b',
            r'/b(api|database|frontend|backend)/b'
        ]
    }
    
    def __init__(self, default_model: ClaudeModel = ClaudeModel.SONNET):
        self.default_model = default_model
        self.usage_stats = {}  # Track model usage for optimization
        
    def analyze_task(self, prompt: str, context: Dict[str, Any] = None) -> TaskRequirements:
        """Analyze a task prompt to determine requirements"""
        prompt_lower = prompt.lower()
        context = context or {}
        
        # Estimate token counts
        input_length = len(prompt.split()) * 1.3  # Rough token estimation
        estimated_output = self._estimate_output_length(prompt)
        
        # Analyze complexity
        complexity = self._analyze_complexity(prompt_lower)
        
        # Check for creativity needs
        creativity_needed = any(
            re.search(pattern, prompt_lower) 
            for pattern in self.TASK_PATTERNS['creative']
        )
        
        # Check for domain expertise needs
        domain_expertise = self._needs_domain_expertise(prompt_lower, context)
        
        # Analyze reasoning depth needed
        reasoning_depth = self._analyze_reasoning_depth(prompt_lower)
        
        # Determine quality requirements
        quality_needed = context.get('quality_level', 'standard')
        
        # Cost sensitivity from context
        cost_sensitivity = context.get('cost_sensitivity', 'medium')
        
        return TaskRequirements(
            complexity=complexity,
            input_length=int(input_length),
            output_length=estimated_output,
            quality_needed=quality_needed,
            domain_expertise=domain_expertise,
            reasoning_depth=reasoning_depth,
            creativity_needed=creativity_needed,
            cost_sensitivity=cost_sensitivity
        )
    
    def select_model(self, prompt: str, context: Dict[str, Any] = None) -> Tuple[ClaudeModel, str]:
        """Select the optimal Claude model for a given task"""
        requirements = self.analyze_task(prompt, context)
        
        # Calculate scores for each model
        scores = {}
        for model in ClaudeModel:
            score = self._calculate_model_score(model, requirements)
            scores[model] = score
        
        # Select the highest scoring model
        best_model = max(scores, key=scores.get)
        
        # Generate reasoning for the selection
        reasoning = self._generate_selection_reasoning(best_model, requirements, scores)
        
        # Log the selection for optimization
        self._log_selection(best_model, requirements, reasoning)
        
        return best_model, reasoning
    
    def _analyze_complexity(self, prompt_lower: str) -> str:
        """Determine task complexity from prompt"""
        simple_score = sum(
            1 for pattern in self.TASK_PATTERNS['simple']
            if re.search(pattern, prompt_lower)
        )
        
        complex_score = sum(
            1 for pattern in self.TASK_PATTERNS['complex']
            if re.search(pattern, prompt_lower)
        )
        
        # Long prompts tend to be more complex
        length_factor = len(prompt_lower.split())
        if length_factor > 200:
            complex_score += 2
        elif length_factor < 50:
            simple_score += 1
        
        if complex_score > simple_score:
            return 'complex'
        elif simple_score > 0:
            return 'simple'
        else:
            return 'moderate'
    
    def _estimate_output_length(self, prompt: str) -> int:
        """Estimate expected output length in tokens"""
        # Look for length indicators in prompt
        if any(word in prompt.lower() for word in ['brief', 'short', 'summary', 'list']):
            return 100
        elif any(word in prompt.lower() for word in ['detailed', 'comprehensive', 'elaborate']):
            return 1000
        elif any(word in prompt.lower() for word in ['report', 'analysis', 'essay', 'article']):
            return 800
        else:
            return 300  # Default moderate length
    
    def _needs_domain_expertise(self, prompt_lower: str, context: Dict[str, Any]) -> bool:
        """Check if task requires specialized domain knowledge"""
        domain_keywords = [
            'medical', 'legal', 'financial', 'scientific', 'academic',
            'technical', 'engineering', 'research', 'analysis',
            'space', 'aerospace', 'manufacturing', 'orbital'
        ]
        
        # Check prompt for domain keywords
        domain_needed = any(keyword in prompt_lower for keyword in domain_keywords)
        
        # Check context for domain indicators
        if context.get('domain') or context.get('specialized_knowledge'):
            domain_needed = True
        
        return domain_needed
    
    def _analyze_reasoning_depth(self, prompt_lower: str) -> str:
        """Determine the depth of reasoning required"""
        deep_indicators = ['analyze', 'evaluate', 'compare', 'critique', 'assess', 
                          'explain why', 'reasoning', 'logic', 'because', 'therefore']
        
        shallow_indicators = ['list', 'show', 'display', 'what is', 'who is']
        
        deep_score = sum(1 for indicator in deep_indicators if indicator in prompt_lower)
        shallow_score = sum(1 for indicator in shallow_indicators if indicator in prompt_lower)
        
        if deep_score >= 2:
            return 'deep'
        elif shallow_score > deep_score:
            return 'shallow'
        else:
            return 'moderate'
    
    def _calculate_model_score(self, model: ClaudeModel, req: TaskRequirements) -> float:
        """Calculate a score for how well a model fits the requirements"""
        capabilities = self.MODEL_CAPABILITIES[model]
        score = 0.0
        
        # Base capability scoring
        if req.complexity == 'simple':
            score += capabilities['speed'] * 0.3
            score += capabilities['cost_efficiency'] * 0.4
        elif req.complexity == 'complex':
            score += capabilities['reasoning'] * 0.3
            score += capabilities['complex_analysis'] * 0.3
        else:  # moderate
            score += capabilities['reasoning'] * 0.2
            score += capabilities['cost_efficiency'] * 0.2
        
        # Quality requirements
        if req.quality_needed == 'premium':
            score += capabilities['reasoning'] * 0.2
            score += capabilities['domain_knowledge'] * 0.2
        elif req.quality_needed == 'basic':
            score += capabilities['cost_efficiency'] * 0.3
        
        # Domain expertise
        if req.domain_expertise:
            score += capabilities['domain_knowledge'] * 0.2
        
        # Creativity needs
        if req.creativity_needed:
            score += capabilities['creativity'] * 0.2
        
        # Reasoning depth
        if req.reasoning_depth == 'deep':
            score += capabilities['reasoning'] * 0.3
        elif req.reasoning_depth == 'shallow':
            score += capabilities['speed'] * 0.2
            score += capabilities['cost_efficiency'] * 0.2
        
        # Cost sensitivity adjustment
        if req.cost_sensitivity == 'high':
            score += capabilities['cost_efficiency'] * 0.4
        elif req.cost_sensitivity == 'low':
            score -= capabilities['cost_efficiency'] * 0.2  # De-prioritize cost
        
        # Token length penalties for inappropriate models
        total_tokens = req.input_length + req.output_length
        if total_tokens > capabilities['max_recommended_tokens']:
            score *= 0.7  # Penalty for exceeding recommended tokens
        
        # Pro subscription considerations for Opus usage
        if model == ClaudeModel.OPUS:
            if self.has_pro_subscription:
                # Pro users get free Opus tokens, so less penalty for usage
                estimated_tokens = req.input_length + req.output_length
                if self.opus_tokens_used_today + estimated_tokens <= self.opus_free_limit_per_period:
                    # Within free tier - slight penalty only for simple tasks
                    if req.complexity == 'simple':
                        score *= 0.7  # Lighter penalty since it's free
                    else:
                        score *= 1.1  # Small boost for complex tasks within free tier
                else:
                    # Would exceed free tier
                    if req.complexity == 'simple':
                        score *= 0.2  # Heavy penalty for paid simple tasks
                    elif req.quality_needed != 'premium':
                        score *= 0.6  # Moderate penalty for non-premium tasks
            else:
                # Non-pro users - original heavy penalty for simple tasks
                if req.complexity == 'simple':
                    score *= 0.3
        
        # Boost Sonnet as the balanced default choice
        if model == ClaudeModel.SONNET and req.complexity == 'moderate':
            score *= 1.1  # Small boost for balanced choice
        
        return score
    
    def _generate_selection_reasoning(self, model: ClaudeModel, req: TaskRequirements, 
                                    scores: Dict[ClaudeModel, float]) -> str:
        """Generate human-readable reasoning for model selection"""
        reasons = []
        estimated_tokens = req.input_length + req.output_length
        
        # Primary reason based on complexity and Pro subscription status
        if req.complexity == 'simple' and model == ClaudeModel.HAIKU:
            reasons.append("Haiku selected for simple task - optimal speed and efficiency")
        elif req.complexity == 'complex' and model == ClaudeModel.OPUS:
            if self.has_pro_subscription and (self.opus_tokens_used_today + estimated_tokens <= self.opus_free_limit_per_period):
                reasons.append("Opus selected for complex task - within Pro free tier")
            else:
                reasons.append("Opus selected for complex task requiring advanced reasoning")
        elif model == ClaudeModel.SONNET:
            reasons.append("Sonnet selected as balanced choice - optimal quality/cost ratio")
        
        # Pro subscription context
        if model == ClaudeModel.OPUS and self.has_pro_subscription:
            remaining_free = self.opus_free_limit_per_period - self.opus_tokens_used_today
            if estimated_tokens <= remaining_free:
                reasons.append(f"Pro free tier: {remaining_free - estimated_tokens:,} tokens remaining after this request")
            else:
                reasons.append("would exceed Pro free tier - consider for premium tasks only")
        
        # Additional factors
        if req.domain_expertise and model != ClaudeModel.HAIKU:
            reasons.append("requires domain expertise")
        
        if req.creativity_needed and model in [ClaudeModel.SONNET, ClaudeModel.OPUS]:
            reasons.append("creative output needed")
        
        if req.cost_sensitivity == 'high' and model == ClaudeModel.HAIKU:
            reasons.append("cost optimization prioritized")
        
        if req.reasoning_depth == 'deep' and model != ClaudeModel.HAIKU:
            reasons.append("deep reasoning required")
        
        # Cost impact
        estimated_cost = self._estimate_cost(model, req.input_length + req.output_length)
        reasons.append(f"estimated cost: ${estimated_cost:.6f}")
        
        return f"{model.value}: {', '.join(reasons)}"
    
    def _estimate_cost(self, model: ClaudeModel, total_tokens: int) -> float:
        """Estimate cost for model and token count"""
        # Using pricing from usage_tracking_service.py
        pricing = {
            ClaudeModel.HAIKU: {'input': 0.00000025, 'output': 0.00000125},
            ClaudeModel.SONNET: {'input': 0.000003, 'output': 0.000015},
            ClaudeModel.OPUS: {'input': 0.000015, 'output': 0.000075}
        }
        
        # Assume 70% input, 30% output split
        input_tokens = int(total_tokens * 0.7)
        output_tokens = int(total_tokens * 0.3)
        
        model_pricing = pricing[model]
        return (input_tokens * model_pricing['input']) + (output_tokens * model_pricing['output'])
    
    def _log_selection(self, model: ClaudeModel, req: TaskRequirements, reasoning: str):
        """Log model selection for analytics and optimization"""
        try:
            # Update usage stats
            if model.value not in self.usage_stats:
                self.usage_stats[model.value] = {
                    'count': 0,
                    'total_tokens': 0,
                    'complexity_breakdown': {'simple': 0, 'moderate': 0, 'complex': 0}
                }
            
            stats = self.usage_stats[model.value]
            stats['count'] += 1
            stats['total_tokens'] += req.input_length + req.output_length
            stats['complexity_breakdown'][req.complexity] += 1
            
            logger.debug(f"Model selected: {reasoning}")
            
        except Exception as e:
            logger.warning(f"Failed to log model selection: {e}")
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get model usage statistics"""
        total_requests = sum(stats['count'] for stats in self.usage_stats.values())
        
        if total_requests == 0:
            return {"message": "No model selections recorded yet"}
        
        # Calculate efficiency metrics
        efficiency_report = {}
        for model_name, stats in self.usage_stats.items():
            percentage = (stats['count'] / total_requests) * 100
            avg_tokens = stats['total_tokens'] / max(stats['count'], 1)
            
            efficiency_report[model_name] = {
                'usage_percentage': round(percentage, 1),
                'total_requests': stats['count'],
                'average_tokens_per_request': round(avg_tokens),
                'complexity_breakdown': stats['complexity_breakdown']
            }
        
        return {
            'total_requests': total_requests,
            'model_distribution': efficiency_report,
            'recommendations': self._generate_usage_recommendations()
        }
    
    def _generate_usage_recommendations(self) -> List[str]:
        """Generate recommendations based on usage patterns"""
        recommendations = []
        total_requests = sum(stats['count'] for stats in self.usage_stats.values())
        
        if total_requests < 10:
            return ["Collect more usage data for meaningful recommendations"]
        
        # Check for potential over-use of expensive models
        opus_percentage = (self.usage_stats.get(ClaudeModel.OPUS.value, {}).get('count', 0) / total_requests) * 100
        if opus_percentage > 20:
            recommendations.append("Consider reducing Opus usage - high percentage detected")
        
        # Check for under-utilization of cost-effective models
        haiku_percentage = (self.usage_stats.get(ClaudeModel.HAIKU.value, {}).get('count', 0) / total_requests) * 100
        if haiku_percentage < 30:
            recommendations.append("Consider using Haiku for more simple tasks")
        
        return recommendations
    
    def track_opus_usage(self, tokens_used: int):
        """Track Opus token usage for Pro tier management"""
        if self.has_pro_subscription:
            self.opus_tokens_used_today += tokens_used
            logger.debug(f"Opus tokens used: {tokens_used:,}, daily total: {self.opus_tokens_used_today:,}")
    
    def reset_daily_opus_counter(self):
        """Reset daily Opus counter (call when free tier refreshes)"""
        previous_usage = self.opus_tokens_used_today
        self.opus_tokens_used_today = 0
        logger.info(f"Opus daily counter reset. Previous usage: {previous_usage:,} tokens")
    
    def get_opus_usage_status(self) -> Dict[str, Any]:
        """Get current Opus usage status for Pro users"""
        if not self.has_pro_subscription:
            return {"message": "Not a Pro subscriber - no free Opus tier"}
        
        remaining_free = self.opus_free_limit_per_period - self.opus_tokens_used_today
        usage_percentage = (self.opus_tokens_used_today / self.opus_free_limit_per_period) * 100
        
        return {
            "tokens_used_today": self.opus_tokens_used_today,
            "free_tier_limit": self.opus_free_limit_per_period,
            "remaining_free_tokens": max(0, remaining_free),
            "usage_percentage": round(usage_percentage, 1),
            "status": "within_free_tier" if remaining_free > 0 else "exceeded_free_tier",
            "recommendation": self._get_opus_usage_recommendation(remaining_free)
        }
    
    def _get_opus_usage_recommendation(self, remaining_free: int) -> str:
        """Get usage recommendation based on remaining free tokens"""
        if remaining_free > 30000:
            return "Plenty of free Opus tokens available - use freely for complex tasks"
        elif remaining_free > 10000:
            return "Moderate free tokens remaining - reserve for important complex tasks"
        elif remaining_free > 0:
            return "Low free tokens remaining - save for critical high-value tasks only"
        else:
            return "Free tier exhausted - Opus usage will incur charges"


# Global instance with Pro subscription enabled
model_selector = ClaudeModelSelector(has_pro_subscription=True)


def select_claude_model(prompt: str, context: Dict[str, Any] = None) -> Tuple[str, str]:
    """Convenient function to select Claude model"""
    model, reasoning = model_selector.select_model(prompt, context)
    return model.value, reasoning


# Decorator for automatic model selection and tracking
def auto_select_model(context: Dict[str, Any] = None):
    """Decorator to automatically select and track Claude model usage"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract prompt from arguments
            prompt = ""
            if args and isinstance(args[0], str):
                prompt = args[0]
            elif 'prompt' in kwargs:
                prompt = kwargs['prompt']
            elif 'text' in kwargs:
                prompt = kwargs['text']
            
            if prompt:
                model, reasoning = model_selector.select_model(prompt, context)
                kwargs['model'] = model.value
                logger.info(f"Auto-selected model: {reasoning}")
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator