"""
Error Categorization and Automatic Retry Logic
"""
import asyncio
import logging
from enum import Enum
from typing import Any, Callable, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

class ErrorCategory(Enum):
    NETWORK = "network"
    DATABASE = "database" 
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    SYSTEM = "system"
    UNKNOWN = "unknown"

@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    
class ErrorCategorizer:
    """Categorize errors and determine retry strategies"""
    
    def __init__(self):
        self.error_patterns = {
            ErrorCategory.NETWORK: [
                "connection refused", "timeout", "network unreachable",
                "connection reset", "host unreachable"
            ],
            ErrorCategory.DATABASE: [
                "database not initialized", "connection lost", "table does not exist",
                "database locked", "constraint violation"
            ],
            ErrorCategory.AUTHENTICATION: [
                "credentials not found", "invalid token", "authentication failed",
                "permission denied", "unauthorized"
            ],
            ErrorCategory.VALIDATION: [
                "invalid input", "validation error", "missing required field",
                "format error", "type error"
            ],
            ErrorCategory.SYSTEM: [
                "out of memory", "disk full", "resource unavailable",
                "system overload", "internal server error"
            ]
        }
        
        self.retry_configs = {
            ErrorCategory.NETWORK: RetryConfig(max_attempts=3, base_delay=1.0),
            ErrorCategory.DATABASE: RetryConfig(max_attempts=2, base_delay=2.0),
            ErrorCategory.AUTHENTICATION: RetryConfig(max_attempts=1, base_delay=0.5),
            ErrorCategory.VALIDATION: RetryConfig(max_attempts=1, base_delay=0.0),
            ErrorCategory.SYSTEM: RetryConfig(max_attempts=3, base_delay=5.0),
            ErrorCategory.UNKNOWN: RetryConfig(max_attempts=2, base_delay=1.0)
        }
    
    def categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize an error based on its message and type"""
        error_text = str(error).lower()
        
        for category, patterns in self.error_patterns.items():
            if any(pattern in error_text for pattern in patterns):
                return category
                
        return ErrorCategory.UNKNOWN
    
    def get_retry_config(self, category: ErrorCategory) -> RetryConfig:
        """Get retry configuration for error category"""
        return self.retry_configs.get(category, self.retry_configs[ErrorCategory.UNKNOWN])

async def retry_with_categorization(
    func: Callable, 
    *args, 
    categorizer: Optional[ErrorCategorizer] = None,
    **kwargs
) -> Any:
    """Execute function with intelligent retry based on error categorization"""
    
    if categorizer is None:
        categorizer = ErrorCategorizer()
    
    last_error = None
    
    for attempt in range(1, 10):  # Max 10 attempts across all categories
        try:
            return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
        except Exception as e:
            last_error = e
            category = categorizer.categorize_error(e)
            config = categorizer.get_retry_config(category)
            
            if attempt >= config.max_attempts:
                logger.error(f"Max retries exceeded for {category.value} error: {e}")
                break
                
            delay = min(
                config.base_delay * (config.exponential_base ** (attempt - 1)),
                config.max_delay
            )
            
            logger.warning(f"Attempt {attempt} failed ({category.value}), retrying in {delay}s: {e}")
            await asyncio.sleep(delay) if asyncio.iscoroutinefunction(func) else time.sleep(delay)
    
    raise last_error

# Global categorizer instance
error_categorizer = ErrorCategorizer()
