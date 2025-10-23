"""
Enhanced Automated Error Pattern Learning and Classification
Extension to the existing ML Error Learning System
"""
import asyncio
import logging
import re
from typing import Dict, List, Any, Set
from datetime import datetime, timezone
from collections import defaultdict, Counter
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class ErrorPatternLearner:
    """Learn and classify error patterns automatically"""
    
    def __init__(self, patterns_file: str = "learned_error_patterns.json"):
        self.patterns_file = Path(patterns_file)
        self.learned_patterns: Dict[str, List[str]] = defaultdict(list)
        self.error_frequency: Counter = Counter()
        self.pattern_confidence: Dict[str, float] = {}
        self.load_patterns()
    
    def load_patterns(self):
        """Load previously learned patterns"""
        if self.patterns_file.exists():
            try:
                with open(self.patterns_file, 'r') as f:
                    data = json.load(f)
                    self.learned_patterns.update(data.get("patterns", {}))
                    self.error_frequency.update(data.get("frequency", {}))
                    self.pattern_confidence.update(data.get("confidence", {}))
            except Exception as e:
                logger.warning(f"Failed to load error patterns: {e}")
    
    def save_patterns(self):
        """Save learned patterns"""
        try:
            data = {
                "patterns": dict(self.learned_patterns),
                "frequency": dict(self.error_frequency),
                "confidence": self.pattern_confidence,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            with open(self.patterns_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save error patterns: {e}")
    
    def extract_patterns(self, error_message: str) -> List[str]:
        """Extract patterns from error message"""
        patterns = []
        
        # Common error pattern templates
        pattern_templates = [
            r"connection /w+ to /w+",  # Connection errors
            r"/w+ not found",  # Not found errors
            r"cannot import /w+ from /w+",  # Import errors
            r"/w+ is not /w+",  # Type/state errors
            r"/w+ already exists",  # Conflict errors
            r"permission denied /w*",  # Permission errors
            r"timeout /w*",  # Timeout errors
            r"invalid /w+",  # Validation errors
        ]
        
        error_lower = error_message.lower()
        
        for template in pattern_templates:
            matches = re.findall(template, error_lower)
            patterns.extend(matches)
        
        # Extract key phrases (3-5 word sequences)
        words = re.findall(r'/b/w+/b', error_lower)
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            if len(phrase) > 8:  # Skip very short phrases
                patterns.append(phrase)
        
        return list(set(patterns))  # Remove duplicates
    
    def learn_error_pattern(self, error_message: str, error_category: str = "unknown"):
        """Learn new error patterns from error message"""
        patterns = self.extract_patterns(error_message)
        
        for pattern in patterns:
            self.learned_patterns[error_category].append(pattern)
            self.error_frequency[pattern] += 1
            
            # Calculate confidence based on frequency
            self.pattern_confidence[pattern] = min(
                self.error_frequency[pattern] / 10.0,  # Max confidence at 10 occurrences
                1.0
            )
        
        self.save_patterns()
    
    def classify_error(self, error_message: str) -> Dict[str, Any]:
        """Classify error based on learned patterns"""
        error_lower = error_message.lower()
        
        category_scores = defaultdict(float)
        matched_patterns = []
        
        for category, patterns in self.learned_patterns.items():
            for pattern in patterns:
                if pattern in error_lower:
                    confidence = self.pattern_confidence.get(pattern, 0.1)
                    frequency_weight = min(self.error_frequency[pattern] / 100.0, 1.0)
                    score = confidence * frequency_weight
                    
                    category_scores[category] += score
                    matched_patterns.append({
                        "pattern": pattern,
                        "category": category,
                        "confidence": confidence,
                        "frequency": self.error_frequency[pattern]
                    })
        
        if not category_scores:
            return {
                "category": "unknown", 
                "confidence": 0.0,
                "matched_patterns": [],
                "suggestion": "New error pattern - will be learned"
            }
        
        best_category = max(category_scores.items(), key=lambda x: x[1])
        
        return {
            "category": best_category[0],
            "confidence": best_category[1],
            "matched_patterns": matched_patterns,
            "all_scores": dict(category_scores),
            "suggestion": self._generate_suggestion(best_category[0], best_category[1])
        }
    
    def _generate_suggestion(self, category: str, confidence: float) -> str:
        """Generate suggestion based on category and confidence"""
        suggestions = {
            "connection": "Check network connectivity and service availability",
            "database": "Verify database connections and initialization",
            "authentication": "Review credentials and permissions",
            "import": "Check package installations and dependencies",
            "validation": "Validate input data and parameters",
            "timeout": "Increase timeout values or check system load",
            "permission": "Check file/directory permissions",
            "unknown": "Monitor this error pattern for classification"
        }
        
        base_suggestion = suggestions.get(category, suggestions["unknown"])
        
        if confidence > 0.7:
            return f"High confidence: {base_suggestion}"
        elif confidence > 0.3:
            return f"Medium confidence: {base_suggestion}"
        else:
            return f"Low confidence: {base_suggestion}. Pattern needs more data."
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about learned patterns"""
        return {
            "total_patterns": sum(len(patterns) for patterns in self.learned_patterns.values()),
            "categories": len(self.learned_patterns),
            "most_frequent_errors": dict(self.error_frequency.most_common(10)),
            "high_confidence_patterns": {
                pattern: confidence for pattern, confidence in self.pattern_confidence.items()
                if confidence > 0.7
            }
        }

# Global error pattern learner
error_pattern_learner = ErrorPatternLearner()
