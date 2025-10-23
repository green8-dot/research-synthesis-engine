"""
Automation Ideas API Router
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from loguru import logger
from datetime import datetime
import asyncio

try:
    from research_synthesis.services.automation_ideas import discover_automation_ideas, AutomationIdea
    AUTOMATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Automation ideas service not available: {e}")
    AUTOMATION_AVAILABLE = False

# Database imports for persistence
try:
    from research_synthesis.database.connection import get_postgres_session
    from research_synthesis.database.models import AutomationIdea as AutomationIdeaModel
    from sqlalchemy import select, desc, func
    DATABASE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Database not available for automation ideas: {e}")
    DATABASE_AVAILABLE = False

router = APIRouter()

@router.get("/")
async def get_automation_ideas_status():
    """Get automation ideas service status"""
    return {
        "service": "Automation Ideas Discovery API",
        "status": "active" if AUTOMATION_AVAILABLE else "unavailable",
        "description": "Discovers productivity automation ideas from multiple sources",
        "endpoints": [
            "/discover - Discover new automation ideas",
            "/discover/{limit} - Discover ideas with custom limit per source",
            "/categories - Get automation categories",
            "/sources - Get source rankings and statistics"
        ],
        "sources": [
            "Reddit r/Automate",
            "Reddit r/productivity", 
            "Reddit r/selfhosted",
            "Reddit r/sysadmin",
            "Reddit r/DevOps",
            "Reddit r/Python",
            "Reddit r/PowerShell",
            "Reddit r/bash",
            "Reddit r/homeautomation",
            "Reddit r/homelab",
            "Reddit r/HomeAssistant",
            "Reddit r/workflow",
            "Reddit r/IFTTT",
            "Reddit r/zapier",
            "Reddit r/tasker",
            "Reddit r/shortcuts",
            "Reddit r/MachineLearning",
            "Reddit r/datascience",
            "Reddit r/technology",
            "Reddit r/software"
        ]
    }

@router.post("/discover")
async def discover_automation_ideas_endpoint(
    limit_per_source: int = Query(25, description="Number of ideas to fetch per source", ge=5, le=100),
    persist: bool = Query(True, description="Whether to store discovered ideas in database")
):
    """Discover new automation ideas from all sources"""
    if not AUTOMATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Automation ideas service unavailable")
    
    try:
        logger.info(f"Starting automation ideas discovery (limit: {limit_per_source} per source)...")
        
        try:
            ideas, source_rankings = await asyncio.wait_for(
                discover_automation_ideas(limit_per_source), 
                timeout=120  # 2 minute timeout to allow full discovery
            )
        except asyncio.TimeoutError:
            logger.warning("Automation discovery timed out, returning sample ideas")
            return await _get_fallback_automation_ideas()
        
        # Convert ideas to serializable format
        ideas_data = []
        for idea in ideas:
            ideas_data.append({
                "title": idea.title,
                "description": idea.description[:200] + "..." if len(idea.description) > 200 else idea.description,
                "source": idea.source,
                "url": idea.url,
                "category": idea.category,
                "difficulty": idea.difficulty,
                "productivity_estimate": idea.productivity_estimate,
                "implementation_time_days": idea.implementation_time_days,
                "priority_score": idea.priority_score,
                "tags": idea.tags,
                "upvotes": idea.upvotes,
                "engagement_score": idea.engagement_score,
                "discovered_at": idea.discovered_at.isoformat()
            })
        
        # Format source rankings
        source_rankings_data = {}
        for source_name, stats in source_rankings['rankings']:
            source_rankings_data[source_name] = {
                "rank": len(source_rankings_data) + 1,
                "quality_score": round(stats['quality_score'], 2),
                "total_ideas": stats['total_ideas'],
                "avg_productivity": stats['avg_productivity'],
                "avg_priority_score": stats['avg_priority_score'],
                "avg_engagement": stats['avg_engagement'],
                "top_categories": sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True)[:3],
                "difficulty_distribution": stats['difficulties']
            }
        
        # Store ideas in database if persist is enabled
        stored_count = 0
        if persist and ideas_data:
            stored_count = await store_automation_ideas_db(ideas_data)
            logger.info(f"Stored {stored_count} automation ideas in database")
        
        return {
            "status": "success",
            "discovery_summary": {
                "total_ideas_found": len(ideas_data),
                "sources_scanned": len(source_rankings_data),
                "avg_productivity_estimate": source_rankings['summary']['avg_productivity_all'],
                "best_source": source_rankings['summary']['best_source'],
                "discovery_timestamp": datetime.now().isoformat(),
                "ideas_persisted": persist,
                "stored_count": stored_count
            },
            "top_ideas": ideas_data[:20],  # Top 20 ideas by priority
            "all_ideas": ideas_data,
            "source_rankings": source_rankings_data,
            "categories_summary": _get_categories_summary(ideas_data),
            "recommendations": _generate_implementation_recommendations(ideas_data)
        }
        
    except Exception as e:
        logger.error(f"Automation ideas discovery failed: {e}")
        raise HTTPException(status_code=500, detail=f"Discovery failed: {str(e)}")

@router.post("/recommend-sources")
async def recommend_automation_sources(
    user_interests: List[str] = Query([], description="User's areas of interest"),
    skill_level: str = Query("intermediate", description="User's technical skill level"),
    time_available: int = Query(10, description="Hours per week available for automation")
):
    """Recommend best automation sources based on user profile"""
    
    # Base source recommendations with metadata
    all_sources = {
        "Reddit r/Automate": {
            "url": "https://reddit.com/r/automate",
            "description": "General automation discussions and ideas",
            "skill_level": ["beginner", "intermediate", "advanced"],
            "categories": ["workflow", "data", "general"],
            "activity": "high",
            "quality_score": 0.85
        },
        "Reddit r/selfhosted": {
            "url": "https://reddit.com/r/selfhosted",
            "description": "Self-hosted solutions and home automation",
            "skill_level": ["intermediate", "advanced"],
            "categories": ["infrastructure", "monitoring", "integration"],
            "activity": "very high",
            "quality_score": 0.90
        },
        "Reddit r/DevOps": {
            "url": "https://reddit.com/r/devops",
            "description": "DevOps automation and CI/CD pipelines",
            "skill_level": ["intermediate", "advanced"],
            "categories": ["deployment", "monitoring", "infrastructure"],
            "activity": "high",
            "quality_score": 0.88
        },
        "Reddit r/Python": {
            "url": "https://reddit.com/r/python",
            "description": "Python automation scripts and tools",
            "skill_level": ["beginner", "intermediate", "advanced"],
            "categories": ["data", "scripting", "testing"],
            "activity": "very high",
            "quality_score": 0.87
        },
        "Reddit r/PowerShell": {
            "url": "https://reddit.com/r/powershell",
            "description": "Windows automation with PowerShell",
            "skill_level": ["intermediate", "advanced"],
            "categories": ["windows", "scripting", "system"],
            "activity": "medium",
            "quality_score": 0.82
        },
        "Reddit r/homeautomation": {
            "url": "https://reddit.com/r/homeautomation",
            "description": "Home and IoT automation projects",
            "skill_level": ["beginner", "intermediate"],
            "categories": ["iot", "home", "integration"],
            "activity": "high",
            "quality_score": 0.80
        },
        "Reddit r/productivity": {
            "url": "https://reddit.com/r/productivity",
            "description": "Personal productivity and workflow automation",
            "skill_level": ["beginner", "intermediate"],
            "categories": ["workflow", "personal", "productivity"],
            "activity": "high",
            "quality_score": 0.75
        },
        "Reddit r/sysadmin": {
            "url": "https://reddit.com/r/sysadmin",
            "description": "System administration automation",
            "skill_level": ["intermediate", "advanced"],
            "categories": ["infrastructure", "security", "monitoring"],
            "activity": "very high",
            "quality_score": 0.86
        },
        "Reddit r/DataEngineering": {
            "url": "https://reddit.com/r/dataengineering",
            "description": "Data pipeline and ETL automation",
            "skill_level": ["intermediate", "advanced"],
            "categories": ["data", "etl", "analytics"],
            "activity": "high",
            "quality_score": 0.89
        },
        "Reddit r/MachineLearning": {
            "url": "https://reddit.com/r/machinelearning",
            "description": "ML model automation and MLOps",
            "skill_level": ["advanced"],
            "categories": ["ml", "data", "analytics"],
            "activity": "very high",
            "quality_score": 0.91
        }
    }
    
    # Score sources based on user profile
    recommendations = []
    
    for source_name, source_info in all_sources.items():
        score = 0.0
        reasons = []
        
        # Check skill level match
        if skill_level in source_info["skill_level"]:
            score += 0.3
            reasons.append(f"Matches your {skill_level} skill level")
        
        # Check interest overlap
        interest_overlap = set(user_interests) & set(source_info["categories"])
        if interest_overlap:
            score += 0.4 * (len(interest_overlap) / len(user_interests) if user_interests else 0.5)
            reasons.append(f"Covers: {', '.join(interest_overlap)}")
        
        # Factor in source quality
        score += source_info["quality_score"] * 0.2
        
        # Factor in activity level
        activity_scores = {"very high": 0.1, "high": 0.08, "medium": 0.05}
        score += activity_scores.get(source_info["activity"], 0.03)
        
        # Time-based recommendation
        if time_available < 5 and skill_level == "beginner":
            if "beginner" in source_info["skill_level"]:
                score += 0.05
                reasons.append("Good for limited time commitment")
        elif time_available > 15 and skill_level == "advanced":
            if "advanced" in source_info["skill_level"]:
                score += 0.05
                reasons.append("Rich content for deep exploration")
        
        recommendations.append({
            "source": source_name,
            "url": source_info["url"],
            "description": source_info["description"],
            "score": round(score, 2),
            "reasons": reasons,
            "categories": source_info["categories"],
            "activity": source_info["activity"]
        })
    
    # Sort by score
    recommendations.sort(key=lambda x: x["score"], reverse=True)
    
    # Generate personalized advice
    advice = []
    if skill_level == "beginner":
        advice.append("Start with general automation communities like r/Automate and r/productivity")
        advice.append("Focus on simple workflow automation before moving to complex DevOps")
    elif skill_level == "intermediate":
        advice.append("Explore specialized communities like r/selfhosted and r/DevOps")
        advice.append("Consider contributing your own automation solutions")
    else:  # advanced
        advice.append("Engage with technical communities like r/DataEngineering and r/MachineLearning")
        advice.append("Share your expertise and learn from complex implementations")
    
    if time_available < 5:
        advice.append("Focus on quick-win automations with immediate impact")
    elif time_available > 15:
        advice.append("Consider tackling comprehensive automation projects")
    
    return {
        "recommendations": recommendations[:5],  # Top 5 sources
        "all_sources": recommendations,
        "personalized_advice": advice,
        "user_profile": {
            "interests": user_interests,
            "skill_level": skill_level,
            "time_available_hours": time_available
        },
        "suggestion": f"Based on your profile, start with {recommendations[0]['source'] if recommendations else 'r/Automate'}"
    }


@router.get("/categories")
async def get_automation_categories():
    """Get available automation categories with descriptions"""
    categories = {
        "Data Processing": {
            "description": "Automate data collection, transformation, and migration tasks",
            "examples": ["ETL pipelines", "Data cleaning", "File processing", "Database migrations"],
            "typical_productivity": "15-25%",
            "common_tools": ["Python", "SQL", "Pandas", "Airflow"]
        },
        "Workflow Automation": {
            "description": "Streamline business processes and task sequences", 
            "examples": ["Approval workflows", "Task orchestration", "Process chains"],
            "typical_productivity": "20-35%",
            "common_tools": ["Zapier", "n8n", "Microsoft Power Automate", "IFTTT"]
        },
        "Monitoring & Alerts": {
            "description": "Automated monitoring and notification systems",
            "examples": ["System health checks", "Performance monitoring", "Error alerting"],
            "typical_productivity": "10-20%", 
            "common_tools": ["Prometheus", "Grafana", "Nagios", "PagerDuty"]
        },
        "Deployment & DevOps": {
            "description": "Automate software deployment and infrastructure management",
            "examples": ["CI/CD pipelines", "Infrastructure as Code", "Auto-scaling"],
            "typical_productivity": "25-40%",
            "common_tools": ["Docker", "Kubernetes", "Ansible", "Terraform", "Jenkins"]
        },
        "Communication & Notifications": {
            "description": "Automated messaging and communication workflows",
            "examples": ["Slack bots", "Email automation", "Status updates"],
            "typical_productivity": "5-15%",
            "common_tools": ["Slack API", "Email services", "SMS APIs", "Webhooks"]
        },
        "Analytics & Reporting": {
            "description": "Automated report generation and data visualization",
            "examples": ["Dashboard automation", "Scheduled reports", "KPI tracking"],
            "typical_productivity": "15-30%",
            "common_tools": ["Tableau", "Power BI", "Python", "R", "SQL"]
        },
        "Testing & QA": {
            "description": "Automated testing and quality assurance processes", 
            "examples": ["Unit testing", "Integration tests", "Performance testing"],
            "typical_productivity": "20-35%",
            "common_tools": ["Selenium", "Jest", "PyTest", "Postman", "JMeter"]
        },
        "Security & Compliance": {
            "description": "Automated security monitoring and compliance checking",
            "examples": ["Vulnerability scanning", "Access auditing", "Backup automation"],
            "typical_productivity": "10-25%",
            "common_tools": ["Security scanners", "Backup tools", "Compliance frameworks"]
        },
        "System Integration": {
            "description": "Connect and synchronize different systems and services",
            "examples": ["API integrations", "Data synchronization", "Third-party connections"],
            "typical_productivity": "15-25%", 
            "common_tools": ["REST APIs", "GraphQL", "Message queues", "ESB"]
        },
        "General Automation": {
            "description": "Miscellaneous automation tasks and productivity improvements",
            "examples": ["File organization", "Routine tasks", "Personal productivity"],
            "typical_productivity": "5-20%",
            "common_tools": ["Scripts", "Cron jobs", "Task schedulers", "Macros"]
        }
    }
    
    return {
        "automation_categories": categories,
        "total_categories": len(categories),
        "selection_guidance": {
            "high_impact": ["Deployment & DevOps", "Workflow Automation", "Testing & QA"],
            "quick_wins": ["General Automation", "Communication & Notifications"],
            "enterprise_focus": ["Security & Compliance", "System Integration", "Analytics & Reporting"]
        }
    }

@router.post("/expand-sources")
async def expand_automation_sources():
    """Expand automation sources by discovering related subreddits"""
    try:
        from research_synthesis.services.automation_ideas import AutomationIdeasDiscovery
        
        logger.info("Expanding automation sources...")
        async with AutomationIdeasDiscovery() as discovery:
            await discovery.expand_automation_sources()
            
            # Get the expanded sources
            expanded_sources = []
            for name, config in discovery.automation_sources.items():
                expanded_sources.append({
                    "name": name,
                    "url": config["url"],
                    "weight": config["weight"],
                    "discovered": config.get("discovered", False)
                })
                
            discovered_count = len([s for s in expanded_sources if s["discovered"]])
            
            return {
                "status": "success",
                "total_sources": len(expanded_sources),
                "discovered_sources": discovered_count,
                "original_sources": len(expanded_sources) - discovered_count,
                "sources": expanded_sources,
                "message": f"Expanded to {len(expanded_sources)} total sources ({discovered_count} newly discovered)"
            }
            
    except Exception as e:
        logger.error(f"Failed to expand automation sources: {e}")
        raise HTTPException(status_code=500, detail=f"Source expansion failed: {str(e)}")

@router.get("/sources")
async def get_source_information():
    """Get information about automation idea sources"""
    sources_info = {
        "Reddit r/Automate": {
            "description": "Primary automation community focused on personal and business automation",
            "focus_areas": ["Workflow automation", "Tool recommendations", "Success stories"],
            "typical_quality": "High",
            "member_count": "~50K",
            "update_frequency": "Daily",
            "best_for": "General automation ideas and tool discovery"
        },
        "Reddit r/productivity": {
            "description": "Productivity tips and automation for personal efficiency",
            "focus_areas": ["Personal productivity", "Time management", "Efficiency hacks"],
            "typical_quality": "Medium-High", 
            "member_count": "~200K",
            "update_frequency": "Very Active",
            "best_for": "Personal automation and productivity improvements"
        },
        "Reddit r/selfhosted": {
            "description": "Self-hosted solutions and home automation",
            "focus_areas": ["Server automation", "Home lab", "Self-hosted tools"],
            "typical_quality": "High",
            "member_count": "~150K", 
            "update_frequency": "Very Active",
            "best_for": "Infrastructure and server automation"
        },
        "Reddit r/sysadmin": {
            "description": "System administration automation and tools",
            "focus_areas": ["Enterprise automation", "System management", "IT operations"],
            "typical_quality": "Very High",
            "member_count": "~400K",
            "update_frequency": "Very Active", 
            "best_for": "Enterprise and system-level automation"
        },
        "Reddit r/DevOps": {
            "description": "DevOps practices and automation tools",
            "focus_areas": ["CI/CD", "Infrastructure as Code", "Deployment automation"],
            "typical_quality": "Very High",
            "member_count": "~100K",
            "update_frequency": "Active",
            "best_for": "Development and deployment automation"
        }
    }
    
    return {
        "sources": sources_info,
        "total_sources": len(sources_info),
        "discovery_strategy": {
            "recommended_order": ["Reddit r/sysadmin", "Reddit r/DevOps", "Reddit r/Automate", "Reddit r/selfhosted", "Reddit r/productivity"],
            "refresh_frequency": "Daily for high-activity sources, Weekly for others",
            "quality_ranking": "Based on engagement, technical depth, and implementation success rates"
        }
    }

async def store_automation_ideas_db(ideas: List[Dict]) -> int:
    """Store automation ideas in database for persistence"""
    if not DATABASE_AVAILABLE:
        logger.warning("Database not available - cannot store automation ideas")
        return 0
    
    stored_count = 0
    try:
        async for db_session in get_postgres_session():
            for idea in ideas:
                # Check if idea already exists by title
                existing = await db_session.execute(
                    select(AutomationIdeaModel).filter(AutomationIdeaModel.title == idea.get('title'))
                )
                if not existing.scalar_one_or_none():
                    # Create new automation idea
                    db_idea = AutomationIdeaModel(
                        title=idea.get('title', 'Untitled'),
                        description=idea.get('description', ''),
                        source=idea.get('source', 'Unknown'),
                        source_url=idea.get('url', ''),
                        category=idea.get('category', 'general'),
                        complexity=idea.get('difficulty', 'moderate'),
                        potential_impact=f"{idea.get('productivity_estimate', 0)}% productivity gain",
                        tools_required=[],  # Could be extracted from description
                        implementation_notes=f"Implementation time: {idea.get('implementation_time_days', 0)} days",
                        priority_score=idea.get('priority_score', 0.5),
                        feasibility_score=idea.get('engagement_score', 0.5) / 100 if idea.get('engagement_score') else 0.5,
                        votes=idea.get('upvotes', 0)
                    )
                    db_session.add(db_idea)
                    stored_count += 1
            
            await db_session.commit()
            break
    except Exception as e:
        logger.error(f"Failed to store automation ideas: {e}")
    
    return stored_count


@router.post("/ideas/{idea_id}/to-todo")
async def convert_idea_to_todo(idea_id: int):
    """Convert an automation idea to a todo item"""
    try:
        async for db_session in get_postgres_session():
            result = await db_session.execute(
                select(AutomationIdeaModel).filter(AutomationIdeaModel.id == idea_id)
            )
            idea = result.scalar_one_or_none()
            
            if not idea:
                raise HTTPException(status_code=404, detail="Automation idea not found")
            
            # Create todo item from idea
            todo_item = {
                "title": f"Implement: {idea.title}",
                "description": idea.description,
                "category": idea.category,
                "priority": idea.priority_score,
                "complexity": idea.complexity,
                "estimated_time": idea.implementation_notes,
                "source": f"Automation Idea #{idea.id}",
                "created_from_idea": True,
                "idea_id": idea.id
            }
            
            # Mark idea as being worked on
            idea.implemented = True
            idea.implementation_date = datetime.utcnow()
            await db_session.commit()
            
            return {
                "success": True,
                "message": f"Created todo from automation idea: {idea.title}",
                "todo": todo_item
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to convert idea to todo: {e}")
        raise HTTPException(status_code=500, detail="Failed to create todo")


@router.post("/ideas/batch-to-todo")
async def convert_multiple_ideas_to_todos(idea_ids: List[int]):
    """Convert multiple automation ideas to todo items"""
    todos_created = []
    
    try:
        async for db_session in get_postgres_session():
            for idea_id in idea_ids:
                result = await db_session.execute(
                    select(AutomationIdeaModel).filter(AutomationIdeaModel.id == idea_id)
                )
                idea = result.scalar_one_or_none()
                
                if idea and not idea.implemented:
                    todo_item = {
                        "title": f"Implement: {idea.title}",
                        "description": idea.description,
                        "category": idea.category,
                        "priority": idea.priority_score,
                        "complexity": idea.complexity,
                        "estimated_time": idea.implementation_notes,
                        "source": f"Automation Idea #{idea.id}",
                        "created_from_idea": True,
                        "idea_id": idea.id
                    }
                    
                    # Mark as being worked on
                    idea.implemented = True
                    idea.implementation_date = datetime.utcnow()
                    todos_created.append(todo_item)
            
            await db_session.commit()
            
            return {
                "success": True,
                "message": f"Created {len(todos_created)} todos from automation ideas",
                "todos": todos_created,
                "skipped": len(idea_ids) - len(todos_created)
            }
            
    except Exception as e:
        logger.error(f"Failed to convert ideas to todos: {e}")
        raise HTTPException(status_code=500, detail="Failed to create todos")


@router.get("/stored")
async def get_stored_automation_ideas(
    limit: int = Query(50, description="Number of ideas to retrieve"),
    category: Optional[str] = Query(None, description="Filter by category"),
    show_unimplemented_only: bool = Query(False, description="Show only unimplemented ideas")
):
    """Get stored automation ideas from database"""
    if not DATABASE_AVAILABLE:
        return {
            "ideas": [],
            "total": 0,
            "categories": {},
            "timestamp": datetime.now().isoformat(),
            "error": "Database not available"
        }
    
    try:
        async for db_session in get_postgres_session():
            # Build query with filters
            query = select(AutomationIdeaModel)
            
            if category:
                query = query.filter(AutomationIdeaModel.category.ilike(f"%{category}%"))
            
            if show_unimplemented_only:
                query = query.filter(AutomationIdeaModel.implemented == False)
            
            # Order by priority score and creation date
            query = query.order_by(desc(AutomationIdeaModel.priority_score), desc(AutomationIdeaModel.created_at))
            query = query.limit(limit)
            
            result = await db_session.execute(query)
            ideas = result.scalars().all()
            
            # Convert to JSON serializable format
            ideas_data = []
            categories = {}
            
            for idea in ideas:
                idea_data = {
                    "id": idea.id,
                    "title": idea.title,
                    "description": idea.description,
                    "source": idea.source,
                    "category": idea.category,
                    "complexity": idea.complexity,
                    "impact": idea.potential_impact,
                    "tools": idea.tools_required or [],
                    "priority": float(idea.priority_score or 0.5),
                    "implemented": idea.implemented,
                    "created_at": idea.created_at.isoformat() if idea.created_at else datetime.now().isoformat()
                }
                ideas_data.append(idea_data)
                
                # Count categories
                cat = idea.category or "General"
                categories[cat] = categories.get(cat, 0) + 1
            
            # Get total count
            total_query = select(func.count(AutomationIdeaModel.id))
            if category:
                total_query = total_query.filter(AutomationIdeaModel.category.ilike(f"%{category}%"))
            if show_unimplemented_only:
                total_query = total_query.filter(AutomationIdeaModel.implemented == False)
            
            total_result = await db_session.execute(total_query)
            total_count = total_result.scalar()
            
            return {
                "ideas": ideas_data,
                "total": total_count,
                "categories": categories,
                "timestamp": datetime.now().isoformat(),
                "note": f"Retrieved from database - {len(ideas_data)} ideas returned"
            }
            
    except Exception as e:
        logger.error(f"Failed to retrieve stored automation ideas: {e}")
        return {
            "ideas": [],
            "total": 0,
            "categories": {},
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


def _get_categories_summary(ideas: List) -> Dict[str, Any]:
    """Generate summary of ideas by category"""
    category_stats = {}
    
    for idea in ideas:
        # Handle both AutomationIdea objects and dictionaries
        if hasattr(idea, 'category'):
            category = idea.category
            productivity = idea.productivity_estimate
            priority = idea.priority_score
        else:
            category = idea.get('category', 'Unknown')
            productivity = idea.get('productivity_estimate', 0)
            priority = idea.get('priority_score', 0)
            
        if category not in category_stats:
            category_stats[category] = {
                'count': 0,
                'avg_productivity': 0,
                'avg_priority': 0,
                'total_productivity': 0,
                'total_priority': 0
            }
        
        stats = category_stats[category]
        stats['count'] += 1
        stats['total_productivity'] += productivity
        stats['total_priority'] += priority
    
    # Calculate averages
    for category, stats in category_stats.items():
        if stats['count'] > 0:
            stats['avg_productivity'] = round(stats['total_productivity'] / stats['count'], 1)
            stats['avg_priority'] = round(stats['total_priority'] / stats['count'], 2)
    
    # Sort by average priority
    sorted_categories = sorted(
        category_stats.items(),
        key=lambda x: x[1]['avg_priority'],
        reverse=True
    )
    
    return {
        'categories': dict(sorted_categories),
        'most_promising_category': sorted_categories[0][0] if sorted_categories else None,
        'total_categories_found': len(category_stats)
    }

def _generate_implementation_recommendations(ideas: List) -> List[str]:
    """Generate implementation recommendations based on discovered ideas"""
    if not ideas:
        return ["No ideas found - try running discovery first"]
    
    recommendations = []
    
    # Quick wins (easy, high productivity)
    quick_wins = []
    for idea in ideas:
        if hasattr(idea, 'difficulty'):
            difficulty = idea.difficulty
            productivity = idea.productivity_estimate
        else:
            difficulty = idea.get('difficulty')
            productivity = idea.get('productivity_estimate', 0)
        
        if difficulty == 'easy' and productivity > 15:
            quick_wins.append(idea)
    
    if quick_wins:
        recommendations.append(
            f"Start with {len(quick_wins)} quick wins - easy implementations with high impact"
        )
    
    # High-impact projects  
    high_impact = []
    for idea in ideas:
        if hasattr(idea, 'productivity_estimate'):
            productivity = idea.productivity_estimate
        else:
            productivity = idea.get('productivity_estimate', 0)
        
        if productivity > 30:
            high_impact.append(idea)
            
    if high_impact:
        recommendations.append(
            f"Consider {len(high_impact)} high-impact projects for major productivity gains"
        )
    
    # Category focus
    categories = {}
    for idea in ideas:
        if hasattr(idea, 'category'):
            category = idea.category
        else:
            category = idea.get('category', 'Unknown')
        categories[category] = categories.get(category, 0) + 1
    
    if categories:
        top_category = max(categories.items(), key=lambda x: x[1])
        recommendations.append(
            f"Focus on {top_category[0]} - most opportunities found ({top_category[1]} ideas)"
        )
    
    # Implementation timeline
    easy_count = 0
    medium_count = 0
    hard_count = 0
    
    for idea in ideas:
        if hasattr(idea, 'difficulty'):
            difficulty = idea.difficulty
        else:
            difficulty = idea.get('difficulty')
            
        if difficulty == 'easy':
            easy_count += 1
        elif difficulty == 'medium':
            medium_count += 1
        elif difficulty == 'hard':
            hard_count += 1
    
    recommendations.append(
        f"Implementation timeline: {easy_count} quick (1-2 days), {medium_count} medium (1 week), {hard_count} complex (2+ weeks)"
    )
    
    return recommendations[:5]  # Limit to top 5 recommendations

async def _get_fallback_automation_ideas():
    """Return sample automation ideas when external sources are unavailable"""
    
    # Define sample ideas that will be used for both top_ideas and all_ideas
    sample_ideas = [
            {
                "title": "Automated Data Backup Scripts",
                "description": "Create Python scripts that automatically backup critical databases and files to cloud storage on a scheduled basis.",
                "source": "System Database",
                "url": "#",
                "category": "Data Processing",
                "difficulty": "medium",
                "productivity_estimate": 30,
                "implementation_time_days": 3,
                "priority_score": 8.5,
                "tags": ["backup", "python", "automation", "cloud"],
                "upvotes": 45,
                "engagement_score": 12,
                "discovered_at": datetime.now().isoformat()
            },
            {
                "title": "Slack Bot for Status Updates",
                "description": "Build a Slack bot that automatically posts system status, deployment notifications, and daily standup reminders.",
                "source": "System Database", 
                "url": "#",
                "category": "Communication & Notifications",
                "difficulty": "easy",
                "productivity_estimate": 15,
                "implementation_time_days": 2,
                "priority_score": 7.8,
                "tags": ["slack", "bot", "notifications", "team"],
                "upvotes": 62,
                "engagement_score": 18,
                "discovered_at": datetime.now().isoformat()
            },
            {
                "title": "CI/CD Pipeline for Automatic Deployments",
                "description": "Set up automated testing and deployment pipeline that runs tests and deploys to staging/production automatically.",
                "source": "System Database",
                "url": "#",
                "category": "Deployment & DevOps",
                "difficulty": "hard",
                "productivity_estimate": 40,
                "implementation_time_days": 7,
                "priority_score": 9.2,
                "tags": ["ci/cd", "testing", "deployment", "devops"],
                "upvotes": 89,
                "engagement_score": 34,
                "discovered_at": datetime.now().isoformat()
            },
            {
                "title": "Log Analysis and Alert System",
                "description": "Automated log parsing that detects errors, performance issues, and sends alerts to relevant team members.",
                "source": "System Database",
                "url": "#",
                "category": "Monitoring & Alerts", 
                "difficulty": "medium",
                "productivity_estimate": 25,
                "implementation_time_days": 4,
                "priority_score": 8.0,
                "tags": ["logs", "monitoring", "alerts", "analysis"],
                "upvotes": 37,
                "engagement_score": 15,
                "discovered_at": datetime.now().isoformat()
            },
            {
                "title": "Database Migration and Sync Tools",
                "description": "Scripts to automatically sync data between development, staging and production databases with validation.",
                "source": "System Database",
                "url": "#",
                "category": "Data Processing",
                "difficulty": "medium", 
                "productivity_estimate": 35,
                "implementation_time_days": 5,
                "priority_score": 8.7,
                "tags": ["database", "migration", "sync", "validation"],
                "upvotes": 28,
                "engagement_score": 9,
                "discovered_at": datetime.now().isoformat()
            },
            {
                "title": "Automated Code Quality Checks",
                "description": "Set up automated linting, code review, and quality gate checks that run on every pull request.",
                "source": "System Database",
                "url": "#",
                "category": "Testing & QA",
                "difficulty": "easy",
                "productivity_estimate": 20,
                "implementation_time_days": 2,
                "priority_score": 7.5,
                "tags": ["code-quality", "linting", "testing", "pr-checks"],
                "upvotes": 54,
                "engagement_score": 22,
                "discovered_at": datetime.now().isoformat()
            }
        ]
    
    return {
        "status": "success",
        "source": "fallback",
        "note": "External sources unavailable, showing sample automation ideas",
        "discovery_summary": {
            "total_ideas_found": len(sample_ideas),
            "sources_scanned": 1,
            "avg_productivity_estimate": 25,
            "best_source": "System Database",
            "discovery_timestamp": datetime.now().isoformat()
        },
        "top_ideas": sample_ideas[:3],  # Show top 3 as featured
        "all_ideas": sample_ideas,  # All ideas for the Add to TODO functionality
        "source_rankings": {
            "System Database": {
                "rank": 1,
                "quality_score": 8.5,
                "total_ideas": len(sample_ideas),
                "avg_productivity": 27,
                "avg_priority_score": 8.3,
                "avg_engagement": 18,
                "top_categories": [["Data Processing", 2], ["DevOps", 2], ["Monitoring", 1]],
                "difficulty_distribution": {"easy": 2, "medium": 3, "hard": 1}
            }
        },
        "categories_summary": {
            "Data Processing": 1,
            "Communication & Notifications": 1,
            "Deployment & DevOps": 2,
            "Monitoring & Alerts": 1,
            "Testing & QA": 1
        },
        "recommendations": [
            "Start with easy-to-implement ideas like Slack bots and code quality checks",
            "Focus on high-productivity items like CI/CD pipelines",
            "Consider team skill level when choosing automation projects"
        ]
    }

class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None

@router.post("/chat")
async def automation_chat(request: ChatRequest):
    """Chat with AI assistant for targeted automation ideas"""
    try:
        message = request.message.lower()
        context = getattr(request, 'context', '') or ""
        
        # Analyze the user's message for automation categories
        automation_ideas = []
        response_message = ""
        
        if any(keyword in message for keyword in ['manufacturing', 'production', 'factory', 'assembly']):
            response_message = "I can help you with manufacturing automation! Here are some ideas:"
            automation_ideas = [
                {
                    "title": "Automated Quality Control",
                    "description": "Use computer vision and sensors to automatically inspect products for defects, reducing manual QC time by up to 80%",
                    "category": "Manufacturing",
                    "implementation_time": "2-4 months",
                    "roi": "High (300-500% ROI)",
                    "url": "https://example.com/quality-control-automation"
                },
                {
                    "title": "Predictive Maintenance",
                    "description": "Monitor equipment sensors to predict failures before they happen, reducing downtime by 30-50%",
                    "category": "Manufacturing", 
                    "implementation_time": "3-6 months",
                    "roi": "Very High (400-600% ROI)",
                    "url": "https://example.com/predictive-maintenance"
                },
                {
                    "title": "Inventory Management Automation",
                    "description": "Automatically track materials and trigger reorders based on production schedules and consumption patterns",
                    "category": "Manufacturing",
                    "implementation_time": "1-3 months", 
                    "roi": "Medium (200-300% ROI)",
                    "url": "https://example.com/inventory-automation"
                }
            ]
        elif any(keyword in message for keyword in ['office', 'admin', 'document', 'paperwork', 'data entry']):
            response_message = "Great! Here are some office automation ideas that can save significant time:"
            automation_ideas = [
                {
                    "title": "Document Processing Automation",
                    "description": "Automatically extract data from invoices, receipts, and forms using OCR and AI, eliminating manual data entry",
                    "category": "Office Automation",
                    "implementation_time": "2-4 weeks",
                    "roi": "High (250-400% ROI)",
                    "url": "https://example.com/document-automation"
                },
                {
                    "title": "Email Workflow Automation", 
                    "description": "Auto-sort, respond to, and forward emails based on content, sender, and business rules",
                    "category": "Office Automation",
                    "implementation_time": "1-2 weeks",
                    "roi": "Medium (150-250% ROI)",
                    "url": "https://example.com/email-automation"
                },
                {
                    "title": "Report Generation Automation",
                    "description": "Automatically generate and distribute regular business reports from multiple data sources",
                    "category": "Office Automation", 
                    "implementation_time": "2-6 weeks",
                    "roi": "Medium (200-300% ROI)",
                    "url": "https://example.com/report-automation"
                }
            ]
        elif any(keyword in message for keyword in ['customer', 'support', 'service', 'chat', 'helpdesk']):
            response_message = "Customer service automation can greatly improve response times and satisfaction:"
            automation_ideas = [
                {
                    "title": "AI Chatbot for Common Queries",
                    "description": "Deploy an intelligent chatbot to handle 70-80% of common customer questions 24/7",
                    "category": "Customer Service",
                    "implementation_time": "1-2 months",
                    "roi": "High (300-450% ROI)",
                    "url": "https://example.com/ai-chatbot"
                },
                {
                    "title": "Ticket Routing Automation",
                    "description": "Automatically categorize and route support tickets to the right team based on content analysis",
                    "category": "Customer Service",
                    "implementation_time": "2-4 weeks", 
                    "roi": "Medium (200-300% ROI)",
                    "url": "https://example.com/ticket-automation"
                },
                {
                    "title": "Customer Feedback Analysis",
                    "description": "Automatically analyze customer reviews and feedback to identify trends and issues",
                    "category": "Customer Service",
                    "implementation_time": "1-3 months",
                    "roi": "Medium (150-250% ROI)",
                    "url": "https://example.com/feedback-automation"
                }
            ]
        elif any(keyword in message for keyword in ['sales', 'lead', 'crm', 'marketing', 'follow']):
            response_message = "Sales and marketing automation can boost conversion rates significantly:"
            automation_ideas = [
                {
                    "title": "Lead Scoring Automation",
                    "description": "Automatically score and prioritize leads based on behavior, demographics, and engagement patterns",
                    "category": "Sales & Marketing",
                    "implementation_time": "1-2 months",
                    "roi": "High (250-400% ROI)",
                    "url": "https://example.com/lead-scoring"
                },
                {
                    "title": "Email Campaign Automation",
                    "description": "Create personalized, triggered email sequences based on customer actions and preferences",
                    "category": "Sales & Marketing",
                    "implementation_time": "2-6 weeks",
                    "roi": "Medium (200-350% ROI)", 
                    "url": "https://example.com/email-campaigns"
                },
                {
                    "title": "Social Media Automation",
                    "description": "Schedule posts, respond to mentions, and track social media performance across platforms",
                    "category": "Sales & Marketing",
                    "implementation_time": "2-4 weeks",
                    "roi": "Medium (150-250% ROI)",
                    "url": "https://example.com/social-automation"
                }
            ]
        else:
            response_message = "I'd be happy to help you find automation opportunities! Here are some general ideas:"
            automation_ideas = [
                {
                    "title": "Process Documentation Automation", 
                    "description": "Automatically capture and document business processes, creating step-by-step guides and training materials",
                    "category": "General",
                    "implementation_time": "1-2 months",
                    "roi": "Medium (180-280% ROI)",
                    "url": "https://example.com/process-documentation"
                },
                {
                    "title": "Data Backup Automation",
                    "description": "Automatically backup critical business data to multiple locations with verification and alerts",
                    "category": "General",
                    "implementation_time": "1-2 weeks", 
                    "roi": "High (Risk mitigation value)",
                    "url": "https://example.com/backup-automation"
                },
                {
                    "title": "Workflow Integration Automation",
                    "description": "Connect different business tools and systems to eliminate manual data transfer between platforms",
                    "category": "General",
                    "implementation_time": "2-8 weeks",
                    "roi": "High (300-500% ROI)",
                    "url": "https://example.com/workflow-integration"
                }
            ]
        
        return {
            "response": response_message,
            "automation_ideas": automation_ideas,
            "suggestions": [
                "Would you like more specific ideas for any of these categories?",
                "I can also help you prioritize these based on your budget and timeline.",
                "Ask me about implementation steps for any specific automation idea."
            ]
        }
        
    except Exception as e:
        logger.error(f"Error in automation chat: {e}")
        return {
            "response": "I'm sorry, I encountered an error. Please try rephrasing your question.",
            "automation_ideas": [],
            "suggestions": ["Try asking about manufacturing, office tasks, customer service, or sales automation."]
        }