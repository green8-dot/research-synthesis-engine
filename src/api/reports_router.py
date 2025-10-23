
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
import json
import os
from pathlib import Path
from research_synthesis.database.connection import get_postgres_session
from research_synthesis.database.models import Report
from research_synthesis.utils.database_optimizer import database_optimizer, OperationType
from sqlalchemy import select, func, desc

router = APIRouter()

# Reports storage directory
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# Database reports storage with intelligent database selection
# Using database optimizer for cost-effective storage

@router.get("/")
async def get_reports():
    """Get all available reports from database"""
    reports_list = []
    
    try:
        # Get optimal database for read operations
        optimal_db = database_optimizer.choose_optimal_database(
            OperationType.READ,
            data_size_mb=1.0,  # Small read operation
            priority="cost"
        )
        
        async for db_session in get_postgres_session():
            # Get reports from database
            result = await db_session.execute(
                select(Report)
                .order_by(desc(Report.created_at))
                .limit(50)  # Limit to recent reports
            )
            reports = result.scalars().all()
            
            for report in reports:
                reports_list.append({
                    "id": str(report.id),
                    "title": report.title or f"Report {report.id}",
                    "type": report.report_type or "unknown",
                    "status": "completed",
                    "generated_at": report.created_at.isoformat() if report.created_at else None,
                    "focus_area": "general",
                    "word_count": report.word_count,
                    "confidence_score": report.confidence_score
                })
            break
            
    except Exception as e:
        print(f"Error loading reports from database: {e}")
        # Fallback to empty list
    
    return {
        "reports": reports_list,
        "total_reports": len(reports_list),
        "generating_reports": 0,
        "report_types": [
            {
                "id": "market_analysis",
                "name": "Market Analysis",
                "description": "Comprehensive market trends and competitive analysis",
                "icon": "fas fa-chart-line"
            },
            {
                "id": "company_profile",
                "name": "Company Profile",
                "description": "Detailed company analysis and strategic insights",
                "icon": "fas fa-building"
            },
            {
                "id": "technology_trends",
                "name": "Technology Trends",
                "description": "Emerging technology analysis and innovation tracking",
                "icon": "fas fa-rocket"
            },
            {
                "id": "investment_brief",
                "name": "Investment Brief",
                "description": "Investment opportunities and financial analysis",
                "icon": "fas fa-dollar-sign"
            }
        ]
    }

class ReportRequest(BaseModel):
    report_type: str
    title: Optional[str] = None
    time_period: str = "last_month"
    focus_area: str = "general"
    companies: Optional[str] = None
    notes: Optional[str] = None
    include_charts: bool = True
    executive_summary: bool = True

class ReportResponse(BaseModel):
    message: str
    report_id: str
    status: str = "generating"
    estimated_completion: str

@router.post("/generate", response_model=ReportResponse)
async def generate_report(request: ReportRequest):
    """Generate a new intelligence report"""
    report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Generate actual report content based on knowledge graph data
    try:
        report_content = await generate_report_content(request, report_id)
        
        # Get optimal database for write operations
        optimal_db = database_optimizer.choose_optimal_database(
            OperationType.WRITE,
            data_size_mb=len(str(report_content)) / (1024 * 1024),  # Estimate content size
            priority="balanced"
        )
        
        # Save report to database for persistence
        async for db_session in get_postgres_session():
            try:
                # Create database report record
                db_report = Report(
                    title=request.title or f"{request.report_type.replace('_', ' ').title()} Report",
                    report_type=request.report_type,
                    executive_summary=json.dumps(report_content.get('executive_summary', {})),
                    full_content=json.dumps(report_content),
                    word_count=len(str(report_content).split()),
                    confidence_score=0.85,  # Default confidence
                    sections=json.dumps(report_content.get('sections', [])),
                    key_findings=json.dumps(report_content.get('executive_summary', {}).get('key_findings', [])),
                    recommendations=json.dumps(["Analysis complete based on current data"]),
                    generation_params=json.dumps(request.dict()),
                    generation_time=1.0,  # Placeholder
                    model_used="Research Synthesis Engine v1.0"
                )
                
                db_session.add(db_report)
                await db_session.commit()
                await db_session.refresh(db_report)
                
                # Update report_id to use database ID
                report_id = str(db_report.id)
                
                print(f"Report {report_id} saved to database successfully")
                
            except Exception as db_error:
                print(f"Database save failed: {db_error}")
                await db_session.rollback()
                # Continue with file-based fallback
            
            break
        
        # Also save to file as backup
        report_data = {
            "id": report_id,
            "title": request.title or f"{request.report_type.replace('_', ' ').title()} Report",
            "type": request.report_type,
            "status": "completed",
            "generated_at": datetime.now().isoformat(),
            "focus_area": request.focus_area,
            "time_period": request.time_period,
            "content": report_content,
            "request_data": request.dict()
        }
        
        report_file = REPORTS_DIR / f"{report_id}.json"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, default=str)
        except Exception as file_error:
            print(f"File save failed: {file_error}")
        
        # Clear dashboard cache since new report was generated
        try:
            from research_synthesis.database.connection import get_redis
            redis_client = get_redis()
            if redis_client:
                await redis_client.delete("dashboard_stats")
        except Exception as e:
            print(f"Cache invalidation failed: {e}")
        
        return ReportResponse(
            message=f"Generated {request.report_type} report successfully",
            report_id=report_id,
            status="completed",
            estimated_completion="Report ready now"
        )
    except Exception as e:
        return ReportResponse(
            message=f"Error generating {request.report_type} report: {str(e)}",
            report_id=report_id,
            status="failed",
            estimated_completion="N/A"
        )

@router.get("/view/{report_id}")
async def get_report_details(report_id: str):
    """Get detailed report by ID"""
    try:
        async for session in get_postgres_session():
            result = await session.execute(
                select(Report).filter(Report.id == report_id)
            )
            report = result.scalar_one_or_none()
            
            if not report:
                raise HTTPException(status_code=404, detail="Report not found")
            
            return {
                "id": str(report.id),
                "title": report.title,
                "report_type": report.report_type,
                "executive_summary": report.executive_summary,
                "full_content": report.full_content,
                "word_count": report.word_count,
                "reading_time": report.reading_time,
                "confidence_score": report.confidence_score,
                "sections": report.sections,
                "key_findings": report.key_findings,
                "recommendations": report.recommendations,
                "risks": report.risks,
                "opportunities": report.opportunities,
                "created_at": report.created_at.isoformat()
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving report {report_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve report")

@router.get("/download/{report_id}")
async def download_report(report_id: str):
    """Download generated report from database"""
    # Validate report_id
    if not report_id or report_id in ["undefined", "null", ""]:
        raise HTTPException(status_code=400, detail="Invalid report ID provided")
    
    try:
        report_id_int = int(report_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Report ID must be a valid integer, got: {report_id}")
    
    try:
        # Get optimal database for read operations
        optimal_db = database_optimizer.choose_optimal_database(
            OperationType.READ,
            data_size_mb=5.0,  # Assume larger read for full report
            priority="cost"
        )
        
        async for db_session in get_postgres_session():
            # Try to get from database first
            result = await db_session.execute(
                select(Report).filter(Report.id == report_id_int)
            )
            db_report = result.scalar_one_or_none()
            
            if db_report:
                full_content = json.loads(db_report.full_content) if db_report.full_content else {}
                return {
                    "id": str(db_report.id),
                    "report_id": str(db_report.id),  # Add report_id for consistency with frontend
                    "title": db_report.title,
                    "type": db_report.report_type,
                    "status": "completed",
                    "generated_at": db_report.created_at.isoformat(),
                    "content": full_content,
                    "word_count": db_report.word_count,
                    "confidence_score": db_report.confidence_score,
                    "format": "json"
                }
            break
            
    except Exception as e:
        print(f"Database retrieval failed: {e}")
    
    # Try to load from file as fallback
    report_file = REPORTS_DIR / f"{report_id}.json"
    if report_file.exists():
        with open(report_file, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        return report_data
    
    # Fallback to simple structure
    return {
        "report_id": report_id,
        "title": f"Space Industry Intelligence Report - {report_id}",
        "generated_at": datetime.now().isoformat(),
        "content": await get_report_content(report_id),
        "format": "json"
    }

async def generate_report_content(request: ReportRequest, report_id: str):
    """Generate actual report content from knowledge graph data with learning insights"""
    content = {}
    
    # Get learning insights for report improvement
    learning_insights = None
    try:
        from research_synthesis.services.learning_system import learning_system
        if learning_system:
            learning_insights = await learning_system.get_insights_for_report_generation(request.report_type)
    except Exception as e:
        print(f"Learning insights not available: {e}")
    
    try:
        from research_synthesis.database.connection import get_postgres_session
        from research_synthesis.database.models import Article, Entity
        from sqlalchemy import select, func
        
        async for db_session in get_postgres_session():
            # Get entity statistics
            total_entities_result = await db_session.execute(select(func.count(Entity.id)))
            total_entities = total_entities_result.scalar() or 0
            
            # Get entity distribution
            entity_types_query = await db_session.execute(
                select(Entity.entity_type, func.count(Entity.id).label('count'))
                .group_by(Entity.entity_type)
            )
            entity_distribution = {et[0]: et[1] for et in entity_types_query.fetchall()}
            
            # Get top companies by mention count
            top_companies_query = await db_session.execute(
                select(Entity.name, Entity.mention_count)
                .filter(Entity.entity_type == 'company')
                .order_by(Entity.mention_count.desc())
                .limit(10)
            )
            top_companies = [{"name": c[0], "mentions": c[1]} for c in top_companies_query.fetchall()]
            
            # Get recent articles
            articles_query = await db_session.execute(
                select(Article.title, Article.published_date, Article.summary)
                .order_by(Article.created_at.desc())
                .limit(5)
            )
            recent_articles = [
                {
                    "title": a[0], 
                    "date": a[1].isoformat() if a[1] else None,
                    "summary": a[2]
                } 
                for a in articles_query.fetchall()
            ]
            
            # Build enhanced content with learning insights
            key_findings = [
                f"Tracking {total_entities} space industry entities",
                f"Coverage across {len(entity_distribution)} entity categories",
                f"Top company by mentions: {top_companies[0]['name'] if top_companies else 'N/A'}",
                f"Recent analysis of {len(recent_articles)} articles"
            ]
            
            # Add learning-based insights if available
            if learning_insights and learning_insights.get('relevant_insights'):
                for insight in learning_insights['relevant_insights'][:2]:  # Add top 2 insights
                    key_findings.append(f"Learning insight: {insight.get('summary', 'N/A')}")
            
            content = {
                "executive_summary": {
                    "total_entities": total_entities,
                    "entity_types": len(entity_distribution),
                    "key_findings": key_findings
                },
                "entity_analysis": {
                    "distribution": entity_distribution,
                    "top_companies": top_companies
                },
                "recent_activity": {
                    "articles": recent_articles
                },
                "learning_insights": learning_insights if learning_insights else {},
                "report_metadata": {
                    "type": request.report_type,
                    "focus_area": request.focus_area,
                    "time_period": request.time_period,
                    "generated_at": datetime.now().isoformat(),
                    "enhanced_with_learning": learning_insights is not None
                }
            }
            break
            
    except Exception as e:
        content = {
            "error": f"Failed to generate report content: {str(e)}",
            "fallback_content": {
                "message": "Report generation system active",
                "note": "Limited data available in current environment"
            }
        }
    
    return content

async def get_report_content(report_id: str):
    """Get report content by ID (simplified for demo)"""
    return {
        "summary": f"Space Industry Intelligence Report {report_id}",
        "sections": [
            {
                "title": "Executive Summary",
                "content": "This report provides insights into current space industry trends and key players based on real-time data analysis."
            },
            {
                "title": "Market Analysis", 
                "content": "Analysis of space industry entities, companies, and recent developments."
            },
            {
                "title": "Recommendations",
                "content": "Strategic recommendations based on current market intelligence and trends."
            }
        ],
        "generated_at": datetime.now().isoformat()
    }
