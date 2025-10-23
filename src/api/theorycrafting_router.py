"""
Theorycrafting API Router
Provides intelligent research assistance through theoretical analysis
"""
from fastapi import APIRouter, HTTPException, Query, Body
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from loguru import logger
from datetime import datetime

try:
    from research_synthesis.services.theorycrafting_service import theorycrafting_service
    THEORYCRAFTING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Theorycrafting service not available: {e}")
    THEORYCRAFTING_AVAILABLE = False

router = APIRouter()

class ResearchTopicRequest(BaseModel):
    topic: str
    context: str = ""
    user_background: str = ""
    analysis_depth: str = "standard"  # "quick", "standard", "comprehensive"
    focus_areas: List[str] = []  # e.g., ["hypotheses", "methodologies", "applications"]

class ConceptAnalysisRequest(BaseModel):
    concepts: List[str]
    domain: str = "general"
    connection_analysis: bool = True

@router.get("/")
async def get_theorycrafting_info():
    """Get theorycrafting service information"""
    return {
        "service": "Theorycrafting Logic System",
        "status": "active" if THEORYCRAFTING_AVAILABLE else "unavailable",
        "description": "Intelligent research assistance through theoretical analysis and hypothesis generation",
        "capabilities": [
            "Research topic analysis and theorycrafting",
            "Hypothesis generation and evaluation", 
            "Conceptual connection mapping",
            "Research direction suggestions",
            "Methodological recommendations",
            "Interdisciplinary insight generation",
            "Knowledge gap identification",
            "Theoretical framework construction"
        ],
        "endpoints": [
            "/analyze - Comprehensive research topic analysis",
            "/concepts - Analyze and connect research concepts", 
            "/hypotheses - Generate research hypotheses",
            "/directions - Suggest research directions",
            "/sources - Get ML-powered source recommendations",
            "/connections - Find conceptual connections",
            "/methodologies - Recommend research methodologies",
            "/gaps - Identify knowledge gaps",
            "/framework - Build theoretical framework"
        ]
    }

@router.post("/analyze")
async def analyze_research_topic(request: ResearchTopicRequest):
    """
    Comprehensive theorycrafting analysis of a research topic
    
    Provides:
    - Theoretical framework construction
    - Hypothesis generation
    - Research direction suggestions
    - Conceptual connection mapping
    - Knowledge gap identification
    - Methodological recommendations
    """
    if not THEORYCRAFTING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Theorycrafting service unavailable")
    
    try:
        logger.info(f"Starting comprehensive theorycrafting analysis for: {request.topic}")
        
        # Perform comprehensive analysis
        analysis = await theorycrafting_service.analyze_research_topic(
            topic=request.topic,
            context=request.context,
            user_background=request.user_background
        )
        
        # Filter results based on focus areas if specified
        if request.focus_areas:
            filtered_analysis = {}
            for key, value in analysis.items():
                if key in request.focus_areas or key in ['topic', 'timestamp', 'meta_analysis']:
                    filtered_analysis[key] = value
            analysis = filtered_analysis
        
        # Adjust depth based on request
        if request.analysis_depth == "quick":
            # Limit results for quick analysis
            for key in ['hypotheses', 'research_directions', 'conceptual_connections']:
                if key in analysis:
                    analysis[key] = analysis[key][:3]
        elif request.analysis_depth == "comprehensive":
            # Add extra analysis for comprehensive request
            analysis['detailed_insights'] = await theorycrafting_service._generate_detailed_insights(
                request.topic, request.context
            )
        
        logger.info(f"Completed theorycrafting analysis with {len(analysis.get('hypotheses', []))} hypotheses")
        
        return {
            "status": "success",
            "analysis": analysis,
            "analysis_metadata": {
                "topic": request.topic,
                "analysis_depth": request.analysis_depth,
                "focus_areas": request.focus_areas or ["all"],
                "timestamp": analysis.get('timestamp'),
                "processing_stats": {
                    "hypotheses_generated": len(analysis.get('hypotheses', [])),
                    "research_directions": len(analysis.get('research_directions', [])),
                    "concepts_analyzed": len(analysis.get('concept_analysis', [])),
                    "connections_found": len(analysis.get('conceptual_connections', []))
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error in theorycrafting analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/concepts")
async def analyze_concepts(request: ConceptAnalysisRequest):
    """Analyze research concepts and find connections between them"""
    if not THEORYCRAFTING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Theorycrafting service unavailable")
    
    try:
        logger.info(f"Analyzing {len(request.concepts)} concepts in domain: {request.domain}")
        
        results = []
        
        # Analyze each concept
        for concept in request.concepts:
            concept_analysis = {
                'concept': concept,
                'definition': theorycrafting_service._generate_concept_definition(concept),
                'domain_classification': theorycrafting_service._classify_concept_domain(concept),
                'maturity_level': theorycrafting_service._assess_concept_maturity(concept),
                'applications': theorycrafting_service._identify_concept_applications(concept),
                'limitations': theorycrafting_service._identify_concept_limitations(concept),
                'future_potential': theorycrafting_service._assess_concept_future_potential(concept)
            }
            results.append(concept_analysis)
        
        response = {
            "status": "success",
            "concept_analyses": results
        }
        
        # Add connection analysis if requested
        if request.connection_analysis and len(request.concepts) > 1:
            connections = []
            for i, concept_a in enumerate(request.concepts):
                for concept_b in request.concepts[i+1:]:
                    connection = theorycrafting_service._analyze_concept_pair(
                        concept_a, concept_b, request.domain
                    )
                    if connection and connection.strength > 0.2:
                        connections.append({
                            'concept_a': concept_a,
                            'concept_b': concept_b,
                            'connection_type': connection.connection_type,
                            'strength': connection.strength,
                            'mechanism': connection.mechanism,
                            'implications': connection.implications
                        })
            
            response["conceptual_connections"] = connections
            logger.info(f"Found {len(connections)} significant connections between concepts")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in concept analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Concept analysis failed: {str(e)}")

@router.post("/hypotheses")
async def generate_hypotheses(
    topic: str = Body(..., embed=True),
    context: str = Body("", embed=True),
    hypothesis_types: List[str] = Body(["causal", "correlational", "predictive"], embed=True),
    max_hypotheses: int = Body(8, embed=True)
):
    """Generate research hypotheses for a given topic"""
    if not THEORYCRAFTING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Theorycrafting service unavailable")
    
    try:
        logger.info(f"Generating {max_hypotheses} hypotheses of types {hypothesis_types} for: {topic}")
        
        all_hypotheses = []
        
        # Generate different types of hypotheses based on request
        if "causal" in hypothesis_types:
            causal_hypotheses = theorycrafting_service._generate_causal_hypotheses(topic, context)
            all_hypotheses.extend(causal_hypotheses)
        
        if "correlational" in hypothesis_types:
            correlational_hypotheses = theorycrafting_service._generate_correlational_hypotheses(topic, context)
            all_hypotheses.extend(correlational_hypotheses)
        
        if "predictive" in hypothesis_types:
            predictive_hypotheses = theorycrafting_service._generate_predictive_hypotheses(topic, context)
            all_hypotheses.extend(predictive_hypotheses)
        
        if "comparative" in hypothesis_types:
            comparative_hypotheses = theorycrafting_service._generate_comparative_hypotheses(topic, context)
            all_hypotheses.extend(comparative_hypotheses)
        
        # Rank and select best hypotheses
        ranked_hypotheses = sorted(all_hypotheses, key=lambda h: h.confidence_level, reverse=True)
        selected_hypotheses = [asdict(h) for h in ranked_hypotheses[:max_hypotheses]]
        
        return {
            "status": "success",
            "hypotheses": selected_hypotheses,
            "generation_metadata": {
                "topic": topic,
                "hypothesis_types": hypothesis_types,
                "total_generated": len(all_hypotheses),
                "selected_count": len(selected_hypotheses),
                "avg_confidence": sum(h.confidence_level for h in ranked_hypotheses[:max_hypotheses]) / len(selected_hypotheses) if selected_hypotheses else 0,
                "avg_testability": sum(h.testability_score for h in ranked_hypotheses[:max_hypotheses]) / len(selected_hypotheses) if selected_hypotheses else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating hypotheses: {e}")
        raise HTTPException(status_code=500, detail=f"Hypothesis generation failed: {str(e)}")

@router.post("/directions")
async def suggest_research_directions(
    topic: str = Body(..., embed=True),
    context: str = Body("", embed=True),
    direction_types: List[str] = Body(["empirical", "theoretical", "applied"], embed=True),
    resource_constraints: Dict[str, Any] = Body({}, embed=True),
    max_directions: int = Body(6, embed=True)
):
    """Suggest research directions for a given topic"""
    if not THEORYCRAFTING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Theorycrafting service unavailable")
    
    try:
        logger.info(f"Suggesting {max_directions} research directions for: {topic}")
        
        # Generate research directions
        directions = await theorycrafting_service._suggest_research_directions(topic, context)
        
        # Filter by requested types if specified
        if direction_types and direction_types != ["all"]:
            filtered_directions = []
            for direction in directions:
                direction_type = theorycrafting_service._classify_research_direction_type(direction)
                if direction_type in direction_types:
                    filtered_directions.append(direction)
            directions = filtered_directions
        
        # Apply resource constraints if specified
        if resource_constraints:
            directions = theorycrafting_service._filter_by_constraints(directions, resource_constraints)
        
        # Limit to requested number
        selected_directions = directions[:max_directions]
        
        return {
            "status": "success",
            "research_directions": selected_directions,
            "direction_metadata": {
                "topic": topic,
                "direction_types": direction_types,
                "resource_constraints": resource_constraints,
                "total_generated": len(directions),
                "selected_count": len(selected_directions),
                "avg_difficulty": theorycrafting_service._calculate_avg_difficulty(selected_directions),
                "interdisciplinary_potential": sum(1 for d in selected_directions if len(d.get('interdisciplinary_connections', [])) > 0) / len(selected_directions) if selected_directions else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error suggesting research directions: {e}")
        raise HTTPException(status_code=500, detail=f"Research direction suggestion failed: {str(e)}")

@router.get("/domains")
async def get_domain_expertise():
    """Get information about domain expertise and capabilities"""
    if not THEORYCRAFTING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Theorycrafting service unavailable")
    
    return {
        "status": "success",
        "domain_expertise": theorycrafting_service.domain_expertise,
        "expertise_levels": {
            "expert": "0.8 - 1.0",
            "proficient": "0.6 - 0.8", 
            "intermediate": "0.4 - 0.6",
            "basic": "0.0 - 0.4"
        },
        "supported_domains": list(theorycrafting_service.domain_expertise.keys()),
        "cross_domain_capabilities": [
            "Interdisciplinary insight generation",
            "Cross-domain concept mapping",
            "Technology transfer identification",
            "Innovation opportunity spotting"
        ]
    }

@router.post("/sources")
async def get_ml_source_recommendations(
    topic: str = Body(..., embed=True),
    context: str = Body("", embed=True),
    research_type: str = Body("standard", embed=True),
    time_constraint: int = Body(60, embed=True, ge=10, le=240)
):
    """Get ML-powered source recommendations for research topic"""
    if not THEORYCRAFTING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Theorycrafting service unavailable")
    
    try:
        logger.info(f"Getting ML source recommendations for: {topic}")
        
        # Get recommendations from the theorycrafting service
        recommendations = await theorycrafting_service._get_ml_source_recommendations(topic, context)
        
        return {
            "status": "success",
            "topic": topic,
            "research_type": research_type,
            "time_constraint_minutes": time_constraint,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting ML source recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Source recommendation failed: {str(e)}")

@router.get("/conversation-history")
async def get_conversation_history(limit: int = Query(10, ge=1, le=50)):
    """Get recent theorycrafting conversation history"""
    if not THEORYCRAFTING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Theorycrafting service unavailable")
    
    history = theorycrafting_service.conversation_history[-limit:]
    
    # Anonymize and summarize for response
    summarized_history = []
    for entry in history:
        summary = {
            "topic": entry["topic"],
            "timestamp": entry["timestamp"].isoformat(),
            "analysis_summary": {
                "hypotheses_count": len(entry["analysis"].get("hypotheses", [])),
                "directions_count": len(entry["analysis"].get("research_directions", [])),
                "concepts_count": len(entry["analysis"].get("concept_analysis", [])),
                "primary_domain": entry["analysis"].get("meta_analysis", {}).get("domain_relevance", {})
            }
        }
        summarized_history.append(summary)
    
    return {
        "status": "success",
        "conversation_history": summarized_history,
        "total_conversations": len(theorycrafting_service.conversation_history),
        "showing": len(summarized_history)
    }

# Helper function imports
from dataclasses import asdict