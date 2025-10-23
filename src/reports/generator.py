"""
Automated Research Report Generation System
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path
from loguru import logger
import asyncio
from jinja2 import Template
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import openai

from research_synthesis.config.settings import settings


class ReportGenerator:
    """Generate comprehensive research reports"""
    
    def __init__(self):
        """Initialize report generator"""
        self.template_dir = Path(settings.REPORT_TEMPLATE_DIR)
        self.output_dir = Path(settings.REPORT_OUTPUT_DIR)
        
        # Create directories if they don't exist
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set OpenAI key if available
        if settings.OPENAI_API_KEY:
            openai.api_key = settings.OPENAI_API_KEY
        
        self.report_templates = {
            'executive': self.create_executive_template(),
            'technical': self.create_technical_template(),
            'competitive': self.create_competitive_template(),
            'market': self.create_market_template(),
            'manufacturing': self.create_manufacturing_template()
        }
    
    async def generate_executive_summary(self, 
                                       articles: List[Dict],
                                       time_period: str = "last_week",
                                       focus_area: str = "space_manufacturing") -> Dict[str, Any]:
        """Generate executive summary report"""
        
        # Analyze articles
        analysis = await self.analyze_articles(articles)
        
        # Generate key insights
        key_insights = self.extract_key_insights(analysis, focus_area)
        
        # Create visualizations
        charts = self.create_executive_charts(analysis)
        
        # Generate AI-powered summary
        ai_summary = await self.generate_ai_summary(articles, "executive")
        
        report = {
            'title': f'Executive Brief: {focus_area.replace("_", " ").title()}',
            'period': time_period,
            'generated_at': datetime.utcnow().isoformat(),
            'executive_summary': ai_summary,
            'key_insights': key_insights,
            'risk_factors': self.identify_risks(analysis),
            'opportunities': self.identify_opportunities(analysis),
            'recommendations': self.generate_recommendations(analysis),
            'charts': charts,
            'source_count': len(articles),
            'confidence_score': self.calculate_confidence(analysis)
        }
        
        return report
    
    async def generate_technical_analysis(self, 
                                        articles: List[Dict],
                                        technology: str) -> Dict[str, Any]:
        """Generate detailed technical analysis report"""
        
        # Filter articles by technology relevance
        relevant_articles = self.filter_by_technology(articles, technology)
        
        # Extract technical details
        tech_analysis = self.extract_technical_details(relevant_articles)
        
        # Analyze trends
        trends = self.analyze_technology_trends(relevant_articles)
        
        # Create technical visualizations
        charts = self.create_technical_charts(tech_analysis, trends)
        
        # Generate AI analysis
        ai_analysis = await self.generate_ai_summary(relevant_articles, "technical")
        
        report = {
            'title': f'Technical Analysis: {technology}',
            'technology': technology,
            'generated_at': datetime.utcnow().isoformat(),
            'technical_overview': ai_analysis,
            'maturity_assessment': self.assess_technology_maturity(tech_analysis),
            'performance_metrics': tech_analysis.get('metrics', {}),
            'development_timeline': self.create_timeline(relevant_articles),
            'key_players': self.identify_key_players(relevant_articles),
            'innovations': self.extract_innovations(relevant_articles),
            'challenges': self.identify_challenges(relevant_articles),
            'future_outlook': self.predict_future_developments(trends),
            'charts': charts,
            'source_count': len(relevant_articles)
        }
        
        return report
    
    async def generate_competitive_analysis(self, 
                                          companies: List[str],
                                          articles: List[Dict]) -> Dict[str, Any]:
        """Generate competitive landscape analysis"""
        
        # Filter articles by companies
        company_articles = {}
        for company in companies:
            company_articles[company] = self.filter_by_company(articles, company)
        
        # Analyze each company
        company_analysis = {}
        for company, articles_subset in company_articles.items():
            company_analysis[company] = await self.analyze_company(articles_subset, company)
        
        # Generate competitive comparison
        comparison = self.compare_companies(company_analysis)
        
        # Create competitive charts
        charts = self.create_competitive_charts(company_analysis)
        
        report = {
            'title': 'Competitive Landscape Analysis',
            'companies': companies,
            'generated_at': datetime.utcnow().isoformat(),
            'market_overview': await self.generate_market_overview(articles),
            'company_profiles': company_analysis,
            'competitive_positioning': comparison,
            'market_share': self.estimate_market_share(company_analysis),
            'strategic_moves': self.identify_strategic_moves(company_analysis),
            'partnership_network': self.map_partnerships(articles),
            'investment_activity': self.track_investments(articles),
            'charts': charts,
            'total_articles': len(articles)
        }
        
        return report
    
    async def generate_manufacturing_intelligence(self, 
                                                articles: List[Dict]) -> Dict[str, Any]:
        """Generate space manufacturing intelligence report"""
        
        # Filter manufacturing-related articles
        manufacturing_articles = [
            a for a in articles 
            if a.get('is_manufacturing_related', False)
        ]
        
        # Analyze manufacturing trends
        manufacturing_analysis = self.analyze_manufacturing_trends(manufacturing_articles)
        
        # Extract manufacturing capabilities
        capabilities = self.extract_manufacturing_capabilities(manufacturing_articles)
        
        # Economic analysis
        economics = self.analyze_manufacturing_economics(manufacturing_articles)
        
        # Create manufacturing visualizations
        charts = self.create_manufacturing_charts(manufacturing_analysis, economics)
        
        report = {
            'title': 'Space Manufacturing Intelligence Report',
            'generated_at': datetime.utcnow().isoformat(),
            'executive_summary': await self.generate_ai_summary(manufacturing_articles, "manufacturing"),
            'manufacturing_trends': manufacturing_analysis['trends'],
            'key_technologies': manufacturing_analysis['technologies'],
            'production_capabilities': capabilities,
            'economic_analysis': economics,
            'market_opportunities': self.identify_manufacturing_opportunities(manufacturing_articles),
            'supply_chain': self.analyze_supply_chain(manufacturing_articles),
            'regulatory_landscape': self.analyze_regulations(manufacturing_articles),
            'future_projections': self.project_manufacturing_future(manufacturing_analysis),
            'charts': charts,
            'source_count': len(manufacturing_articles)
        }
        
        return report
    
    async def analyze_articles(self, articles: List[Dict]) -> Dict[str, Any]:
        """Analyze collection of articles"""
        analysis = {
            'total_articles': len(articles),
            'sources': set(),
            'date_range': {'start': None, 'end': None},
            'entities': {},
            'topics': {},
            'sentiment': {'positive': 0, 'negative': 0, 'neutral': 0}
        }
        
        dates = []
        
        for article in articles:
            # Collect sources
            if article.get('source'):
                analysis['sources'].add(article['source'])
            
            # Collect dates
            if article.get('published_date'):
                dates.append(article['published_date'])
            
            # Count entities
            entities = article.get('entities', [])
            for entity in entities:
                entity_text = entity.get('text', '')
                analysis['entities'][entity_text] = analysis['entities'].get(entity_text, 0) + 1
            
            # Analyze sentiment
            sentiment = article.get('sentiment', 'neutral')
            analysis['sentiment'][sentiment] += 1
        
        # Set date range
        if dates:
            analysis['date_range']['start'] = min(dates)
            analysis['date_range']['end'] = max(dates)
        
        analysis['sources'] = list(analysis['sources'])
        
        return analysis
    
    def extract_key_insights(self, analysis: Dict, focus_area: str) -> List[str]:
        """Extract key insights from analysis"""
        insights = []
        
        # Entity-based insights
        if analysis['entities']:
            top_entities = sorted(analysis['entities'].items(), key=lambda x: x[1], reverse=True)[:5]
            insights.append(f"Most mentioned entities: {', '.join([e[0] for e in top_entities])}")
        
        # Sentiment insights
        total_articles = analysis['total_articles']
        if total_articles > 0:
            positive_pct = (analysis['sentiment']['positive'] / total_articles) * 100
            negative_pct = (analysis['sentiment']['negative'] / total_articles) * 100
            insights.append(f"Sentiment: {positive_pct:.1f}% positive, {negative_pct:.1f}% negative")
        
        # Source diversity
        source_count = len(analysis['sources'])
        insights.append(f"Information gathered from {source_count} different sources")
        
        return insights
    
    def identify_risks(self, analysis: Dict) -> List[str]:
        """Identify potential risks from analysis"""
        risks = []
        
        # High negative sentiment
        total = analysis['total_articles']
        if total > 0 and (analysis['sentiment']['negative'] / total) > 0.3:
            risks.append("Significant negative sentiment detected in recent coverage")
        
        # Add more risk identification logic here
        
        return risks
    
    def identify_opportunities(self, analysis: Dict) -> List[str]:
        """Identify opportunities from analysis"""
        opportunities = []
        
        # High positive sentiment
        total = analysis['total_articles']
        if total > 0 and (analysis['sentiment']['positive'] / total) > 0.6:
            opportunities.append("Strong positive sentiment suggests favorable market conditions")
        
        # Add more opportunity identification logic here
        
        return opportunities
    
    def generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate strategic recommendations"""
        recommendations = []
        
        # Based on entity frequency
        if analysis['entities']:
            top_entity = max(analysis['entities'].items(), key=lambda x: x[1])
            recommendations.append(f"Monitor developments related to {top_entity[0]} closely")
        
        return recommendations
    
    def create_executive_charts(self, analysis: Dict) -> List[Dict]:
        """Create charts for executive summary"""
        charts = []
        
        # Sentiment distribution pie chart
        sentiment_data = analysis['sentiment']
        if sum(sentiment_data.values()) > 0:
            fig = px.pie(
                values=list(sentiment_data.values()),
                names=list(sentiment_data.keys()),
                title="Sentiment Distribution"
            )
            charts.append({
                'type': 'pie',
                'title': 'Sentiment Distribution',
                'data': json.loads(fig.to_json())
            })
        
        # Top entities bar chart
        if analysis['entities']:
            top_entities = dict(sorted(analysis['entities'].items(), key=lambda x: x[1], reverse=True)[:10])
            fig = px.bar(
                x=list(top_entities.keys()),
                y=list(top_entities.values()),
                title="Most Mentioned Entities"
            )
            charts.append({
                'type': 'bar',
                'title': 'Most Mentioned Entities',
                'data': json.loads(fig.to_json())
            })
        
        return charts
    
    async def generate_ai_summary(self, articles: List[Dict], report_type: str) -> str:
        """Generate AI-powered summary"""
        if not settings.OPENAI_API_KEY:
            return f"AI summary not available (OpenAI API key not configured)"
        
        # Prepare content for summarization
        content_preview = []
        for article in articles[:10]:  # Limit to first 10 articles
            title = article.get('title', '')
            summary = article.get('summary', '')
            if title and summary:
                content_preview.append(f"Title: {title}/nSummary: {summary}")
        
        combined_content = "/n/n".join(content_preview)
        
        # Create prompt based on report type
        prompts = {
            'executive': "Provide a concise executive summary of the key developments and trends:",
            'technical': "Analyze the technical developments and provide insights on technological progress:",
            'manufacturing': "Summarize the space manufacturing developments and market implications:",
            'competitive': "Analyze the competitive landscape and strategic implications:"
        }
        
        prompt = prompts.get(report_type, "Summarize the key information:")
        
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": combined_content[:4000]}  # Limit content length
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"AI summary generation failed: {e}")
            return f"AI summary generation failed: {str(e)}"
    
    def calculate_confidence(self, analysis: Dict) -> float:
        """Calculate overall confidence score for the report"""
        factors = []
        
        # Source diversity
        source_count = len(analysis.get('sources', []))
        source_score = min(source_count / 5.0, 1.0)  # Max at 5 sources
        factors.append(source_score)
        
        # Article count
        article_count = analysis.get('total_articles', 0)
        article_score = min(article_count / 20.0, 1.0)  # Max at 20 articles
        factors.append(article_score)
        
        # Date recency (if date range available)
        if analysis.get('date_range', {}).get('end'):
            try:
                end_date = datetime.fromisoformat(analysis['date_range']['end'])
                days_old = (datetime.utcnow() - end_date).days
                recency_score = max(0, 1.0 - (days_old / 30.0))  # Decay over 30 days
                factors.append(recency_score)
            except:
                factors.append(0.5)
        
        return sum(factors) / len(factors) if factors else 0.5
    
    def create_executive_template(self) -> str:
        """Create executive summary template"""
        return """
# {{ title }}

**Report Period:** {{ period }}  
**Generated:** {{ generated_at }}  
**Confidence Score:** {{ "%.1f"|format(confidence_score * 100) }}%

## Executive Summary

{{ executive_summary }}

## Key Insights

{% for insight in key_insights %}
- {{ insight }}
{% endfor %}

## Risk Factors

{% for risk in risk_factors %}
- {{ risk }}
{% endfor %}

## Opportunities

{% for opportunity in opportunities %}
- {{ opportunity }}
{% endfor %}

## Strategic Recommendations

{% for recommendation in recommendations %}
- {{ recommendation }}
{% endfor %}

---

*This report was generated from {{ source_count }} sources using AI-powered analysis.*
"""
    
    def create_technical_template(self) -> str:
        """Create technical analysis template"""
        return """
# {{ title }}

**Technology:** {{ technology }}  
**Generated:** {{ generated_at }}

## Technical Overview

{{ technical_overview }}

## Maturity Assessment

{{ maturity_assessment }}

## Key Performance Metrics

{% for metric, value in performance_metrics.items() %}
- **{{ metric }}:** {{ value }}
{% endfor %}

## Key Players

{% for player in key_players %}
- {{ player }}
{% endfor %}

## Recent Innovations

{% for innovation in innovations %}
- {{ innovation }}
{% endfor %}

## Technical Challenges

{% for challenge in challenges %}
- {{ challenge }}
{% endfor %}

## Future Outlook

{{ future_outlook }}

---

*Analysis based on {{ source_count }} technical sources.*
"""
    
    def create_competitive_template(self) -> str:
        """Create competitive analysis template"""
        return """
# {{ title }}

**Companies Analyzed:** {{ companies|join(", ") }}  
**Generated:** {{ generated_at }}

## Market Overview

{{ market_overview }}

## Company Profiles

{% for company, profile in company_profiles.items() %}
### {{ company }}

{{ profile.summary }}

**Key Strengths:**
{% for strength in profile.strengths %}
- {{ strength }}
{% endfor %}

{% endfor %}

## Competitive Positioning

{{ competitive_positioning }}

---

*Analysis based on {{ total_articles }} articles.*
"""
    
    def create_market_template(self) -> str:
        """Create market analysis template"""
        return """
# Market Analysis Report

**Generated:** {{ generated_at }}

## Market Overview

{{ market_overview }}

## Key Market Trends

{% for trend in market_trends %}
- {{ trend }}
{% endfor %}

## Market Size and Growth

{{ market_size }}

## Competitive Landscape

{{ competitive_landscape }}

---

*Market analysis based on comprehensive data sources.*
"""
    
    def create_manufacturing_template(self) -> str:
        """Create manufacturing intelligence template"""
        return """
# {{ title }}

**Generated:** {{ generated_at }}

## Executive Summary

{{ executive_summary }}

## Manufacturing Trends

{% for trend in manufacturing_trends %}
- {{ trend }}
{% endfor %}

## Key Technologies

{% for tech in key_technologies %}
- {{ tech }}
{% endfor %}

## Economic Analysis

{{ economic_analysis.summary }}

## Market Opportunities

{% for opportunity in market_opportunities %}
- {{ opportunity }}
{% endfor %}

## Future Projections

{{ future_projections }}

---

*Intelligence gathered from {{ source_count }} manufacturing sources.*
"""
    
    # Placeholder methods for various analysis functions
    def filter_by_technology(self, articles: List[Dict], technology: str) -> List[Dict]:
        """Filter articles by technology relevance"""
        # Implementation would filter articles based on technology keywords
        return articles
    
    def filter_by_company(self, articles: List[Dict], company: str) -> List[Dict]:
        """Filter articles by company mentions"""
        # Implementation would filter articles mentioning specific company
        return [a for a in articles if company.lower() in a.get('content', '').lower()]
    
    def extract_technical_details(self, articles: List[Dict]) -> Dict:
        """Extract technical details from articles"""
        return {'metrics': {}, 'developments': []}
    
    def analyze_technology_trends(self, articles: List[Dict]) -> Dict:
        """Analyze technology trends"""
        return {'trends': [], 'predictions': []}
    
    def assess_technology_maturity(self, analysis: Dict) -> str:
        """Assess technology maturity level"""
        return "Emerging technology with rapid development"
    
    def create_timeline(self, articles: List[Dict]) -> List[Dict]:
        """Create development timeline"""
        return []
    
    def identify_key_players(self, articles: List[Dict]) -> List[str]:
        """Identify key players in the field"""
        return []
    
    def extract_innovations(self, articles: List[Dict]) -> List[str]:
        """Extract recent innovations"""
        return []
    
    def identify_challenges(self, articles: List[Dict]) -> List[str]:
        """Identify technical challenges"""
        return []
    
    def predict_future_developments(self, trends: Dict) -> str:
        """Predict future developments"""
        return "Continued innovation expected in this space"
    
    def analyze_company(self, articles: List[Dict], company: str) -> Dict:
        """Analyze a specific company"""
        return {
            'summary': f'Analysis of {company}',
            'strengths': ['Market leader', 'Strong technology'],
            'activities': []
        }
    
    def compare_companies(self, company_analysis: Dict) -> str:
        """Compare companies competitively"""
        return "Competitive analysis summary"
    
    def estimate_market_share(self, company_analysis: Dict) -> Dict:
        """Estimate market share"""
        return {}
    
    def identify_strategic_moves(self, company_analysis: Dict) -> List[str]:
        """Identify strategic moves"""
        return []
    
    def map_partnerships(self, articles: List[Dict]) -> Dict:
        """Map partnership networks"""
        return {}
    
    def track_investments(self, articles: List[Dict]) -> Dict:
        """Track investment activity"""
        return {}
    
    def generate_market_overview(self, articles: List[Dict]) -> str:
        """Generate market overview"""
        return "Market overview based on recent developments"
    
    def create_technical_charts(self, analysis: Dict, trends: Dict) -> List[Dict]:
        """Create technical analysis charts"""
        return []
    
    def create_competitive_charts(self, analysis: Dict) -> List[Dict]:
        """Create competitive analysis charts"""
        return []
    
    def analyze_manufacturing_trends(self, articles: List[Dict]) -> Dict:
        """Analyze manufacturing trends"""
        return {
            'trends': ['Increased automation', 'Cost reduction focus'],
            'technologies': ['3D printing', 'Robotics']
        }
    
    def extract_manufacturing_capabilities(self, articles: List[Dict]) -> Dict:
        """Extract manufacturing capabilities"""
        return {}
    
    def analyze_manufacturing_economics(self, articles: List[Dict]) -> Dict:
        """Analyze manufacturing economics"""
        return {'summary': 'Economic analysis of space manufacturing'}
    
    def create_manufacturing_charts(self, analysis: Dict, economics: Dict) -> List[Dict]:
        """Create manufacturing charts"""
        return []
    
    def identify_manufacturing_opportunities(self, articles: List[Dict]) -> List[str]:
        """Identify manufacturing opportunities"""
        return ['High-value products', 'Specialized materials']
    
    def analyze_supply_chain(self, articles: List[Dict]) -> Dict:
        """Analyze supply chain"""
        return {}
    
    def analyze_regulations(self, articles: List[Dict]) -> Dict:
        """Analyze regulatory landscape"""
        return {}
    
    def project_manufacturing_future(self, analysis: Dict) -> str:
        """Project future of manufacturing"""
        return "Continued growth in space manufacturing capabilities expected"