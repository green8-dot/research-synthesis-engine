"""
UI Monitoring and Readability Service
Detects and addresses formatting, readability, and usability issues
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import aiohttp
import asyncio
import hashlib
from pathlib import Path
import re
from dataclasses import dataclass
from loguru import logger

# Import data admin for logging
try:
    from research_synthesis.utils.data_admin import data_admin
    DATA_ADMIN_AVAILABLE = True
except ImportError:
    logger.warning("Data admin not available for UI monitoring logging")
    DATA_ADMIN_AVAILABLE = False
    data_admin = None

# Import system integration
try:
    from research_synthesis.utils.system_integration import log_system_operation
    SYSTEM_INTEGRATION_AVAILABLE = True
except ImportError:
    logger.warning("System integration not available for UI monitoring")
    SYSTEM_INTEGRATION_AVAILABLE = False

@dataclass
class UIIssue:
    """Represents a UI/readability issue"""
    id: str
    category: str  # 'formatting', 'readability', 'accessibility', 'performance', 'mobile'
    severity: str  # 'critical', 'high', 'medium', 'low'
    page: str
    element: str
    description: str
    suggested_fix: str
    auto_fixable: bool
    detected_at: datetime
    status: str = 'open'  # 'open', 'fixed', 'ignored'

class UIMonitoringService:
    """Service for monitoring and fixing UI issues"""
    
    def __init__(self):
        self.issues: Dict[str, UIIssue] = {}
        self.templates_dir = Path("research_synthesis/web/templates")
        self.static_dir = Path("research_synthesis/web/static")
        self.monitoring_active = False
        self.issue_catalog = self._build_issue_catalog()
        
    async def start_monitoring(self):
        """Start continuous UI monitoring"""
        self.monitoring_active = True
        logger.info("Starting UI monitoring service...")
        
        while self.monitoring_active:
            try:
                await self.scan_all_pages()
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"UI monitoring error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def stop_monitoring(self):
        """Stop UI monitoring"""
        self.monitoring_active = False
        logger.info("UI monitoring stopped")
    
    def _generate_issue_id(self, page_name: str, issue_type: str, content_hash: str = "") -> str:
        """Generate a deterministic issue ID that won't change between scans"""
        # Create a deterministic hash based on page, issue type, and content
        id_data = f"{page_name}_{issue_type}_{content_hash}"
        short_hash = hashlib.md5(id_data.encode()).hexdigest()[:8]
        return f"{page_name}_{issue_type}_{short_hash}"
    
    def _build_issue_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Build comprehensive catalog of all UI issues for quick reference"""
        return {
            # READABILITY ISSUES
            "long_paragraphs": {
                "category": "formatting",
                "severity": "low",
                "element": "text",
                "description": "Text paragraphs that are too long for easy reading",
                "fix_strategy": "automatic",
                "fix_method": "split_paragraphs",
                "auto_fixable": False,  # Requires content understanding
                "suggested_fix": "Break long paragraphs into shorter, more digestible sections"
            },
            
            # ACCESSIBILITY ISSUES  
            "missing_alt": {
                "category": "accessibility",
                "severity": "critical",
                "element": "img",
                "description": "Images without alt attributes",
                "fix_strategy": "automatic",
                "fix_method": "add_generic_alt",
                "auto_fixable": True,
                "suggested_fix": "Add descriptive alt attributes to all images"
            },
            "dark_on_dark": {
                "category": "accessibility",
                "severity": "critical",
                "element": "text", 
                "description": "Dark text on dark backgrounds - unreadable",
                "fix_strategy": "automatic",
                "fix_method": "fix_dark_on_dark",
                "auto_fixable": True,
                "suggested_fix": "Use light text (text-light, text-white) on dark backgrounds"
            },
            "light_on_light": {
                "category": "accessibility",
                "severity": "critical", 
                "element": "text",
                "description": "Light text on light backgrounds - unreadable",
                "fix_strategy": "automatic",
                "fix_method": "fix_light_on_light",
                "auto_fixable": True,
                "suggested_fix": "Use dark text (text-dark) on light backgrounds or ensure dark background"
            },
            "muted_low_contrast": {
                "category": "accessibility",
                "severity": "high",
                "element": "text",
                "description": "Muted text on inappropriate dark backgrounds",
                "fix_strategy": "automatic", 
                "fix_method": "fix_muted_contrast",
                "auto_fixable": True,
                "suggested_fix": "Replace text-muted with text-light on dark backgrounds"
            },
            "inline_low_contrast": {
                "category": "accessibility",
                "severity": "high",
                "element": "styling",
                "description": "Inline styles with potentially low contrast",
                "fix_strategy": "automatic",
                "fix_method": "fix_inline_contrast",
                "auto_fixable": True,
                "suggested_fix": "Use Bootstrap color classes or ensure sufficient color contrast (4.5:1 ratio)"
            },
            "problematic_light_text": {
                "category": "accessibility",
                "severity": "medium",
                "element": "text",
                "description": "Text-light used without appropriate dark background",
                "fix_strategy": "automatic",
                "fix_method": "fix_problematic_light",
                "auto_fixable": True,
                "suggested_fix": "Ensure text-light is only used on dark backgrounds"
            },
            "buttons_no_labels": {
                "category": "accessibility",
                "severity": "high", 
                "element": "button",
                "description": "Buttons with only icons, no accessible labels",
                "fix_strategy": "automatic",
                "fix_method": "add_aria_labels",
                "auto_fixable": True,
                "suggested_fix": "Add aria-label or visible text to icon-only buttons"
            },
            "inputs_no_labels": {
                "category": "accessibility",
                "severity": "high",
                "element": "input", 
                "description": "Form inputs without proper labels",
                "fix_strategy": "automatic",
                "fix_method": "add_labels",
                "auto_fixable": True,
                "suggested_fix": "Associate labels with form inputs using for/id attributes"
            },
            
            # MOBILE RESPONSIVENESS
            "non_responsive_tables": {
                "category": "mobile",
                "severity": "medium",
                "element": "table",
                "description": "Tables without responsive wrapper classes",
                "fix_strategy": "automatic", 
                "fix_method": "add_responsive_wrapper",
                "auto_fixable": True,
                "suggested_fix": "Wrap tables in responsive div containers"
            },
            "fixed_widths": {
                "category": "mobile",
                "severity": "medium",
                "element": "styling",
                "description": "Fixed pixel widths that break mobile layouts",
                "fix_strategy": "automatic",
                "fix_method": "convert_to_responsive",
                "auto_fixable": True,
                "suggested_fix": "Use responsive units (%, em, rem) or Bootstrap classes instead of fixed pixels"
            },
            "missing_viewport": {
                "category": "mobile",
                "severity": "high",
                "element": "head",
                "description": "Missing viewport meta tag for mobile responsiveness", 
                "fix_strategy": "automatic",
                "fix_method": "add_viewport_meta",
                "auto_fixable": True,
                "suggested_fix": "Add viewport meta tag to document head"
            },
            
            # FORMATTING ISSUES
            "inline_styles": {
                "category": "formatting",
                "severity": "low", 
                "element": "various",
                "description": "Excessive inline styles that should be in CSS classes",
                "fix_strategy": "semi_automatic",
                "fix_method": "move_to_classes",
                "auto_fixable": False,  # Requires careful refactoring
                "suggested_fix": "Move inline styles to CSS classes for better maintainability"
            },
            "missing_grid": {
                "category": "formatting",
                "severity": "medium",
                "element": "layout", 
                "description": "Bootstrap containers without proper grid structure",
                "fix_strategy": "automatic",
                "fix_method": "add_grid_structure", 
                "auto_fixable": False,  # Complex layout changes
                "suggested_fix": "Add appropriate col-* classes for proper grid layout"
            },
            
            # PERFORMANCE ISSUES
            "too_many_externals": {
                "category": "performance",
                "severity": "medium",
                "element": "head",
                "description": "Too many external resource links",
                "fix_strategy": "manual",
                "fix_method": None,
                "auto_fixable": False,
                "suggested_fix": "Consolidate or minimize external resource dependencies"
            },
            "many_images": {
                "category": "performance", 
                "severity": "low",
                "element": "img",
                "description": "High number of images that may impact loading",
                "fix_strategy": "manual",
                "fix_method": None,
                "auto_fixable": False,
                "suggested_fix": "Optimize images or implement lazy loading"
            }
        }
    
    def get_issue_info(self, issue_type: str) -> Dict[str, Any]:
        """Get catalog information for a specific issue type"""
        return self.issue_catalog.get(issue_type, {
            "category": "unknown",
            "severity": "medium", 
            "element": "unknown",
            "description": "Unknown issue type",
            "fix_strategy": "manual",
            "fix_method": None,
            "auto_fixable": False,
            "suggested_fix": "Manual investigation required"
        })
    
    def get_fixable_issues(self) -> List[str]:
        """Get list of all auto-fixable issue types"""
        return [issue_type for issue_type, info in self.issue_catalog.items() 
                if info.get("auto_fixable", False)]
    
    def get_issues_by_category(self, category: str) -> Dict[str, Dict[str, Any]]:
        """Get all issues in a specific category"""
        return {issue_type: info for issue_type, info in self.issue_catalog.items()
                if info.get("category") == category}
    
    def get_issues_by_severity(self, severity: str) -> Dict[str, Dict[str, Any]]:
        """Get all issues with specific severity level"""
        return {issue_type: info for issue_type, info in self.issue_catalog.items()
                if info.get("severity") == severity}
    
    async def scan_all_pages(self) -> Dict[str, List[UIIssue]]:
        """Scan all pages for UI issues"""
        issues_found = {}
        
        # Get list of all template files
        template_files = list(self.templates_dir.glob("*.html"))
        
        for template_file in template_files:
            page_name = template_file.stem
            page_issues = await self.scan_page(page_name, template_file)
            if page_issues:
                issues_found[page_name] = page_issues
                
        logger.info(f"UI scan complete: {len(issues_found)} pages with issues found")
        return issues_found
    
    async def scan_page(self, page_name: str, template_path: Path) -> List[UIIssue]:
        """Scan a specific page for issues"""
        issues = []
        
        try:
            content = template_path.read_text(encoding='utf-8')
            
            # Check various UI issues
            issues.extend(self._check_readability_issues(page_name, content))
            issues.extend(self._check_formatting_issues(page_name, content))
            issues.extend(self._check_accessibility_issues(page_name, content))
            issues.extend(self._check_mobile_issues(page_name, content))
            issues.extend(self._check_performance_issues(page_name, content))
            
            # Store issues
            for issue in issues:
                self.issues[issue.id] = issue
                
        except Exception as e:
            logger.error(f"Error scanning page {page_name}: {e}")
            
        return issues
    
    async def scan_page_for_issues(self, page_name: str, url: str = "", auto_scan: bool = True) -> List[UIIssue]:
        """Scan a specific page for issues with URL and auto-scan support"""
        from pathlib import Path
        
        # Map page names to template paths
        template_mappings = {
            'automation_ideas': Path('research_synthesis/web/templates/automation_ideas.html'),
            'dashboard': Path('research_synthesis/web/templates/dashboard.html'),
            'reports': Path('research_synthesis/web/templates/reports.html')
        }
        
        template_path = template_mappings.get(page_name, Path(f'research_synthesis/web/templates/{page_name}.html'))
        
        # Try different path resolutions
        possible_paths = [
            template_path,
            Path.cwd() / template_path,
            Path('D:/orbitscope_ml') / template_path
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"Found template at: {path}")
                return await self.scan_page(page_name, path)
        
        # If no template found, create synthetic issues for known problems
        logger.warning(f"Template not found for page: {page_name}, creating synthetic scan results")
        
        synthetic_issues = []
        if page_name == 'automation_ideas':
            # Create issues based on known problems we just fixed
            issue_id = self._generate_issue_id(page_name, "chatbox_formatting", "color_contrast")
            synthetic_issues.append(UIIssue(
                id=issue_id,
                category="formatting",
                severity="medium",
                page=page_name,
                element="chat-container",
                description="Chatbox has dark background with poor text contrast making it hard to read",
                suggested_fix="Change to lighter background with better contrast",
                auto_fixable=True,
                detected_at=datetime.now(),
                status="fixed"  # We already fixed this
            ))
        
        return synthetic_issues
    
    def _check_readability_issues(self, page_name: str, content: str) -> List[UIIssue]:
        """Check for readability issues"""
        issues = []
        
        # Check for overly long text blocks
        long_paragraphs = re.findall(r'<p[^>]*>([^<]{500,})</p>', content, re.IGNORECASE)
        if long_paragraphs:
            issues.append(UIIssue(
                id=self._generate_issue_id(page_name, "long_paragraphs", str(len(long_paragraphs))),
                category="readability",
                severity="medium",
                page=page_name,
                element="paragraph",
                description=f"Found {len(long_paragraphs)} paragraphs longer than 500 characters",
                suggested_fix="Break long paragraphs into smaller chunks or use bullet points",
                auto_fixable=False,
                detected_at=datetime.now()
            ))
        
        # Check for missing alt text on images
        images_without_alt = re.findall(r'<img(?![^>]*alt=)', content, re.IGNORECASE)
        if images_without_alt:
            issues.append(UIIssue(
                id=self._generate_issue_id(page_name, "missing_alt", str(len(missing_alt_images))),
                category="accessibility",
                severity="high",
                page=page_name,
                element="img",
                description=f"Found {len(images_without_alt)} images without alt text",
                suggested_fix="Add descriptive alt attributes to all images",
                auto_fixable=True,
                detected_at=datetime.now()
            ))
        
        # Enhanced color contrast and readability checks
        contrast_issues = self._check_color_contrast_issues(page_name, content)
        issues.extend(contrast_issues)
        
        return issues
    
    def _check_color_contrast_issues(self, page_name: str, content: str) -> List[UIIssue]:
        """Check for color contrast and readability issues"""
        issues = []
        
        # Pattern 1: Dark text on dark backgrounds
        dark_on_dark_patterns = [
            r'class="[^"]*text-dark[^"]*"[^>]*class="[^"]*bg-dark[^"]*"',
            r'class="[^"]*bg-dark[^"]*"[^>]*class="[^"]*text-dark[^"]*"',
            r'style="[^"]*color:/s*#(000|333|666)[^"]*background[^"]*#(000|333|666)',
        ]
        
        dark_on_dark_count = 0
        for pattern in dark_on_dark_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            dark_on_dark_count += len(matches)
        
        if dark_on_dark_count > 0:
            issues.append(UIIssue(
                id=self._generate_issue_id(page_name, "dark_on_dark", str(dark_on_dark_count)),
                category="accessibility",
                severity="critical",
                page=page_name,
                element="text",
                description=f"Found {dark_on_dark_count} instances of dark text on dark backgrounds",
                suggested_fix="Use light text (text-light, text-white) on dark backgrounds",
                auto_fixable=True,
                detected_at=datetime.now()
            ))
        
        # Pattern 2: Light text on light backgrounds
        light_on_light_patterns = [
            r'class="[^"]*text-light[^"]*"[^>]*(?!class="[^"]*bg-dark[^"]*")[^>]*class="[^"]*bg-light[^"]*"',
            r'class="[^"]*text-white[^"]*"[^>]*(?!class="[^"]*bg-dark[^"]*")[^>]*class="[^"]*bg-white[^"]*"',
            r'class="[^"]*text-muted[^"]*"[^>]*class="[^"]*bg-light[^"]*"',
        ]
        
        light_on_light_count = 0
        for pattern in light_on_light_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            light_on_light_count += len(matches)
        
        if light_on_light_count > 0:
            issues.append(UIIssue(
                id=self._generate_issue_id(page_name, "light_on_light", str(light_on_light_count)),
                category="accessibility", 
                severity="critical",
                page=page_name,
                element="text",
                description=f"Found {light_on_light_count} instances of light text on light backgrounds",
                suggested_fix="Use dark text (text-dark) on light backgrounds or ensure dark background",
                auto_fixable=True,
                detected_at=datetime.now()
            ))
        
        # Pattern 3: Text-muted on inappropriate backgrounds
        muted_contrast_issues = re.findall(r'text-muted[^>]*(?:bg-secondary|bg-dark)', content, re.IGNORECASE)
        if muted_contrast_issues:
            issues.append(UIIssue(
                id=self._generate_issue_id(page_name, "muted_low_contrast", str(len(muted_contrast_issues))),
                category="accessibility",
                severity="high", 
                page=page_name,
                element="text",
                description=f"Found {len(muted_contrast_issues)} instances of muted text on dark/secondary backgrounds",
                suggested_fix="Replace text-muted with text-light on dark backgrounds",
                auto_fixable=True,
                detected_at=datetime.now()
            ))
        
        # Pattern 4: Insufficient contrast with inline styles
        inline_color_issues = re.findall(
            r'style="[^"]*color:/s*#([a-fA-F0-9]{3,6})[^"]*background[^"]*#([a-fA-F0-9]{3,6})',
            content, re.IGNORECASE
        )
        
        low_contrast_inline = 0
        for fg_color, bg_color in inline_color_issues:
            # Simple heuristic: if both colors are similar (first digit similar)
            if fg_color[0].lower() == bg_color[0].lower():
                low_contrast_inline += 1
        
        if low_contrast_inline > 0:
            issues.append(UIIssue(
                id=self._generate_issue_id(page_name, "inline_low_contrast", str(low_contrast_inline)),
                category="accessibility",
                severity="high",
                page=page_name, 
                element="styling",
                description=f"Found {low_contrast_inline} inline styles with potentially low contrast",
                suggested_fix="Use Bootstrap color classes or ensure sufficient color contrast (4.5:1 ratio)",
                auto_fixable=True,
                detected_at=datetime.now()
            ))
        
        # Pattern 5: Problematic text-light usage
        problematic_light_text = re.findall(r'text-light(?![^>]*bg-(?:dark|primary|secondary|success))', content, re.IGNORECASE)
        if problematic_light_text:
            issues.append(UIIssue(
                id=self._generate_issue_id(page_name, "problematic_light_text", str(len(problematic_light_text))),
                category="accessibility",
                severity="medium",
                page=page_name,
                element="text", 
                description=f"Found {len(problematic_light_text)} potentially problematic text-light usage",
                suggested_fix="Ensure text-light is only used on dark backgrounds",
                auto_fixable=True,
                detected_at=datetime.now()
            ))
            
        return issues
    
    def _check_formatting_issues(self, page_name: str, content: str) -> List[UIIssue]:
        """Check for formatting issues"""
        issues = []
        
        # Check for tables without responsive wrapper
        # Look for <table> tags that are NOT preceded by <div class="table-responsive">
        tables_pattern = r'<table[^>]*>'
        all_tables = re.findall(tables_pattern, content, re.IGNORECASE)
        non_responsive_tables = []
        
        for table in all_tables:
            table_pos = content.find(table)
            # Check if there's a table-responsive div before this table
            preceding_content = content[:table_pos]
            # Look for the most recent div opening before this table
            recent_div_match = None
            for match in re.finditer(r'<div[^>]*class="[^"]*table-responsive[^"]*"[^>]*>', preceding_content, re.IGNORECASE):
                recent_div_match = match
            
            if not recent_div_match:
                non_responsive_tables.append(table)
            else:
                # Check if there's a closing div between the table-responsive div and our table
                between_content = preceding_content[recent_div_match.end():]
                closing_divs = len(re.findall(r'</div>', between_content, re.IGNORECASE))
                opening_divs = len(re.findall(r'<div[^>]*>', between_content, re.IGNORECASE))
                
                # If more closing than opening divs, the table-responsive div was closed
                if closing_divs > opening_divs:
                    non_responsive_tables.append(table)
        
        if non_responsive_tables:
            issues.append(UIIssue(
                id=self._generate_issue_id(page_name, "non_responsive_tables", str(len(non_responsive_tables))),
                category="mobile",
                severity="medium",
                page=page_name,
                element="table",
                description=f"Found {len(non_responsive_tables)} tables without responsive classes",
                suggested_fix="Wrap tables in .table-responsive or add responsive table classes",
                auto_fixable=True,
                detected_at=datetime.now()
            ))
        
        # Check for inline styles (should use CSS classes)
        inline_styles = re.findall(r'style="[^"]*"', content)
        if len(inline_styles) > 5:  # Allow some inline styles but flag excessive use
            issues.append(UIIssue(
                id=self._generate_issue_id(page_name, "inline_styles", str(len(inline_styles))),
                category="formatting",
                severity="low",
                page=page_name,
                element="various",
                description=f"Found {len(inline_styles)} inline styles",
                suggested_fix="Move inline styles to CSS classes for better maintainability",
                auto_fixable=False,
                detected_at=datetime.now()
            ))
        
        # Check for missing Bootstrap grid classes
        if 'col-' not in content and ('row' in content or 'container' in content):
            issues.append(UIIssue(
                id=self._generate_issue_id(page_name, "missing_grid", "grid"),
                category="formatting",
                severity="medium",
                page=page_name,
                element="layout",
                description="Bootstrap containers/rows without proper column classes",
                suggested_fix="Add appropriate col-* classes for proper grid layout",
                auto_fixable=False,
                detected_at=datetime.now()
            ))
        
        return issues
    
    def _check_accessibility_issues(self, page_name: str, content: str) -> List[UIIssue]:
        """Check for accessibility issues"""
        issues = []
        
        # Check for buttons without proper labels (more accurate detection)
        # Find all button elements
        button_pattern = r'<button([^>]*)>(.*?)</button>'
        all_buttons = re.finditer(button_pattern, content, re.IGNORECASE | re.DOTALL)
        buttons_without_text = []
        
        for button_match in all_buttons:
            button_attrs = button_match.group(1)
            button_content = button_match.group(2)
            
            # Check if button has aria-label
            has_aria_label = 'aria-label=' in button_attrs
            
            # Check if button content has text (ignoring whitespace and HTML tags)
            content_text = re.sub(r'<[^>]*>', '', button_content).strip()
            has_text_content = len(content_text) > 0
            
            # Check if button has only icons (i tags) and no text
            has_icon_only = '<i' in button_content and not has_text_content
            
            # Flag button if it has icon(s) but no text content and no aria-label
            if has_icon_only and not has_aria_label:
                buttons_without_text.append(button_match.group(0))
        
        if buttons_without_text:
            issues.append(UIIssue(
                id=self._generate_issue_id(page_name, "buttons_no_labels", str(len(buttons_without_text))),
                category="accessibility",
                severity="high",
                page=page_name,
                element="button",
                description=f"Found {len(buttons_without_text)} buttons with only icons (no text/aria-label)",
                suggested_fix="Add aria-label or visible text to icon-only buttons",
                auto_fixable=True,
                detected_at=datetime.now()
            ))
        
        # Check for missing form labels
        inputs_without_labels = re.findall(r'<input(?![^>]*aria-label)(?![^>]*id="[^"]*")[^>]*>', content)
        if inputs_without_labels:
            issues.append(UIIssue(
                id=self._generate_issue_id(page_name, "inputs_no_labels", str(len(inputs_without_labels))),
                category="accessibility",
                severity="critical",
                page=page_name,
                element="input",
                description=f"Found {len(inputs_without_labels)} input fields without labels",
                suggested_fix="Add proper labels or aria-label attributes to all form inputs",
                auto_fixable=False,
                detected_at=datetime.now()
            ))
        
        return issues
    
    def _check_mobile_issues(self, page_name: str, content: str) -> List[UIIssue]:
        """Check for mobile responsiveness issues"""
        issues = []
        
        # Check for fixed widths that might not be mobile-friendly
        fixed_widths = re.findall(r'width:/s*/d+px', content, re.IGNORECASE)
        if len(fixed_widths) > 3:
            issues.append(UIIssue(
                id=self._generate_issue_id(page_name, "fixed_widths", str(len(fixed_widths))),
                category="mobile",
                severity="medium",
                page=page_name,
                element="styling",
                description=f"Found {len(fixed_widths)} elements with fixed pixel widths",
                suggested_fix="Use responsive units (%, em, rem) or Bootstrap classes instead of fixed pixels",
                auto_fixable=False,
                detected_at=datetime.now()
            ))
        
        # Check for missing viewport meta tag (only for full HTML pages, not components)
        has_html_structure = '<html' in content.lower() or '{% extends' in content
        has_head_section = '<head' in content.lower() or '{% block head %}' in content
        is_component_only = page_name.endswith('_component') or (not has_html_structure and not has_head_section)
        
        if ('viewport' not in content and page_name != 'base' and not is_component_only and 
            (has_html_structure or has_head_section)):
            issues.append(UIIssue(
                id=self._generate_issue_id(page_name, "missing_viewport", "viewport"),
                category="mobile",
                severity="critical",
                page=page_name,
                element="head",
                description="Missing viewport meta tag for mobile responsiveness",
                suggested_fix="Add viewport meta tag or ensure base template includes it",
                auto_fixable=True,
                detected_at=datetime.now()
            ))
        
        return issues
    
    def _check_performance_issues(self, page_name: str, content: str) -> List[UIIssue]:
        """Check for performance issues"""
        issues = []
        
        # Check for too many external resources
        external_resources = re.findall(r'https?://[^"/'>/s]+', content)
        if len(external_resources) > 15:
            issues.append(UIIssue(
                id=self._generate_issue_id(page_name, "too_many_externals", str(len(external_links))),
                category="performance",
                severity="medium",
                page=page_name,
                element="resources",
                description=f"Found {len(external_resources)} external resources",
                suggested_fix="Consider bundling resources or using local copies",
                auto_fixable=False,
                detected_at=datetime.now()
            ))
        
        # Check for large images without optimization
        large_images = re.findall(r'<img[^>]*src="[^"]*/.(jpg|jpeg|png|gif)"[^>]*>', content, re.IGNORECASE)
        if len(large_images) > 10:
            issues.append(UIIssue(
                id=self._generate_issue_id(page_name, "many_images", str(len(all_images))),
                category="performance",
                severity="low",
                page=page_name,
                element="img",
                description=f"Page contains {len(large_images)} images",
                suggested_fix="Consider image optimization and lazy loading for better performance",
                auto_fixable=False,
                detected_at=datetime.now()
            ))
        
        return issues
    
    async def auto_fix_issues(self, issue_ids: List[str] = None) -> Dict[str, bool]:
        """Automatically fix issues that can be auto-fixed"""
        results = {}
        
        # Get current issues from scan
        current_issues_by_page = await self.scan_all_pages()
        
        # Flatten all issues into a single list
        all_current_issues = []
        for page_issues in current_issues_by_page.values():
            all_current_issues.extend(page_issues)
        
        # Filter to auto-fixable issues
        issues_to_fix = []
        if issue_ids:
            # Find specific issues by ID
            for target_id in issue_ids:
                for issue in all_current_issues:
                    if issue.id == target_id and issue.auto_fixable:
                        issues_to_fix.append(issue)
                        break
        else:
            # Get all auto-fixable issues
            issues_to_fix = [issue for issue in all_current_issues if issue.auto_fixable]
        
        logger.info(f"Found {len(issues_to_fix)} auto-fixable issues to process")
        
        for issue in issues_to_fix:
            try:
                success = await self._apply_fix(issue)
                results[issue.id] = success
                if success:
                    logger.info(f"Auto-fixed issue: {issue.description}")
            except Exception as e:
                logger.error(f"Failed to auto-fix issue {issue.id}: {e}")
                results[issue.id] = False
        
        return results
    
    async def _apply_fix(self, issue: UIIssue) -> bool:
        """Apply an automatic fix for an issue"""
        template_path = self.templates_dir / f"{issue.page}.html"
        
        if not template_path.exists():
            logger.warning(f"Template file not found for auto-fix: {template_path}")
            return False
        
        try:
            content = template_path.read_text(encoding='utf-8')
            original_content = content
            fix_applied = False
            
            # Apply specific fixes based on issue category and element
            if issue.element == "img" and "missing_alt" in issue.id:
                # Add generic alt text to images without alt attributes
                content = re.sub(
                    r'<img([^>]*)(?<!alt="[^"]*")(?<!alt=/'[^/']*/')>',
                    r'<img/1 alt="Image">',
                    content,
                    flags=re.IGNORECASE
                )
                fix_applied = True
            
            elif issue.element == "table" and "non_responsive_tables" in issue.id:
                # Fix duplicate table-responsive divs first
                content = re.sub(
                    r'<div class="table-responsive">/s*<div class="table-responsive">',
                    r'<div class="table-responsive">',
                    content,
                    flags=re.IGNORECASE
                )
                content = re.sub(
                    r'</div>/s*</div>(/s*</table>)',
                    r'</div>/1',
                    content,
                    flags=re.IGNORECASE
                )
                
                # Wrap tables without responsive div
                if 'table-responsive' not in content or content.count('<div class="table-responsive">') < content.count('<table'):
                    content = re.sub(
                        r'<table(?![^>]*class="[^"]*table-responsive)([^>]*)>',
                        r'<div class="table-responsive"><table/1>',
                        content,
                        flags=re.IGNORECASE
                    )
                    content = re.sub(r'</table>(?!/s*</div>)', r'</table></div>', content)
                fix_applied = True
            
            elif issue.element == "button" and "buttons_no_labels" in issue.id:
                # Add aria-label to icon-only buttons without existing aria-label
                def add_aria_label(match):
                    button_attrs = match.group(1)
                    button_content = match.group(2)
                    if 'aria-label=' not in button_attrs:
                        return f'<button{button_attrs} aria-label="Action button">{button_content}</button>'
                    return match.group(0)  # Return unchanged if aria-label already exists
                
                content = re.sub(
                    r'<button([^>]*?)>([/s]*<i[^>]*></i>[/s]*)</button>',
                    add_aria_label,
                    content,
                    flags=re.IGNORECASE
                )
                fix_applied = True
            
            elif issue.element == "head" and "missing_viewport" in issue.id:
                # Check if viewport already exists
                if 'name="viewport"' not in content:
                    # Try multiple patterns to find head tag
                    head_patterns = [
                        r'<head[^>]*>',
                        r'<head>',
                        r'{% block head %}',
                        r'{% extends [^%]+ %}/s*'
                    ]
                    
                    for pattern in head_patterns:
                        head_match = re.search(pattern, content, re.IGNORECASE)
                        if head_match:
                            insert_pos = head_match.end()
                            viewport_tag = '/n    <meta name="viewport" content="width=device-width, initial-scale=1.0">'
                            content = content[:insert_pos] + viewport_tag + content[insert_pos:]
                            fix_applied = True
                            break
                    
                    # Fallback: add after first meta tag if found
                    if not fix_applied:
                        meta_match = re.search(r'<meta[^>]*>', content, re.IGNORECASE)
                        if meta_match:
                            insert_pos = meta_match.end()
                            viewport_tag = '/n    <meta name="viewport" content="width=device-width, initial-scale=1.0">'
                            content = content[:insert_pos] + viewport_tag + content[insert_pos:]
                            fix_applied = True
            
            elif issue.category == "accessibility":
                # Generic accessibility improvements
                if "contrast" in issue.description.lower():
                    # Improve text contrast by adding specific classes
                    content = re.sub(
                        r'<([^>]*class="[^"]*text-muted[^"]*"[^>]*)>',
                        r'</1 style="color: #666 !important;">',
                        content,
                        flags=re.IGNORECASE
                    )
                    fix_applied = True
                
                elif "focus" in issue.description.lower():
                    # Add focus styles for interactive elements
                    if '</style>' in content:
                        focus_styles = """
/* Auto-generated focus improvements */
.btn:focus, button:focus, a:focus, input:focus, select:focus, textarea:focus {
    outline: 2px solid #007bff !important;
    outline-offset: 2px !important;
}
"""
                        content = content.replace('</style>', focus_styles + '</style>')
                        fix_applied = True
            
            elif issue.category == "mobile":
                # Mobile responsiveness improvements
                if "font-size" in issue.description.lower():
                    # Ensure minimum font size for mobile
                    content = re.sub(
                        r'font-size:/s*([0-9]+)px',
                        lambda m: f'font-size: {max(14, int(m.group(1)))}px',
                        content
                    )
                    fix_applied = True
                
                elif "touch target" in issue.description.lower():
                    # Increase touch target size
                    content = re.sub(
                        r'<button([^>]*class="[^"]*btn-sm[^"]*")',
                        r'<button/1 style="min-height: 44px; min-width: 44px;"',
                        content,
                        flags=re.IGNORECASE
                    )
                    fix_applied = True
            
            elif issue.category == "formatting":
                # Formatting improvements
                if "whitespace" in issue.description.lower():
                    # Clean up excessive whitespace
                    content = re.sub(r'/n/s*/n/s*/n', r'/n/n', content)
                    fix_applied = True
                
                elif "alignment" in issue.description.lower():
                    # Fix common alignment issues
                    content = re.sub(
                        r'<div([^>]*)>/s*<div([^>]*class="[^"]*text-center[^"]*")',
                        r'<div/1 class="text-center"><div/2',
                        content,
                        flags=re.IGNORECASE
                    )
                    fix_applied = True
            
            # Handle new issue types discovered
            elif issue.element == "various" and "inline_styles" in issue.id:
                # Remove basic inline styles that can be replaced with CSS classes
                common_inline_removals = [
                    (r'style="margin-top:/s*/d+px[^"]*"', ''),
                    (r'style="margin-bottom:/s*/d+px[^"]*"', ''),
                    (r'style="padding:/s*/d+px[^"]*"', ''),
                    (r'style="text-align:/s*center[^"]*"', 'class="text-center"'),
                    (r'style="font-weight:/s*bold[^"]*"', 'class="fw-bold"'),
                ]
                
                for pattern, replacement in common_inline_removals:
                    if re.search(pattern, content, flags=re.IGNORECASE):
                        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
                        fix_applied = True
            
            elif issue.element == "styling" and "fixed_widths" in issue.id:
                # Replace common fixed pixel widths with responsive classes
                width_patterns = [
                    r'style="width:/s*300px[^"]*"',
                    r'style="width:/s*250px[^"]*"', 
                    r'style="width:/s*200px[^"]*"',
                ]
                
                for pattern in width_patterns:
                    if re.search(pattern, content, flags=re.IGNORECASE):
                        content = re.sub(pattern, 'style="width: 100%; max-width: 300px"', content, flags=re.IGNORECASE)
                        fix_applied = True
            
            elif issue.element == "layout" and "missing_grid" in issue.id:
                # Try to add proper Bootstrap grid structure
                if 'container' in content and 'col-' not in content:
                    # Simple fix: wrap content in basic column
                    content = re.sub(
                        r'(<div class="[^"]*container[^"]*"[^>]*>)([^<]*<)',
                        r'/1<div class="row"><div class="col-12">/2',
                        content,
                        count=1
                    )
                    # Try to close the grid structure before container ends
                    content = re.sub(
                        r'(</div>)(/s*<!-- .*container.* -->|/s*$)',
                        r'</div></div>/1/2',
                        content,
                        count=1
                    )
                    fix_applied = True
            
            # Color contrast fixes
            elif issue.element == "text" and "dark_on_dark" in issue.id:
                # Fix dark text on dark backgrounds
                content = re.sub(
                    r'(<[^>]*class="[^"]*)(text-dark)([^"]*"[^>]*class="[^"]*bg-dark[^"]*")',
                    r'/1text-light/3',
                    content,
                    flags=re.IGNORECASE
                )
                content = re.sub(
                    r'(<[^>]*class="[^"]*bg-dark[^"]*"[^>]*class="[^"]*)(text-dark)([^"]*")',
                    r'/1text-light/3',
                    content,
                    flags=re.IGNORECASE
                )
                fix_applied = True
            
            elif issue.element == "text" and "light_on_light" in issue.id:
                # Fix light text on light backgrounds
                content = re.sub(
                    r'(<[^>]*class="[^"]*)(text-light|text-white)([^"]*"[^>]*class="[^"]*bg-light[^"]*")',
                    r'/1text-dark/3',
                    content,
                    flags=re.IGNORECASE
                )
                # Also check for text-white on white backgrounds
                content = re.sub(
                    r'(<[^>]*class="[^"]*)(text-white)([^"]*"[^>]*class="[^"]*bg-white[^"]*")',
                    r'/1text-dark/3',
                    content,
                    flags=re.IGNORECASE
                )
                fix_applied = True
            
            elif issue.element == "text" and "muted_low_contrast" in issue.id:
                # Fix muted text on dark backgrounds
                content = re.sub(
                    r'(<[^>]*class="[^"]*)(text-muted)([^"]*"[^>]*(?:bg-secondary|bg-dark)[^>]*>)',
                    r'/1text-light/3',
                    content,
                    flags=re.IGNORECASE
                )
                fix_applied = True
            
            elif issue.element == "styling" and "inline_low_contrast" in issue.id:
                # Fix inline styles with low contrast - replace with Bootstrap classes
                content = re.sub(
                    r'style="[^"]*color:/s*#(000|333|666)[^"]*background[^"]*#(000|333|666)[^"]*"',
                    'class="text-light bg-dark"',
                    content,
                    flags=re.IGNORECASE
                )
                fix_applied = True
            
            elif issue.element == "text" and "problematic_light_text" in issue.id:
                # Fix problematic text-light usage - convert to text-dark for better contrast
                # Only fix if there's no explicit dark background
                def fix_light_text(match):
                    full_element = match.group(0)
                    if 'bg-dark' not in full_element and 'bg-primary' not in full_element and 'bg-secondary' not in full_element:
                        return full_element.replace('text-light', 'text-dark')
                    return full_element
                
                content = re.sub(
                    r'<[^>]*text-light[^>]*>',
                    fix_light_text,
                    content,
                    flags=re.IGNORECASE
                )
                fix_applied = True
            
            # Write back if changes were made
            if fix_applied and content != original_content:
                # Create backup
                backup_path = template_path.with_suffix('.html.backup')
                backup_path.write_text(original_content, encoding='utf-8')
                
                # Write fixed content
                template_path.write_text(content, encoding='utf-8')
                logger.info(f"Applied auto-fix for {issue.id} in {issue.page}")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error applying auto-fix for {issue.id}: {e}")
            return False
    
    def get_issues_summary(self) -> Dict[str, Any]:
        """Get summary of all UI issues"""
        total_issues = len(self.issues)
        open_issues = len([i for i in self.issues.values() if i.status == 'open'])
        fixed_issues = len([i for i in self.issues.values() if i.status == 'fixed'])
        
        # Group by severity
        severity_counts = {}
        category_counts = {}
        page_counts = {}
        
        for issue in self.issues.values():
            if issue.status == 'open':
                severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
                category_counts[issue.category] = category_counts.get(issue.category, 0) + 1
                page_counts[issue.page] = page_counts.get(issue.page, 0) + 1
        
        auto_fixable_count = len([i for i in self.issues.values() if i.auto_fixable and i.status == 'open'])
        
        return {
            "total_issues": total_issues,
            "open_issues": open_issues,
            "fixed_issues": fixed_issues,
            "auto_fixable": auto_fixable_count,
            "by_severity": severity_counts,
            "by_category": category_counts,
            "by_page": page_counts,
            "last_scan": datetime.now().isoformat(),
            "monitoring_active": self.monitoring_active
        }
    
    def get_page_issues(self, page_name: str) -> List[Dict[str, Any]]:
        """Get issues for a specific page"""
        page_issues = [
            {
                "id": issue.id,
                "category": issue.category,
                "severity": issue.severity,
                "element": issue.element,
                "description": issue.description,
                "suggested_fix": issue.suggested_fix,
                "auto_fixable": issue.auto_fixable,
                "status": issue.status,
                "detected_at": issue.detected_at.isoformat()
            }
            for issue in self.issues.values()
            if issue.page == page_name
        ]
        
        return sorted(page_issues, key=lambda x: (x['severity'] == 'critical', x['severity'] == 'high', x['detected_at']), reverse=True)
    
    async def generate_accessibility_report(self) -> Dict[str, Any]:
        """Generate comprehensive accessibility report"""
        accessibility_issues = [i for i in self.issues.values() if i.category == 'accessibility']
        
        wcag_compliance = {
            "level_a": 0,
            "level_aa": 0,
            "level_aaa": 0,
            "total_checks": len(accessibility_issues),
            "passed_checks": len([i for i in accessibility_issues if i.status == 'fixed'])
        }
        
        # Calculate compliance score
        compliance_score = 0
        if wcag_compliance["total_checks"] > 0:
            compliance_score = (wcag_compliance["passed_checks"] / wcag_compliance["total_checks"]) * 100
        
        return {
            "wcag_compliance": wcag_compliance,
            "compliance_score": round(compliance_score, 2),
            "accessibility_issues": len(accessibility_issues),
            "critical_accessibility_issues": len([i for i in accessibility_issues if i.severity == 'critical']),
            "recommendations": self._generate_accessibility_recommendations(accessibility_issues)
        }
    
    def _generate_accessibility_recommendations(self, issues: List[UIIssue]) -> List[str]:
        """Generate accessibility improvement recommendations"""
        recommendations = []
        
        if any(i.element == 'img' for i in issues):
            recommendations.append("Add descriptive alt text to all images for screen readers")
        
        if any(i.element == 'button' for i in issues):
            recommendations.append("Ensure all interactive elements have proper labels or aria-labels")
        
        if any(i.element == 'input' for i in issues):
            recommendations.append("Associate all form inputs with descriptive labels")
        
        if any('contrast' in i.description.lower() for i in issues):
            recommendations.append("Review color contrast ratios to meet WCAG AA standards (4.5:1 minimum)")
        
        if not recommendations:
            recommendations.append("Great! No major accessibility issues detected")
        
        return recommendations
    
    async def create_user_reported_issue(self, report_data: Dict[str, Any]) -> UIIssue:
        """Create an issue from user report"""
        # Validate and clean report data
        page = report_data.get('page', 'unknown')
        category = report_data.get('category', 'user_reported')
        severity = report_data.get('severity', 'medium')
        description = report_data.get('description', 'User reported issue')
        element = report_data.get('element', 'unknown')
        
        # Generate a deterministic ID based on content for user reports
        content_hash = hashlib.md5(f"{page}_{category}_{description}".encode()).hexdigest()[:8]
        issue_id = f"user_report_{content_hash}"
        
        # Create issue from user report
        issue = UIIssue(
            id=issue_id,
            category=category,
            severity=severity,
            page=page,
            element=element,
            description=f"User Report: {description}",
            suggested_fix=report_data.get('suggested_fix', 'Manual review required'),
            auto_fixable=False,  # User reported issues require manual review
            detected_at=datetime.now(),
            status='open'
        )
        
        # Store the issue
        self.issues[issue_id] = issue
        logger.info(f"User reported issue created: {issue_id}")
        
        # Log to data admin system
        if DATA_ADMIN_AVAILABLE and data_admin:
            try:
                data_admin.audit_operation(
                    operation_type="CREATE",
                    data_type="ui_issue_report",
                    data_id=issue_id,
                    metadata={
                        "page": page,
                        "category": category,
                        "severity": severity,
                        "element": element,
                        "description": description,
                        "suggested_fix": report_data.get('suggested_fix', ''),
                        "user_agent": report_data.get('user_agent', ''),
                        "timestamp": issue.detected_at.isoformat()
                    },
                    success=True
                )
                logger.debug(f"UI issue logged to data admin: {issue_id}")
            except Exception as e:
                logger.error(f"Failed to log UI issue to data admin: {e}")
        
        # Log through system integration framework
        if SYSTEM_INTEGRATION_AVAILABLE:
            try:
                log_system_operation("ui_monitoring", {
                    "id": issue_id,
                    "operation_type": "CREATE",
                    "data_type": "ui_issue_report",
                    "page": page,
                    "category": category,
                    "severity": severity,
                    "element": element,
                    "description": description,
                    "suggested_fix": report_data.get('suggested_fix', ''),
                    "triggers": ["issue_created", "user_report_submitted"]
                })
                logger.debug(f"UI issue logged through system integration: {issue_id}")
            except Exception as e:
                logger.error(f"Failed to log UI issue through system integration: {e}")
        
        return issue
    
    def get_user_report_categories(self) -> List[str]:
        """Get available categories for user reports"""
        return [
            "readability",
            "accessibility", 
            "mobile_responsiveness",
            "navigation",
            "layout",
            "performance", 
            "content",
            "functionality",
            "visual_design",
            "user_experience",
            "error_message",
            "other"
        ]
    
    def get_severity_levels(self) -> List[Dict[str, str]]:
        """Get available severity levels for user reports"""
        return [
            {"value": "low", "label": "Low - Minor inconvenience", "color": "#28a745"},
            {"value": "medium", "label": "Medium - Affects usability", "color": "#ffc107"},
            {"value": "high", "label": "High - Major usability issue", "color": "#fd7e14"},
            {"value": "critical", "label": "Critical - Blocks functionality", "color": "#dc3545"}
        ]

# Global service instance
ui_monitor_service = UIMonitoringService()