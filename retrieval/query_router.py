"""
Query router to determine optimal search strategy
"""
import re
from typing import Dict, List
from loguru import logger


class QueryRouter:
    """Analyze queries and determine search strategy"""
    
    def __init__(self):
        # Temporal keywords
        self.temporal_keywords = [
            'when', 'date', 'time', 'yesterday', 'today', 'last', 'recent',
            'before', 'after', 'during', 'ago', 'year', 'month', 'week'
        ]
        
        # Cross-modal keywords
        self.cross_modal_keywords = [
            'image', 'picture', 'photo', 'screenshot', 'chart', 'diagram',
            'audio', 'recording', 'call', 'meeting', 'transcript', 'said'
        ]
        
        # Exact match indicators
        self.exact_keywords = [
            'exactly', 'precise', 'specific', 'exact', 'verbatim'
        ]
    
    def analyze_query(self, query: str) -> Dict:
        """
        Analyze query to determine intent and strategy
        
        Args:
            query: User's query string
            
        Returns:
            Dict with query analysis
        """
        query_lower = query.lower()
        
        analysis = {
            "query": query,
            "is_temporal": self._check_temporal(query_lower),
            "is_cross_modal": self._check_cross_modal(query_lower),
            "needs_exact_match": self._check_exact_match(query_lower),
            "target_modalities": self._detect_modalities(query_lower),
            "search_strategy": "hybrid",  # Default
            "filters": {}
        }
        
        # Determine optimal search strategy
        if analysis["needs_exact_match"]:
            analysis["search_strategy"] = "keyword"
        elif analysis["is_cross_modal"]:
            analysis["search_strategy"] = "cross_modal"
        else:
            analysis["search_strategy"] = "hybrid"
        
        # Extract temporal filters
        if analysis["is_temporal"]:
            analysis["filters"]["temporal"] = self._extract_temporal_filter(query)
        
        logger.info(f"Query analysis: {analysis['search_strategy']} search, "
                   f"modalities: {analysis['target_modalities']}")
        
        return analysis
    
    def _check_temporal(self, query: str) -> bool:
        """Check if query has temporal component"""
        return any(keyword in query for keyword in self.temporal_keywords)
    
    def _check_cross_modal(self, query: str) -> bool:
        """Check if query spans multiple modalities"""
        return any(keyword in query for keyword in self.cross_modal_keywords)
    
    def _check_exact_match(self, query: str) -> bool:
        """Check if query needs exact matching"""
        # Check for quotes
        if '"' in query or "'" in query:
            return True
        return any(keyword in query for keyword in self.exact_keywords)
    
    def _detect_modalities(self, query: str) -> List[str]:
        """Detect which modalities to search"""
        modalities = set()
        
        # Check for specific modality mentions
        if any(word in query for word in ['image', 'picture', 'photo', 'screenshot', 'chart']):
            modalities.add('image')
        
        if any(word in query for word in ['audio', 'recording', 'call', 'meeting', 'said']):
            modalities.add('audio')
        
        if any(word in query for word in ['document', 'report', 'pdf', 'text', 'page']):
            modalities.add('text')
        
        # Default to all if none specified
        if not modalities:
            modalities = {'text', 'image', 'audio'}
        
        return list(modalities)
    
    def _extract_temporal_filter(self, query: str) -> Dict:
        """Extract temporal constraints from query"""
        # Simple regex patterns for dates
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        dates = re.findall(date_pattern, query)
        
        filter_dict = {}
        if dates:
            if len(dates) == 1:
                filter_dict["date"] = dates[0]
            elif len(dates) >= 2:
                filter_dict["start_date"] = dates[0]
                filter_dict["end_date"] = dates[1]
        
        # Relative time extraction (simplified)
        if 'last week' in query.lower():
            filter_dict["relative"] = "last_week"
        elif 'last month' in query.lower():
            filter_dict["relative"] = "last_month"
        elif 'today' in query.lower():
            filter_dict["relative"] = "today"
        
        return filter_dict