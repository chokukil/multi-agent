"""
User-Personalized Result Optimizer Module

This module provides user-specific result optimization by learning from user preferences,
interaction history, and contextual requirements to deliver the most relevant and
personalized results for each individual user.

Author: CherryAI Development Team
Version: 1.0.0
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import hashlib

from .domain_specific_answer_formatter import FormattedAnswer, FormattingContext, DomainType as FormatterDomainType
from .holistic_answer_synthesis_engine import HolisticAnswer
from .intent_analyzer import DetailedIntentAnalysis
from .domain_extractor import EnhancedDomainKnowledge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersonalizationLevel(Enum):
    """Levels of personalization intensity"""
    BASIC = "basic"           # Basic preference application
    MODERATE = "moderate"     # Moderate customization with history
    ADVANCED = "advanced"     # Advanced ML-based personalization
    EXPERT = "expert"         # Expert-level deep personalization


class UserRole(Enum):
    """User role classifications for personalization"""
    EXECUTIVE = "executive"
    MANAGER = "manager"
    ANALYST = "analyst"
    ENGINEER = "engineer"
    RESEARCHER = "researcher"
    CONSULTANT = "consultant"
    STUDENT = "student"
    GENERAL_USER = "general_user"


class InteractionType(Enum):
    """Types of user interactions for learning"""
    QUERY_SUBMISSION = "query_submission"
    RESULT_RATING = "result_rating"
    SECTION_EXPANSION = "section_expansion"
    EXPORT_ACTION = "export_action"
    SHARING_ACTION = "sharing_action"
    TIME_SPENT = "time_spent"
    FOLLOW_UP_QUERY = "follow_up_query"


class OptimizationStrategy(Enum):
    """Optimization strategies for result personalization"""
    CONTENT_BASED = "content_based"      # Based on content preferences
    COLLABORATIVE = "collaborative"      # Based on similar users
    HYBRID = "hybrid"                   # Combination approach
    CONTEXTUAL = "contextual"           # Context-aware optimization
    ADAPTIVE = "adaptive"               # Self-learning optimization


@dataclass
class UserPreference:
    """Individual user preference item"""
    preference_type: str
    value: Any
    weight: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    last_updated: datetime = field(default_factory=datetime.now)
    source: str = "explicit"  # explicit, inferred, learned


@dataclass
class UserInteraction:
    """User interaction record for learning"""
    interaction_id: str
    user_id: str
    interaction_type: InteractionType
    query: str
    result_id: str
    rating: Optional[float] = None  # 0.0 - 5.0
    feedback: Optional[str] = None
    time_spent: Optional[int] = None  # seconds
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class UserProfile:
    """Comprehensive user profile for personalization"""
    user_id: str
    role: UserRole
    domain_expertise: Dict[str, float]  # domain -> expertise level (0.0-1.0)
    preferences: Dict[str, UserPreference]
    interaction_history: List[UserInteraction]
    learning_weights: Dict[str, float]
    personalization_level: PersonalizationLevel
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationContext:
    """Context for result optimization"""
    user_profile: UserProfile
    current_query: str
    domain_context: EnhancedDomainKnowledge
    intent_analysis: DetailedIntentAnalysis
    time_constraints: Optional[int] = None  # seconds
    device_type: str = "desktop"
    session_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PersonalizationInsights:
    """Insights about applied personalization"""
    applied_optimizations: List[str]
    preference_matches: Dict[str, float]
    learning_contributions: Dict[str, float]
    confidence_score: float
    optimization_impact: float  # estimated improvement (0.0-1.0)


@dataclass
class OptimizedResult:
    """Personalized and optimized result"""
    original_result: FormattedAnswer
    optimized_content: str
    personalization_metadata: Dict[str, Any]
    optimization_context: OptimizationContext
    personalization_insights: PersonalizationInsights
    optimization_score: float  # 0.0 - 1.0
    generated_at: datetime = field(default_factory=datetime.now)


class UserPersonalizedResultOptimizer:
    """
    Advanced result optimizer that personalizes responses based on user preferences,
    history, and contextual requirements
    """
    
    def __init__(self):
        """Initialize the User-Personalized Result Optimizer"""
        self.logger = logging.getLogger(__name__)
        self.user_profiles: Dict[str, UserProfile] = {}
        self.optimization_strategies = self._initialize_optimization_strategies()
        self.personalization_rules = self._initialize_personalization_rules()
        self.learning_algorithms = self._initialize_learning_algorithms()
        
    def _initialize_optimization_strategies(self) -> Dict[OptimizationStrategy, Dict[str, Any]]:
        """Initialize optimization strategies configuration"""
        return {
            OptimizationStrategy.CONTENT_BASED: {
                "description": "Optimize based on content preferences",
                "weight_factors": {
                    "content_length": 0.3,
                    "technical_depth": 0.4,
                    "visual_elements": 0.3
                },
                "applicable_roles": [UserRole.ANALYST, UserRole.RESEARCHER, UserRole.ENGINEER]
            },
            OptimizationStrategy.COLLABORATIVE: {
                "description": "Optimize based on similar user preferences",
                "weight_factors": {
                    "role_similarity": 0.4,
                    "domain_similarity": 0.3,
                    "interaction_patterns": 0.3
                },
                "applicable_roles": [UserRole.MANAGER, UserRole.CONSULTANT, UserRole.GENERAL_USER]
            },
            OptimizationStrategy.CONTEXTUAL: {
                "description": "Optimize based on current context",
                "weight_factors": {
                    "time_constraints": 0.3,
                    "device_type": 0.2,
                    "domain_urgency": 0.5
                },
                "applicable_roles": [UserRole.EXECUTIVE, UserRole.MANAGER]
            },
            OptimizationStrategy.ADAPTIVE: {
                "description": "Self-learning optimization",
                "weight_factors": {
                    "learning_rate": 0.4,
                    "feedback_incorporation": 0.3,
                    "pattern_recognition": 0.3
                },
                "applicable_roles": list(UserRole)
            }
        }
    
    def _initialize_personalization_rules(self) -> Dict[UserRole, Dict[str, Any]]:
        """Initialize role-based personalization rules"""
        return {
            UserRole.EXECUTIVE: {
                "preferred_length": "short",
                "focus_areas": ["business_impact", "roi", "strategic_implications"],
                "visualization_preference": "high_level_charts",
                "detail_level": "executive_summary",
                "time_sensitivity": "high"
            },
            UserRole.MANAGER: {
                "preferred_length": "medium",
                "focus_areas": ["actionable_recommendations", "team_impact", "resource_requirements"],
                "visualization_preference": "operational_dashboards",
                "detail_level": "balanced",
                "time_sensitivity": "medium"
            },
            UserRole.ANALYST: {
                "preferred_length": "detailed",
                "focus_areas": ["data_analysis", "statistical_insights", "methodology"],
                "visualization_preference": "detailed_charts",
                "detail_level": "comprehensive",
                "time_sensitivity": "low"
            },
            UserRole.ENGINEER: {
                "preferred_length": "technical",
                "focus_areas": ["technical_implementation", "architecture", "performance"],
                "visualization_preference": "technical_diagrams",
                "detail_level": "technical_deep_dive",
                "time_sensitivity": "medium"
            },
            UserRole.RESEARCHER: {
                "preferred_length": "comprehensive",
                "focus_areas": ["methodology", "research_insights", "academic_rigor"],
                "visualization_preference": "research_charts",
                "detail_level": "academic",
                "time_sensitivity": "low"
            }
        }
    
    def _initialize_learning_algorithms(self) -> Dict[str, Dict[str, Any]]:
        """Initialize learning algorithms configuration"""
        return {
            "preference_learning": {
                "algorithm": "weighted_average",
                "decay_factor": 0.95,  # older interactions have less weight
                "minimum_interactions": 3,
                "confidence_threshold": 0.7
            },
            "pattern_recognition": {
                "algorithm": "frequency_analysis",
                "pattern_window": 30,  # days
                "minimum_pattern_strength": 0.6
            },
            "collaborative_filtering": {
                "algorithm": "user_similarity",
                "similarity_threshold": 0.75,
                "neighbor_count": 5
            }
        }
    
    def optimize_result(self,
                       formatted_answer: FormattedAnswer,
                       user_id: str,
                       optimization_context: OptimizationContext) -> OptimizedResult:
        """
        Optimize a formatted answer for a specific user
        
        Args:
            formatted_answer: The formatted answer to optimize
            user_id: Unique user identifier
            optimization_context: Context for optimization
            
        Returns:
            OptimizedResult: Personalized and optimized result
        """
        try:
            self.logger.info(f"Optimizing result for user: {user_id}")
            
            # Get or create user profile
            user_profile = self._get_or_create_user_profile(user_id, optimization_context)
            
            # Determine optimization strategy
            optimization_strategy = self._determine_optimization_strategy(user_profile, optimization_context)
            
            # Apply personalization
            personalized_content = self._apply_personalization(
                formatted_answer, user_profile, optimization_context, optimization_strategy
            )
            
            # Generate personalization insights
            personalization_insights = self._generate_personalization_insights(
                formatted_answer, personalized_content, user_profile, optimization_strategy
            )
            
            # Calculate optimization score
            optimization_score = self._calculate_optimization_score(
                formatted_answer, personalized_content, personalization_insights
            )
            
            # Create optimized result
            optimized_result = OptimizedResult(
                original_result=formatted_answer,
                optimized_content=personalized_content,
                personalization_metadata={
                    "user_profile_version": user_profile.updated_at.isoformat(),
                    "optimization_strategy": optimization_strategy.value,
                    "personalization_level": user_profile.personalization_level.value,
                    "applied_rules": self._get_applied_rules(user_profile, optimization_strategy)
                },
                optimization_context=optimization_context,
                personalization_insights=personalization_insights,
                optimization_score=optimization_score
            )
            
            # Record interaction for learning
            self._record_optimization_interaction(user_id, optimization_context, optimized_result)
            
            self.logger.info(f"Result optimization completed for user: {user_id}")
            return optimized_result
            
        except Exception as e:
            self.logger.error(f"Error optimizing result for user {user_id}: {str(e)}")
            raise
    
    def _get_or_create_user_profile(self, user_id: str, context: OptimizationContext) -> UserProfile:
        """Get existing user profile or create a new one"""
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            profile.updated_at = datetime.now()
            return profile
        
        # Create new user profile
        new_profile = UserProfile(
            user_id=user_id,
            role=self._infer_user_role(context),
            domain_expertise=self._infer_domain_expertise(context),
            preferences=self._initialize_default_preferences(context),
            interaction_history=[],
            learning_weights=self._initialize_learning_weights(),
            personalization_level=PersonalizationLevel.BASIC
        )
        
        self.user_profiles[user_id] = new_profile
        return new_profile
    
    def _infer_user_role(self, context: OptimizationContext) -> UserRole:
        """Infer user role from context"""
        # Analyze query patterns and intent to infer role
        intent = context.intent_analysis.primary_intent.lower()
        
        if any(keyword in intent for keyword in ["strategy", "business", "roi", "executive"]):
            return UserRole.EXECUTIVE
        elif any(keyword in intent for keyword in ["manage", "team", "resource", "plan"]):
            return UserRole.MANAGER
        elif any(keyword in intent for keyword in ["analyze", "data", "statistical", "metrics"]):
            return UserRole.ANALYST
        elif any(keyword in intent for keyword in ["implement", "technical", "system", "code"]):
            return UserRole.ENGINEER
        elif any(keyword in intent for keyword in ["research", "study", "academic", "methodology"]):
            return UserRole.RESEARCHER
        else:
            return UserRole.GENERAL_USER
    
    def _infer_domain_expertise(self, context: OptimizationContext) -> Dict[str, float]:
        """Infer domain expertise from context"""
        domain_expertise = {}
        
        # Base expertise on domain knowledge confidence
        primary_domain = context.domain_context.taxonomy.primary_domain.value
        expertise_level = context.domain_context.extraction_confidence
        
        domain_expertise[primary_domain] = expertise_level
        
        # Add related domains with lower confidence
        for sub_domain in context.domain_context.taxonomy.sub_domains:
            domain_expertise[sub_domain] = expertise_level * 0.7
        
        return domain_expertise
    
    def _initialize_default_preferences(self, context: OptimizationContext) -> Dict[str, UserPreference]:
        """Initialize default user preferences"""
        preferences = {}
        
        # Content length preference
        preferences["content_length"] = UserPreference(
            preference_type="content_length",
            value="medium",
            weight=0.8,
            confidence=0.5,
            source="inferred"
        )
        
        # Detail level preference
        preferences["detail_level"] = UserPreference(
            preference_type="detail_level",
            value="balanced",
            weight=0.7,
            confidence=0.5,
            source="inferred"
        )
        
        # Visualization preference
        preferences["visualization_type"] = UserPreference(
            preference_type="visualization_type",
            value="charts",
            weight=0.6,
            confidence=0.4,
            source="inferred"
        )
        
        return preferences
    
    def _initialize_learning_weights(self) -> Dict[str, float]:
        """Initialize learning algorithm weights"""
        return {
            "content_preferences": 0.4,
            "interaction_patterns": 0.3,
            "feedback_signals": 0.2,
            "contextual_factors": 0.1
        }
    
    def _determine_optimization_strategy(self, user_profile: UserProfile, context: OptimizationContext) -> OptimizationStrategy:
        """Determine the best optimization strategy for the user"""
        
        # Consider personalization level
        if user_profile.personalization_level == PersonalizationLevel.BASIC:
            return OptimizationStrategy.CONTENT_BASED
        
        # Consider interaction history
        if len(user_profile.interaction_history) < 5:
            return OptimizationStrategy.CONTENT_BASED
        elif len(user_profile.interaction_history) < 20:
            return OptimizationStrategy.CONTEXTUAL
        else:
            return OptimizationStrategy.ADAPTIVE
    
    def _apply_personalization(self,
                              formatted_answer: FormattedAnswer,
                              user_profile: UserProfile,
                              context: OptimizationContext,
                              strategy: OptimizationStrategy) -> str:
        """Apply personalization to the formatted answer"""
        
        # Get role-based rules
        role_rules = self.personalization_rules.get(user_profile.role, {})
        
        # Start with original content
        personalized_content = formatted_answer.content
        
        # Apply length preference
        personalized_content = self._apply_length_preference(
            personalized_content, user_profile, role_rules
        )
        
        # Apply focus area emphasis
        personalized_content = self._apply_focus_emphasis(
            personalized_content, user_profile, role_rules, context
        )
        
        # Apply detail level adjustment
        personalized_content = self._apply_detail_level_adjustment(
            personalized_content, user_profile, role_rules
        )
        
        # Apply time sensitivity adjustments
        personalized_content = self._apply_time_sensitivity(
            personalized_content, user_profile, context
        )
        
        return personalized_content
    
    def _apply_length_preference(self, content: str, user_profile: UserProfile, role_rules: Dict[str, Any]) -> str:
        """Apply length preference to content"""
        
        length_pref = user_profile.preferences.get("content_length")
        role_length = role_rules.get("preferred_length", "medium")
        
        # Determine target length
        if length_pref and length_pref.weight > 0.7:
            target_length = length_pref.value
        else:
            target_length = role_length
        
        # Apply length adjustment
        if target_length == "short":
            # Summarize to shorter version
            sections = content.split('\n\n')
            if len(sections) > 3:
                # Keep only the most important sections
                important_sections = sections[:2] + [sections[-1]]
                content = '\n\n'.join(important_sections)
        elif target_length == "detailed":
            # Add more detailed explanations
            content = self._add_detailed_explanations(content)
        
        return content
    
    def _apply_focus_emphasis(self, content: str, user_profile: UserProfile, role_rules: Dict[str, Any], context: OptimizationContext) -> str:
        """Apply focus area emphasis to content"""
        
        focus_areas = role_rules.get("focus_areas", [])
        
        for focus_area in focus_areas:
            if focus_area in content.lower():
                # Emphasize sections related to focus areas
                content = self._emphasize_focus_area(content, focus_area)
        
        return content
    
    def _apply_detail_level_adjustment(self, content: str, user_profile: UserProfile, role_rules: Dict[str, Any]) -> str:
        """Apply detail level adjustment to content"""
        
        detail_level = role_rules.get("detail_level", "balanced")
        
        if detail_level == "executive_summary":
            # Focus on high-level insights
            content = self._create_executive_focus(content)
        elif detail_level == "technical_deep_dive":
            # Add technical details
            content = self._add_technical_details(content)
        elif detail_level == "academic":
            # Add methodology and references
            content = self._add_academic_rigor(content)
        
        return content
    
    def _apply_time_sensitivity(self, content: str, user_profile: UserProfile, context: OptimizationContext) -> str:
        """Apply time sensitivity adjustments"""
        
        if context.time_constraints and context.time_constraints < 300:  # Less than 5 minutes
            # Prioritize most critical information
            content = self._prioritize_critical_info(content)
        
        return content
    
    def _add_detailed_explanations(self, content: str) -> str:
        """Add detailed explanations to content"""
        # Implementation would add more detailed explanations
        return content + "\n\n**Additional Details:**\nDetailed explanations and supporting information would be added here."
    
    def _emphasize_focus_area(self, content: str, focus_area: str) -> str:
        """Emphasize specific focus areas in content"""
        # Implementation would emphasize relevant sections
        return content.replace(focus_area, f"**{focus_area.upper()}**")
    
    def _create_executive_focus(self, content: str) -> str:
        """Create executive-focused version of content"""
        # Implementation would extract key business insights
        sections = content.split('\n\n')
        executive_sections = []
        
        for section in sections:
            if any(keyword in section.lower() for keyword in ["summary", "recommendation", "impact", "roi"]):
                executive_sections.append(section)
        
        return '\n\n'.join(executive_sections) if executive_sections else content[:500] + "..."
    
    def _add_technical_details(self, content: str) -> str:
        """Add technical details to content"""
        return content + "\n\n**Technical Implementation Notes:**\nTechnical specifications and implementation details would be added here."
    
    def _add_academic_rigor(self, content: str) -> str:
        """Add academic rigor to content"""
        return content + "\n\n**Methodology & References:**\nMethodological considerations and academic references would be added here."
    
    def _prioritize_critical_info(self, content: str) -> str:
        """Prioritize critical information for time-sensitive contexts"""
        sections = content.split('\n\n')
        if len(sections) > 1:
            # Return first section (usually summary) and recommendations
            critical_sections = [sections[0]]
            for section in sections[1:]:
                if "recommendation" in section.lower():
                    critical_sections.append(section)
                    break
            return '\n\n'.join(critical_sections)
        return content
    
    def _generate_personalization_insights(self,
                                         formatted_answer: FormattedAnswer,
                                         personalized_content: str,
                                         user_profile: UserProfile,
                                         strategy: OptimizationStrategy) -> PersonalizationInsights:
        """Generate insights about applied personalization"""
        
        applied_optimizations = []
        preference_matches = {}
        learning_contributions = {}
        
        # Analyze applied optimizations
        if len(personalized_content) != len(formatted_answer.content):
            applied_optimizations.append("length_adjustment")
        
        # Calculate preference matches
        for pref_name, preference in user_profile.preferences.items():
            preference_matches[pref_name] = preference.confidence * preference.weight
        
        # Calculate learning contributions
        for weight_name, weight in user_profile.learning_weights.items():
            learning_contributions[weight_name] = weight
        
        # Calculate overall confidence
        confidence_score = sum(preference_matches.values()) / len(preference_matches) if preference_matches else 0.5
        
        # Estimate optimization impact
        optimization_impact = min(confidence_score * 0.8, 0.9)  # Conservative estimate
        
        return PersonalizationInsights(
            applied_optimizations=applied_optimizations,
            preference_matches=preference_matches,
            learning_contributions=learning_contributions,
            confidence_score=confidence_score,
            optimization_impact=optimization_impact
        )
    
    def _calculate_optimization_score(self,
                                    formatted_answer: FormattedAnswer,
                                    personalized_content: str,
                                    insights: PersonalizationInsights) -> float:
        """Calculate optimization effectiveness score"""
        
        # Base score on personalization insights
        base_score = insights.confidence_score
        
        # Adjust based on optimization impact
        impact_bonus = insights.optimization_impact * 0.2
        
        # Penalize if no optimizations were applied
        if not insights.applied_optimizations:
            base_score *= 0.8
        
        return min(base_score + impact_bonus, 1.0)
    
    def _get_applied_rules(self, user_profile: UserProfile, strategy: OptimizationStrategy) -> List[str]:
        """Get list of applied personalization rules"""
        applied_rules = []
        
        applied_rules.append(f"role_based_{user_profile.role.value}")
        applied_rules.append(f"strategy_{strategy.value}")
        applied_rules.append(f"level_{user_profile.personalization_level.value}")
        
        return applied_rules
    
    def _record_optimization_interaction(self, user_id: str, context: OptimizationContext, result: OptimizedResult):
        """Record optimization interaction for learning"""
        
        interaction = UserInteraction(
            interaction_id=f"opt_{int(datetime.now().timestamp())}",
            user_id=user_id,
            interaction_type=InteractionType.QUERY_SUBMISSION,
            query=context.current_query,
            result_id=result.original_result.metadata.get("result_id", "unknown"),
            context={
                "optimization_score": result.optimization_score,
                "personalization_level": context.user_profile.personalization_level.value,
                "domain": context.domain_context.taxonomy.primary_domain.value
            }
        )
        
        if user_id in self.user_profiles:
            self.user_profiles[user_id].interaction_history.append(interaction)
            
            # Limit history size
            if len(self.user_profiles[user_id].interaction_history) > 100:
                self.user_profiles[user_id].interaction_history = self.user_profiles[user_id].interaction_history[-100:]
    
    def update_user_feedback(self, user_id: str, result_id: str, rating: float, feedback: Optional[str] = None):
        """Update user feedback for learning"""
        
        if user_id not in self.user_profiles:
            return
        
        # Find the corresponding interaction
        for interaction in self.user_profiles[user_id].interaction_history:
            if interaction.result_id == result_id:
                interaction.rating = rating
                interaction.feedback = feedback
                break
        
        # Update learning weights based on feedback
        self._update_learning_weights(user_id, rating, feedback)
    
    def _update_learning_weights(self, user_id: str, rating: float, feedback: Optional[str]):
        """Update learning weights based on user feedback"""
        
        if user_id not in self.user_profiles:
            return
        
        profile = self.user_profiles[user_id]
        
        # Adjust learning weights based on rating
        if rating >= 4.0:  # Positive feedback
            # Increase weights for current preferences
            for weight_name in profile.learning_weights:
                profile.learning_weights[weight_name] = min(profile.learning_weights[weight_name] * 1.1, 1.0)
        elif rating <= 2.0:  # Negative feedback
            # Decrease weights and encourage exploration
            for weight_name in profile.learning_weights:
                profile.learning_weights[weight_name] = max(profile.learning_weights[weight_name] * 0.9, 0.1)
        
        # Update personalization level based on interaction history
        if len(profile.interaction_history) >= 10:
            avg_rating = sum(i.rating for i in profile.interaction_history if i.rating) / len([i for i in profile.interaction_history if i.rating])
            
            if avg_rating >= 4.0 and profile.personalization_level == PersonalizationLevel.BASIC:
                profile.personalization_level = PersonalizationLevel.MODERATE
            elif avg_rating >= 4.5 and profile.personalization_level == PersonalizationLevel.MODERATE:
                profile.personalization_level = PersonalizationLevel.ADVANCED
        
        profile.updated_at = datetime.now()
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID"""
        return self.user_profiles.get(user_id)
    
    def get_optimization_strategies(self) -> List[OptimizationStrategy]:
        """Get available optimization strategies"""
        return list(OptimizationStrategy)
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get user interaction statistics"""
        
        if user_id not in self.user_profiles:
            return {}
        
        profile = self.user_profiles[user_id]
        interactions = profile.interaction_history
        
        stats = {
            "total_interactions": len(interactions),
            "average_rating": 0.0,
            "personalization_level": profile.personalization_level.value,
            "domain_expertise": profile.domain_expertise,
            "last_interaction": None
        }
        
        if interactions:
            ratings = [i.rating for i in interactions if i.rating is not None]
            if ratings:
                stats["average_rating"] = sum(ratings) / len(ratings)
            
            stats["last_interaction"] = max(interactions, key=lambda x: x.timestamp).timestamp.isoformat()
        
        return stats 