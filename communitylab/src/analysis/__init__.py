"""AI analysis layer: sentiment, themes, relevance and their consolidation."""

from src.analysis.base import AnalysisError, MessageInput, StructuredAnalyzer, message_to_prompt_input
from src.analysis.consolidator import Consolidator, EnrichedMessage, to_frame, to_records
from src.analysis.relevance import RelevanceAnalyzer, RelevanceResult, RelevanceTier, analyze_relevance, tier_for_score
from src.analysis.sentiment import SentimentAnalyzer, SentimentLabel, SentimentResult, analyze_sentiment
from src.analysis.themes import TOPIC_CATEGORIES, ThemesAnalyzer, ThemesResult, TopicCategory, analyze_themes

__all__ = [
    "TOPIC_CATEGORIES",
    "AnalysisError",
    "Consolidator",
    "EnrichedMessage",
    "MessageInput",
    "RelevanceAnalyzer",
    "RelevanceResult",
    "RelevanceTier",
    "SentimentAnalyzer",
    "SentimentLabel",
    "SentimentResult",
    "StructuredAnalyzer",
    "ThemesAnalyzer",
    "ThemesResult",
    "TopicCategory",
    "analyze_relevance",
    "analyze_sentiment",
    "analyze_themes",
    "message_to_prompt_input",
    "tier_for_score",
    "to_frame",
    "to_records",
]
