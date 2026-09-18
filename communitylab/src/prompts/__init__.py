"""System prompts for analysis and channel-specific copywriting."""

from src.prompts.system_prompts import (
    ANALYSIS_PROMPTS,
    CHANNELS,
    COPYWRITING_PROMPTS,
    FAQ_SYSTEM_PROMPT,
    LINKEDIN_POST_SYSTEM_PROMPT,
    MESSAGE_HUMAN_TEMPLATE,
    NEWSLETTER_SYSTEM_PROMPT,
    RELEVANCE_SYSTEM_PROMPT,
    SENTIMENT_SYSTEM_PROMPT,
    TESTIMONIAL_SYSTEM_PROMPT,
    TOPIC_EXTRACTION_SYSTEM_PROMPT,
    Channel,
    build_message_prompt,
    get_copywriting_prompt,
)

__all__ = [
    "ANALYSIS_PROMPTS",
    "CHANNELS",
    "COPYWRITING_PROMPTS",
    "FAQ_SYSTEM_PROMPT",
    "LINKEDIN_POST_SYSTEM_PROMPT",
    "MESSAGE_HUMAN_TEMPLATE",
    "NEWSLETTER_SYSTEM_PROMPT",
    "RELEVANCE_SYSTEM_PROMPT",
    "SENTIMENT_SYSTEM_PROMPT",
    "TESTIMONIAL_SYSTEM_PROMPT",
    "TOPIC_EXTRACTION_SYSTEM_PROMPT",
    "Channel",
    "build_message_prompt",
    "get_copywriting_prompt",
]
