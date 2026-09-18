"""Content generators: turn analysed messages into publishable assets."""

from src.generators.base import (
    BRIEF_HUMAN_TEMPLATE,
    ContentGenerator,
    GeneratedAsset,
    GenerationError,
    build_brief,
    format_message_block,
    slugify,
)
from src.generators.faq import FAQEntry, FAQGenerator, thread_messages
from src.generators.linkedin import LinkedInGenerator, LinkedInPost, select_success_stories
from src.generators.newsletter import NewsletterGenerator, NewsletterSection, select_highlights
from src.generators.testimonials import Testimonial, TestimonialGenerator, select_testimonial_candidates

__all__ = [
    "BRIEF_HUMAN_TEMPLATE",
    "ContentGenerator",
    "FAQEntry",
    "FAQGenerator",
    "GeneratedAsset",
    "GenerationError",
    "LinkedInGenerator",
    "LinkedInPost",
    "NewsletterGenerator",
    "NewsletterSection",
    "Testimonial",
    "TestimonialGenerator",
    "build_brief",
    "format_message_block",
    "select_highlights",
    "select_success_stories",
    "select_testimonial_candidates",
    "slugify",
    "thread_messages",
]
