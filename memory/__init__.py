"""
Long-term Memory Module for Multi-Agent Medical Assistant

This module provides intelligent long-term memory capabilities using mem0.
"""

from .long_term_memory import (
    LongTermMemoryManager,
    MemoryEnhancedPromptBuilder,
    MemoryType,
    MemoryEntry
)

__all__ = [
    "LongTermMemoryManager",
    "MemoryEnhancedPromptBuilder", 
    "MemoryType",
    "MemoryEntry"
] 