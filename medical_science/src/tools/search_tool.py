from typing import Any
from agents.web_search_processor_agent import WebSearchProcessorAgent
from config import Config


def run_web_search(query: str, chat_history: str = "") -> Any:
    processor = WebSearchProcessorAgent(Config())
    return processor.process_web_search_results(query=query, chat_history=chat_history) 