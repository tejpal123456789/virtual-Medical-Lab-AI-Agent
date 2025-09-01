from typing import Any, Dict, Union, List
from langchain_core.messages import BaseMessage

from agents.agent_decision import process_query as _process_query


def process(query: Union[str, Dict], conversation_history: List[BaseMessage] | None = None) -> Any:
    return _process_query(query, conversation_history) 