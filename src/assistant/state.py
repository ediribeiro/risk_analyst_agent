from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import add_messages

class State(TypedDict):
    """Core state management for the optimization workflow"""
    input_file: str
    context: Annotated[List[str], add_messages]
    risk_list: str
    risk_analysis: List[str]
    iteration: int
    token_usage: Dict[str, int]
