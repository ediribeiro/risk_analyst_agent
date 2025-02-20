from tenacity import retry, stop_after_attempt
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict

class BaseAgent:
    def __init__(self, llm):
        self.llm = llm
        
    @retry(stop=stop_after_attempt(3))
    def invoke(self, input_data: Dict) -> Dict:
        raise NotImplementedError 