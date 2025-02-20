from typing import Dict
import logging
import json
from tenacity import retry, stop_after_attempt
from ..utils import extract_json_from_response, rate_limit, split_json_array, merge_json_responses, count_tokens
from .base import BaseAgent
from ..prompts import OPTIMIZER_PROMPT
import traceback

logger = logging.getLogger(__name__)

class OptimizationAgent(BaseAgent):
    """Agent responsible for optimizing and finalizing risk analysis"""
    
    def __init__(self, llm):
        super().__init__(llm)
        self.prompt = OPTIMIZER_PROMPT

    @rate_limit(max_calls=5, period=60)
    @retry(stop=stop_after_attempt(3))
    def optimize(self, state: Dict) -> Dict:
        """Optimize risk analysis with scoring and categorization"""
        logger.info("Starting report optimization")
        try:
            # Split input if needed
            chunks = split_json_array(state["risk_analysis"])
            all_responses = []
            
            for chunk in chunks:
                chain = self.prompt | self.llm
                response = chain.invoke({"risk_analysis": chunk})
                
                # Extract JSON from response
                risks = extract_json_from_response(response)
                all_responses.append(risks)
            
            # Merge responses
            optimized_risks = merge_json_responses(all_responses)
            tokens = count_tokens(optimized_risks)
            
            logger.info("[Optimizer] Successfully extracted optimized risks")
            
            return {
                "risk_list": optimized_risks,
                "iteration": state["iteration"] + 1,
                "token_usage": {"optimization": tokens}
            }
            
        except Exception as e:
            logger.error(f"[Optimizer] Failed: {str(e)}")
            logger.error(f"[Optimizer] Traceback: {traceback.format_exc()}")
            raise
