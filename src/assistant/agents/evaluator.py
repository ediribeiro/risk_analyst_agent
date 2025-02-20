from typing import Dict
import logging
import traceback
from tenacity import retry, stop_after_attempt
from ..utils import extract_json_from_response, rate_limit, split_json_array, merge_json_responses, count_tokens, process_risk_data
from .base import BaseAgent
from ..prompts import EVALUATOR_PROMPT

logger = logging.getLogger(__name__)

class EvaluatorAgent(BaseAgent):
    """Agent responsible for evaluating identified risks"""
    
    def __init__(self, llm):
        super().__init__(llm)
        self.prompt = EVALUATOR_PROMPT

    @rate_limit(max_calls=10, period=60)
    @retry(stop=stop_after_attempt(2))
    def evaluate(self, state: Dict) -> Dict:
        """Evaluate risks and assign impact scores"""
        logger.info("Starting report evaluation")
        try:
            # Split input if needed
            chunks = split_json_array(state["risk_list"])
            all_responses = []
            
            for chunk in chunks:
                chain = self.prompt | self.llm
                response = chain.invoke({"risk_list": chunk})
                
                # Extract JSON from response text
                risks = extract_json_from_response(response)
                all_responses.append(risks)
            
            # Merge all responses
            evaluated_risks = merge_json_responses(all_responses)
            
            # Process risk data
            df = process_risk_data(evaluated_risks)
            evaluated_risks = df.to_json(orient='records', force_ascii=False)
            tokens = count_tokens(evaluated_risks)
            
            logger.info(f"[Evaluator] Successfully processed risk evaluations")
            
            return {
                "risk_analysis": evaluated_risks,
                "iteration": state["iteration"] + 1,
                "token_usage": {"evaluation": tokens}
            }
            
        except Exception as e:
            logger.error(f"[Evaluator] Failed: {str(e)}")
            logger.error(f"[Evaluator] Traceback: {traceback.format_exc()}")
            error_msg = f"Error in evaluation: {str(e)}"
            return {"validation_errors": [error_msg]}
