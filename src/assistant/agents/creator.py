from typing import Dict
import traceback
import logging
import json
from ..utils import extract_json_from_response
from .base import BaseAgent
from ..prompts import CREATOR_PROMPT

logger = logging.getLogger(__name__)

class CreatorAgent(BaseAgent):
    """Agent responsible for initial risk identification"""
    
    def __init__(self, llm):
        super().__init__(llm)
        self.prompt = CREATOR_PROMPT

    def generate(self, state: Dict) -> Dict:
        try:
            if not state.get("context"):
                logger.error("No context found in state")
                raise ValueError("No context found from document search")
 
            all_risks = []
            risk_counter = 1
 
            total_chunks = len(state['context'])
            logger.info(f"=== Starting risk analysis with {total_chunks} context chunks ===")
            
            # Process each context chunk
            for context in state["context"]:
                try:
                    logger.info(f"\n[Chunk {risk_counter}/{total_chunks}] Processing...")
                    
                    context_content = context.content if hasattr(context, 'content') else str(context)
                    logger.info(f"[Chunk {risk_counter}/{total_chunks}] Content preview: {context_content[:200]}...")
                    
                    chain = self.prompt | self.llm
                    response = chain.invoke({
                        "context": context_content
                    })
                    
                    # Extract JSON from response text
                    response_text = str(response.content if hasattr(response, 'content') else response)
                    stage_risks = extract_json_from_response(response_text)
                    
                    # Add IDs to risks
                    for risk in stage_risks:
                        risk["Id"] = f"R{risk_counter:03d}"
                        risk_counter += 1
                    all_risks.extend(stage_risks)
                    
                except Exception as e:
                    logger.error(f"[Chunk {risk_counter}/{total_chunks}] Failed to process:")
                    logger.error(f"  Error: {str(e)}")
                    logger.error(f"  Context: {str(context)[:200]}...")
                    logger.error(f"  Traceback: {traceback.format_exc()}")
                    continue
 
            if not all_risks:
                logger.error(f"=== Risk analysis failed: No risks were generated from {total_chunks} chunks ===")
                raise ValueError(f"No risks were generated from {total_chunks} context chunks")
            
            logger.info(f"=== Risk analysis completed: Generated {len(all_risks)} total risks ===")
 
            # Convert final list to JSON string
            risk_list = extract_json_from_response(all_risks)
            tokens = len(risk_list)
            
            return {
                "risk_list": risk_list,
                "iteration": 1,
                "token_usage": {"generation": tokens}
            }
 
        except Exception as e:
            logger.error(f"Error in risk generation: {str(e)}")
            raise 