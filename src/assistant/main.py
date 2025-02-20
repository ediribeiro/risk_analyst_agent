import os
import json
from src.assistant.configuration import logger
from src.assistant.graph import create_workflow

# Initialize workflow graph
workflow = create_workflow()
agent = workflow.compile()

# Export the agent for LangGraph Studio
__all__ = ["agent"]

def main():
    try:
        # Get script directory and set up paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_file = os.path.join(script_dir, "documents", "TERMO_DE_REFERENCIA.pdf")
        output_dir = os.path.join(script_dir, "risk_analysis")
        
        logger.info(f"Using input file: {input_file}")
        
        # Run the workflow
        final_state = agent.invoke({
            "input_file": input_file,
            "risk_list": "",
            "iteration": 0
        })
        
        logger.info("Workflow completed successfully")
        
        # Save output
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "risk_analysis_report.json")
        
        # Parse the final risk list from the state
        try:
            # The risk_list should already be a JSON string from the optimizer
            final_risks = json.loads(final_state["risk_list"])
            
            # Write with proper formatting
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(final_risks, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Report saved to: {output_file}")
            logger.info(f"Report contains {len(final_risks)} risks")
            logger.info(f"Token usage by stage: {json.dumps(final_state['token_usage'], indent=2)}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse final risks: {str(e)}")
            logger.error(f"Raw content: {final_state['risk_list'][:500]}")
            raise
            
        if final_state.get("validation_errors"):
            logger.warning(f"Validation errors: {final_state['validation_errors']}")
            
    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 