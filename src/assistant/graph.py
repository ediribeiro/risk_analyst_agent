import logging
from typing import Dict
from langgraph.graph import StateGraph, END
from .state import State
from .agents.creator import CreatorAgent
from .agents.evaluator import EvaluatorAgent 
from .agents.optimizator import OptimizationAgent
from .utils import load_and_process_pdf, perform_rag_search
from .configuration import RISK_ANALYSIS_QUERIES, logger, models
import json

def create_report(state: State) -> Dict:
    """Node function for creating initial risk report"""
    try:
        creator = CreatorAgent(models["small_model"])
        update = creator.generate(state)
        
        if not update or not update.get("risk_list"):
            raise ValueError("Failed to generate report content")
                
        additional_update = {
            "token_usage": {"generation": len(update["risk_list"])},
        }
        update.update(additional_update)
        return update
        
    except Exception as e:
        error_msg = f"Error in create_report: {str(e)}"
        logger.error(error_msg)
        return {
            "risk_list": "",
            "risk_analysis": [],
            "iteration": 0,
            "token_usage": {"generation": 0},
        }

def evaluate_report(state: State) -> Dict:
    """Node function for evaluating risks"""
    try:
        if not state.get("risk_list"):
            raise ValueError("No report content to evaluate")
            
        evaluator = EvaluatorAgent(models["large_model"])
        update = evaluator.evaluate(state)
        
        if not update:
            raise ValueError("Failed to generate evaluation result")
        
        return update
        
    except Exception as e:
        error_msg = f"Error in evaluate_report: {str(e)}"
        logger.error(error_msg)
        return {
            "risk_list": state.get("risk_list", ""),
            "risk_analysis": state.get("risk_analysis", []),
            "iteration": state.get("iteration", 0),
            "validation_errors": state.get("validation_errors", []),
            "token_usage": state.get("token_usage", {}),
        }

def optimize_report(state: State) -> Dict:
    """Node function for optimizing risk analysis"""
    try:
        if not state.get("risk_list") or not state.get("risk_analysis"):
            raise ValueError("Missing required state: risk_list or risk_analysis")
            
        optimizer = OptimizationAgent(models["small_model"])
        update = optimizer.optimize(state)
        
        if not update.get("risk_list"):
            raise ValueError("Failed to generate optimized report")
                
        return update
        
    except Exception as e:
        error_msg = f"Error in optimize_report: {str(e)}"
        logger.error(error_msg)
        return {
            "risk_list": state.get("risk_list", ""),
            "risk_analysis": state.get("risk_analysis", []),
            "iteration": state.get("iteration", 0),
            "validation_errors": state.get("validation_errors", []),
            "token_usage": state.get("token_usage", {}),
        }

def load_document(state: State) -> Dict:
    """Initial node that loads and processes document"""
    try:
        if not state.get("input_file"):
            raise ValueError("No input file provided")
            
        # Load and process document
        vectorstore = load_and_process_pdf(state["input_file"])
        
        # Perform RAG search
        contexts = perform_rag_search(vectorstore, RISK_ANALYSIS_QUERIES)
        
        return {
            "context": contexts,
            "token_usage": {"search": sum(len(c) for c in contexts)}
        }
        
    except Exception as e:
        error_msg = f"Error loading document: {str(e)}"
        logger.error(error_msg)
        return {"validation_errors": [error_msg]}

def get_initial_state() -> Dict:
    """Returns minimal required state to start workflow"""
    return {
        "input_file": "",
        "risk_list": "",
        "iteration": 0
    }

def create_workflow() -> StateGraph:
    """Creates and configures the workflow graph"""
    # Create workflow graph
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("load_document", load_document)
    workflow.add_node("create_report", create_report)
    workflow.add_node("evaluate_report", evaluate_report)
    workflow.add_node("optimize_report", optimize_report)

    # Set entry point
    workflow.set_entry_point("load_document")

    # Add edges
    workflow.add_edge("load_document", "create_report")
    workflow.add_edge("create_report", "evaluate_report")
    workflow.add_edge("evaluate_report", "optimize_report")
    workflow.add_edge("optimize_report", END)

    return workflow
