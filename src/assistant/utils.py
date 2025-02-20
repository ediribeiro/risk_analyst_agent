import hashlib
import tiktoken
import logging
import json
import os
import time
import traceback
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from typing import List, Dict, Union
from langchain.schema import Document
from .configuration import CACHE_DIR, embeddings, logger
import pandas as pd

# Define the tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Define the count_tokens function
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Define the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=8000,
    chunk_overlap=400,
    length_function=count_tokens,
    separators=[
        "\n\n\n",
        "\n\n",
        "\n",
        ". ",
        "! ",
        "? ",
        ";",
        ",",
        " ",
        ""
    ]
)

def get_document_hash(file_path: str) -> str:
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def resolve_file_path(file_path: str) -> str:
    if os.path.exists(file_path):
        return file_path
    raise FileNotFoundError(f"Could not find file at path: {file_path}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def load_and_process_pdf(file_path: str) -> FAISS:
    """Load and process PDF document with caching
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        FAISS: Vector store containing document embeddings
    """
    try:
        # Check cache first
        file_path = resolve_file_path(file_path)
        doc_hash = get_document_hash(file_path)
        cache_file = os.path.join(CACHE_DIR, f"{doc_hash}.faiss")

        if os.path.exists(cache_file):
            logger.info(f"Loading cached embeddings from {cache_file}")
            return FAISS.load_local(cache_file, embeddings, allow_dangerous_deserialization=True)

        # Process new document
        logger.info(f"Processing new document: {os.path.basename(file_path)}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Clean and add metadata
        processed_docs = [
            Document(
                page_content=doc.page_content.strip(),
                metadata={**doc.metadata, 'source': os.path.basename(file_path)}
            )
            for doc in documents 
            if doc.page_content.strip()
        ]

        # Create and cache vector store
        splits = text_splitter.split_documents(processed_docs)
        logger.info(f"Created {len(splits)} splits for vector search")
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(cache_file)
        
        return vectorstore

    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def perform_rag_search(vectorstore: FAISS, queries: Dict[str, List[str]]) -> List[str]:
    """Execute similarity search for given queries
    
    Args:
        vectorstore: FAISS vector store containing document embeddings
        queries: Dictionary mapping section names to search queries
        
    Returns:
        List[str]: List of formatted context strings, each containing:
            - Section title
            - Document content
            - Source metadata (filename and page number)
    """
    try:
        contexts = []
        for stage_name, stage_queries in queries.items():
            for query in stage_queries:
                docs = vectorstore.similarity_search(
                    query,
                    k=2,
                    search_kwargs={"score_threshold": 0.7}
                )
                
                for doc in docs:
                    # Format context with metadata
                    source = doc.metadata.get('source', 'unknown')
                    page = doc.metadata.get('page', 1)
                    
                    context = f"""Termo de busca: {query}:
                    
                    {doc.page_content.strip()}
                    
                    [Fonte: {source}, Página: {page}]
                    """
                    
                    contexts.append(context)
                    
        return contexts
        
    except Exception as e:
        logger.error(f"RAG search failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def rate_limit(max_calls: int, period: int):
    """Decorator to enforce rate limiting on function calls"""
    def decorator(func):
        calls = []
        def wrapper(*args, **kwargs):
            now = time.time()
            calls[:] = [call for call in calls if call > now - period]
            if len(calls) >= max_calls:
                sleep_time = period - (now - calls[0])
                time.sleep(sleep_time)
            result = func(*args, **kwargs)
            calls.append(time.time())
            return result
        return wrapper
    return decorator

def extract_json_from_response(response) -> List[Dict]:
    """Extract and parse JSON from LLM response"""
    try:
        # Handle different response types
        if hasattr(response, 'content'):
            text = str(response.content)
        elif isinstance(response, str):
            text = response
        else:
            text = str(response)
            
        # Clean and find JSON
        text = text.strip()
        start = text.rfind('[')
        end = text.rfind(']') + 1
        
        if start >= 0 and end > start:
            json_str = text[start:end]
            return json.loads(json_str)
            
        raise ValueError("No valid JSON array found in response")
        
    except Exception as e:
        logger.error(f"Failed to parse response: {str(e)}")
        logger.error(f"Raw response: {text[:500]}")
        raise

def split_json_array(json_str: str, max_tokens: int = 4096) -> List[str]:
    """Split JSON array into chunks respecting token limits"""
    try:
        risks = json.loads(json_str)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for risk in risks:
            risk_str = json.dumps([risk], ensure_ascii=False)
            risk_tokens = count_tokens(risk_str)
            
            if current_tokens + risk_tokens > max_tokens:
                chunks.append(json.dumps(current_chunk, ensure_ascii=False))
                current_chunk = [risk]
                current_tokens = risk_tokens
            else:
                current_chunk.append(risk)
                current_tokens += risk_tokens
        
        if current_chunk:
            chunks.append(json.dumps(current_chunk, ensure_ascii=False))
            
        return chunks
        
    except Exception as e:
        logger.error(f"Error splitting JSON: {str(e)}")
        raise

def merge_json_responses(responses: List[Union[str, List, Dict]]) -> str:
    """Merge multiple JSON array responses into single array"""
    try:
        merged = []
        for resp in responses:
            # Handle different input types
            if isinstance(resp, str):
                risks = json.loads(resp)
            elif isinstance(resp, list):
                risks = resp
            elif isinstance(resp, dict):
                risks = [resp]
            else:
                raise TypeError(f"Unexpected response type: {type(resp)}")
                
            merged.extend(risks)
            
        return json.dumps(merged, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error merging responses: {str(e)}")
        raise

def process_risk_data(evaluated_risks: str) -> pd.DataFrame:
    """Process evaluated risks data and calculate risk scores"""
    try:
        # Load JSON to DataFrame
        df = pd.read_json(evaluated_risks)
        
        # Handle column name variations
        if 'Relacionado' in df.columns:
            if 'Relacionado ao' in df.columns:
                # Merge columns, prefer 'Relacionado ao' where both exist
                df['Relacionado ao'] = df['Relacionado ao'].combine_first(df['Relacionado'])
                df.drop(columns=['Relacionado'], inplace=True)
            else:
                # Rename if only 'Relacionado' exists
                df.rename(columns={'Relacionado': 'Relacionado ao'}, inplace=True)
        
        # Calculate derived columns
        df['Impacto Geral'] = df['Impacto Financeiro'] + df['Impacto no Cronograma'] + df['Impacto Reputacional']
        df['Pontuação Geral'] = df['Probabilidade'] * df['Impacto Geral']
        
        # Classify risk levels
        bins = [-1, 10, 20, float('inf')]
        labels = ['Baixo', 'Médio', 'Alto']
        df['Nível de Risco'] = pd.cut(df['Pontuação Geral'], bins=bins, labels=labels)
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing risk data: {str(e)}")
        raise
