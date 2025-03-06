import os
import logging
from typing import Dict
from google.oauth2 import service_account
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from google.cloud import aiplatform
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Suppress gRPC and TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('absl').setLevel(logging.ERROR)

# Configuration constants
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

RISK_ANALYSIS_QUERIES = {
    "requirements": ["REQUISITOS DA CONTRATAÇÃO"],
    "responsibilities": ["PAPÉIS E RESPONSABILIDADES"],
    "execution": ["MODELO DE EXECUÇÃO DO CONTRATO"],
    "management": ["MODELO DE GESTÃO DO CONTRATO"],
    "selection": ["CRITÉRIOS DE SELEÇÃO DO FORNECEDOR"]
}

# Google Cloud configuration
GOOGLE_CREDENTIALS_PATH = os.getenv('GOOGLE_CLOUD_CREDENTIALS_PATH')
if not GOOGLE_CREDENTIALS_PATH:
    raise ValueError("GOOGLE_CLOUD_CREDENTIALS_PATH environment variable is not set")

credentials = service_account.Credentials.from_service_account_file(
    GOOGLE_CREDENTIALS_PATH
)

# Initialize Google Cloud SDK
aiplatform.init(
    project="gen-lang-client-0178129527",
    location="us-central1",
    credentials=credentials
)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    credentials=credentials,
    project="gen-lang-client-0178129527"
)

# Model initialization with retries
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def initialize_models() -> Dict:
    """Initialize and return model configurations"""
    return {
        "small_model": ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            max_output_tokens=8192,
            top_p=0.95,
            top_k=40
            ),
        "large_model": ChatGoogleGenerativeAI(
            model="gemini-2.0-pro-exp-02-05",
            temperature=0.3,
            max_output_tokens=8192,
            top_p=0.95,
            top_k=40
        ),
        "thinking_model": ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-thinking-exp-01-21",
            temperature=0.3,
            max_output_tokens=8192,
            top_p=0.95,
            top_k=40
        )
    }

# Initialize models
models = initialize_models()

# Export commonly used items
__all__ = [
    'logger',
    'embeddings',
    'models',
    'CACHE_DIR',
    'RISK_ANALYSIS_QUERIES'
]
