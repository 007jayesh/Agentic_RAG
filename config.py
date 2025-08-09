"""Configuration management for Agentic RAG System"""
import os
from pathlib import Path


class Config:
    """Production configuration management"""

    def __init__(self):
        # API Configuration
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # Model Configuration
        self.EMBEDDING_MODEL = "text-embedding-3-small"
        self.EMBEDDING_DIMENSIONS = 1536
        self.LLM_MODEL = "gpt-4o-mini"
        self.LLM_TEMPERATURE = 0.0

        # Processing Configuration
        self.CHUNK_SIZE = 1000
        self.CHUNK_OVERLAP = 200
        self.MAX_RETRIES = 2
        self.RATE_LIMIT_DELAY = 2.0

        # Retrieval Configuration
        self.DEFAULT_K = 4
        self.SIMILARITY_THRESHOLD = 0.3
        self.MMR_DIVERSITY_FACTOR = 0.5

        # Quality Thresholds
        self.MIN_DOCUMENT_SCORE = 4.0
        self.HIGH_CONFIDENCE_THRESHOLD = 7.0
        self.MEDIUM_CONFIDENCE_THRESHOLD = 5.0

        # Paths
        self.DATA_PATH = Path("data")
        self.VECTORSTORE_PATH = Path("vectorstore")
        self.RESULTS_PATH = Path("results")
        
        # Ensure directories exist
        for path in [self.DATA_PATH, self.VECTORSTORE_PATH, self.RESULTS_PATH]:
            path.mkdir(exist_ok=True)

    def validate(self):
        """Validate configuration"""
        try:
            from langchain_openai import OpenAIEmbeddings
            test_embeddings = OpenAIEmbeddings(
                openai_api_key=self.OPENAI_API_KEY,
                model=self.EMBEDDING_MODEL
            )
            test_embeddings.embed_query("test")
            return True
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return False