import os
from pathlib import Path
from dotenv import load_dotenv

# Set the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load the .env file from the project root
dotenv_path = BASE_DIR / '.env'
load_dotenv(dotenv_path=dotenv_path)

class Settings:
    # API Keys
    LLAMA_PARSE_API_KEY = os.getenv("LLAMA_PARSE_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # File Paths as Path objects (matching your original naming)
    raw_pdfs_dir = BASE_DIR / os.getenv("RAW_PDFS_DIR", "data/raw_pdfs")
    parsed_docs_dir = BASE_DIR / os.getenv("PARSED_DOCS_DIR", "data/processed/parsed_docs")
    chunks_dir = BASE_DIR / os.getenv("CHUNKS_DIR", "data/processed/chunks")
    metadata_dir = BASE_DIR / os.getenv("METADATA_DIR", "data/processed/metadata")
    vector_db_dir = BASE_DIR / os.getenv("VECTOR_DB_DIR", "data/vector_db")
    
    # Also provide _PATH versions for consistency
    RAW_PDFS_PATH = raw_pdfs_dir
    PARSED_DOCS_PATH = parsed_docs_dir
    CHUNKS_PATH = chunks_dir
    METADATA_PATH = metadata_dir
    VECTOR_DB_PATH = vector_db_dir
    
    # Processing Parameters
    chunk_size = int(os.getenv("CHUNK_SIZE", 1024))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Also provide uppercase versions
    CHUNK_SIZE = chunk_size
    CHUNK_OVERLAP = chunk_overlap
    EMBEDDING_MODEL = embedding_model

    def validate_api_keys(self):
        """Check if API keys are properly set"""
        missing = []
        if not self.LLAMA_PARSE_API_KEY:
            missing.append("LLAMA_PARSE_API_KEY")
        if not self.GOOGLE_API_KEY:
            missing.append("GOOGLE_API_KEY")
        
        if missing:
            print(f"❌ Missing API keys: {', '.join(missing)}")
            return False
        print("✅ All API keys are set")
        return True

# Create settings instance
settings = Settings()