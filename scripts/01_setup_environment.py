import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnvironmentSetup:
    """Sets up the IRC RAG system environment."""

    REQUIRED_FOLDERS = [
        "data/raw_pdfs",
        "data/processed/parsed_docs",
        "data/processed/chunks",
        "data/processed/metadata",
        "data/vector_db",
        "src/config",
        "src/processing",
        "src/database",
        "src/api",
        "src/utils",
        "scripts",
        "tests"
    ]

    ENV_TEMPLATE = """# API Keys
LLAMA_PARSE_API_KEY=your_llama_parse_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# File Paths
RAW_PDFS_DIR=data/raw_pdfs
PARSED_DOCS_DIR=data/processed/parsed_docs
CHUNKS_DIR=data/processed/chunks
METADATA_DIR=data/processed/metadata
VECTOR_DB_DIR=data/vector_db

# Processing Parameters
CHUNK_SIZE=1024
CHUNK_OVERLAP=200
EMBEDDING_MODEL=intfloat/e5-large-v2

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True
"""

    def __init__(self):
        """Initialize setup with project root directory."""
        self.root_dir = Path(__file__).resolve().parent.parent
        self.env_file = self.root_dir / ".env"
        self.requirements_file = self.root_dir / "requirements.txt"

    def create_folders(self) -> List[Tuple[str, bool]]:
        """
        Create all required folders if they don't exist.
        
        Returns:
            List of tuples containing (folder_path, creation_success)
        """
        results = []
        for folder in self.REQUIRED_FOLDERS:
            folder_path = self.root_dir / folder
            try:
                folder_path.mkdir(parents=True, exist_ok=True)
                results.append((folder, True))
                logger.info(f"✅ Created/verified folder: {folder}")
            except Exception as e:
                results.append((folder, False))
                logger.error(f"❌ Failed to create folder {folder}: {e}")
        return results

    def setup_env_file(self) -> bool:
        """
        Create .env file from template if it doesn't exist.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.env_file.exists():
                with open(self.env_file, 'w', encoding='utf-8') as f:
                    f.write(self.ENV_TEMPLATE)
                logger.info(f"✅ Created {self.env_file}")
                logger.warning("⚠️  Please update the .env file with your actual API keys!")
            else:
                logger.info(f"✅ .env file already exists")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create .env file: {e}")
            return False

    def create_requirements_if_missing(self) -> bool:
        """Create requirements.txt if it doesn't exist."""
        requirements_content = """llama-parse==0.4.0
fastapi==0.104.1
uvicorn==0.24.0
chromadb==0.4.15
google-generativeai==0.8.0
pydantic==2.5.0
python-dotenv==1.0.0
aiofiles==23.2.1
tiktoken==0.5.1
sentence-transformers==2.2.2
tqdm==4.66.1
numpy
pandas
"""
        try:
            if not self.requirements_file.exists():
                with open(self.requirements_file, 'w', encoding='utf-8') as f:
                    f.write(requirements_content)
                logger.info(f"✅ Created {self.requirements_file}")
            else:
                logger.info(f"✅ requirements.txt already exists")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create requirements.txt: {e}")
            return False

    def install_dependencies(self) -> bool:
        """
        Install Python dependencies from requirements.txt.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("📦 Installing dependencies...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e.stderr}")
            logger.info("💡 You can try installing manually: pip install -r requirements.txt")
            return False
        except Exception as e:
            logger.error(f"❌ Error during dependency installation: {e}")
            return False

    def validate_setup(self) -> bool:
        """
        Validate the environment setup.
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        validation_passed = True

        logger.info("🔍 Validating setup...")

        # Check folders
        for folder in self.REQUIRED_FOLDERS:
            folder_path = self.root_dir / folder
            if folder_path.exists():
                logger.info(f"✅ Folder exists: {folder}")
            else:
                logger.error(f"❌ Required folder missing: {folder}")
                validation_passed = False

        # Check .env file
        if self.env_file.exists():
            logger.info("✅ .env file exists")
            
            # Check if API keys are set (not the template values)
            try:
                with open(self.env_file, 'r') as f:
                    env_content = f.read()
                    if "your_llama_parse_api_key_here" in env_content:
                        logger.warning("⚠️  Please update LLAMA_PARSE_API_KEY in .env file")
                    if "your_google_api_key_here" in env_content:
                        logger.warning("⚠️  Please update GOOGLE_API_KEY in .env file")
            except Exception as e:
                logger.warning(f"⚠️  Could not read .env file: {e}")
        else:
            logger.error("❌ .env file is missing")
            validation_passed = False

        # Check requirements.txt
        if self.requirements_file.exists():
            logger.info("✅ requirements.txt exists")
        else:
            logger.error("❌ requirements.txt is missing")
            validation_passed = False

        return validation_passed

    def check_api_keys(self) -> bool:
        """Check if API keys are properly set."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            llama_key = os.getenv("LLAMA_PARSE_API_KEY")
            google_key = os.getenv("GOOGLE_API_KEY")
            
            logger.info("🔑 Checking API keys...")
            
            if llama_key and llama_key != "your_llama_parse_api_key_here":
                logger.info("✅ LLAMA_PARSE_API_KEY is set")
            else:
                logger.warning("⚠️  LLAMA_PARSE_API_KEY not set or using template value")
                
            if google_key and google_key != "your_google_api_key_here":
                logger.info("✅ GOOGLE_API_KEY is set")
            else:
                logger.warning("⚠️  GOOGLE_API_KEY not set or using template value")
                
            return True
        except ImportError:
            logger.warning("⚠️  python-dotenv not installed, cannot check API keys")
            return False
        except Exception as e:
            logger.error(f"❌ Error checking API keys: {e}")
            return False

    def print_next_steps(self):
        """Print instructions for next steps."""
        print("\n" + "="*60)
        print("🎉 SETUP COMPLETE! Next Steps:")
        print("="*60)
        print("1. 🔑 Update API keys in .env file:")
        print("   - Get LlamaParse key: https://cloud.llamaindex.ai/")
        print("   - Get Google key: https://aistudio.google.com/app/apikey")
        print()
        print("2. 📄 Add your IRC PDF files to:")
        print("   data/raw_pdfs/")
        print()
        print("3. 🚀 Run the processing scripts:")
        print("   python scripts/02_process_documents.py")
        print("   python scripts/03_create_vector_db.py")
        print("   python scripts/04_test_system.py")
        print("   python scripts/05_start_api.py")
        print()
        print("4. 🔍 Quick test:")
        print("   python -c \"from dotenv import load_dotenv; load_dotenv(); import os; print('Keys set:', bool(os.getenv('LLAMA_PARSE_API_KEY')))\"")
        print("="*60 + "\n")

def main():
    """Main setup function."""
    setup = EnvironmentSetup()
    
    logger.info("🚀 Starting IRC RAG System setup...")
    
    # Create requirements.txt if missing
    if not setup.create_requirements_if_missing():
        logger.error("❌ Failed to create requirements.txt")
        return

    # Create folders
    folder_results = setup.create_folders()
    failed_folders = [f for f, success in folder_results if not success]
    
    if failed_folders:
        logger.error(f"❌ Failed to create folders: {failed_folders}")
        return

    # Setup .env file
    if not setup.setup_env_file():
        logger.error("❌ Failed to setup .env file")
        return

    # Install dependencies (optional - may fail on Windows)
    logger.info("📦 Attempting to install dependencies...")
    deps_installed = setup.install_dependencies()
    if not deps_installed:
        logger.warning("⚠️  Dependency installation failed. You can install manually later.")

    # Validate setup
    if not setup.validate_setup():
        logger.error("❌ Setup validation failed")
        return

    # Check API keys
    setup.check_api_keys()

    logger.info("✅ Environment setup completed!")
    setup.print_next_steps()

if __name__ == "__main__":
    main()