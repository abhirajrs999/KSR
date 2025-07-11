#!/usr/bin/env python3
"""
Script to gradually restore full RAG functionality to Railway deployment.
This script helps you test each component step by step.
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

class RAGRestorer:
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.backup_dir = self.root_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
    def backup_current_state(self):
        """Backup current working state."""
        print("📁 Backing up current state...")
        
        # Backup key files
        files_to_backup = [
            "main-minimal.py",
            "requirements.txt", 
            "Dockerfile",
            "railway.toml"
        ]
        
        for file in files_to_backup:
            src = self.root_dir / file
            if src.exists():
                dst = self.backup_dir / f"{file}.working"
                shutil.copy2(src, dst)
                print(f"  ✅ Backed up {file}")
    
    def step1_add_basic_dependencies(self):
        """Step 1: Add basic ML dependencies."""
        print("\n🔧 STEP 1: Adding basic dependencies...")
        
        basic_deps = [
            "chromadb==0.4.15",
            "sentence-transformers==2.2.2",
            "torch==2.1.0",
            "transformers==4.36.0",
            "numpy==1.24.3",
            "pandas==2.0.3"
        ]
        
        # Read current requirements
        req_file = self.root_dir / "requirements.txt"
        with open(req_file, 'r') as f:
            current_reqs = f.read().strip()
        
        # Add new dependencies
        new_reqs = current_reqs + "\n" + "\n".join(basic_deps)
        
        with open(req_file, 'w') as f:
            f.write(new_reqs)
        
        print("  ✅ Added basic ML dependencies to requirements.txt")
        print("  📝 Next: git add, commit, push, and check Railway build")
        
    def step2_create_hybrid_main(self):
        """Step 2: Create hybrid main.py with minimal + some RAG features."""
        print("\n🔧 STEP 2: Creating hybrid main.py...")
        
        hybrid_main = '''"""
FastAPI app with gradual RAG functionality restoration.
Stage 2: Basic structure + health checks.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add src to path
import sys
sys.path.append(str(Path(__file__).resolve().parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Response models
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    message: str
    environment: str
    components: dict

class QueryRequest(BaseModel):
    query: str = Field(..., description="Query to search IRC documents")
    irc_code: Optional[str] = Field(None, description="Optional IRC code filter")
    limit: Optional[int] = Field(5, description="Max results", ge=1, le=20)

# Create FastAPI app
app = FastAPI(
    title="IRC RAG API",
    description="IRC Document Retrieval-Augmented Generation API",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("=== IRC RAG API STARTING ===")
    logger.info(f"Environment: {os.getenv('RAILWAY_ENVIRONMENT', 'local')}")
    logger.info(f"Port: {os.getenv('PORT', '8000')}")
    
    # Try to import and test components
    try:
        import chromadb
        logger.info("✅ ChromaDB available")
    except ImportError as e:
        logger.warning(f"❌ ChromaDB not available: {e}")
    
    try:
        import sentence_transformers
        logger.info("✅ Sentence Transformers available")
    except ImportError as e:
        logger.warning(f"❌ Sentence Transformers not available: {e}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "IRC RAG API - Stage 2",
        "status": "running",
        "environment": os.getenv('RAILWAY_ENVIRONMENT', 'local'),
        "stage": "basic-dependencies"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check with component status."""
    
    components = {}
    
    # Check ChromaDB
    try:
        import chromadb
        components["chromadb"] = "available"
    except ImportError:
        components["chromadb"] = "not_available"
    
    # Check Sentence Transformers
    try:
        import sentence_transformers
        components["sentence_transformers"] = "available"
    except ImportError:
        components["sentence_transformers"] = "not_available"
    
    # Check data directory
    data_path = Path("/data" if os.getenv('RAILWAY_ENVIRONMENT') else "data")
    components["data_directory"] = "exists" if data_path.exists() else "missing"
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        message="IRC RAG API is running - Stage 2",
        environment=os.getenv('RAILWAY_ENVIRONMENT', 'local'),
        components=components
    )

@app.get("/test/dependencies")
async def test_dependencies():
    """Test endpoint to check all dependencies."""
    
    deps = {}
    
    # Test imports
    test_imports = [
        "chromadb",
        "sentence_transformers", 
        "torch",
        "transformers",
        "numpy",
        "pandas"
    ]
    
    for dep in test_imports:
        try:
            __import__(dep)
            deps[dep] = "✅ Available"
        except ImportError as e:
            deps[dep] = f"❌ Error: {str(e)}"
    
    return {
        "status": "dependency_check_complete",
        "dependencies": deps,
        "stage": "basic-dependencies"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', '8000'))
    logger.info(f"Starting uvicorn on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
'''
        
        # Write hybrid main
        main_file = self.root_dir / "main.py"
        with open(main_file, 'w') as f:
            f.write(hybrid_main)
        
        print("  ✅ Created hybrid main.py (Stage 2)")
        print("  📝 Next: Deploy and test /health and /test/dependencies endpoints")
    
    def step3_add_vector_db_support(self):
        """Step 3: Add vector database initialization."""
        print("\n🔧 STEP 3: Adding vector DB support...")
        # This will be implemented after Step 2 success
        print("  🎯 To be implemented after Step 2 deployment success")
    
    def step4_add_document_processing(self):
        """Step 4: Add document processing capabilities."""
        print("\n🔧 STEP 4: Adding document processing...")
        # This will be implemented after Step 3 success
        print("  🎯 To be implemented after Step 3 success")
    
    def step5_full_rag_pipeline(self):
        """Step 5: Full RAG pipeline with query capabilities."""
        print("\n🔧 STEP 5: Full RAG pipeline...")
        # This will be implemented after Step 4 success
        print("  🎯 To be implemented after Step 4 success")
    
    def git_deploy(self, message: str) -> None:
        """Helper to git add, commit, and push."""
        print(f"\n📤 Deploying: {message}")
        
        try:
            subprocess.run(["git", "add", "."], check=True, cwd=self.root_dir)
            subprocess.run(["git", "commit", "-m", message], check=True, cwd=self.root_dir)
            subprocess.run(["git", "push"], check=True, cwd=self.root_dir)
            print("  ✅ Deployed successfully")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Git error: {e}")
    
    def print_instructions(self):
        """Print step-by-step instructions."""
        print("""
🚀 IRC RAG RESTORATION PLAN

Current Status: ✅ Minimal FastAPI app deployed and working

RESTORATION STEPS:

1️⃣ STEP 1: Add Basic Dependencies
   - Run: python restore_functionality.py step1
   - Adds: chromadb, sentence-transformers, torch, etc.
   - Deploy and verify build succeeds

2️⃣ STEP 2: Hybrid Main App  
   - Run: python restore_functionality.py step2
   - Creates main.py with dependency checks
   - Test: /health and /test/dependencies endpoints

3️⃣ STEP 3: Vector Database (Future)
   - Initialize ChromaDB
   - Test vector storage

4️⃣ STEP 4: Document Processing (Future)
   - Add upload capabilities
   - Test document parsing

5️⃣ STEP 5: Full RAG Pipeline (Future)
   - Query engine
   - Complete functionality

📋 USAGE:
   python restore_functionality.py backup     # Backup current state
   python restore_functionality.py step1      # Add dependencies
   python restore_functionality.py step2      # Create hybrid main
   python restore_functionality.py deploy     # Git add/commit/push

🔍 TESTING URLS (replace with your Railway URL):
   https://your-app.railway.app/health
   https://your-app.railway.app/test/dependencies
""")

def main():
    restorer = RAGRestorer()
    
    if len(sys.argv) < 2:
        restorer.print_instructions()
        return
    
    command = sys.argv[1].lower()
    
    if command == "backup":
        restorer.backup_current_state()
    elif command == "step1":
        restorer.backup_current_state()
        restorer.step1_add_basic_dependencies()
        print("\n📝 Next: Run 'python restore_functionality.py deploy' to push changes")
    elif command == "step2":
        restorer.step2_create_hybrid_main()
        print("\n📝 Next: Run 'python restore_functionality.py deploy' to push changes")
    elif command == "deploy":
        message = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Restore RAG functionality step"
        restorer.git_deploy(message)
    else:
        print(f"❌ Unknown command: {command}")
        restorer.print_instructions()

if __name__ == "__main__":
    main()
