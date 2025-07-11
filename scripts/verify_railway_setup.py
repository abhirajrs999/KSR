#!/usr/bin/env python3
"""
Pre-deployment verification script for Railway deployment.

This script checks that all necessary files and configurations are in place
before deploying the IRC RAG system to Railway.

Usage:
    python scripts/verify_railway_setup.py
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RailwaySetupVerifier:
    """Verifies Railway deployment setup."""
    
    def __init__(self):
        """Initialize verifier."""
        self.project_root = Path(__file__).parent.parent
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = []
        self.errors = []
    
    def check_file_exists(self, file_path: Path, required: bool = True) -> bool:
        """Check if a file exists."""
        if file_path.exists():
            logger.info(f"✅ Found: {file_path.relative_to(self.project_root)}")
            return True
        else:
            if required:
                logger.error(f"❌ Missing required file: {file_path.relative_to(self.project_root)}")
                self.errors.append(f"Missing required file: {file_path.relative_to(self.project_root)}")
            else:
                logger.warning(f"⚠️  Optional file missing: {file_path.relative_to(self.project_root)}")
                self.warnings.append(f"Optional file missing: {file_path.relative_to(self.project_root)}")
            return False
    
    def check_gitignore(self) -> bool:
        """Check .gitignore configuration."""
        logger.info("Checking .gitignore configuration...")
        gitignore_path = self.project_root / ".gitignore"
        
        if not self.check_file_exists(gitignore_path):
            return False
        
        try:
            with open(gitignore_path, 'r') as f:
                content = f.read()
            
            required_patterns = [
                'data/',
                '*.log',
                '__pycache__/',
                '.env'
            ]
            
            missing_patterns = []
            for pattern in required_patterns:
                if pattern not in content:
                    missing_patterns.append(pattern)
            
            if missing_patterns:
                logger.warning(f"⚠️  .gitignore missing patterns: {', '.join(missing_patterns)}")
                self.warnings.append(f".gitignore missing patterns: {', '.join(missing_patterns)}")
                return False
            else:
                logger.info("✅ .gitignore properly configured")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error reading .gitignore: {e}")
            self.errors.append(f"Error reading .gitignore: {e}")
            return False
    
    def check_railway_toml(self) -> bool:
        """Check railway.toml configuration."""
        logger.info("Checking railway.toml configuration...")
        railway_toml_path = self.project_root / "railway.toml"
        
        if not self.check_file_exists(railway_toml_path):
            return False
        
        try:
            with open(railway_toml_path, 'r') as f:
                content = f.read()
            
            required_sections = [
                '[build]',
                '[deploy]',
                '[[mounts]]',
                'mountPath = "/data"'
            ]
            
            missing_sections = []
            for section in required_sections:
                if section not in content:
                    missing_sections.append(section)
            
            if missing_sections:
                logger.error(f"❌ railway.toml missing sections: {', '.join(missing_sections)}")
                self.errors.append(f"railway.toml missing sections: {', '.join(missing_sections)}")
                return False
            else:
                logger.info("✅ railway.toml properly configured")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error reading railway.toml: {e}")
            self.errors.append(f"Error reading railway.toml: {e}")
            return False
    
    def check_dockerfile(self) -> bool:
        """Check Dockerfile configuration."""
        logger.info("Checking Dockerfile...")
        dockerfile_path = self.project_root / "Dockerfile"
        
        if not self.check_file_exists(dockerfile_path):
            return False
        
        try:
            with open(dockerfile_path, 'r') as f:
                content = f.read()
            
            required_elements = [
                'FROM python:',
                'WORKDIR /app',
                'COPY requirements.txt',
                'RUN pip install',
                'EXPOSE',
                'CMD'
            ]
            
            missing_elements = []
            for element in required_elements:
                if element not in content:
                    missing_elements.append(element)
            
            if missing_elements:
                logger.warning(f"⚠️  Dockerfile missing elements: {', '.join(missing_elements)}")
                self.warnings.append(f"Dockerfile missing elements: {', '.join(missing_elements)}")
                return False
            else:
                logger.info("✅ Dockerfile properly configured")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error reading Dockerfile: {e}")
            self.errors.append(f"Error reading Dockerfile: {e}")
            return False
    
    def check_requirements_txt(self) -> bool:
        """Check requirements.txt."""
        logger.info("Checking requirements.txt...")
        requirements_path = self.project_root / "requirements.txt"
        
        if not self.check_file_exists(requirements_path):
            return False
        
        try:
            with open(requirements_path, 'r') as f:
                content = f.read()
            
            critical_packages = [
                'fastapi',
                'uvicorn',
                'chromadb',
                'google-generativeai'
            ]
            
            missing_packages = []
            for package in critical_packages:
                if package not in content.lower():
                    missing_packages.append(package)
            
            if missing_packages:
                logger.warning(f"⚠️  requirements.txt missing packages: {', '.join(missing_packages)}")
                self.warnings.append(f"requirements.txt missing packages: {', '.join(missing_packages)}")
                return False
            else:
                logger.info("✅ requirements.txt includes critical packages")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error reading requirements.txt: {e}")
            self.errors.append(f"Error reading requirements.txt: {e}")
            return False
    
    def check_main_py(self) -> bool:
        """Check main.py FastAPI entry point."""
        logger.info("Checking main.py...")
        main_py_path = self.project_root / "main.py"
        
        if not self.check_file_exists(main_py_path):
            return False
        
        try:
            with open(main_py_path, 'r') as f:
                content = f.read()
            
            required_elements = [
                'FastAPI',
                'RAILWAY_ENVIRONMENT',
                '/data',
                'health',
                'app = '
            ]
            
            missing_elements = []
            for element in required_elements:
                if element not in content:
                    missing_elements.append(element)
            
            if missing_elements:
                logger.warning(f"⚠️  main.py missing elements: {', '.join(missing_elements)}")
                self.warnings.append(f"main.py missing elements: {', '.join(missing_elements)}")
                return False
            else:
                logger.info("✅ main.py properly configured for Railway")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error reading main.py: {e}")
            self.errors.append(f"Error reading main.py: {e}")
            return False
    
    def check_source_structure(self) -> bool:
        """Check source code structure."""
        logger.info("Checking source code structure...")
        
        required_directories = [
            self.project_root / "src",
            self.project_root / "src" / "api",
            self.project_root / "src" / "database",
            self.project_root / "src" / "processing",
            self.project_root / "scripts"
        ]
        
        all_exist = True
        for directory in required_directories:
            if not directory.exists():
                logger.error(f"❌ Missing directory: {directory.relative_to(self.project_root)}")
                self.errors.append(f"Missing directory: {directory.relative_to(self.project_root)}")
                all_exist = False
            else:
                logger.info(f"✅ Found directory: {directory.relative_to(self.project_root)}")
        
        return all_exist
    
    def check_environment_examples(self) -> bool:
        """Check for environment variable examples."""
        logger.info("Checking environment variable documentation...")
        
        env_docs = [
            self.project_root / "RAILWAY_DEPLOYMENT_GUIDE_COMPLETE.md",
            self.project_root / "RAILWAY_DEPLOYMENT_CHECKLIST.md"
        ]
        
        has_env_docs = False
        for doc_path in env_docs:
            if doc_path.exists():
                has_env_docs = True
                logger.info(f"✅ Found deployment documentation: {doc_path.name}")
        
        if not has_env_docs:
            logger.warning("⚠️  No deployment documentation found")
            self.warnings.append("No deployment documentation found")
        
        return has_env_docs
    
    def check_data_directory_excluded(self) -> bool:
        """Check that data directory is not in Git."""
        logger.info("Checking data directory exclusion...")
        
        data_dir = self.project_root / "data"
        if data_dir.exists():
            # Check if data directory has .git tracking
            git_dir = self.project_root / ".git"
            if git_dir.exists():
                try:
                    import subprocess
                    result = subprocess.run(
                        ["git", "ls-files", "data/"],
                        cwd=self.project_root,
                        capture_output=True,
                        text=True
                    )
                    
                    if result.stdout.strip():
                        logger.error("❌ Data directory is tracked by Git - should be in .gitignore")
                        self.errors.append("Data directory is tracked by Git")
                        return False
                    else:
                        logger.info("✅ Data directory properly excluded from Git")
                        return True
                        
                except Exception as e:
                    logger.warning(f"⚠️  Could not check Git status: {e}")
                    self.warnings.append(f"Could not check Git status: {e}")
                    return True
            else:
                logger.info("ℹ️  No Git repository detected")
                return True
        else:
            logger.info("ℹ️  No data directory found (will be created on Railway)")
            return True
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all verification checks."""
        logger.info("=" * 60)
        logger.info("RAILWAY DEPLOYMENT VERIFICATION")
        logger.info("=" * 60)
        
        checks = [
            ("Core Files", [
                ("main.py", lambda: self.check_main_py()),
                ("requirements.txt", lambda: self.check_requirements_txt()),
                ("Dockerfile", lambda: self.check_dockerfile()),
                ("railway.toml", lambda: self.check_railway_toml()),
                (".gitignore", lambda: self.check_gitignore())
            ]),
            ("Project Structure", [
                ("Source directories", lambda: self.check_source_structure()),
                ("Environment docs", lambda: self.check_environment_examples()),
                ("Data exclusion", lambda: self.check_data_directory_excluded())
            ])
        ]
        
        results = {}
        
        for category, category_checks in checks:
            logger.info(f"\n--- {category} ---")
            category_results = {}
            
            for check_name, check_func in category_checks:
                try:
                    result = check_func()
                    category_results[check_name] = result
                    if result:
                        self.checks_passed += 1
                    else:
                        self.checks_failed += 1
                except Exception as e:
                    logger.error(f"❌ Check '{check_name}' failed with error: {e}")
                    category_results[check_name] = False
                    self.checks_failed += 1
                    self.errors.append(f"Check '{check_name}' failed: {e}")
            
            results[category] = category_results
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("VERIFICATION SUMMARY")
        logger.info("=" * 60)
        
        total_checks = self.checks_passed + self.checks_failed
        logger.info(f"Total checks: {total_checks}")
        logger.info(f"Passed: {self.checks_passed}")
        logger.info(f"Failed: {self.checks_failed}")
        logger.info(f"Warnings: {len(self.warnings)}")
        
        if self.checks_failed == 0:
            logger.info("\n🎉 ALL CHECKS PASSED! Ready for Railway deployment.")
        else:
            logger.error(f"\n❌ {self.checks_failed} CRITICAL ISSUES FOUND!")
            logger.error("Please fix the following errors before deployment:")
            for error in self.errors:
                logger.error(f"  • {error}")
        
        if self.warnings:
            logger.warning(f"\n⚠️  {len(self.warnings)} WARNINGS:")
            for warning in self.warnings:
                logger.warning(f"  • {warning}")
        
        return {
            "passed": self.checks_passed,
            "failed": self.checks_failed,
            "warnings": len(self.warnings),
            "ready_for_deployment": self.checks_failed == 0,
            "details": results,
            "errors": self.errors,
            "warnings": self.warnings
        }


def main():
    """Main function."""
    verifier = RailwaySetupVerifier()
    results = verifier.run_all_checks()
    
    # Exit with appropriate code
    if results["ready_for_deployment"]:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
