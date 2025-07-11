# Railway Deployment Checklist for IRC RAG API

This checklist ensures a successful deployment of the IRC RAG API on Railway with persistent storage.

## Pre-Deployment Checklist

### 1. Repository Preparation
- [ ] Code is committed to Git repository (GitHub, GitLab, etc.)
- [ ] `.gitignore` excludes sensitive files and large data
- [ ] All required files are present:
  - [ ] `main.py` (Railway-optimized FastAPI app)
  - [ ] `Dockerfile` (container configuration)
  - [ ] `railway.toml` (Railway deployment configuration)
  - [ ] `requirements.txt` (Python dependencies)
  - [ ] `deploy_init.py` (initialization script)
  - [ ] `src/` directory with all source code
  - [ ] `RAILWAY_DEPLOYMENT_GUIDE.md` (deployment guide)

### 2. Environment Variables Preparation
- [ ] Gemini API key obtained from Google AI Studio
- [ ] LlamaParse API key obtained (optional, for advanced parsing)
- [ ] List of required environment variables prepared:
  ```
  GEMINI_API_KEY=your_gemini_api_key_here
  RAILWAY_ENVIRONMENT=production
  PORT=8000
  LLAMAPARSE_API_KEY=your_llamaparse_key (optional)
  ```

### 3. Data Preparation
- [ ] PDF documents ready for upload
- [ ] Document descriptions prepared (optional)
- [ ] Bulk upload script tested locally (if using large datasets)

## Deployment Steps

### 1. Railway Project Setup
- [ ] Railway account created
- [ ] Railway CLI installed (`npm install -g @railway/cli`)
- [ ] Logged into Railway CLI (`railway login`)
- [ ] New Railway project created (`railway create irc-rag-api`)
- [ ] Local repository linked to Railway project (`railway link`)

### 2. Persistent Volume Configuration
- [ ] Railway volume created:
  - **Name**: `irc-rag-data`
  - **Size**: 10GB (minimum recommended)
  - **Mount Path**: `/data`
- [ ] Volume properly mounted in `railway.toml`

### 3. Environment Variables Setup
- [ ] All required environment variables set in Railway dashboard
- [ ] API keys verified and working
- [ ] Environment-specific variables configured

### 4. Initial Deployment
- [ ] Repository connected to Railway (auto-deploy enabled)
- [ ] First deployment triggered (`git push` or `railway up`)
- [ ] Build logs monitored for errors
- [ ] Deployment successful notification received

### 5. Post-Deployment Verification
- [ ] Application accessible at Railway-provided URL
- [ ] Health check endpoint working (`/health`)
- [ ] API documentation accessible (`/docs`)
- [ ] Basic API functionality tested

## Testing Checklist

### 1. API Endpoint Testing
- [ ] Root endpoint (`/`) returns correct information
- [ ] Health check (`/health`) shows healthy status
- [ ] Documents endpoint (`/documents`) returns empty list (initially)
- [ ] Search endpoint (`/search`) handles queries without errors
- [ ] Query endpoint (`/query`) processes RAG queries (may have no results initially)

### 2. Document Upload Testing
- [ ] Small test PDF upload successful
- [ ] Document processing completes in background
- [ ] Processed document appears in `/documents` endpoint
- [ ] Vector database contains new document chunks

### 3. End-to-End Testing
- [ ] Multiple documents uploaded successfully
- [ ] Search returns relevant results
- [ ] RAG queries generate meaningful responses
- [ ] Citations and source references working correctly

## Bulk Upload Checklist (For Large Datasets)

### 1. Preparation
- [ ] Bulk upload script (`scripts/bulk_upload_railway.py`) ready
- [ ] PDF documents organized in directory structure
- [ ] Railway API URL noted
- [ ] Upload strategy planned (batch sizes, timing)

### 2. Execution
- [ ] API health verified before bulk upload
- [ ] Bulk upload script executed with appropriate parameters
- [ ] Upload progress monitored
- [ ] Failed uploads identified and retried if necessary

### 3. Validation
- [ ] All documents listed in `/documents` endpoint
- [ ] Vector database document count matches expected
- [ ] Sample queries return relevant results from uploaded documents
- [ ] System performance acceptable with full dataset

## Performance Optimization Checklist

### 1. Resource Monitoring
- [ ] Memory usage monitored in Railway dashboard
- [ ] CPU usage within acceptable limits
- [ ] Disk usage tracked (persistent volume)
- [ ] Response times measured and acceptable

### 2. Optimization Actions (if needed)
- [ ] Railway plan upgraded for more resources
- [ ] Chunking parameters optimized
- [ ] Concurrent upload limits adjusted
- [ ] Query response caching implemented

## Security Checklist

### 1. API Security
- [ ] CORS configured appropriately for production
- [ ] API keys stored securely in Railway environment variables
- [ ] No sensitive information in logs or responses
- [ ] Rate limiting considered for production use

### 2. Data Security
- [ ] Uploaded documents contain no sensitive information
- [ ] Vector database properly isolated
- [ ] Backup strategy planned for persistent data

## Monitoring and Maintenance Checklist

### 1. Monitoring Setup
- [ ] Railway application metrics monitored
- [ ] Error logging functional
- [ ] Health check endpoints monitored
- [ ] Performance alerts configured

### 2. Maintenance Planning
- [ ] Update strategy planned for application code
- [ ] Data backup procedures established
- [ ] Disaster recovery plan documented
- [ ] Regular health checks scheduled

## Troubleshooting Checklist

### Common Issues
- [ ] **Build Failures**: Check Dockerfile and requirements.txt
- [ ] **Import Errors**: Verify Python path and module structure
- [ ] **Volume Mount Issues**: Confirm volume configuration in railway.toml
- [ ] **API Key Issues**: Verify environment variables are set correctly
- [ ] **Memory Issues**: Monitor resource usage and consider upgrading plan

### Debug Tools
- [ ] Railway logs accessible (`railway logs`)
- [ ] Test script available (`scripts/test_api_railway.py`)
- [ ] Health check provides detailed deployment information
- [ ] Debug endpoints functional for troubleshooting

## Success Criteria

### Deployment Success
- [ ] Application deployed without errors
- [ ] All API endpoints functional
- [ ] Health check shows healthy status
- [ ] Persistent storage working correctly

### Functionality Success
- [ ] Documents can be uploaded and processed
- [ ] Vector database stores document chunks
- [ ] Search functionality returns relevant results
- [ ] RAG queries generate coherent responses
- [ ] Citations and sources properly attributed

### Performance Success
- [ ] Response times acceptable (< 5s for queries)
- [ ] System stable under normal load
- [ ] Memory usage within allocated limits
- [ ] Persistent storage adequate for dataset

## Post-Deployment Tasks

### 1. Documentation Updates
- [ ] Update README with deployed API URL
- [ ] Document any custom configuration
- [ ] Update API documentation with production examples
- [ ] Share deployment guide with team

### 2. User Onboarding
- [ ] API access instructions provided
- [ ] Example queries documented
- [ ] Upload procedures documented
- [ ] Support contact information provided

### 3. Continuous Improvement
- [ ] User feedback collection planned
- [ ] Performance monitoring ongoing
- [ ] Feature enhancement roadmap planned
- [ ] Regular maintenance schedule established

---

**Deployment Complete!** ✅

Your IRC RAG API is now successfully deployed on Railway with persistent storage and ready for production use.
