# Railway Deployment Guide - IRC RAG System

This guide provides step-by-step instructions for deploying the IRC RAG system to Railway with persistent storage for large datasets.

## Overview

This deployment uses **Railway Volumes** for persistent storage, which means:
- ✅ Vector database and processed data persist across deployments
- ✅ Large datasets don't need to be in Git repository
- ✅ Faster deployments and better version control
- ✅ Production-ready with health checks and proper configuration

## Prerequisites

1. **Railway Account**: Sign up at [railway.app](https://railway.app)
2. **GitHub Repository**: Code should be pushed to GitHub (excluding large data files)
3. **Environment Variables**: You'll need API keys for LLM services
4. **Data Preparation**: PDF documents should be ready for upload

## Deployment Steps

### Step 1: Prepare Repository

Ensure your repository structure is correct:

```
d:\RAG\
├── src/                    # Source code (included in Git)
├── scripts/                # Setup and utility scripts (included in Git)
├── main.py                 # FastAPI app entry point (included in Git)
├── requirements.txt        # Python dependencies (included in Git)
├── Dockerfile             # Container configuration (included in Git)
├── railway.toml           # Railway configuration (included in Git)
├── .gitignore             # Excludes data/, logs/, cache/ (included in Git)
└── data/                  # Large data files (EXCLUDED from Git)
    ├── raw_pdfs/          # PDF documents
    ├── processed/         # Processed documents
    └── vector_db/         # Vector database
```

**Important**: The `data/` directory and its contents should NOT be in Git due to `.gitignore`.

### Step 2: Push Code to GitHub

```bash
# If not already initialized
git init
git remote add origin https://github.com/yourusername/irc-rag-system.git

# Add and commit code (data/ is excluded by .gitignore)
git add .
git commit -m "Railway deployment ready"
git push -u origin main
```

### Step 3: Create Railway Project

1. Go to [railway.app](https://railway.app) and sign in
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your IRC RAG repository
5. Railway will automatically detect the Dockerfile

### Step 4: Configure Persistent Volume

Railway will automatically create the volume based on your `railway.toml` configuration:

```toml
[[mounts]]
  mountPath = "/data"
  volumeName = "irc-rag-data"
```

This creates a persistent 5GB volume mounted at `/data` in your container.

### Step 5: Set Environment Variables

In the Railway dashboard, go to your project and set these environment variables:

#### Required Variables:
```
GOOGLE_API_KEY=your_gemini_api_key_here
HUGGINGFACE_API_TOKEN=your_hf_token_here
RAILWAY_ENVIRONMENT=production
PORT=8000
```

#### Optional Variables (with defaults):
```
RAW_PDFS_DIR=/data/raw_pdfs
PROCESSED_DOCS_DIR=/data/processed/parsed_docs
CHUNKS_DIR=/data/processed/chunks
METADATA_DIR=/data/processed/metadata
VECTOR_DB_DIR=/data/vector_db
CHROMA_DB_PATH=/data/vector_db/chroma
```

### Step 6: Deploy and Initialize

1. **Deploy**: Railway will automatically deploy after setting environment variables
2. **Wait for Deployment**: Check the deployment logs for any issues
3. **Get Service URL**: Railway will provide a public URL (e.g., `https://your-service.railway.app`)

### Step 7: Initialize Vector Database

After successful deployment, initialize the vector database using one of these methods:

#### Option A: Use Railway's Run Command
1. In Railway dashboard, go to your service
2. Click on "Settings" → "Deploy"
3. Add a one-time deployment command:
```bash
python scripts/railway_setup.py
```

#### Option B: Use Railway CLI (if installed)
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and link to your project
railway login
railway link

# Run setup command
railway run python scripts/railway_setup.py
```

#### Option C: API Call (if service is running)
The setup can also be triggered via API if you add an admin endpoint to your FastAPI app.

### Step 8: Upload PDF Documents

After initialization, upload your PDF documents using one of these methods:

#### Option A: Upload via API
```bash
# Upload a single PDF
curl -X POST "https://your-service.railway.app/upload" \
  -F "file=@/path/to/your/document.pdf" \
  -F "description=IRC Document Description"
```

#### Option B: Bulk Upload Script
If you have many PDFs, you can use the bulk upload script:

```bash
# Copy PDFs to a local directory first, then use the bulk upload script
python scripts/bulk_upload_railway.py \
  --api-url https://your-service.railway.app \
  --pdf-directory /path/to/your/pdfs
```

#### Option C: Manual Upload to Volume
If you have Railway CLI access, you can copy files directly:

```bash
# This requires special Railway volume access - contact Railway support
railway volume:mount irc-rag-data /tmp/volume
cp your-pdfs/*.pdf /tmp/volume/raw_pdfs/
railway run python scripts/railway_setup.py --force
```

### Step 9: Verify Deployment

1. **Health Check**: Visit `https://your-service.railway.app/health`
   - Should return status "healthy" with document count > 0

2. **API Documentation**: Visit `https://your-service.railway.app/docs`
   - Interactive API documentation

3. **Test Query**: Test the API with a sample query
```bash
curl -X POST "https://your-service.railway.app/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the requirements for road construction?",
    "limit": 5
  }'
```

### Step 10: Monitor and Maintain

1. **Logs**: Monitor application logs in Railway dashboard
2. **Metrics**: Check CPU, memory, and volume usage
3. **Updates**: For code updates, just push to GitHub - Railway will auto-deploy
4. **Data Backup**: Consider periodic backups of the `/data` volume

## Important Configuration Files

### `.gitignore`
Ensures large data files are not tracked in Git:
```gitignore
# Data directories (use Railway volumes)
data/
vector_db/
*.db
*.chroma

# Logs and cache
*.log
__pycache__/
.pytest_cache/

# Environment and secrets
.env
.env.*
```

### `railway.toml`
Configures Railway deployment:
```toml
[build]
  builder = "dockerfile"
  
[deploy]
  startCommand = "python -m uvicorn main:app --host 0.0.0.0 --port $PORT"
  
[[mounts]]
  mountPath = "/data"
  volumeName = "irc-rag-data"

[environments.production]
  PORT = "8000"
  RAILWAY_ENVIRONMENT = "production"
```

### `Dockerfile`
Optimized for Railway deployment:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Create data directories with proper permissions
RUN mkdir -p /data/raw_pdfs /data/processed/parsed_docs /data/processed/chunks \
             /data/processed/metadata /data/vector_db && \
    chmod -R 755 /data

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:$PORT/health || exit 1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Troubleshooting

### Common Issues:

1. **Volume Not Mounting**
   - Check `railway.toml` volume configuration
   - Verify volume name matches in Railway dashboard

2. **Environment Variables Not Set**
   - Double-check all required environment variables in Railway dashboard
   - Restart service after setting new variables

3. **PDF Processing Fails**
   - Check that PDFs are valid and not corrupted
   - Monitor memory usage - large PDFs may require more RAM

4. **Vector Database Empty**
   - Run the setup script: `python scripts/railway_setup.py --force`
   - Check logs for embedding generation errors

5. **API Timeouts**
   - Large queries may take time - increase client timeout
   - Consider pagination for large result sets

### Useful Commands:

```bash
# Check deployment status
railway status

# View logs
railway logs

# Connect to service shell
railway shell

# Check environment variables
railway variables

# Restart service
railway up --detach
```

## Production Considerations

1. **Security**:
   - Use proper CORS settings for production
   - Implement API authentication if needed
   - Secure environment variables

2. **Performance**:
   - Monitor resource usage and scale as needed
   - Consider caching for frequent queries
   - Optimize embedding generation

3. **Backup**:
   - Regular backups of the volume data
   - Version control for configuration changes

4. **Monitoring**:
   - Set up alerts for service health
   - Monitor API response times
   - Track error rates

## Cost Optimization

- **Compute**: Railway charges based on resource usage
- **Storage**: Volume storage is billed separately (5GB included)
- **Data Transfer**: Minimal for API usage
- **Sleep Policy**: Service will sleep after inactivity (Hobby plan)

For production workloads, consider upgrading to Railway Pro for:
- No sleep policy
- More compute resources
- Priority support
- Advanced analytics

## Next Steps

After successful deployment:

1. **Custom Domain**: Set up a custom domain in Railway dashboard
2. **SSL Certificate**: Railway provides automatic HTTPS
3. **CI/CD Pipeline**: Set up automated testing before deployment
4. **Monitoring**: Integrate with monitoring services
5. **Scaling**: Monitor usage and scale resources as needed

---

**Need Help?**
- Railway Documentation: [docs.railway.app](https://docs.railway.app)
- Railway Discord: [discord.gg/railway](https://discord.gg/railway)
- GitHub Issues: For code-specific problems
