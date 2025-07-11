# Railway Deployment Guide for IRC RAG API

This guide provides step-by-step instructions for deploying the IRC RAG API on Railway with persistent storage (Option B).

## Prerequisites

1. **Railway Account**: Sign up at [railway.app](https://railway.app)
2. **Railway CLI**: Install the Railway CLI
   ```bash
   npm install -g @railway/cli
   ```
3. **Git Repository**: Your code should be in a Git repository (GitHub, GitLab, etc.)

## Deployment Steps

### Step 1: Create Railway Project

1. Log in to Railway CLI:
   ```bash
   railway login
   ```

2. Create a new project:
   ```bash
   railway create irc-rag-api
   ```

3. Link your local repository:
   ```bash
   railway link
   ```

### Step 2: Set Up Persistent Volume

1. In the Railway dashboard, go to your project
2. Navigate to the **Volumes** section
3. Create a new volume:
   - **Name**: `irc-rag-data`
   - **Size**: 10GB (recommended minimum for PDF storage and vector DB)
   - **Mount Path**: `/data`

### Step 3: Configure Environment Variables

Set the following environment variables in Railway dashboard:

#### Required Variables
```bash
# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Environment Configuration
RAILWAY_ENVIRONMENT=production
PORT=8000

# LlamaParse Configuration (Optional)
LLAMAPARSE_API_KEY=your_llamaparse_api_key_here

# Embedding Model Configuration (Optional)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
```

#### Optional Variables for Advanced Configuration
```bash
# Chunking Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Vector Database Configuration
VECTOR_DB_COLLECTION_NAME=irc_documents

# Logging Level
LOG_LEVEL=INFO
```

### Step 4: Deploy

1. **Automatic Deployment** (Recommended):
   - Connect your GitHub repository to Railway
   - Railway will automatically deploy when you push to the main branch
   - The `railway.toml` file configures the deployment

2. **Manual Deployment**:
   ```bash
   railway up
   ```

### Step 5: Monitor Deployment

1. Check deployment logs in Railway dashboard
2. Verify the health endpoint:
   ```bash
   curl https://your-app-url.up.railway.app/health
   ```

3. Test the API documentation:
   ```bash
   https://your-app-url.up.railway.app/docs
   ```

## Post-Deployment Setup

### Upload Initial Documents

1. **Via API** (recommended for large datasets):
   ```bash
   curl -X POST "https://your-app-url.up.railway.app/upload" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@path/to/your/document.pdf" \
        -F "description=Document description"
   ```

2. **Bulk Upload Script** (for multiple documents):
   ```python
   import requests
   import os
   from pathlib import Path
   
   API_URL = "https://your-app-url.up.railway.app"
   PDF_DIR = "path/to/your/pdfs"
   
   for pdf_file in Path(PDF_DIR).glob("*.pdf"):
       with open(pdf_file, 'rb') as f:
           files = {'file': f}
           data = {'description': f"IRC document: {pdf_file.stem}"}
           response = requests.post(f"{API_URL}/upload", files=files, data=data)
           print(f"Uploaded {pdf_file.name}: {response.status_code}")
   ```

### Initialize Vector Database

If you have existing processed data, you can initialize the vector database:

1. Upload your PDFs via the API (they will be processed automatically)
2. Or run the bulk processing script if you have many documents
3. Monitor processing through the `/documents` endpoint

## Configuration Files Overview

### `railway.toml`
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
  ENVIRONMENT = "production"
```

### `Dockerfile`
- Uses Python 3.12 slim image
- Installs system dependencies
- Sets up the application environment
- Configures persistent data directory
- Includes health check

### `main.py`
- Railway-optimized FastAPI application
- Automatic path configuration for persistent storage
- Enhanced health checks with deployment info
- Proper error handling and logging

### `deploy_init.py`
- Deployment initialization script
- Sets up directory structure
- Initializes vector database
- Runs system health checks

## Monitoring and Maintenance

### Health Monitoring

The API includes comprehensive health monitoring:

- **Basic Health**: `GET /health`
- **Deployment Info**: Includes Railway environment details
- **Document Count**: Shows number of documents in vector DB
- **System Status**: Verifies all components are working

### Logging

Logs are available in the Railway dashboard:
- Application logs
- Deployment logs
- Error logs

### Scaling

Railway provides automatic scaling:
- **Vertical Scaling**: Automatically scales resources based on usage
- **Volume Storage**: Persistent data is maintained across deployments
- **Multiple Regions**: Can deploy to different regions for better performance

### Backup Strategy

1. **Volume Snapshots**: Railway provides automatic volume snapshots
2. **Data Export**: Use the `/documents` endpoint to export metadata
3. **Raw PDFs**: Keep original PDFs as backup
4. **Vector DB**: Can be rebuilt from processed data if needed

## Troubleshooting

### Common Issues

1. **Import Errors**:
   - Ensure all dependencies are in `requirements.txt`
   - Check Python path configuration

2. **Volume Mount Issues**:
   - Verify volume is properly mounted to `/data`
   - Check directory permissions

3. **API Key Issues**:
   - Verify environment variables are set correctly
   - Check API key validity

4. **Memory Issues**:
   - Monitor resource usage in Railway dashboard
   - Consider upgrading Railway plan for larger datasets

### Debug Commands

```bash
# Check deployment status
railway status

# View logs
railway logs

# Connect to container (if needed)
railway shell

# Check environment variables
railway variables
```

## Performance Optimization

### For Large Datasets

1. **Batch Processing**: Upload documents in batches
2. **Async Processing**: Use background tasks for document processing
3. **Memory Management**: Monitor memory usage and optimize chunk sizes
4. **Caching**: Implement result caching for frequent queries

### Vector Database Optimization

1. **Embedding Model**: Choose appropriate embedding model for your use case
2. **Chunk Size**: Optimize chunk size for your documents
3. **Indexing**: Ensure proper indexing in ChromaDB
4. **Query Optimization**: Use filters to narrow search scope

## Security Considerations

1. **API Keys**: Store in Railway environment variables, never in code
2. **CORS**: Configure appropriately for production
3. **Rate Limiting**: Implement rate limiting for production use
4. **Authentication**: Add authentication for production deployment
5. **Input Validation**: Validate all file uploads and queries

## Cost Optimization

1. **Resource Planning**: Start with smaller resources and scale as needed
2. **Volume Size**: Estimate storage needs accurately
3. **Monitoring**: Use Railway's usage monitoring
4. **Cleanup**: Regularly clean up unused data

## Support and Updates

- **Railway Documentation**: [docs.railway.app](https://docs.railway.app)
- **API Documentation**: Available at `/docs` endpoint
- **Updates**: Deploy updates by pushing to your connected Git repository

---

This deployment guide provides a comprehensive setup for the IRC RAG API on Railway with persistent storage, ensuring your vector database and uploaded documents persist across deployments.
