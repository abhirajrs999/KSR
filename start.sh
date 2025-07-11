#!/bin/bash
# Railway startup script for IRC RAG API

echo "🚀 Starting IRC RAG API deployment on Railway..."

# Print environment info
echo "Environment: ${RAILWAY_ENVIRONMENT:-local}"
echo "Service: ${RAILWAY_SERVICE_NAME:-irc-rag-api}"
echo "Port: ${PORT:-8000}"

# Run deployment initialization
echo "📋 Running deployment initialization..."
python deploy_init.py

if [ $? -eq 0 ]; then
    echo "✅ Initialization completed successfully"
else
    echo "❌ Initialization failed"
    exit 1
fi

# Start the API server
echo "🔧 Starting API server..."
exec python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
