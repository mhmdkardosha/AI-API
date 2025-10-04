#!/bin/bash
# ============================================================================
# Unified Multi-Modal API Server - Unix/Mac Startup Script
# ============================================================================

echo ""
echo "========================================================================"
echo "   Unified Multi-Modal API Server - Startup Script"
echo "========================================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ from https://www.python.org/downloads/"
    exit 1
fi

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "[ERROR] Ollama is not installed or not in PATH"
    echo "Please install Ollama from https://ollama.ai/download"
    exit 1
fi

echo "[OK] Python installed"
echo "[OK] Ollama installed"
echo ""

sudo apt install ffmpeg -y
pip3 install -r requirements.txt

echo "[OK] Dependencies installed"
echo ""

# Check if models are available
echo "Checking Ollama models..."
MISSING_MODELS=0

if ! ollama list | grep -q "gemma3:4b"; then
    echo "[WARNING] Model 'gemma3:4b' not found"
    echo "Pull it with: ollama pull gemma3:4b"
    MISSING_MODELS=1
fi

if ! ollama list | grep -q "embeddinggemma:300m"; then
    echo "[WARNING] Model 'embeddinggemma:300m' not found"
    echo "Pull it with: ollama pull embeddinggemma:300m"
    MISSING_MODELS=1
fi

if ! ollama list | grep -q "llava:7b"; then
    echo "[WARNING] Model 'llava:7b' not found"
    echo "Pull it with: ollama pull llava:7b"
    MISSING_MODELS=1
fi

if [ $MISSING_MODELS -eq 1 ]; then
    echo ""
    echo "[WARNING] Some models are missing"
    echo ""
    read -p "Do you want to pull the missing models now? [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Pulling models... This may take several minutes..."
        ollama pull gemma3:4b
        ollama pull embeddinggemma:300m
        ollama pull llava:7b
        echo ""
        echo "[OK] Models downloaded"
    else
        echo ""
        echo "[WARNING] Starting without all models"
        echo "Some features may not work correctly"
        echo ""
    fi
fi

echo ""
echo "========================================================================"
echo "   Starting Unified Multi-Modal API Server"
echo "========================================================================"
echo ""
echo "Server will start on: http://localhost:8000"
echo "API Documentation:    http://localhost:8000/docs"
echo "Health Check:         http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "========================================================================"
echo ""

# Start the server
python3 api_server_integrated.py

# If server exits
echo ""
echo "========================================================================"
echo "   Server Stopped"
echo "========================================================================"
echo ""
