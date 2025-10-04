# Quick Start Guide

## Get Your Unified API Running in 3 Steps

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Pull Required Models

```bash
ollama pull gemma3:4b
ollama pull embeddinggemma:300m
ollama pull llava:7b
```

### Step 3: Start the Server

```bash
python api_server_integrated.py
```

That's it!

## Direct approach

### Linux:

```bash
chmod +x start_server.sh
./start_server.sh
```

### Windows:
```cmd
start_server.bat
```

## Access Your API

- **API Server**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc


## Example Usage

### Python

```python
import requests

# Summarization
response = requests.post(
    "http://localhost:8000/summarize",
    json={
        "question": "What is microgravity?",
        "top_k_texts": ["Microgravity is..."]
    }
)
print(response.json()["answer"])

# Text Embedding
response = requests.post(
    "http://localhost:8000/embed/text",
    data={"text": "Hello world"}
)
print(f"Dimension: {response.json()['dimension']}")
```

### cURL

```bash
# Health Check
curl http://localhost:8000/health

# Summarization
curl -X POST "http://localhost:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is microgravity?",
    "top_k_texts": ["Microgravity is a condition..."]
  }'

# Text Embedding
curl -X POST "http://localhost:8000/embed/text" \
  -F "text=Hello world"

# Image Embedding
curl -X POST "http://localhost:8000/embed/image" \
  -F "file=@image.jpg"

# Audio Embedding
curl -X POST "http://localhost:8000/embed/audio" \
  -F "file=@audio.mp3"
```

## Available Endpoints

1. **POST** `/summarize` - Summarize retrieved document chunks
2. **POST** `/embed/text` - Generate text embeddings
3. **POST** `/embed/image` - Generate image embeddings
4. **POST** `/embed/audio` - Generate audio embeddings
5. **GET** `/health` - Check server health
6. **GET** `/` - API information
7. **GET** `/docs` - Interactive API documentation

## Troubleshooting

### Server won't start?

```bash
# Check if Ollama is installed
ollama --version

# If not installed, download from:
# https://ollama.ai/download
```

### Missing models?

```bash
# List installed models
ollama list

# Pull missing models
ollama pull gemma3:4b
ollama pull embeddinggemma:300m
ollama pull llava:7b
```

### Port already in use?

Edit `api_server_integrated.py` and change:
```python
uvicorn.run(app, host="0.0.0.0", port=8001)  # Change to 8001
```

### Audio processing fails?

Install FFmpeg:
- **Windows**: Download from https://ffmpeg.org/download.html
- **Mac**: `brew install ffmpeg`
- **Linux**: `sudo apt-get install ffmpeg`

  
## What's Included?

This unified API combines:

✅ **Content Summarization**
- Answer questions using retrieved documents
- Powered by Gemma 3 4B model

✅ **Multi-Modal Embeddings**
- Text embeddings
- Image embeddings (via LLaVA vision model)
- Audio embeddings (via Whisper transcription)

✅ **Automatic Ollama Management**
- No need to run `ollama serve` manually
- Automatic startup and cleanup

✅ **Production Ready**
- Comprehensive logging
- Error handling
- Health checks
- CORS enabled
- API documentation

## Next Steps

1. ✅ Server running? → Test with `test_integrated_api.py`
2. ✅ Tests passing? → Check docs at `/docs`
3. ✅ Ready to integrate? → See `README_INTEGRATED_API.md` for examples
4. ✅ Deploying? → See deployment section in README

## Tips

- **Performance**: GPU-enabled Ollama models run faster
- **Monitoring**: Check `api_server.log` for detailed logs
- **Development**: Use `/docs` for interactive testing
- **Integration**: All endpoints return JSON

---

Made for NASA Space Apps
