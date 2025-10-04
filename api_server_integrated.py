"""
Unified Multi-Modal FastAPI Server
===================================

Combines content summarization and multi-modal embedding generation.
Automatic Ollama server management - no need to run 'ollama serve' manually!

Endpoints:
- POST /summarize - Summarize retrieved document chunks
- POST /embed/text - Generate text embeddings
- POST /embed/image - Generate image embeddings
- POST /embed/audio - Generate audio embeddings
- GET /health - Health check
"""

from fastapi import FastAPI, HTTPException, status, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import requests
import subprocess
import time
import os
import signal
import sys
import atexit
import tempfile
import base64
import logging
from logging.handlers import RotatingFileHandler
from contextlib import asynccontextmanager

# LangChain imports
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

# Whisper for audio processing
import whisper

# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler('api_server.log', maxBytes=10485760, backupCount=5)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Summarization models
SUMMARIZATION_MODEL = "gemma3:4b"
TEMPERATURE = 0.2
MAX_CONTEXT_LENGTH = 10000

# Embedding models
EMBEDDING_MODEL = "embeddinggemma:300m"
TEXT_MODEL = "gemma3:4b"
VISION_MODEL = "llava:7b"
WHISPER_MODEL = "base"

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"

# ============================================================================
# Global Variables
# ============================================================================

# Summarization
llm = None
summarization_chain = None

# Multi-modal embedding
embeddings = None
text_llm = None
vision_llm = None
whisper_model = None

# Ollama process
ollama_process = None

# ============================================================================
# Ollama Server Management
# ============================================================================

def start_ollama_server():
    """Start Ollama server if not already running."""
    global ollama_process
    
    logger.info("Checking if Ollama server is running...")
    
    # Check if already running
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if response.status_code == 200:
            logger.info("Ollama server is already running")
            return True
    except:
        pass
    
    # Start Ollama server
    logger.info("Starting Ollama server...")
    
    try:
        if os.name == 'nt':  # Windows
            ollama_process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:  # Unix/Linux/Mac
            ollama_process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid
            )
        
        # Wait for server to be ready
        logger.info("Waiting for Ollama server to be ready...")
        for i in range(30):
            try:
                response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=1)
                if response.status_code == 200:
                    logger.info("Ollama server started successfully")
                    return True
            except:
                pass
            time.sleep(1)
        
        logger.error("Ollama server did not start in time")
        return False
        
    except FileNotFoundError:
        logger.error("Ollama is not installed")
        print("❌ Ollama is not installed!")
        print("   Install from: https://ollama.ai/download")
        return False
    except Exception as e:
        logger.error(f"Error starting Ollama: {e}")
        return False


def stop_ollama_server():
    """Stop Ollama server if we started it."""
    global ollama_process
    
    if ollama_process is None:
        return
    
    logger.info("Stopping Ollama server...")
    
    try:
        if os.name == 'nt':  # Windows
            ollama_process.send_signal(signal.CTRL_C_EVENT)
        else:  # Unix/Linux/Mac
            os.killpg(os.getpgid(ollama_process.pid), signal.SIGTERM)
        
        ollama_process.wait(timeout=5)
        logger.info("Ollama server stopped")
    except:
        try:
            ollama_process.kill()
        except:
            pass


# Register cleanup
atexit.register(stop_ollama_server)


def check_model_available(model_name: str) -> bool:
    """Check if model is available."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = [m['name'] for m in response.json().get('models', [])]
            
            # Check various forms
            if model_name in models:
                return True
            if f"{model_name}:latest" in models:
                return True
            
            # Check base model
            base = model_name.split(':')[0]
            for m in models:
                if m.startswith(base):
                    return True
    except:
        pass
    return False

# ============================================================================
# Lifespan Events
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup."""
    global llm, summarization_chain, embeddings, text_llm, vision_llm, whisper_model
    
    print("=" * 80)
    print("🚀 Starting Unified Multi-Modal API Server")
    print("=" * 80)
    
    # Start Ollama
    if not start_ollama_server():
        logger.error("Failed to start Ollama")
        sys.exit(1)
    
    # Check required models
    required_models = [
        SUMMARIZATION_MODEL,
        EMBEDDING_MODEL,
        TEXT_MODEL,
        VISION_MODEL
    ]
    
    print(f"\n🔍 Checking required models...")
    missing_models = []
    for model in required_models:
        if not check_model_available(model):
            missing_models.append(model)
            logger.warning(f"Model '{model}' not found")
    
    if missing_models:
        print(f"\n⚠️  Missing models: {', '.join(missing_models)}")
        print(f"\n📥 Please pull the missing models:")
        for model in missing_models:
            print(f"   ollama pull {model}")
        stop_ollama_server()
        sys.exit(1)
    
    print(f"✅ All required models available")
    
    # Initialize summarization models
    logger.info("Initializing summarization models...")
    llm = ChatOllama(
        model=SUMMARIZATION_MODEL,
        temperature=TEMPERATURE,
        base_url=OLLAMA_BASE_URL,
    )
    
    prompt_template = """
You are an AI assistant helping to answer questions based on retrieved documents.

You will be given:
1. A user's question
2. Relevant text chunks retrieved from documents

Your task:
- Analyze the provided context carefully
- Answer the question based ONLY on the given context
- Provide a comprehensive and well-structured answer
- Use Markdown formatting for better readability
- Do NOT cite sources or say "Based on the resource" or similar phrases
- If the context doesn't contain enough information, say so clearly

Context from retrieved documents:
{context}

Question: {question}

Answer:
""".strip()
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    summarization_chain = prompt | llm | StrOutputParser()
    logger.info("Summarization models initialized")
    
    # Initialize multi-modal models
    logger.info("Initializing multi-modal embedding models...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    text_llm = ChatOllama(model=TEXT_MODEL, base_url=OLLAMA_BASE_URL)
    vision_llm = ChatOllama(model=VISION_MODEL, base_url=OLLAMA_BASE_URL)
    logger.info("Multi-modal embedding models initialized")
    
    # Initialize Whisper
    logger.info(f"Loading Whisper model: {WHISPER_MODEL}...")
    whisper_model = whisper.load_model(WHISPER_MODEL)
    logger.info("Whisper model loaded")
    
    print("\n" + "=" * 80)
    print("✅ All models initialized successfully!")
    print("✅ Server ready to accept requests!")
    print("=" * 80)
    
    yield
    
    # Shutdown
    print("\n" + "=" * 80)
    print("🔌 Shutting down...")
    print("=" * 80)
    stop_ollama_server()

# ============================================================================
# Initialize FastAPI App
# ============================================================================

app = FastAPI(
    title="Unified Multi-Modal API",
    description="Content summarization and multi-modal embedding generation in one API",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Request/Response Models
# ============================================================================

class SummarizeRequest(BaseModel):
    """Summarization request."""
    question: str = Field(..., description="User's question", min_length=1)
    top_k_texts: List[str] = Field(..., description="Retrieved text chunks", min_items=1)
    max_context_length: Optional[int] = Field(MAX_CONTEXT_LENGTH, description="Max context length", gt=0)
    show_context: Optional[bool] = Field(False, description="Include context in response")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the effects of microgravity?",
                "top_k_texts": [
                    "Microgravity causes bone density loss...",
                    "Research shows 1-2% bone loss per month..."
                ]
            }
        }


class SummarizeResponse(BaseModel):
    """Summarization response."""
    answer: str
    num_chunks: int
    context_length: int
    success: bool
    error: Optional[str] = None
    context: Optional[str] = None


class EmbeddingResponse(BaseModel):
    """Embedding response."""
    embedding: List[float]
    dimension: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models: Dict[str, str]
    ollama_connected: bool

# ============================================================================
# Core Functions - Summarization
# ============================================================================

def summarize_retrieved_content(
    question: str,
    top_k_texts: List[str],
    max_context_length: int = MAX_CONTEXT_LENGTH,
    show_context: bool = False
) -> Dict[str, Any]:
    """Summarize retrieved content."""
    
    if not top_k_texts:
        return {
            "answer": "No context provided.",
            "num_chunks": 0,
            "context_length": 0,
            "success": False,
            "error": "Empty top_k_texts list"
        }
    
    # Format context
    formatted_chunks = []
    for i, text in enumerate(top_k_texts, 1):
        text = str(text).strip()
        if text:
            formatted_chunks.append(f"[Document {i}]\n{text}")
    
    context = "\n\n".join(formatted_chunks)
    
    # Truncate if needed
    if len(context) > max_context_length:
        context = context[:max_context_length] + "\n\n[Context truncated...]"
    
    # Generate answer
    try:
        answer = summarization_chain.invoke({
            "context": context,
            "question": question
        })
        
        result = {
            "answer": answer,
            "num_chunks": len(top_k_texts),
            "context_length": len(context),
            "success": True
        }
        
        if show_context:
            result["context"] = context
        
        return result
        
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        return {
            "answer": f"Error: {str(e)}",
            "num_chunks": len(top_k_texts),
            "context_length": len(context),
            "success": False,
            "error": str(e)
        }

# ============================================================================
# Core Functions - Multi-Modal Embeddings
# ============================================================================

def process_text(text: str) -> List[float]:
    """Process text and return embedding."""
    logger.info(f"Processing text input (length: {len(text)})")
    try:
        embedding = embeddings.embed_query(text)
        logger.info(f"Text embedding generated: dimension={len(embedding)}")
        return embedding
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        raise


def process_image(image_path: str) -> List[float]:
    """Process image and return embedding."""
    logger.info(f"Processing image: {image_path}")
    
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        raise FileNotFoundError(f"Image file {image_path} not found")
    
    try:
        # Read and encode image
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        logger.info("Image encoded, sending to vision model")
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe this image in detail. Don't mention any text like: 'this image describes or represnts', just respond with the detailed description"},
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{image_data}"
                }
            ]
        )
        response = vision_llm.invoke([message])
        description = response.content
        logger.info(f"Image description generated: {description[:100]}...")
        
        # Generate embedding from description
        embedding = embeddings.embed_query(description)
        logger.info(f"Image embedding generated: dimension={len(embedding)}")
        return embedding
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise


def process_audio(audio_path: str) -> List[float]:
    """Process audio and return embedding."""
    logger.info(f"Processing audio: {audio_path}")
    
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        raise FileNotFoundError(f"Audio file {audio_path} not found")
    
    try:
        # Transcribe audio
        logger.info("Transcribing audio with Whisper")
        result = whisper_model.transcribe(audio_path)
        transcript = result["text"]
        logger.info(f"Transcript generated: {transcript[:100]}...")
        
        # Enhance transcript with LLM
        logger.info("Enhancing transcript with LLM")
        messages = [
            SystemMessage(content="Summarize and enhance this transcript. Don't add sentences like 'This is a summarized transcript' or anything like it. Respond with just the transcribed text."),
            HumanMessage(content=transcript)
        ]
        response = text_llm.invoke(messages)
        enhanced_text = response.content
        logger.info(f"Enhanced text: {enhanced_text[:100]}...")
        
        # Generate embedding
        embedding = embeddings.embed_query(enhanced_text)
        logger.info(f"Audio embedding generated: dimension={len(embedding)}")
        return embedding
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Unified Multi-Modal API",
        "version": "3.0.0",
        "status": "running",
        "endpoints": {
            "summarization": {
                "summarize": "POST /summarize"
            },
            "embeddings": {
                "text": "POST /embed/text",
                "image": "POST /embed/image",
                "audio": "POST /embed/audio"
            },
            "info": {
                "health": "GET /health",
                "docs": "GET /docs"
            }
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    ollama_connected = False
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        ollama_connected = response.status_code == 200
    except:
        pass
    
    return HealthResponse(
        status="healthy" if ollama_connected else "degraded",
        models={
            "summarization": SUMMARIZATION_MODEL,
            "embedding": EMBEDDING_MODEL,
            "text": TEXT_MODEL,
            "vision": VISION_MODEL,
            "whisper": WHISPER_MODEL
        },
        ollama_connected=ollama_connected
    )


# ============================================================================
# Summarization Endpoints
# ============================================================================

@app.post("/summarize", response_model=SummarizeResponse, tags=["Summarization"])
async def summarize(request: SummarizeRequest):
    """
    Summarize retrieved content and generate an answer.
    
    **Example Request:**
    ```json
    {
        "question": "What are the effects of microgravity?",
        "top_k_texts": [
            "Document 1 content...",
            "Document 2 content..."
        ]
    }
    ```
    """
    try:
        result = summarize_retrieved_content(
            question=request.question,
            top_k_texts=request.top_k_texts,
            max_context_length=request.max_context_length,
            show_context=request.show_context
        )
        return SummarizeResponse(**result)
    
    except Exception as e:
        logger.error(f"Summarization endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization error: {str(e)}"
        )


# ============================================================================
# Embedding Endpoints
# ============================================================================

@app.post("/embed/text", response_model=EmbeddingResponse, tags=["Embeddings"])
async def embed_text(text: str = Form(...)):
    """
    Generate embedding for text input.
    
    **Example:**
    - text: "What are the effects of microgravity on astronauts?"
    """
    try:
        embedding = process_text(text)
        return EmbeddingResponse(embedding=embedding, dimension=len(embedding))
    except Exception as e:
        logger.error(f"Text embedding endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/image", response_model=EmbeddingResponse, tags=["Embeddings"])
async def embed_image(file: UploadFile = File(...)):
    """
    Generate embedding for image input.
    
    Upload an image file to get its embedding vector.
    """
    logger.info(f"Received image upload: {file.filename}")
    tmp_path = None
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=os.path.splitext(file.filename)[1], 
            mode='wb'
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp.flush()
            tmp_path = tmp.name
            logger.info(f"Image saved to temp file: {tmp_path}")
        
        # Process image
        embedding = process_image(tmp_path)
        return EmbeddingResponse(embedding=embedding, dimension=len(embedding))
    
    except Exception as e:
        logger.error(f"Image embedding endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temp file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.info(f"Cleaned up temp file: {tmp_path}")


@app.post("/embed/audio", response_model=EmbeddingResponse, tags=["Embeddings"])
async def embed_audio(file: UploadFile = File(...)):
    """
    Generate embedding for audio input.
    
    Upload an audio file (mp3, wav, m4a, etc.) to get its embedding vector.
    The audio will be transcribed and converted to an embedding.
    """
    logger.info(f"Received audio upload: {file.filename}")
    tmp_path = None
    
    try:
        # Determine file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']:
            file_ext = '.wav'  # Default to .wav
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=file_ext, 
            mode='wb'
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp.flush()
            tmp_path = tmp.name
            logger.info(f"Audio saved to temp file: {tmp_path} (size: {len(content)} bytes)")
        
        # Process audio
        embedding = process_audio(tmp_path)
        return EmbeddingResponse(embedding=embedding, dimension=len(embedding))
    
    except Exception as e:
        logger.error(f"Audio embedding endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temp file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.info(f"Cleaned up temp file: {tmp_path}")

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 Unified Multi-Modal API Server")
    print("=" * 80)
    print(f"\n📦 Models:")
    print(f"   • Summarization: {SUMMARIZATION_MODEL}")
    print(f"   • Embedding: {EMBEDDING_MODEL}")
    print(f"   • Text LLM: {TEXT_MODEL}")
    print(f"   • Vision: {VISION_MODEL}")
    print(f"   • Whisper: {WHISPER_MODEL}")
    print(f"\n🔗 Ollama: Auto-start enabled")
    print(f"\n🌐 Server: http://localhost:8000")
    print(f"📚 Docs: http://localhost:8000/docs")
    print("\n💡 Features:")
    print("   • Content Summarization")
    print("   • Text Embedding")
    print("   • Image Embedding")
    print("   • Audio Embedding")
    print("   • Automatic Ollama management")
    print("=" * 80 + "\n")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    finally:
        stop_ollama_server()
