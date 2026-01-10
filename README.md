# Sinhala Mathematics Question Generator

An advanced AI-powered system that generates O/L (Ordinary Level) mathematics questions in Sinhala using Retrieval-Augmented Generation (RAG) and Google Gemini API.

## 🎯 Overview

This project combines modern AI techniques with contextual learning materials to generate high-quality, contextually relevant mathematics problems in Sinhala. It leverages:

- **Retrieval-Augmented Generation (RAG)**: Retrieves relevant context from educational materials
- **Google Gemini 2.5 Flash API**: Generates coherent and contextually appropriate questions
- **ChromaDB**: Vector database for efficient semantic search
- **FastAPI**: RESTful API for easy integration
- **Multilingual Embeddings**: Support for Sinhala and other languages

## ✨ Features

- Generate mathematics questions with configurable difficulty levels (easy, medium, hard)
- RAG-based context retrieval for relevant problem generation
- Batch question generation (1-10 questions per request)
- REST API with comprehensive documentation
- Support for various mathematics topics in Sinhala
- Automatic rate limiting and error handling
- Health check and system status endpoints
- ChromaDB integration for efficient data retrieval

## 📋 Prerequisites

- Python 3.8+
- Google Gemini API key
- 2GB RAM minimum
- Windows, macOS, or Linux

## 🚀 Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd backend
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_google_gemini_api_key_here
```

## 📁 Project Structure

```
backend/
├── apis/
│   └── api.py                 # FastAPI application and endpoints
├── models/
│   └── rag_model.py           # SinhalaRAGSystem core implementation
├── data/
│   ├── extracted_text/        # Extracted content from PDFs
│   │   ├── extracted_examples.json
│   │   ├── exteacted_exercises.json
│   │   ├── guidelines.json
│   │   └── paragraphs_and_tables.json
│   ├── pdfs/                  # Source PDF documents
│   └── vector_store/          # ChromaDB vector storage
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── generated_questions.json   # Output file for generated questions
```

## 🔧 Key Components

### SinhalaRAGSystem (rag_model.py)

The core RAG system that:

- Initializes Google Gemini API connection
- Sets up ChromaDB for vector storage and retrieval
- Loads educational materials (examples, exercises, paragraphs)
- Generates questions using contextual retrieval
- Implements rate limiting and retry logic

**Main Methods:**

- `load_all_data()` - Loads all educational materials
- `generate_questions()` - Generates questions based on topic
- `retrieve_context()` - Retrieves relevant context using embeddings

### FastAPI Application (api.py)

Provides REST endpoints for:

- **`GET /`** - API information and status
- **`GET /health`** - System health check
- **`POST /initialize`** - Initialize/reinitialize RAG system
- **`POST /generate`** - Generate mathematics questions
- **`GET /docs`** - Interactive API documentation (Swagger UI)

## 📊 API Usage

### Health Check

```bash
curl -X GET "http://localhost:8000/health"
```

### Initialize System

```bash
curl -X POST "http://localhost:8000/initialize" \
  -H "Content-Type: application/json" \
  -d '{"load_data": true}'
```

### Generate Questions

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "වාරික ගණනය",
    "difficulty": "medium",
    "num_questions": 5
  }'
```

### Response Example

```json
{
  "success": true,
  "topic": "වාරික ගණනය",
  "difficulty": "medium",
  "questions": [
    {
      "question": "...",
      "solution": "...",
      "answer": "..."
    }
  ],
  "count": 5,
  "generation_time_seconds": 4.23,
  "model_used": "gemini-2.5-flash",
  "rag_context_used": true
}
```

## ▶️ Running the Application

1. Start the FastAPI server:

```bash
python -m uvicorn apis.api:app --reload
```

2. Access the API:

- **Interactive Documentation**: http://localhost:8000/docs
- **Alternative Documentation**: http://localhost:8000/redoc

## 🔑 Configuration

### Model Settings (rag_model.py)

- **Model**: `gemini-2.5-flash`
- **Temperature**: 0.8 (balanced creativity and coherence)
- **Max Output Tokens**: 16,384
- **Top P**: 0.95
- **Top K**: 40

### Embedding Model

- **SentenceTransformer**: `paraphrase-multilingual-mpnet-base-v2`
- Supports Sinhala and multilingual text encoding

### Rate Limiting

- Minimum 2 seconds between API requests to avoid quota exhaustion

## 📦 Dependencies

Key packages:

- `fastapi` - Web framework
- `google-generativeai` - Google Gemini API
- `chromadb` - Vector database
- `sentence-transformers` - Multilingual embeddings
- `pandas` - Data processing
- `pydantic` - Data validation
- `pdfplumber`, `pdfminer.six`, `PyMuPDF` - PDF processing

See `requirements.txt` for complete list.

## 🧪 Testing

To test the API endpoints, use the built-in Swagger UI:

1. Navigate to `http://localhost:8000/docs`
2. Try out the endpoints directly from the browser interface

## ⚙️ Supported Difficulty Levels

- **Easy**: Basic problems for beginners
- **Medium**: Standard O/L level problems
- **Hard**: Advanced challenging problems

## 🤝 Error Handling

The system includes comprehensive error handling for:

- Missing API keys
- API rate limiting
- Invalid requests
- Data loading failures
- ChromaDB initialization issues

Check the `/health` endpoint for current system status and last error.

## 🔐 Security Features

- CORS middleware configured
- Safety settings for harmful content filtering
- API key validation
- Environment variable protection

## 📈 Performance

- Average generation time: 3-5 seconds per question batch
- ChromaDB provides sub-100ms context retrieval
- Supports concurrent requests with async/await

## 🐛 Troubleshooting

**Issue**: `GEMINI_API_KEY not configured`

- Solution: Create `.env` file with valid API key

**Issue**: ChromaDB initialization fails

- Solution: Install with `pip install chromadb sentence-transformers`

**Issue**: Slow question generation

- Solution: Reduce `num_questions` or check API rate limits

## 👥 Contributors

[Add contributor information here]

## 🔄 Version History

- **v0.3.0** - Current development version
- **v2.0.0** - API version (gemini-2.5-flash)

---

**Last Updated**: January 2026
