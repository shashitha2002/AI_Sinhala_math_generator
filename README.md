# Sinhala Mathematics Question Generator API v2.0.0

An advanced AI-powered system that generates O/L (Ordinary Level) mathematics questions in Sinhala using Retrieval-Augmented Generation (RAG), Google Gemini API, and JWT-based authentication.

## 🎯 Overview

This project combines modern AI techniques with contextual learning materials to generate high-quality, contextually relevant mathematics problems in Sinhala. It leverages:

- **Retrieval-Augmented Generation (RAG)**: Retrieves relevant context from educational materials
- **Google Gemini 2.5 Flash API**: Generates coherent and contextually appropriate questions
- **ChromaDB**: Vector database for efficient semantic search
- **FastAPI**: RESTful API with modular router architecture
- **JWT Authentication**: Secure user authentication and authorization
- **MongoDB**: User and data persistence
- **Multilingual Embeddings**: Support for Sinhala and other languages

## ✨ Features

- **User Authentication**: Secure signup and login with JWT tokens
- **Generate mathematics questions** with configurable difficulty levels (easy, medium, hard)
- **RAG-based context retrieval** for relevant problem generation
- **Batch question generation** (1-10 questions per request)
- **Modular API Architecture** with separate routers for auth and math generation
- **Protected Endpoints** - All math generation endpoints require authentication
- **Support for various mathematics topics** in Sinhala
- **Automatic rate limiting** and error handling
- **MongoDB Integration** for user persistence
- **ChromaDB integration** for efficient data retrieval

## 📋 Prerequisites

- Python 3.8+
- Google Gemini API key
- MongoDB instance (local or cloud)
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
MONGO_URL=mongodb://localhost:27017
SECRET_KEY=your_secret_key_for_jwt_tokens
ALGORITHM=HS256
```

## 📁 Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application entry point
│   ├── auth_utils.py              # Authentication utilities
│   ├── database.py                # MongoDB configuration
│   ├── dependencies.py            # Dependency injection (Auth & RAG)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── auth.py                # Auth data models
│   │   ├── math.py                # Math question models
│   │   ├── rag_model.py           # SinhalaRAGSystem core implementation
│   │   └── generated_questions.py # Generated questions model
│   └── routers/
│       ├── __init__.py
│       ├── auth.py                # Authentication endpoints (/auth)
│       └── math_gen.py            # Math generation endpoints (/math)
├── apis/
│   └── api.py                     # Legacy API (standalone version)
├── data/
│   ├── extracted_text/            # Extracted content from PDFs
│   │   ├── extracted_examples.json
│   │   ├── exteacted_exercises.json
│   │   ├── guidelines.json
│   │   └── paragraphs_and_tables.json
│   ├── pdfs/                      # Source PDF documents
│   └── vector_store/              # ChromaDB vector storage
├── requirements.txt               # Python dependencies
├── .env                           # Environment variables (create this)
├── README.md                      # This file
└── generated_questions.json       # Output file for generated questions
```

## 🔧 Key Components

### Core Architecture

**app/main.py** - FastAPI Application
- FastAPI application with CORS middleware
- Router integration for auth and math generation
- Startup event to pre-load RAG data

**app/dependencies.py** - Dependency Injection
- `get_current_user()` - JWT authentication dependency
- `get_rag_system()` - RAG system singleton instance
- OAuth2 password bearer scheme

**app/database.py** - MongoDB Connection
- AsyncIO MongoDB client setup
- Collections: users_collection for user persistence

**app/auth_utils.py** - Authentication Utilities
- Password hashing and verification (bcrypt)
- JWT token creation and validation
- SECRET_KEY and ALGORITHM configuration

### API Routers

**app/routers/auth.py** - Authentication Endpoints
- `POST /auth/signup` - User registration
- `POST /auth/token` - User login (OAuth2)

**app/routers/math_gen.py** - Math Generation Endpoints (Protected)
- `POST /math/generate` - Generate mathematics questions
- All endpoints require valid JWT token

### Data Models

**app/models/math.py**
- `DifficultyLevel` - Enum (easy, medium, hard)
- `QuestionRequest` - Input model for generation requests
- `Question` - Individual question model
- `QuestionResponse` - API response model

**app/models/auth.py**
- `UserSignUp` - User registration model
- `Token` - JWT token response model

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

## 📊 API Usage

### 1. Authentication

#### Sign Up

```bash
curl -X POST "http://localhost:8000/auth/signup" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "user123",
    "email": "user@example.com",
    "password": "securepassword"
  }'
```

Response:
```json
{
  "message": "User created"
}
```

#### Login (Get Access Token)

```bash
curl -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=securepassword"
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### 2. Generate Questions (Protected Endpoint)

**Requires valid JWT token from login**

```bash
curl -X POST "http://localhost:8000/math/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -d '{
    "topic": "වාරික ගණනය",
    "difficulty": "medium",
    "num_questions": 5
  }'
```

Response:
```json
{
  "success": true,
  "topic": "වාරික ගණනය",
  "questions": [
    {
      "question": "...",
      "solution": "...",
      "answer": "..."
    }
  ],
  "generation_time_seconds": 4.23,
  "model_used": "gemini-2.5-flash"
}
```

## ▶️ Running the Application

### Using the New Modular Architecture (Recommended)

1. Make sure MongoDB is running:

```bash
mongod
```

2. Start the FastAPI server:

```bash
python -m uvicorn app.main:app --reload
```

3. Access the API:

- **Interactive Documentation**: http://localhost:8000/docs
- **Alternative Documentation**: http://localhost:8000/redoc

### Using the Standalone API

For backward compatibility, you can still use the standalone api.py:

```bash
python -m uvicorn apis.api:app --reload
```

## 🔑 Configuration

### Environment Variables (.env)

```env
# Google Gemini API
GEMINI_API_KEY=your_api_key_here

# MongoDB Connection
MONGO_URL=mongodb://localhost:27017

# JWT Configuration
SECRET_KEY=your_super_secret_key_change_this_in_production
ALGORITHM=HS256
```

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

### Authentication

- **JWT Token Expiration**: Configure in `auth_utils.py`
- **Password Hashing**: Uses bcrypt for secure password storage
- **OAuth2 Password Bearer**: Standard OAuth2 scheme for token-based auth

## 📦 Dependencies

Key packages:

- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `google-generativeai` - Google Gemini API
- `chromadb` - Vector database
- `sentence-transformers` - Multilingual embeddings
- `motor` - Async MongoDB driver
- `pymongo` - MongoDB
- `python-jose` - JWT tokens
- `python-multipart` - Form data parsing
- `bcrypt` - Password hashing
- `pandas` - Data processing
- `pydantic` - Data validation
- `python-dotenv` - Environment variables
- `pdfplumber`, `pdfminer.six`, `PyMuPDF` - PDF processing

See `requirements.txt` for complete list.

## 🧪 Testing

### Using Swagger UI

1. Navigate to `http://localhost:8000/docs`
2. Authorize by clicking the "Authorize" button
3. Sign up for an account at `/auth/signup`
4. Get access token from `/auth/token`
5. Paste the token and test protected endpoints

### Using cURL

1. First sign up:
```bash
curl -X POST "http://localhost:8000/auth/signup" \
  -H "Content-Type: application/json" \
  -d '{"username":"test","email":"test@example.com","password":"test123"}'
```

2. Get token:
```bash
curl -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test@example.com&password=test123"
```

3. Use token to generate questions:
```bash
curl -X POST "http://localhost:8000/math/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_TOKEN>" \
  -d '{"topic":"වාරික ගණනය","difficulty":"medium","num_questions":3}'
```

## ⚙️ Supported Difficulty Levels

- **Easy**: Basic problems for beginners
- **Medium**: Standard O/L level problems
- **Hard**: Advanced challenging problems

## 🤝 Error Handling

The system includes comprehensive error handling for:

- **Missing API keys** - Returns 500 error if GEMINI_API_KEY not configured
- **Authentication errors** - 401 Unauthorized for invalid/missing tokens
- **Invalid requests** - 422 Unprocessable Entity for malformed requests
- **User already exists** - 400 Bad Request for duplicate email
- **Invalid credentials** - 401 Unauthorized for wrong password
- **Data loading failures** - Warnings logged but non-blocking
- **ChromaDB initialization issues** - Logged errors with fallback

Common responses:

```json
{
  "detail": "Could not validate credentials"
}
```

```json
{
  "detail": "Email already registered"
}
```

## 🔐 Security Features

- **JWT Authentication** - Token-based secure authentication
- **Password Hashing** - Bcrypt for secure password storage
- **CORS Middleware** - Configured for controlled cross-origin requests
- **OAuth2 Integration** - Standard OAuth2 password bearer scheme
- **Protected Routes** - Dependency injection for auth validation
- **API Key Validation** - Environment variable protection for secrets
- **Async Database Operations** - Non-blocking MongoDB queries
- **Safety settings for harmful content** filtering (Gemini API)

## 📈 Performance

- **Question Generation**: 3-5 seconds per batch (including RAG retrieval)
- **Token Generation**: <100ms for JWT token creation
- **Context Retrieval**: <100ms using ChromaDB embeddings
- **Async Operations**: Non-blocking with async/await throughout
- **MongoDB Queries**: Fast with AsyncIO motor driver
- **Concurrent Requests**: Fully supports multiple simultaneous users

Performance Tips:
- Keep MONGO_URL connection close to your server
- Batch multiple questions in single request (up to 10)
- Use appropriate difficulty levels to minimize processing time

## 🐛 Troubleshooting

### Issue: `GEMINI_API_KEY not configured`

- Solution: Create `.env` file with valid API key

### Issue: MongoDB connection error

- Solution: Ensure MongoDB is running on `localhost:27017` or update `MONGO_URL` in `.env`

### Issue: ChromaDB initialization fails

- Solution: Install with `pip install chromadb sentence-transformers`

### Issue: "Could not validate credentials"

- Solution: Make sure token is included in Authorization header as `Bearer <token>`

### Issue: "Email already registered"

- Solution: Use a different email address or login with existing account

### Issue: Slow question generation

- Solution: Reduce `num_questions` or check API rate limits

### Issue: Port 8000 already in use

- Solution: Use different port: `python -m uvicorn app.main:app --reload --port 8001`

## � Version History

- **v2.0.0** - Current release
  - Modular architecture with routers
  - JWT authentication system
  - MongoDB integration
  - Protected endpoints
  - Async database operations
  
- **v0.3.0** - Initial version
  - Basic RAG system
  - Standalone API

---

**Last Updated**: January 2026
**Architecture**: FastAPI + MongoDB + ChromaDB + Google Gemini
**Status**: Production Ready
