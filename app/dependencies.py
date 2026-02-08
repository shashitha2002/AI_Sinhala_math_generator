import os
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from typing import Optional

from app.database import users_collection
from app.auth_utils import SECRET_KEY, ALGORITHM
from app.models.rag_model import SinhalaRAGSystem
from app.models.model_paper_generator import ModelPaperGenerator

# ==================== Auth Dependency ====================

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = await users_collection.find_one({"email": email})
    if user is None:
        raise credentials_exception
    return user


# ==================== RAG System Dependency (Singleton) ====================

_rag_instance: Optional[SinhalaRAGSystem] = None

def get_rag_system() -> SinhalaRAGSystem:
    global _rag_instance
    if _rag_instance is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")
        print("🚀 Initializing RAG System...")
        _rag_instance = SinhalaRAGSystem(api_key=api_key)
    return _rag_instance


# ==================== Model Paper Generator Dependency (Singleton) ====================

_model_paper_generator_instance: Optional[ModelPaperGenerator] = None

def get_model_paper_generator() -> ModelPaperGenerator:
    global _model_paper_generator_instance
    if _model_paper_generator_instance is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")
        print("🚀 Initializing Model Paper Generator...")
        _model_paper_generator_instance = ModelPaperGenerator(api_key=api_key)
    return _model_paper_generator_instance