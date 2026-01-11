import os
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from app.database import users_collection
from app.auth_utils import SECRET_KEY, ALGORITHM
from app.models.rag_model import SinhalaRAGSystem  # Importing your class

# 1. Auth Dependency
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

# 2. RAG System Dependency (Singleton Pattern)
# We store the instance here so it loads only once
rag_instance = None

def get_rag_system():
    global rag_instance
    if rag_instance is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="API Key not configured")
        print("Initializing RAG System...")
        rag_instance = SinhalaRAGSystem(api_key=api_key)
        # You can also call rag_instance.load_all_data() here if you want auto-load
    return rag_instance