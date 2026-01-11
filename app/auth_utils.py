import os
import hashlib
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import jwt
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "unsafe_secret_key")
ALGORITHM = "HS256"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str) -> str:
    # DEBUG: Print to terminal to see what is actually coming in
    print(f"DEBUG: Hashing type: {type(password)}")
    print(f"DEBUG: Hashing value: {password}")

    # 1. SHA256 Hash (Turn any length into 64 characters)
    sha256_password = hashlib.sha256(password.encode('utf-8')).hexdigest()
    
    # 2. Bcrypt Hash (Hash the 64-char string)
    return pwd_context.hash(sha256_password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    sha256_password = hashlib.sha256(plain_password.encode('utf-8')).hexdigest()
    return pwd_context.verify(sha256_password, hashed_password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=30)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)