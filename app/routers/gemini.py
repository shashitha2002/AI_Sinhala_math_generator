from fastapi import APIRouter, HTTPException, Body
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(tags=["AI"])

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

@router.post("/gemini")
async def chat_with_gemini(question: str = Body(..., embed=True)):
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    if not GROQ_API_KEY:
        return {
            "success": True,
            "answer": f"මට ඔබගේ ප්‍රශ්නය ලැබුණි: '{question}'\n\nමේ වන විට AI සම්බන්ධතාවය ස්ථාපිත කර ඇත. ශ්‍රී ලංකාවේ සිසුන් සදහා ගණිත ප්‍රශ්න විසදීමට සූදානම්."
        }
    
    system_prompt = """සිසුන්ගේ ගණිත ප්‍රශ්නයට පැහැදිලි, සරල උත්තරයක් දෙන්න.

  උත්තරය සැපයීමේදී:
  1. පියවරෙන් පියවර විස්තර කරන්න
  2. සිංහල භාෂාවෙන් පැහැදිලිව ලියන්න
  3. ගණිත සංකේත හා සිංහල මිශ්‍ර කළ හැක
  4. සරල උදාහරණ භාවිතා කරන්න
  5. අවසානයේ රටාව හා නීති පැහැදිලි කරන්න"""

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                GROQ_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {GROQ_API_KEY}"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1024,
                    "top_p": 0.9
                },
                timeout=30.0
            )
            
            if response.status_code != 200:
                print(f"❌ Groq API Error: {response.status_code} - {response.text}")
                return {
                    "success": True,
                    "answer": f"මට ඔබගේ ප්‍රශ්නය ලැබුණි: '{question}'\n\nදැනට AI සේවාවට සම්බන්ධ වීමට නොහැකි විය. කරුණාකර පසුව උත්සාහ කරන්න."
                }
            
            data = response.json()
            ai_response = data["choices"][0]["message"]["content"]
            
            return {
                "success": True,
                "answer": ai_response
            }
            
    except Exception as e:
        print(f"❌ Gemini route error: {e}")
        return {
            "success": True,
            "answer": "කණගාටුයි, තාක්ෂණික දෝෂයක් හේතුවෙන් පිළිතුර ලබා දිය නොහැක."
        }
