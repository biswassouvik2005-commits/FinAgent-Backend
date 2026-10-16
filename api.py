from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import jwt
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from finance_assistant.main import process_message # Imports your actual multi-agent router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = "secure-finance-secret-key"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
users_db = {"souvik": "password123"}

class ChatRequest(BaseModel):
    message: str

# 1. Global Exception Handler
# Prevents the FastAPI server from shutting down if a severe error occurs
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"CRITICAL ERROR: {exc}")
    return JSONResponse(
        status_code=500,
        content={"ai_reply": "I am currently experiencing technical difficulties. Your data is safe. Please try again in a moment.", "error": str(exc)},
    )

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if users_db.get(form_data.username) != form_data.password:
        raise HTTPException(status_code=400, detail="Incorrect credentials")
    
    token = jwt.encode({"sub": form_data.username}, SECRET_KEY, algorithm="HS256")
    return {"access_token": token, "token_type": "bearer"}

# 2. Automatic Retry Logic for the Model
# Retries up to 3 times, waiting exponentially (2s, 4s, 8s) between attempts
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
def call_ai_model(message: str):
    print(f"Routing message to LangGraph agents: {message}")
    
    # Pass the message to your actual Python agents
    # process_message should return a dict matching your React Native schema 
    # (e.g., {"ai_reply": "...", "budget_status": "...", etc.})
    actual_response = process_message(message) 
    
    return actual_response

@app.post("/api/chat")
def chat(request: ChatRequest, token: str = Depends(oauth2_scheme)):
    try:
        reply = call_ai_model(request.message)
        return reply
    except Exception as e:
        # If all 3 retries fail, it falls back here safely
        print(f"Model generation failed after retries: {e}")
        return {"ai_reply": "The AI model is temporarily unavailable. Please try your request again."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)