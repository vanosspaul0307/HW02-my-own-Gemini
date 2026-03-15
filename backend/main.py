import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("找不到 GEMINI_API_KEY，請檢查 .env 檔案！")

client = genai.Client(api_key=api_key)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageRecord(BaseModel):
    role: str
    text: str

class ChatRequest(BaseModel):
    model: str = "gemini-2.5-flash"
    system_prompt: str = ""
    temperature: float = 0.7
    history: list[MessageRecord] = []
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    
    config_params = {"temperature": request.temperature}
    if request.system_prompt:
        config_params["system_instruction"] = request.system_prompt
    
    config = types.GenerateContentConfig(**config_params)

    formatted_history = []
    for msg in request.history:
        formatted_history.append(
            types.Content(role=msg.role, parts=[types.Part.from_text(text=msg.text)])
        )

    chat = client.chats.create(
        model=request.model,
        config=config,
        history=formatted_history
    )

    def generate():
        response_stream = chat.send_message_stream(request.message)
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    return StreamingResponse(generate(), media_type="text/event-stream")