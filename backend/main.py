import os
import base64  
import datetime 
import requests 
import asyncio
import json
from typing import Optional, List  
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors  

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

HISTORY_FILE = "chat_history.json"

class MessageRecord(BaseModel):
    role: str
    text: str

class ImageData(BaseModel):
    image_base64: str
    image_mime_type: str

class ChatRequest(BaseModel):
    model: str = "auto"
    system_prompt: str = ""
    temperature: float = 0.7
    history: list[MessageRecord] = []
    message: str
    images: List[ImageData] = []

@app.get("/history")
def get_history():
    """讀取本地端的歷史對話紀錄"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

@app.post("/history")
def save_history(history: list[MessageRecord]):
    """將最新的對話紀錄寫入本地端 JSON 檔案"""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([h.dict() for h in history], f, ensure_ascii=False, indent=2)
    return {"status": "success"}

@app.delete("/history")
def clear_history():
    """清除所有歷史記憶檔案"""
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return {"status": "cleared"}

def get_current_time() -> str:
    """取得伺服器目前的即時日期與時間。"""
    print("[系統日誌] AI 觸發工具：get_current_time")
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_weather(location_en: str) -> str:
    """查詢指定地點即時天氣。參數必須是「英文地名」（如 Hsinchu）。"""
    print(f"[系統日誌] AI 觸發工具：get_weather (地點: {location_en})")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location_en}&count=1&language=zh-TW"
        geo_res = requests.get(geocode_url, headers=headers, timeout=5).json()
        if not geo_res.get("results"): return f"找不到 {location_en} 的地理座標"
        
        res = geo_res["results"][0]
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={res['latitude']}&longitude={res['longitude']}&current_weather=true"
        w_data = requests.get(weather_url, headers=headers, timeout=5).json()
        
        current = w_data.get('current_weather', {})
        temp = current.get('temperature')
        weather_code = current.get('weathercode')
        weather_map = {0: "晴朗", 1: "稍有雲", 2: "多雲", 3: "陰天", 45: "霧", 61: "微雨", 95: "雷雨"}
        weather_desc = weather_map.get(weather_code, "多變")
        
        return f"【真實氣象】{res['name']} 目前氣溫：{temp}°C，天氣狀態：{weather_desc}。"
    except Exception as e:
        return f"天氣查詢失敗：{str(e)}"

def get_financial_quote(symbol: str) -> str:
    """查詢虛擬貨幣即時報價 (BTC, ETH, SOL, BNB)。"""
    print(f"[系統日誌] AI 觸發工具：get_financial_quote (標的: {symbol})")
    try:
        symbol = symbol.upper().replace(" ", "")
        if symbol in ["BTC", "ETH", "SOL", "BNB"]:
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}USDT"
            data = requests.get(url, timeout=5).json()
            return f"【{symbol} 即時報價】${float(data['price']):,.2f} USD"
        return "目前僅支援 BTC, ETH, SOL, BNB 報價查詢"
    except Exception as e:
        return f"報價查詢失敗：{str(e)}"

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    
    tool_keywords = ["天氣", "溫度", "下雨", "時間", "幾點", "今天", "報價", "btc", "eth", "sol", "bnb", "幣"]
    msg_lower = request.message.lower()
    needs_tool = any(kw in msg_lower for kw in tool_keywords)
    has_images = len(request.images) > 0
    auto_log_msg = ""

    if request.model == "auto":
        if needs_tool:
            request.model = "gemini-2.5-pro"
            auto_log_msg = "[Auto Routing] 偵測到「工具呼叫」，自動切換至強邏輯模型 Gemini 2.5 Pro。"
        elif has_images:
            request.model = "gemini-2.5-pro"
            auto_log_msg = "[Auto Routing] 偵測到「多模態圖片」，分配至視覺旗艦模型 Gemini 2.5 Pro。"
        else:
            request.model = "gemini-2.5-flash"
            auto_log_msg = "[Auto Routing] 一般對話，分配至極速模型 Gemini 2.5 Flash。"
    elif request.model == "gemini-2.5-flash" and needs_tool:
        request.model = "gemini-2.5-pro"
        auto_log_msg = "[系統強制重定向] 為確保工具呼叫精準度，已自動升級至 Gemini 2.5 Pro。"

    config_params = {"temperature": request.temperature}
    
    if request.model == "gemini-2.5-pro":
        config_params["tools"] = [get_current_time, get_weather, get_financial_quote]
        
    if request.system_prompt:
        config_params["system_instruction"] = request.system_prompt
    
    config = types.GenerateContentConfig(**config_params)

    formatted_history = [
        types.Content(role=msg.role, parts=[types.Part.from_text(text=msg.text)])
        for msg in request.history
    ]

    chat = client.aio.chats.create(model=request.model, config=config, history=formatted_history)

    message_contents = []
    for img_data in request.images:
        message_contents.append(
            types.Part.from_bytes(
                data=base64.b64decode(img_data.image_base64), 
                mime_type=img_data.image_mime_type
            )
        )
    message_contents.append(request.message if request.message else "請處理此內容")

    async def generate():
        try:
            if auto_log_msg:
                yield f"<span class='system-log'>{auto_log_msg}</span>\n\n"

            response_stream = await asyncio.wait_for(
                chat.send_message_stream(message_contents),
                timeout=60.0
            )
            
            async for chunk in response_stream:
                if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                    for part in chunk.candidates[0].content.parts:
                        if part.function_call:
                            call = part.function_call
                            args_str = ", ".join([f"{k}={v}" for k, v in call.args.items()]) if call.args else ""
                            yield f'<span class="system-log">🛠️ [系統：執行工具] {call.name}({args_str})</span>\n\n'

                if chunk.text:
                    yield chunk.text

        except asyncio.TimeoutError:
            yield "\n\n<span class='system-log'>[系統錯誤] 模型回應超時，請檢查連線或縮短輸入內容。</span>"
        except errors.APIError as e:
            yield f"\n\n<span class='system-log'>[系統錯誤] 遠端伺服器異常 ({e.code})。</span>"
        except Exception as e:
            yield f"\n\n<span class='system-log'>[系統錯誤] 發生未知錯誤：{str(e)}</span>"

    return StreamingResponse(generate(), media_type="text/event-stream")
