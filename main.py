from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import os
from groq import Groq
import asyncio

load_dotenv()

app = FastAPI()

templates = Jinja2Templates(directory="templates")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
USE_GROQ = GROQ_API_KEY is not None

client = Groq(api_key=GROQ_API_KEY) if USE_GROQ else None

async def groq_response(user_text: str) -> str:
    """Call Groq API using the Python client and stream the output."""
    if not client:
        return "Groq client not initialized. Please set GROQ_API_KEY."

    content = ""

    def run_completion():
        nonlocal content
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a helpful, concise assistant answering interview questions in a professional, human tone."},
                {"role": "user", "content": user_text}
            ],
            temperature=0.7,
            max_completion_tokens=1024,
            top_p=1,
            reasoning_effort="medium",
            stream=True,
            stop=None
        )
        for chunk in completion:
            delta = chunk.choices[0].delta.content or ""  # <-- fixed
            content += delta

    await asyncio.to_thread(run_completion)
    return content or "No response from Groq API."


def fallback_response(user_text: str) -> str:
    """Fallback if no Groq key is set."""
    return "Problem with the api connection please check"

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/respond")
async def respond(text: str = Form(...)):
    try:
        if USE_GROQ:
            resp = await groq_response(text)
        else:
            resp = fallback_response(text)
        return JSONResponse({"reply": resp})
    except Exception as e:
        return JSONResponse({"reply": f"Error: {str(e)}"})
