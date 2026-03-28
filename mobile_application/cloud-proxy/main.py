"""
TenantShield Cloud Run Proxy — handles Gemini API with ADC.
The Android app calls this service instead of connecting directly to Vertex AI.
No API keys or service account files needed — Cloud Run uses ADC automatically.
"""

import os
import json
import uuid
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from google import genai
from google.genai import types

app = FastAPI(title="TenantShield Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "crucial-bucksaw-371623")
REGION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
MODEL = "gemini-2.0-flash"

# Initialize the Gemini client with ADC (automatic on Cloud Run)
client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=REGION,
)

# In-memory session storage (conversation history per session)
sessions = {}


class StartSessionRequest(BaseModel):
    system_prompt: str


class StartSessionResponse(BaseModel):
    session_id: str


class SendMessageRequest(BaseModel):
    session_id: str
    message: str


class SendMessageResponse(BaseModel):
    response_text: str


class AnalyzeImagesRequest(BaseModel):
    system_prompt: str
    user_message: str
    images_base64: List[str]


class AnalyzeImagesResponse(BaseModel):
    response_text: str


class GenerateRequest(BaseModel):
    system_prompt: str
    user_message: str


class GenerateResponse(BaseModel):
    response_text: str


@app.get("/health")
def health():
    return {"status": "ok", "project": PROJECT_ID, "region": REGION, "model": MODEL}


@app.post("/session/start", response_model=StartSessionResponse)
def start_session(req: StartSessionRequest):
    """Start a new conversation session (stores system prompt and history)."""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "system_prompt": req.system_prompt,
        "history": [],
    }
    return StartSessionResponse(session_id=session_id)


@app.post("/session/send", response_model=SendMessageResponse)
def send_message(req: SendMessageRequest):
    """Send a message in an existing conversation session."""
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = sessions[req.session_id]
    session_data["history"].append({"role": "user", "text": req.message})

    try:
        # Build conversation contents
        contents = []
        for entry in session_data["history"]:
            contents.append(
                types.Content(
                    role=entry["role"],
                    parts=[types.Part(text=entry["text"])],
                )
            )

        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=session_data["system_prompt"],
                response_mime_type="application/json",
            ),
        )

        response_text = response.text
        session_data["history"].append({"role": "model", "text": response_text})
        return SendMessageResponse(response_text=response_text)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/end")
def end_session(session_id: str):
    """End a conversation session."""
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeImagesResponse)
def analyze_images(req: AnalyzeImagesRequest):
    """Analyze images using Gemini multimodal (for Inspection Agent)."""
    import base64

    try:
        parts = [types.Part(text=req.user_message)]

        for img_b64 in req.images_base64:
            img_bytes = base64.b64decode(img_b64)
            parts.append(
                types.Part(
                    inline_data=types.Blob(
                        mime_type="image/jpeg",
                        data=img_bytes,
                    )
                )
            )

        response = client.models.generate_content(
            model=MODEL,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                system_instruction=req.system_prompt,
                response_mime_type="application/json",
            ),
        )

        return AnalyzeImagesResponse(response_text=response.text)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """Simple text generation (for Filing Agent)."""
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text=req.user_message)],
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=req.system_prompt,
                response_mime_type="application/json",
            ),
        )

        return GenerateResponse(response_text=response.text)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
