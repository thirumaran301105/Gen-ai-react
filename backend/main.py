"""
Rural Advisory System — FastAPI Backend
Handles: image analysis (ML), TTS audio, disease DB, weather data
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import numpy as np
import cv2
import pickle
import json
import io
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Rural Advisory API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load assets ───────────────────────────────────────────────────────────────
BASE = Path(__file__).parent

with open(BASE / "models" / "crop_model.pkl", "rb") as f:
    ML_MODEL = pickle.load(f)
with open(BASE / "models" / "classes.pkl", "rb") as f:
    ML_CLASSES = pickle.load(f)
with open(BASE / "database" / "diseases_db.json", "r", encoding="utf-8") as f:
    DISEASE_DB = json.load(f)

MOCK_WEATHER = {
    "Chennai":   {"temp": 32.5, "humidity": 75, "rainfall": 2.5, "wind": 12},
    "Delhi":     {"temp": 28.3, "humidity": 65, "rainfall": 0.0, "wind":  8},
    "Mumbai":    {"temp": 30.1, "humidity": 80, "rainfall": 5.2, "wind": 15},
    "Bangalore": {"temp": 26.7, "humidity": 70, "rainfall": 1.2, "wind": 10},
    "Kolkata":   {"temp": 29.4, "humidity": 78, "rainfall": 3.8, "wind": 11},
    "Other":     {"temp": 29.0, "humidity": 72, "rainfall": 1.0, "wind": 10},
}

# ── Feature extractor (identical to training) ─────────────────────────────────
def extract_features(rgb: np.ndarray) -> np.ndarray:
    img  = cv2.resize(rgb, (128, 128))
    hsv  = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)

    h_hist = cv2.calcHist([hsv],[0],None,[36],[0,180]).flatten(); h_hist /= h_hist.sum()+1e-6
    s_hist = cv2.calcHist([hsv],[1],None,[32],[0,256]).flatten(); s_hist /= s_hist.sum()+1e-6
    v_hist = cv2.calcHist([hsv],[2],None,[32],[0,256]).flatten(); v_hist /= v_hist.sum()+1e-6

    region_defs = [
        ([35,50,50], [85,255,255]), ([8,60,30],  [25,220,200]),
        ([40,20,0],  [90,255,80]), ([0,0,180],   [180,45,255]),
        ([7,150,50], [22,255,255]),([18,50,100],  [38,255,255]),
        ([5,100,10], [20,255,100]),([0,0,0],      [180,60,60]),
    ]
    ratios = [
        cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)).mean() / 255
        for lo, hi in region_defs
    ]
    lap   = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    sobel = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    stats = np.array([
        gray.mean()/255, gray.std()/255,
        hsv[:,:,0].mean()/180, hsv[:,:,0].std()/180,
        hsv[:,:,1].mean()/255, hsv[:,:,1].std()/255,
        hsv[:,:,2].mean()/255, hsv[:,:,2].std()/255,
        lap.mean()/255, lap.std()/255, lap.max()/255, sobel.mean()/255,
    ], dtype=np.float32)

    return np.concatenate([h_hist, s_hist, v_hist, ratios, stats]).astype(np.float32)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/diseases")
def get_diseases():
    """Return full disease database."""
    return DISEASE_DB


@app.get("/api/weather/{location}")
def get_weather(location: str):
    """Return mock weather + spraying advice for a location."""
    w = MOCK_WEATHER.get(location, MOCK_WEATHER["Other"])
    score, msgs = 1.0, []
    if w["rainfall"] > 3:
        score -= 0.5
        msgs.append(f"Rain detected ({w['rainfall']} mm) — delay spraying 24h")
    if w["wind"] > 18:
        score -= 0.45
        msgs.append(f"High wind ({w['wind']} km/h) — severe drift risk")
    if w["humidity"] < 55:
        msgs.append(f"Low humidity ({w['humidity']}%) — spray early morning")
    status = "ok" if score >= 0.7 else ("wait" if score >= 0.4 else "no")
    return {**w, "status": status, "messages": msgs, "location": location}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Analyze uploaded crop image.
    Returns: disease_key, disease_info, confidence, all_proba, processing_time
    """
    t0 = time.time()
    try:
        contents = await file.read()
        arr = np.frombuffer(contents, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "Could not decode image")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        features = extract_features(rgb).reshape(1, -1)
        proba     = ML_MODEL.predict_proba(features)[0]
        class_idx = int(np.argmax(proba))
        key       = ML_CLASSES[class_idx]
        confidence = float(proba[class_idx])
        all_proba  = {ML_CLASSES[i]: round(float(p), 4) for i, p in enumerate(proba)}

        disease_info = DISEASE_DB.get(key, DISEASE_DB.get("Early_Blight", {}))

        return {
            "disease_key":     key,
            "confidence":      round(confidence, 4),
            "processing_time": round(time.time() - t0, 3),
            "all_proba":       all_proba,
            "disease_info":    disease_info,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"analyze error: {e}")
        raise HTTPException(500, str(e))


class TTSRequest(BaseModel):
    text: str
    language: str  # "English" | "Tamil" | "Hindi"

@app.post("/api/tts")
def text_to_speech(req: TTSRequest):
    """Generate TTS MP3 and stream it back."""
    try:
        from gtts import gTTS
        code = {"Tamil": "ta", "Hindi": "hi", "English": "en"}.get(req.language, "en")
        tts  = gTTS(text=req.text, lang=code, slow=False)
        buf  = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/mpeg",
                                  headers={"Content-Disposition": "inline; filename=remedy.mp3"})
    except ImportError:
        raise HTTPException(503, "gTTS not installed. Run: pip install gtts")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": ML_MODEL is not None,
            "diseases": len(DISEASE_DB), "classes": ML_CLASSES}
