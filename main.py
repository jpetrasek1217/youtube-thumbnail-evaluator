import uvicorn
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import Optional
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import numpy as np
from model import load_model, preprocess_input, predict
import os
from contextlib import asynccontextmanager

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = load_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
    yield
    # Shutdown logic if needed

app = FastAPI(title="YouTube CTR Predictor", lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionInput(BaseModel):
    videoTitle: str
    videoLength: str
    channelSubscribers: int
    totalChannelViews: int
    totalVideos: int
    videoUploadDayofWeek: int
    videoUploadHour: int
    channelAgeYears: int
    thumbnail: str  # base64 string

    # ---------- Validators ----------
    @field_validator('channelSubscribers', 'totalChannelViews', 'totalVideos', 'channelAgeYears')
    @classmethod
    def non_negative(cls, v):
        if v < 0:
            raise ValueError('must be non-negative')
        return v

    @field_validator('videoUploadDayofWeek')
    @classmethod
    def valid_day(cls, v):
        if not 0 <= v <= 6:
            raise ValueError('must be between 0 and 6')
        return v

    @field_validator('videoUploadHour')
    @classmethod
    def valid_hour(cls, v):
        if not 0 <= v <= 23:
            raise ValueError('must be between 0 and 23')
        return v

    @field_validator('thumbnail')
    @classmethod
    def valid_thumbnail(cls, v):
        if not v:
            raise ValueError('thumbnail is required')
        return v

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
async def predict_ctr(data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert Base64 thumbnail to numpy array
        try:
            thumbnail_bytes = base64.b64decode(data.thumbnail)
            image = Image.open(io.BytesIO(thumbnail_bytes)).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.ToTensor()
            ])
            image = transform(image).unsqueeze(0)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid thumbnail image: {e}")

        # Prepare input dictionary
        input_data = {
            "thumbnail": image,
            "videoTitle": data.videoTitle,
            "videoTitleLength": len(data.videoTitle),
            "videoLength": data.videoLength,
            "channelSubscribers": data.channelSubscribers,
            "totalChannelViews": data.totalChannelViews,
            "totalVideos": data.totalVideos,
            "videoUploadDayofWeek": data.videoUploadDayofWeek,
            "videoUploadHour": data.videoUploadHour,
            "channelAgeYears": data.channelAgeYears,
        }

        # Get prediction
        prediction = predict(model, input_data)

        return {
            "success": True,
            "prediction": prediction,
            "message": "Prediction generated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
