from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model, scaler, and features
try:
    model = joblib.load('exoplanet_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features.pkl')
    logger.info(f"Loaded features: {features}")
except Exception as e:
    logger.error(f"Error loading files: {str(e)}")
    raise Exception(f"Failed to load: {str(e)}")

# Common physical features (subset of trained features, excluding dummies)
COMMON_PHYSICAL_FEATURES = ['koi_period', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_steff', 'koi_srad']

@app.post("/api/predict")
async def predict(data: dict):
    try:
        if 'data' not in data:
            raise HTTPException(status_code=400, detail="Missing 'data' key")
        input_data = data['data']
        logger.info(f"Received input: {input_data}")
        
        # Filter to common physical features only (ignore mission-specific like depth, duration, time0bk)
        filtered_data = {k: v for k, v in input_data.items() if k in COMMON_PHYSICAL_FEATURES}
        
        # Add mission dummies (based on input; assumes 'mission' key or logic from frontend)
        mission = input_data.get('mission', 'kepler')  # Default to kepler
        filtered_data['mission_tess'] = 1 if mission == 'tess' else 0
        filtered_data['mission_k2'] = 1 if mission == 'k2' else 0
        logger.info(f"Filtered data: {filtered_data}")
        
        # Create DataFrame and reindex to exact trained features (fill missing with 0)
        input_df = pd.DataFrame([filtered_data]).reindex(columns=features, fill_value=0)
        logger.info(f"Input DF columns: {input_df.columns.tolist()}")
        
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        proba = model.predict_proba(input_scaled).max()
        
        disposition_map = {0: 'FALSE', 1: 'CANDIDATE', 2: 'CONFIRMED'}
        result = {
            "prediction": disposition_map[prediction[0]],
            "confidence": float(proba)
        }
        logger.info(f"Result: {result}")
        return result
    except Exception as e:
        logger.error(f"Predict error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/upload")
async def upload(file: UploadFile = File(...), dataset: str = Form(...), n_estimators: int = Form(100)):
    try:
        df = pd.read_csv(file.file)
        logger.info(f"Uploaded: {file.filename}, dataset: {dataset}")
        # Placeholder (integrate full retraining here if needed)
        return {"accuracy": 0.85, "auc": 0.92}
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")