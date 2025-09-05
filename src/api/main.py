# src/api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from typing import List, Optional
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Canadian Grocery Demand Forecasting API",
    description="AI-powered demand prediction for Canadian grocery stores",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
model = None
feature_list = None
label_encoders = None

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    store_id: str = Field(..., description="Store identifier (e.g., ST_001)")
    product_id: str = Field(..., description="Product identifier (e.g., PR_1001)")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    price: float = Field(..., gt=0, description="Product price")
    promotion_flag: int = Field(0, ge=0, le=1, description="1 if on promotion, 0 otherwise")
    chain: str = Field(..., description="Store chain (e.g., Loblaws)")
    province: str = Field(..., description="Province code (e.g., ON)")
    category: str = Field(..., description="Product category (e.g., Dairy)")
    brand: str = Field(..., description="Product brand (e.g., Brand_A)")

class PredictionResponse(BaseModel):
    store_id: str
    product_id: str
    date: str
    predicted_demand: float
    confidence_lower: float
    confidence_upper: float
    model_used: str
    prediction_timestamp: str

class BatchPredictionRequest(BaseModel):
    predictions: List[PredictionRequest]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    api_version: str
    timestamp: str

@app.on_event("startup")
async def startup_event():
    """Load models and encoders on startup"""
    global model, feature_list, label_encoders
    
    try:
        model_path = "data/models"
        
        # Load LightGBM model (best performer)
        with open(f"{model_path}/lightgbm_model.pkl", "rb") as f:
            model = pickle.load(f)
        
        # Load feature list
        with open(f"{model_path}/feature_list.pkl", "rb") as f:
            feature_list = pickle.load(f)
        
        # Load label encoders
        with open(f"{model_path}/label_encoders.pkl", "rb") as f:
            label_encoders = pickle.load(f)
        
        print("Models loaded successfully!")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        # Initialize with None for graceful error handling
        model = None
        feature_list = None
        label_encoders = None

def create_feature_vector(request: PredictionRequest):
    """Create feature vector from prediction request"""
    
    # Parse date
    date_obj = pd.to_datetime(request.date)
    
    # Basic features that we can derive
    features = {
        'price': request.price,
        'promotion_flag': request.promotion_flag,
        'month': date_obj.month,
        'day': date_obj.day,
        'day_of_week': date_obj.dayofweek,
        'week_of_year': date_obj.isocalendar().week,
        'quarter': date_obj.quarter,
        'is_weekend': int(date_obj.dayofweek >= 5),
        'is_monday': int(date_obj.dayofweek == 0),
        'is_friday': int(date_obj.dayofweek == 4),
        'is_month_start': int(date_obj.day <= 7),
        'is_month_middle': int(7 < date_obj.day <= 21),
        'is_month_end': int(date_obj.day > 21),
        'sin_month': np.sin(2 * np.pi * date_obj.month / 12),
        'cos_month': np.cos(2 * np.pi * date_obj.month / 12),
        'sin_day_of_week': np.sin(2 * np.pi * date_obj.dayofweek / 7),
        'cos_day_of_week': np.cos(2 * np.pi * date_obj.dayofweek / 7),
        'sin_day_of_year': np.sin(2 * np.pi * date_obj.dayofyear / 365),
        'cos_day_of_year': np.cos(2 * np.pi * date_obj.dayofyear / 365),
    }
    
    # Add encoded categorical features
    if label_encoders:
        try:
            features['store_id_encoded'] = label_encoders['store_id'].transform([request.store_id])[0]
        except:
            features['store_id_encoded'] = 0
            
        try:
            features['product_id_encoded'] = label_encoders['product_id'].transform([request.product_id])[0]
        except:
            features['product_id_encoded'] = 0
            
        try:
            features['category_encoded'] = label_encoders['category'].transform([request.category])[0]
        except:
            features['category_encoded'] = 0
            
        try:
            features['brand_encoded'] = label_encoders['brand'].transform([request.brand])[0]
        except:
            features['brand_encoded'] = 0
            
        try:
            features['chain_encoded'] = label_encoders['chain'].transform([request.chain])[0]
        except:
            features['chain_encoded'] = 0
            
        try:
            features['province_encoded'] = label_encoders['province'].transform([request.province])[0]
        except:
            features['province_encoded'] = 0
    
    # Fill missing features with defaults
    for feature in feature_list:
        if feature not in features:
            if 'lag_' in feature or 'rolling_' in feature:
                features[feature] = 10.0  # Default historical value
            elif 'share' in feature:
                features[feature] = 0.1   # Default market share
            elif 'daily_' in feature:
                features[feature] = 100.0 # Default daily volume
            else:
                features[feature] = 0.0   # Default to zero
    
    # Create DataFrame with correct feature order
    X = pd.DataFrame([features])
    X = X[feature_list]  # Ensure correct order
    
    return X

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        api_version="1.0.0",
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_demand(request: PredictionRequest):
    """Make a single demand prediction"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create feature vector
        X = create_feature_vector(request)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Ensure non-negative prediction
        prediction = max(0, prediction)
        
        # Calculate confidence intervals (simplified)
        confidence_lower = max(0, prediction * 0.8)
        confidence_upper = prediction * 1.2
        
        return PredictionResponse(
            store_id=request.store_id,
            product_id=request.product_id,
            date=request.date,
            predicted_demand=round(prediction, 2),
            confidence_lower=round(confidence_lower, 2),
            confidence_upper=round(confidence_upper, 2),
            model_used="LightGBM",
            prediction_timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
    """Make batch predictions"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    predictions = []
    errors = []
    
    for i, pred_request in enumerate(request.predictions):
        try:
            prediction = await predict_demand(pred_request)
            predictions.append(prediction)
        except Exception as e:
            errors.append(f"Request {i}: {str(e)}")
    
    return {
        "predictions": predictions,
        "success_count": len(predictions),
        "error_count": len(errors),
        "errors": errors
    }

@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    
    try:
        with open("data/models/model_metadata.json", "r") as f:
            metadata = json.load(f)
        return metadata
    except:
        return {"error": "Model metadata not found"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Canadian Grocery Demand Forecasting API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "model_info": "/model/info",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)