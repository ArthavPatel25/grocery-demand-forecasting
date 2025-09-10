import pickle
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Global variables
model = None
feature_list = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, feature_list
    try:
        with open("data/models/lightgbm_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("data/models/feature_list.pkl", "rb") as f:
            feature_list = pickle.load(f)
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")

    yield

    # Shutdown (if needed)
    print("Application shutting down...")


app = FastAPI(title="Grocery Demand Forecasting API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    store_id: str
    product_id: str
    date: str
    price: float
    promotion_flag: int
    chain: str
    province: str
    category: str
    brand: str


class PredictionResponse(BaseModel):
    predicted_demand: float
    confidence_lower: float
    confidence_upper: float


def create_simple_features(request: PredictionRequest):
    # Simple feature creation
    features = {}

    # Fill with defaults
    for feature in feature_list:
        features[feature] = 0.0

    # Set actual values
    features["price"] = request.price
    features["promotion_flag"] = request.promotion_flag

    # Simple categorical encoding
    features["store_id_encoded"] = abs(hash(request.store_id)) % 25
    features["product_id_encoded"] = abs(hash(request.product_id)) % 300
    features["category_encoded"] = abs(hash(request.category)) % 6
    features["chain_encoded"] = abs(hash(request.chain)) % 7
    features["province_encoded"] = abs(hash(request.province)) % 10

    # Add some variation based on inputs
    chain_factors = {
        "Loblaws": 1.3,
        "Metro": 1.1,
        "Sobeys": 1.0,
        "FreshCo": 0.8,
        "No Frills": 0.7,
    }
    province_factors = {
        "ON": 1.2,
        "QC": 1.0,
        "BC": 1.1,
        "AB": 0.9,
        "MB": 0.8,
        "SK": 0.7,
        "NS": 0.8,
        "NB": 0.7,
        "NL": 0.6,
        "PE": 0.5,
    }

    base_value = (
        15
        * chain_factors.get(request.chain, 1.0)
        * province_factors.get(request.province, 1.0)
    )

    # Set key features that impact prediction
    features["sales_rolling_mean_30"] = base_value
    features["product_category_share"] = 0.1
    features["revenue"] = base_value * request.price

    return pd.DataFrame([features])[feature_list]


@app.post("/predict", response_model=PredictionResponse)
async def predict_demand(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X = create_simple_features(request)
        prediction = model.predict(X)[0]
        prediction = max(1, prediction)  # Ensure positive

        return PredictionResponse(
            predicted_demand=round(prediction, 2),
            confidence_lower=round(prediction * 0.8, 2),
            confidence_upper=round(prediction * 1.2, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
