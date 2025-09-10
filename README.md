# Grocery Demand Forecasting System

A complete MLOps pipeline for predicting grocery store demand using machine learning. The system provides real-time demand predictions with confidence intervals through a FastAPI backend and interactive Streamlit dashboard.

## Model Performance

- **Algorithm**: LightGBM
- **R² Score**: 94.9%
- **Mean Absolute Error**: 2.84
- **Accuracy within 20%**: 75.4%
- **Test Coverage**: 74% (13 automated tests)

## Tech Stack

- **Backend**: FastAPI with ML model serving
- **Frontend**: Streamlit interactive dashboard
- **Deployment**: Docker containers with docker-compose
- **CI/CD**: GitHub Actions with automated testing and deployment
- **Code Quality**: Black formatting, flake8 linting, isort import sorting

## Prerequisites

- Python 3.11+
- Docker Desktop
- Git

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/grocery-demand-forecasting.git
cd grocery-demand-forecasting
2. Run with Docker (Recommended)
Start both API and Streamlit services:
bashdocker-compose up -d
Check container status:
bashdocker-compose ps
3. Access the Application

Streamlit Dashboard: http://localhost:8501
API Health Check: http://localhost:8000/health
API Documentation: http://localhost:8000/docs

4. Test the API
Health check:
bashcurl http://localhost:8000/health
Make a prediction:
bashcurl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "store_id": "ST_001",
    "product_id": "PR_1001",
    "date": "2024-01-01",
    "price": 5.99,
    "promotion_flag": 1,
    "chain": "Loblaws",
    "province": "ON",
    "category": "Dairy",
    "brand": "TestBrand"
  }'
5. Stop the Application
bashdocker-compose down
Development Setup
1. Create Virtual Environment
bashpython -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
2. Install Dependencies
bashpip install -r requirements.txt
3. Run Tests
bashpytest tests/ -v --cov=src --cov-report=html
4. Run API Locally
bashuvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
5. Run Streamlit Locally
bashstreamlit run src/ui/streamlit_app.py --server.port 8501
Features
API Capabilities

Real-time demand predictions with confidence intervals
RESTful API with automatic documentation
Health monitoring and status checks
Comprehensive input validation

Streamlit Dashboard

Interactive prediction form with business-friendly interface
Real-time demand forecasting with visual confidence bands
Prediction history tracking and analytics
Store performance analysis and trend visualization
Automated business recommendations (stock up, monitor, promote)

Model Features

Product category share analysis
Revenue calculations and pricing impact
Sales rolling statistics and trends
Store and product categorical encoding
Promotional flag impact assessment
Geographic factors (province, chain analysis)

API Usage
Health Check
GET /health
Response: {"status": "healthy", "model_loaded": true}
Prediction
POST /predict
Request: Store details, product info, pricing, and promotional data
Response: Predicted demand with lower and upper confidence bounds
CI/CD Pipeline

Automated Testing: 13 tests with 74% coverage on every commit
Code Quality: Automated formatting and linting checks
Security Scanning: Bandit security analysis
Docker Integration: Multi-container testing and deployment
GitHub Actions: Complete CI/CD workflow from code to deployment

License
This project is licensed under the MIT License.
