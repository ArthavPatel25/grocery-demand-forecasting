Grocery Demand Forecasting System
A complete MLOps pipeline for predicting grocery store demand using machine learning. The system provides real-time demand predictions with confidence intervals through a FastAPI backend and interactive Streamlit dashboard.
Model Performance

Algorithm: LightGBM
R² Score: 94.9%
Mean Absolute Error: 2.84
Accuracy within 20%: 75.4%
Test Coverage: 74% (13 automated tests)

Architecture

Backend: FastAPI with ML model serving
Frontend: Streamlit interactive dashboard
Deployment: Docker containers with docker-compose
CI/CD: GitHub Actions with automated testing and deployment
Code Quality: Black formatting, flake8 linting, isort import sorting

Prerequisites

Python 3.11+
Docker Desktop
Git

Quick Start
1. Clone the Repository
bashgit clone https://github.com/your-username/grocery-demand-forecasting.git
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
# or
source venv/bin/activate  # Linux/Mac
2. Install Dependencies
bashpip install -r requirements.txt
3. Run Tests
bashpytest tests/ -v --cov=src --cov-report=html
4. Code Quality Checks
bash# Format code
black src/

# Check linting
flake8 src/ --max-line-length=88 --extend-ignore=E203

# Sort imports
isort src/
5. Run API Locally
bashcd src
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
6. Run Streamlit Locally
bashstreamlit run src/ui/streamlit_app.py --server.port 8501
Project Structure
grocery-demand-forecasting/
├── .github/
│   └── workflows/
│       ├── ci.yml               # Continuous Integration
│       └── cd.yml               # Continuous Deployment
├── .pytest_cache/              # Pytest cache directory
├── data/                        # Data and models directory
├── htmlcov/                     # HTML test coverage reports
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Data analysis and exploration
│   ├── 02_feature_engineering.ipynb # Feature creation and preprocessing
│   └── 03_model_training.ipynb      # Model training and evaluation
├── src/
│   ├── api/
│   │   ├── __pycache__/         # Python cache files
│   │   └── main.py              # FastAPI application
│   └── ui/
│       └── streamlit_app.py     # Streamlit dashboard
├── tests/
│   ├── __pycache__/             # Python cache files
│   ├── conftest.py              # Test configuration
│   ├── test_additional_coverage.py # Additional test coverage
│   ├── test_api.py              # API tests
│   └── test_model_accuracy.py   # Model validation tests
├── venv/                        # Virtual environment
├── .coverage                    # Coverage data file
├── .dockerignore               # Docker ignore rules
├── coverage.xml                # Coverage XML report
├── docker-compose.yml          # Multi-service orchestration
├── Dockerfile                  # API container configuration
├── Dockerfile.streamlit        # Streamlit container configuration
├── prediction_history.json     # Prediction history storage
├── requirements.txt            # Python dependencies
└── test_docker.py             # Docker testing script
API Endpoints
Health Check
GET /health
Returns API status and model loading state.
Prediction
POST /predict
Request Body:
json{
  "store_id": "ST_001",
  "product_id": "PR_1001",
  "date": "2024-01-01",
  "price": 5.99,
  "promotion_flag": 1,
  "chain": "Loblaws",
  "province": "ON",
  "category": "Dairy",
  "brand": "TestBrand"
}
Response:
json{
  "predicted_demand": 6.37,
  "confidence_lower": 5.1,
  "confidence_upper": 7.65
}
Streamlit Dashboard Features

Interactive Prediction Form: Input store, product, and pricing details
Real-time Predictions: Get demand forecasts with confidence intervals
Business Recommendations: Automated stocking suggestions based on demand
Prediction History: Track and analyze historical predictions
Analytics Dashboard: Visualize trends and store performance
Export Capabilities: Download prediction data

Model Details
Features Used

Product category share
Revenue calculations
Sales rolling statistics
Store and product encodings
Promotional flags
Geographic factors (province, chain)

Training Data

Multiple grocery chains across Canada
Product categories: Dairy, Meat, Produce, Bakery, Frozen, Pantry
Time series data with seasonal patterns
Promotional and pricing information

CI/CD Pipeline
The project includes automated workflows for:
Continuous Integration (CI)

Testing: Automated test suite with 74% coverage
Code Quality: Black formatting, flake8 linting, isort import sorting
Security: Bandit security scanning
Triggers: Every push and pull request

Continuous Deployment (CD)

Docker Build: Multi-stage container builds
Integration Testing: API and Streamlit functionality tests
Deployment: Automated staging and production deployment
Triggers: Successful CI completion on main branch

Docker Configuration
Single Container Deployment
API Only:
bashdocker build -t grocery-api .
docker run -d -p 8000:8000 grocery-api
Streamlit Only:
bashdocker build -f Dockerfile.streamlit -t grocery-streamlit .
docker run -d -p 8501:8501 grocery-streamlit
Multi-Container Deployment
The application uses docker-compose for orchestrating multiple services:
yamlservices:
  api:
    build: .
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  
  streamlit:
    build:
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    depends_on:
      api:
        condition: service_healthy
Environment Variables

API_URL: API endpoint for Streamlit (default: http://api:8000)
PYTHONPATH: Python path configuration

Monitoring and Logging

Health Checks: Built-in endpoint monitoring
Container Logs: Accessible via docker-compose logs
Application Metrics: Model prediction tracking
Error Handling: Comprehensive exception management

Troubleshooting
Common Issues
Docker Desktop not running:
bash# Start Docker Desktop and verify
docker --version
docker info
Port conflicts:
bash# Check port usage
netstat -an | findstr "8000\|8501"

# Use different ports
docker-compose -p custom up -d
Model loading errors:

Ensure model files exist in data/models/
Check file permissions and paths
Verify Python version compatibility

CI/CD failures:

Check GitHub Actions logs
Verify all dependencies in requirements.txt
Ensure code quality checks pass locally

Performance Optimization

Docker: Use multi-stage builds for smaller images
API: Implement caching for frequent predictions
Database: Add persistence for prediction history
Scaling: Use container orchestration for high load

Contributing

Fork the repository
Create a feature branch
Make changes with tests
Ensure code quality checks pass
Submit a pull request

License
This project is licensed under the MIT License.
