# test_api.py
import requests
import json

# Test data
test_request = {
    "store_id": "ST_001",
    "product_id": "PR_1001",
    "date": "2024-01-15",
    "price": 5.99,
    "promotion_flag": 0,
    "chain": "Loblaws",
    "province": "ON",
    "category": "Dairy",
    "brand": "Brand_A"
}

def test_api():
    base_url = "http://localhost:8000"
    
    # Test health check
    print("Testing health check...")
    response = requests.get(f"{base_url}/health")
    print(f"Health check: {response.json()}")
    
    # Test single prediction
    print("\nTesting single prediction...")
    response = requests.post(f"{base_url}/predict", json=test_request)
    print(f"Prediction: {response.json()}")
    
    # Test batch prediction
    print("\nTesting batch prediction...")
    batch_request = {"predictions": [test_request, test_request]}
    response = requests.post(f"{base_url}/predict/batch", json=batch_request)
    print(f"Batch prediction: {response.json()}")

if __name__ == "__main__":
    test_api()