import pytest
import requests
import json
from fastapi.testclient import TestClient
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.main import app

client = TestClient(app)

class TestAPI:
    
    def test_health_endpoint(self):
        """Test API health check"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_prediction_endpoint_valid_input(self):
        """Test prediction with valid input"""
        test_data = {
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
        
        response = client.post("/predict", json=test_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "predicted_demand" in data
            assert "confidence_lower" in data
            assert "confidence_upper" in data
            assert data["predicted_demand"] > 0
        else:
            # If model not loaded, expect 503
            assert response.status_code == 503
    
    def test_prediction_endpoint_invalid_input(self):
        """Test prediction with invalid input"""
        test_data = {
            "store_id": "ST_001",
            "price": -5.99,  # Invalid negative price
            "promotion_flag": 0
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 422  # Validation error
    
    def test_prediction_endpoint_missing_fields(self):
        """Test prediction with missing required fields"""
        test_data = {
            "store_id": "ST_001"
            # Missing required fields
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 422

class TestModelAccuracy:
    
    def test_model_consistency(self):
        """Test that same inputs produce same outputs"""
        test_data = {
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
        
        response1 = client.post("/predict", json=test_data)
        response2 = client.post("/predict", json=test_data)
        
        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()
            assert data1["predicted_demand"] == data2["predicted_demand"]
    
    def test_prediction_reasonableness(self):
        """Test that predictions are in reasonable range"""
        test_data = {
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
        
        response = client.post("/predict", json=test_data)
        
        if response.status_code == 200:
            data = response.json()
            # Check reasonable bounds for grocery demand
            assert 0 < data["predicted_demand"] < 1000
            assert data["confidence_lower"] < data["predicted_demand"]
            assert data["predicted_demand"] < data["confidence_upper"]

if __name__ == "__main__":
    pytest.main([__file__])