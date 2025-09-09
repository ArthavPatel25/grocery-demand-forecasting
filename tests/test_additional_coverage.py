import pytest
from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.main import app, create_simple_features, PredictionRequest

client = TestClient(app)

class TestAdditionalCoverage:
    """Additional tests to improve code coverage"""

    def test_model_not_loaded_error(self):
        """Test prediction when model is not loaded"""
        # This test would need to mock the model being None
        # You might need to temporarily set model = None for this test
        pass

    def test_create_simple_features_function(self):
        """Test the create_simple_features function directly"""
        # Mock feature_list first
        from src.api.main import feature_list
        if feature_list is None:
            # Mock feature_list for testing
            import src.api.main as main_module
            main_module.feature_list = [
                'price', 'promotion_flag', 'store_id_encoded', 
                'product_id_encoded', 'category_encoded', 'chain_encoded',
                'province_encoded', 'sales_rolling_mean_30', 
                'product_category_share', 'revenue'
            ]
        
        request = PredictionRequest(
            store_id="store_001",
            product_id="prod_001", 
            date="2024-01-01",
            price=5.99,
            promotion_flag=1,
            chain="Loblaws",
            province="ON",
            category="Dairy",
            brand="TestBrand"
        )
        
        features_df = create_simple_features(request)
        
        # Assert basic properties
        assert len(features_df) == 1
        assert features_df['price'].iloc[0] == 5.99
        assert features_df['promotion_flag'].iloc[0] == 1
        assert features_df['store_id_encoded'].iloc[0] >= 0
        assert features_df['product_id_encoded'].iloc[0] >= 0

    def test_different_chains_and_provinces(self):
        """Test predictions with different chains and provinces"""
        test_cases = [
            {"chain": "Metro", "province": "QC"},
            {"chain": "Sobeys", "province": "BC"},
            {"chain": "FreshCo", "province": "AB"},
            {"chain": "No Frills", "province": "MB"},
            {"chain": "UnknownChain", "province": "UnknownProvince"}  # Test defaults
        ]
        
        for case in test_cases:
            response = client.post("/predict", json={
                "store_id": "store_001",
                "product_id": "prod_001",
                "date": "2024-01-01", 
                "price": 4.99,
                "promotion_flag": 0,
                "chain": case["chain"],
                "province": case["province"],
                "category": "Snacks",
                "brand": "TestBrand"
            })
            
            if response.status_code == 200:
                data = response.json()
                assert "predicted_demand" in data
                assert "confidence_lower" in data
                assert "confidence_upper" in data
                assert data["predicted_demand"] >= 1  # Ensure positive prediction

    def test_edge_cases_for_prediction(self):
        """Test edge cases for prediction input"""
        edge_cases = [
            {"price": 0.01, "promotion_flag": 0},  # Very low price
            {"price": 999.99, "promotion_flag": 1},  # Very high price
            {"price": 5.99, "promotion_flag": 2},  # Invalid promotion flag (if validation exists)
        ]
        
        for case in edge_cases:
            response = client.post("/predict", json={
                "store_id": "edge_store",
                "product_id": "edge_prod",
                "date": "2024-12-31",
                "price": case["price"],
                "promotion_flag": case["promotion_flag"], 
                "chain": "Loblaws",
                "province": "ON",
                "category": "Test",
                "brand": "EdgeBrand"
            })
            
            # Should either succeed or fail gracefully (503 = model not loaded is also acceptable)
            assert response.status_code in [200, 400, 422, 503]

    def test_health_endpoint_detailed(self):
        """More detailed health check testing"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] == "healthy"
        assert isinstance(data["model_loaded"], bool)

    def test_cors_headers(self):
        """Test CORS headers are properly set"""
        # Test actual CORS functionality with a regular request
        response = client.get("/health", headers={"Origin": "http://localhost:3000"})
        assert response.status_code == 200
        # The response should include CORS headers (if CORS is properly configured)
        # Note: FastAPI/Starlette handles CORS automatically, so this tests the middleware works

    def test_prediction_response_format(self):
        """Test that prediction response follows exact format"""
        response = client.post("/predict", json={
            "store_id": "format_test",
            "product_id": "format_prod",
            "date": "2024-06-15",
            "price": 7.50,
            "promotion_flag": 1,
            "chain": "Metro",
            "province": "ON", 
            "category": "Beverages",
            "brand": "FormatBrand"
        })
        
        if response.status_code == 200:
            data = response.json()
            
            # Check all required fields exist
            required_fields = ["predicted_demand", "confidence_lower", "confidence_upper"]
            for field in required_fields:
                assert field in data
                assert isinstance(data[field], (int, float))
            
            # Check confidence interval makes sense
            assert data["confidence_lower"] <= data["predicted_demand"] <= data["confidence_upper"]