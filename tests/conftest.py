import pytest
import os
import sys

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture(scope="session")
def test_data():
    """Fixture providing test data for multiple tests"""
    return {
        "valid_prediction": {
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
    }