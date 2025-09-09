import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import pytest

class TestModelAccuracy:
    
    @pytest.fixture
    def load_test_data(self):
        """Load test data for model evaluation"""
        try:
            # Load test data that was saved during training
            test_data = pd.read_csv('data/processed/test_data.csv')
            return test_data
        except FileNotFoundError:
            pytest.skip("Test data not found")
    
    @pytest.fixture
    def load_model(self):
        """Load trained model"""
        try:
            with open('data/models/lightgbm_model.pkl', 'rb') as f:
                model = pickle.load(f)
            return model
        except FileNotFoundError:
            pytest.skip("Model not found")
    
    def test_model_accuracy_threshold(self, load_model, load_test_data):
        """Test that model meets accuracy thresholds"""
        model = load_model
        test_data = load_test_data
        
        # Make predictions
        X_test = test_data.drop(['sales_quantity'], axis=1)
        y_test = test_data['sales_quantity']
        
        predictions = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        # Assert accuracy thresholds
        assert mae < 5.0, f"MAE {mae} exceeds threshold of 5.0"
        assert rmse < 8.0, f"RMSE {rmse} exceeds threshold of 8.0"
        assert r2 > 0.8, f"R² {r2} below threshold of 0.8"
        
        print(f"Model Performance: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")
    
    def test_prediction_distribution(self, load_model, load_test_data):
        """Test that predictions follow reasonable distribution"""
        model = load_model
        test_data = load_test_data
        
        X_test = test_data.drop(['sales_quantity'], axis=1)
        predictions = model.predict(X_test)
        
        # Check prediction bounds
        assert np.all(predictions >= 0), "Model produces negative predictions"
        assert np.all(predictions < 1000), "Model produces unreasonably high predictions"
        
        # Check distribution characteristics
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        assert 5 < mean_pred < 50, f"Mean prediction {mean_pred} outside reasonable range"
        assert std_pred > 0, "Predictions have no variance"