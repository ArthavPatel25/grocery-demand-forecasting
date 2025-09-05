
import pickle
import pandas as pd
import numpy as np

class GroceryDemandPredictor:
    def __init__(self, model_path="../data/models"):
        # Load best model
        with open(f"{model_path}/lightgbm_model.pkl", "rb") as f:
            self.model = pickle.load(f)

        # Load feature list
        with open(f"{model_path}/feature_list.pkl", "rb") as f:
            self.features = pickle.load(f)

        # Load encoders
        with open(f"{model_path}/label_encoders.pkl", "rb") as f:
            self.encoders = pickle.load(f)

    def predict_demand(self, store_id, product_id, date, price, promotion_flag, 
                      chain, province, category, brand, **additional_features):
        """
        Predict demand for a single product-store combination

        Args:
            store_id: Store identifier
            product_id: Product identifier  
            date: Date string (YYYY-MM-DD)
            price: Product price
            promotion_flag: 1 if on promotion, 0 otherwise
            chain: Store chain name
            province: Province code
            category: Product category
            brand: Product brand
            **additional_features: Any additional features

        Returns:
            Predicted demand (float)
        """

        # Create feature vector (simplified - would need full feature engineering)
        # This is a template - actual implementation would require
        # the complete feature engineering pipeline

        features_dict = {}

        # Add provided features
        for feature in self.features:
            if feature in locals():
                features_dict[feature] = locals()[feature]
            else:
                # Default values for missing features
                features_dict[feature] = 0

        # Convert to DataFrame
        X = pd.DataFrame([features_dict])

        # Make prediction
        if "LightGBM" == "LightGBM":
            prediction = self.model.predict(X[self.features])[0]
        else:  # XGBoost
            prediction = self.model.predict(X[self.features])[0]

        return max(0, prediction)  # Ensure non-negative prediction

# Example usage:
# predictor = GroceryDemandPredictor()
# demand = predictor.predict_demand(
#     store_id="ST_001",
#     product_id="PR_1001", 
#     date="2024-01-15",
#     price=5.99,
#     promotion_flag=0,
#     chain="Loblaws",
#     province="ON",
#     category="Dairy",
#     brand="Brand_A"
# )
