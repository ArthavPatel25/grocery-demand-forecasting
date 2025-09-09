# src/ui/streamlit_app.py
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from datetime import date, timedelta, datetime

# Page configuration
st.set_page_config(
    page_title="Grocery Demand Forecasting",
    page_icon="📊",
    layout="wide"
)

# Professional CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .metric-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #2E86AB;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration - Fixed for Docker networking
API_URL = os.getenv("API_URL", "http://api:8000")
PREDICTIONS_FILE = "prediction_history.json"

# Store and Product mappings
STORE_MAPPING = {
    "Loblaws Downtown Toronto": "ST_001",
    "Metro Vancouver Center": "ST_002", 
    "Sobeys Montreal Plaza": "ST_003",
    "FreshCo Calgary South": "ST_004",
    "No Frills Ottawa East": "ST_005"
}

PRODUCT_MAPPING = {
    "Milk 2% (2L)": "PR_1001",
    "Ground Beef (1kg)": "PR_1002",
    "Bananas (1kg)": "PR_1003",
    "White Bread (675g)": "PR_1004",
    "Frozen Pizza": "PR_1005",
    "Pasta (500g)": "PR_1006",
    "Cheddar Cheese (400g)": "PR_1007",
    "Chicken Breast (1kg)": "PR_1008",
    "Apples (2kg)": "PR_1009",
    "Bagels (6-pack)": "PR_1010"
}

def load_prediction_history():
    """Load prediction history from JSON file"""
    if os.path.exists(PREDICTIONS_FILE):
        try:
            with open(PREDICTIONS_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_prediction_history(predictions):
    """Save prediction history to JSON file"""
    try:
        with open(PREDICTIONS_FILE, 'w') as f:
            json.dump(predictions, f, indent=2)
    except:
        pass  # Fail silently if can't save

def add_prediction_to_history(prediction_data, result):
    """Add new prediction to history"""
    history = load_prediction_history()
    
    # Find store and product names safely
    store_name = "Unknown Store"
    product_name = "Unknown Product"
    
    for k, v in STORE_MAPPING.items():
        if v == prediction_data["store_id"]:
            store_name = k
            break
    
    for k, v in PRODUCT_MAPPING.items():
        if v == prediction_data["product_id"]:
            product_name = k
            break
    
    new_entry = {
        "timestamp": datetime.now().isoformat(),
        "date": prediction_data["date"],
        "store_name": store_name,
        "product_name": product_name,
        "store_id": prediction_data["store_id"],
        "product_id": prediction_data["product_id"],
        "chain": prediction_data["chain"],
        "province": prediction_data["province"],
        "category": prediction_data["category"],
        "price": prediction_data["price"],
        "promotion_flag": prediction_data["promotion_flag"],
        "predicted_demand": result["predicted_demand"],
        "confidence_lower": result["confidence_lower"],
        "confidence_upper": result["confidence_upper"]
    }
    
    history.append(new_entry)
    save_prediction_history(history)
    return history

def check_api_status():
    """Check API health"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200 and response.json().get("model_loaded", False)
    except Exception as e:
        st.error(f"API Connection Error: {str(e)}")
        return False

def make_prediction(data):
    """Make prediction API call"""
    try:
        response = requests.post(f"{API_URL}/predict", json=data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
        return None

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Grocery Demand Forecasting Platform</h1>
        <p>AI-Powered Demand Prediction System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Debug info
    st.sidebar.write(f"API URL: {API_URL}")
    
    # Sidebar with real data
    with st.sidebar:
        st.title("System Status")
        
        # API Status
        if check_api_status():
            st.success("API Connected")
            st.success("Model Loaded")
        else:
            st.error("API Disconnected")
        
        st.markdown("---")
        
        # Real prediction statistics
        history = load_prediction_history()
        
        st.subheader("Prediction Statistics")
        st.metric("Total Predictions", len(history))
        
        if history:
            today_predictions = [p for p in history if p["timestamp"].startswith(datetime.now().strftime("%Y-%m-%d"))]
            st.metric("Today's Predictions", len(today_predictions))
            
            avg_demand = sum([p["predicted_demand"] for p in history]) / len(history)
            st.metric("Average Predicted Demand", f"{avg_demand:.1f}")
        else:
            st.metric("Today's Predictions", 0)
            st.metric("Average Predicted Demand", "N/A")
    
    # Main tabs
    tab1, tab2 = st.tabs(["Demand Prediction", "Prediction History"])
    
    with tab1:
        st.subheader("Generate New Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Prediction form
            with st.form("prediction_form"):
                form_col1, form_col2 = st.columns(2)
                
                with form_col1:
                    store_display = st.selectbox("Store Location", list(STORE_MAPPING.keys()))
                    product_display = st.selectbox("Product", list(PRODUCT_MAPPING.keys()))
                    category = st.selectbox("Category", ["Dairy", "Meat", "Produce", "Bakery", "Frozen", "Pantry"])
                    brand = st.selectbox("Brand", ["President's Choice", "No Name", "National Brand"])
                
                with form_col2:
                    chain = st.selectbox("Chain", ["Loblaws", "Metro", "Sobeys", "FreshCo", "No Frills"])
                    province = st.selectbox("Province", ["ON", "QC", "BC", "AB", "MB", "SK", "NS", "NB", "NL", "PE"])
                    price = st.number_input("Price ($)", min_value=0.01, value=5.99, step=0.01)
                    promotion = st.checkbox("On Promotion")
                
                prediction_date = st.date_input("Forecast Date", value=date.today() + timedelta(days=1))
                
                submitted = st.form_submit_button("Generate Prediction", use_container_width=True)
                
                if submitted:
                    # Prepare API request
                    prediction_data = {
                        "store_id": STORE_MAPPING[store_display],
                        "product_id": PRODUCT_MAPPING[product_display],
                        "date": prediction_date.strftime("%Y-%m-%d"),
                        "price": price,
                        "promotion_flag": int(promotion),
                        "chain": chain,
                        "province": province,
                        "category": category,
                        "brand": brand
                    }
                    
                    with st.spinner("Generating prediction..."):
                        result = make_prediction(prediction_data)
                        
                        if result:
                            # Save to history
                            add_prediction_to_history(prediction_data, result)
                            st.session_state.latest_prediction = result
                            st.session_state.latest_data = prediction_data
                            st.success("Prediction generated successfully!")
                            st.rerun()
                        else:
                            st.error("Prediction failed. Check API connection.")
        
        with col2:
            # Display latest prediction
            if hasattr(st.session_state, 'latest_prediction'):
                result = st.session_state.latest_prediction
                data = st.session_state.latest_data
                
                st.markdown(f"""
                <div class="prediction-result">
                    <h3>Predicted Demand</h3>
                    <h1 style="font-size: 3rem; margin: 1rem 0;">{result['predicted_demand']}</h1>
                    <p>units expected to be sold</p>
                </div>
                """, unsafe_allow_html=True)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Lower Bound", result['confidence_lower'])
                with col_b:
                    st.metric("Upper Bound", result['confidence_upper'])
                
                # Business recommendation
                demand = result['predicted_demand']
                if demand > 20:
                    st.error("High Demand - Stock Up")
                elif demand > 10:
                    st.warning("Normal Demand - Monitor")
                else:
                    st.info("Low Demand - Consider Promotion")
            else:
                st.info("Submit the form to see prediction results")
    
    with tab2:
        st.subheader("Prediction History")
        
        history = load_prediction_history()
        
        if history:
            # Convert to DataFrame
            df = pd.DataFrame(history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Predictions", len(df))
            
            with col2:
                avg_demand = df['predicted_demand'].mean()
                st.metric("Average Demand", f"{avg_demand:.1f}")
            
            with col3:
                max_demand = df['predicted_demand'].max()
                st.metric("Highest Prediction", f"{max_demand:.1f}")
            
            with col4:
                unique_products = df['product_name'].nunique()
                st.metric("Unique Products", unique_products)
            
            # Prediction trends
            st.subheader("Prediction Trends")
            
            # Time series of predictions
            daily_stats = df.groupby(df['timestamp'].dt.date).agg({
                'predicted_demand': ['count', 'mean']
            }).round(2)
            daily_stats.columns = ['Count', 'Avg_Demand']
            daily_stats = daily_stats.reset_index()
            
            if len(daily_stats) > 0:
                fig = px.line(daily_stats, x='timestamp', y='Avg_Demand', 
                             title="Average Daily Predicted Demand")
                st.plotly_chart(fig, use_container_width=True)
            
            # Store performance
            st.subheader("Store Analysis")
            store_stats = df.groupby('store_name').agg({
                'predicted_demand': ['count', 'mean']
            }).round(2)
            store_stats.columns = ['Predictions', 'Avg_Demand']
            store_stats = store_stats.reset_index()
            
            fig = px.bar(store_stats, x='store_name', y='Avg_Demand', 
                        title="Average Predicted Demand by Store")
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent predictions table
            st.subheader("Recent Predictions")
            recent_df = df.tail(10)[['timestamp', 'store_name', 'product_name', 'predicted_demand', 'price', 'promotion_flag']]
            recent_df['promotion_flag'] = recent_df['promotion_flag'].map({1: 'Yes', 0: 'No'})
            recent_df.columns = ['Time', 'Store', 'Product', 'Predicted Demand', 'Price', 'On Promotion']
            st.dataframe(recent_df, use_container_width=True)
            
            # Clear history button
            if st.button("Clear All History", type="secondary"):
                if os.path.exists(PREDICTIONS_FILE):
                    try:
                        os.remove(PREDICTIONS_FILE)
                        st.success("History cleared!")
                        st.rerun()
                    except:
                        st.error("Could not clear history")
        
        else:
            st.info("No predictions yet. Generate some predictions to see history and analytics.")

if __name__ == "__main__":
    main()