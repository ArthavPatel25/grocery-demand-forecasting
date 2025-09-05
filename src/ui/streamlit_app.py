# src/ui/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime, date, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="Grocery Demand Forecasting",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

def call_api(endpoint, method="GET", data=None):
    """Helper function to call API endpoints"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Please ensure FastAPI server is running at localhost:8000")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🛒 Canadian Grocery Demand Forecasting</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard", "Single Prediction", "Batch Analysis", "Model Info"]
    )
    
    # Check API health
    health_status = call_api("/health")
    if health_status:
        if health_status["model_loaded"]:
            st.sidebar.success("✅ API Connected")
            st.sidebar.success("✅ Model Loaded")
        else:
            st.sidebar.error("❌ Model Not Loaded")
    else:
        st.sidebar.error("❌ API Not Connected")
    
    # Route to pages
    if page == "Dashboard":
        dashboard_page()
    elif page == "Single Prediction":
        prediction_page()
    elif page == "Batch Analysis":
        batch_analysis_page()
    elif page == "Model Info":
        model_info_page()

def dashboard_page():
    """Main dashboard"""
    st.title("📊 Executive Dashboard")
    
    # Key metrics (simulated)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Accuracy", "94.9%", "1.2%")
    
    with col2:
        st.metric("Avg Prediction MAE", "2.84 units", "-0.3")
    
    with col3:
        st.metric("API Uptime", "99.9%", "0.1%")
    
    with col4:
        st.metric("Predictions Today", "1,247", "156")
    
    st.markdown("---")
    
    # Sample charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Prediction Volume")
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=30)
        volumes = np.random.normal(1200, 200, 30).astype(int)
        
        fig = px.line(x=dates, y=volumes, title="Daily Prediction Requests")
        fig.update_traces(line_color='#1f77b4')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏪 Performance by Store Type")
        store_data = {
            'Store Type': ['Large', 'Medium', 'Small'],
            'Accuracy': [96.2, 94.1, 92.3],
            'Count': [245, 156, 89]
        }
        
        fig = px.bar(store_data, x='Store Type', y='Accuracy', 
                     title="Model Accuracy by Store Size")
        st.plotly_chart(fig, use_container_width=True)

def prediction_page():
    """Single prediction interface"""
    st.title("🔮 Single Demand Prediction")
    
    st.markdown("Enter product and store details to get an AI-powered demand forecast.")
    
    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            store_id = st.selectbox(
                "Store ID",
                options=["ST_001", "ST_002", "ST_003", "ST_004", "ST_005"]
            )
            
            product_id = st.selectbox(
                "Product ID", 
                options=[f"PR_{i:04d}" for i in range(1001, 1021)]
            )
            
            category = st.selectbox(
                "Category",
                options=["Dairy", "Meat", "Produce", "Bakery", "Frozen", "Pantry"]
            )
            
            brand = st.selectbox(
                "Brand",
                options=["Brand_A", "Brand_B", "Generic"]
            )
        
        with col2:
            prediction_date = st.date_input(
                "Prediction Date",
                value=date.today() + timedelta(days=1),
                min_value=date.today()
            )
            
            price = st.number_input(
                "Price ($)",
                min_value=0.01,
                max_value=100.00,
                value=5.99,
                step=0.01
            )
            
            chain = st.selectbox(
                "Store Chain",
                options=["Loblaws", "Metro", "Sobeys", "FreshCo", "No Frills"]
            )
            
            province = st.selectbox(
                "Province",
                options=["ON", "QC", "BC", "AB", "MB", "SK", "NS", "NB", "NL", "PE"]
            )
        
        promotion_flag = st.checkbox("On Promotion")
        
        submitted = st.form_submit_button("🔮 Generate Prediction", use_container_width=True)
        
        if submitted:
            # Prepare API request
            request_data = {
                "store_id": store_id,
                "product_id": product_id,
                "date": prediction_date.strftime("%Y-%m-%d"),
                "price": price,
                "promotion_flag": int(promotion_flag),
                "chain": chain,
                "province": province,
                "category": category,
                "brand": brand
            }
            
            # Call API
            with st.spinner("🤖 AI is analyzing demand patterns..."):
                result = call_api("/predict", method="POST", data=request_data)
            
            if result:
                st.success("✅ Prediction Generated Successfully!")
                
                # Display main prediction
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                           color: white; padding: 2rem; border-radius: 15px; text-align: center; margin: 1rem 0;">
                    <h2>🎯 Predicted Demand</h2>
                    <h1 style="font-size: 3rem; margin: 1rem 0;">{result['predicted_demand']}</h1>
                    <p style="font-size: 1.2rem;">units expected to be sold</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Confidence Range",
                        f"{result['confidence_lower']} - {result['confidence_upper']}",
                        help="95% confidence interval"
                    )
                
                with col2:
                    st.metric(
                        "Model Used", 
                        result['model_used']
                    )
                
                with col3:
                    # Calculate recommendation
                    demand = result['predicted_demand']
                    if demand < 10:
                        recommendation = "🟢 Normal Stock"
                    elif demand < 25:
                        recommendation = "🟡 Monitor"
                    else:
                        recommendation = "🔴 High Demand"
                    
                    st.metric("Recommendation", recommendation)
                
                # Show prediction details
                with st.expander("📋 Prediction Details"):
                    st.json(result)

def batch_analysis_page():
    """Batch prediction interface"""
    st.title("📊 Batch Analysis")
    
    st.markdown("Upload a CSV file or create multiple predictions at once.")
    
    # Sample data generator
    if st.button("📝 Generate Sample Data"):
        sample_data = []
        for i in range(10):
            sample_data.append({
                "store_id": f"ST_{(i%5)+1:03d}",
                "product_id": f"PR_{1001+i:04d}",
                "date": (date.today() + timedelta(days=i)).strftime("%Y-%m-%d"),
                "price": round(np.random.uniform(2.99, 19.99), 2),
                "promotion_flag": int(np.random.random() < 0.2),
                "chain": np.random.choice(["Loblaws", "Metro", "Sobeys"]),
                "province": np.random.choice(["ON", "QC", "BC"]),
                "category": np.random.choice(["Dairy", "Meat", "Produce"]),
                "brand": np.random.choice(["Brand_A", "Brand_B", "Generic"])
            })
        
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)
        
        if st.button("🚀 Process Batch Predictions"):
            batch_request = {"predictions": sample_data}
            
            with st.spinner("Processing batch predictions..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                result = call_api("/predict/batch", method="POST", data=batch_request)
            
            if result:
                st.success(f"✅ Processed {result['success_count']} predictions!")
                
                # Display results
                predictions_df = pd.DataFrame(result['predictions'])
                st.dataframe(predictions_df, use_container_width=True)
                
                # Summary stats
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Predicted Demand", 
                             f"{predictions_df['predicted_demand'].sum():.1f}")
                
                with col2:
                    st.metric("Average Demand", 
                             f"{predictions_df['predicted_demand'].mean():.1f}")
                
                with col3:
                    st.metric("Max Demand", 
                             f"{predictions_df['predicted_demand'].max():.1f}")
                
                # Visualization
                fig = px.histogram(predictions_df, x='predicted_demand', 
                                 title="Distribution of Predicted Demands")
                st.plotly_chart(fig, use_container_width=True)

def model_info_page():
    """Model information page"""
    st.title("🔧 Model Information")
    
    # Get model info from API
    model_info = call_api("/model/info")
    
    if model_info:
        st.subheader("📊 Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'models' in model_info and 'lightgbm' in model_info['models']:
                lgb_metrics = model_info['models']['lightgbm']
                st.metric("LightGBM Test MAE", f"{lgb_metrics['test_mae']:.4f}")
                st.metric("LightGBM Test R²", f"{lgb_metrics['test_r2']:.4f}")
        
        with col2:
            if 'models' in model_info and 'xgboost' in model_info['models']:
                xgb_metrics = model_info['models']['xgboost']
                st.metric("XGBoost Test MAE", f"{xgb_metrics['test_mae']:.4f}")
                st.metric("XGBoost Test R²", f"{xgb_metrics['test_r2']:.4f}")
        
        st.subheader("🎯 Feature Importance")
        
        if 'feature_importance' in model_info:
            best_model = model_info.get('best_model', 'lightgbm').lower()
            importance_key = f"{best_model}_top_10"
            
            if importance_key in model_info['feature_importance']:
                importance_data = model_info['feature_importance'][importance_key]
                importance_df = pd.DataFrame(importance_data)
                
                fig = px.bar(importance_df, x='importance', y='feature',
                           orientation='h', title=f"Top 10 Features - {best_model.upper()}")
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📋 Model Metadata")
        st.json(model_info)
    else:
        st.error("Could not retrieve model information")

if __name__ == "__main__":
    main()