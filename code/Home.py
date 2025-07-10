import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from PIL import Image
import io
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

# Set page configuration
st.set_page_config(
    page_title="Instacart Market Basket Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6F61;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2E86C1;
        font-weight: 600;
    }
    .section-header {
        font-size: 1.5rem;
        color: #3498DB;
        font-weight: 500;
    }
    .highlight {
        background-color: #F8F9F9;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #3498DB;
    }
    .info-box {
        background-color: #E8F8F5;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .stDataFrame {
        max-height: 500px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for navigation
#st.sidebar.image("https://d4.alternativeto.net/UbDAm_qiYgG_kzirO3ej_dP_vr5lDSN0h9J_LFQVsYY/rs:fill:280:280:0/g:ce:0:0/YWJzOi8vZGlzdC9pY29ucy9pbnN0YWNhcnRfMTkwNTM5LnBuZw.png", width=100)


# Define paths to data files
# These files are created by the main script
try:
    product_metrics = pd.read_csv('product_metrics.csv')
except FileNotFoundError:
    st.warning("Product metrics file not found. Some features may not work properly.")
    product_metrics = pd.DataFrame()

try:
    user_metrics = pd.read_csv('user_metrics.csv')
except FileNotFoundError:
    st.warning("User metrics file not found. Some features may not work properly.")
    user_metrics = pd.DataFrame()

try:
    association_rules = pd.read_csv('association_rules.csv')
except FileNotFoundError:
    st.warning("Association rules file not found. Some features may not work properly.")
    association_rules = pd.DataFrame()

try:
    with open('reorder_prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.warning("Model file not found. Recommendation features may not work properly.")
    model = None

# Load data (for demonstration purposes, we'll simulate the data if files don't exist)
@st.cache_data
def load_data():
    # If data files are missing, create simulated data for demonstration
    if product_metrics.empty:
        # Simulate product metrics
        product_ids = range(1, 101)
        product_names = [f"Product {i}" for i in product_ids]
        departments = ["Produce", "Dairy", "Bakery", "Snacks", "Beverages", "Frozen", "Canned", "Meat", "Seafood", "Household"]
        aisles = ["Fresh Fruits", "Fresh Vegetables", "Milk", "Bread", "Chips", "Soda", "Frozen Meals", "Canned Soup", "Beef", "Cleaning"]
        
        sim_product_metrics = pd.DataFrame({
            'product_id': product_ids,
            'product_name': product_names,
            'times_purchased': np.random.randint(50, 5000, size=100),
            'times_reordered': np.random.randint(10, 2000, size=100),
            'reorder_rate': np.random.uniform(0.1, 0.9, size=100),
            'department': np.random.choice(departments, size=100),
            'aisle': np.random.choice(aisles, size=100)
        })
        
        return sim_product_metrics
    else:
        return product_metrics

@st.cache_data
def load_user_data():
    # If data files are missing, create simulated data for demonstration
    if user_metrics.empty:
        # Simulate user metrics
        user_ids = range(1, 101)
        
        sim_user_metrics = pd.DataFrame({
            'user_id': user_ids,
            'total_orders': np.random.randint(1, 50, size=100),
            'total_products': np.random.randint(10, 500, size=100),
            'total_reorders': np.random.randint(5, 300, size=100),
            'avg_products_per_order': np.random.uniform(1, 20, size=100),
            'reorder_ratio': np.random.uniform(0.1, 0.9, size=100)
        })
        
        return sim_user_metrics
    else:
        return user_metrics

# Load product data
products_data = load_data()
users_data = load_user_data()

# Simulate recommendation function if real one is not available
def recommend_products(user_id, products_data, n_recommendations=5):
    """Simplified recommendation function for demonstration"""
    if model is None:
        # Simple recommendation based on product popularity
        popular_products = products_data.sort_values('times_purchased', ascending=False).head(n_recommendations)
        return [(p_id, name, score) for p_id, name, score in 
               zip(popular_products['product_id'], popular_products['product_name'], 
                   popular_products['times_purchased'])]
    else:
        # Use the trained model (this code would need to be adapted to your actual model)
        # For demo purposes, we'll return random products
        sampled_products = products_data.sample(n_recommendations)
        return [(p_id, name, np.random.random()) for p_id, name in 
               zip(sampled_products['product_id'], sampled_products['product_name'])]

# HOME PAGE

st.markdown("<h1 class='main-header'>Instacart Market Basket Analysis</h1>", unsafe_allow_html=True)

# Dashboard Overview
col1, col2 = st.columns(2)

with col1:
    
    st.markdown("""
    <div class='info-box' style="
        color: black;
        background: linear-gradient(to right, #E8F8F5, #F2F9F9);
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #2E86C1;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 15px 0;
        font-size: 16px;
        line-height: 1.6;
        ">
        <h3 style="margin-top: 0; color: #2E86C1; font-weight: 600;">Welcome to Customer Insights</h3>
        <p style="color: black; margin-bottom: 10px;">This application provides comprehensive insights into customer purchasing patterns based on Instacart data.</p>
        <p style="color: black; margin-bottom: 0;">Use the navigation panel on the left to explore different analyses and features.</p>
        <div style="margin-top: 15px;">
            <span style="background-color: #2E86C1; color: white; padding: 5px 10px; border-radius: 4px; font-size: 14px;">Get Started →</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    st.markdown("<h3 class='section-header'>Key Metrics</h3>", unsafe_allow_html=True)
    
    metric1, metric2, metric3 = st.columns(3)
    with metric1:
        st.metric(label="Total Products", value=f"{len(products_data):,}")
    with metric2:
        avg_reorder = products_data['reorder_rate'].mean() if 'reorder_rate' in products_data.columns else 0.53
        st.metric(label="Avg Reorder Rate", value=f"{avg_reorder:.2%}")
    with metric3:
        st.metric(label="Total Users", value=f"{len(users_data):,}")

with col2:
    st.markdown("<h2 class='sub-header'>Quick Insights</h2>", unsafe_allow_html=True)
    
    # Create a sample insight chart
    if 'times_purchased' in products_data.columns and 'department' in products_data.columns:
        dept_popularity = products_data.groupby('department')['times_purchased'].sum().sort_values(ascending=False).head(5)
        fig = px.bar(
            x=dept_popularity.index,
            y=dept_popularity.values,
            labels={'x': 'Department', 'y': 'Total Purchases'},
            title='Top 5 Departments by Popularity',
            color=dept_popularity.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Demo chart
        departments = ["Produce", "Dairy", "Snacks", "Bakery", "Beverages"]
        purchases = [18500, 12300, 9800, 7600, 6200]
        fig = px.bar(
            x=departments,
            y=purchases,
            labels={'x': 'Department', 'y': 'Total Purchases'},
            title='Top 5 Departments by Popularity',
            color=purchases,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

# Main Features
st.markdown("<h2 class='sub-header'>Main Features</h2>", unsafe_allow_html=True)

feature_col1, feature_col2, feature_col3 = st.columns(3)

with feature_col1:
    st.markdown("""
    <div class='highlight' style="
        color: black;
        background: linear-gradient(to right, #F0F8FF, #F8F9F9);
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #3498DB;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 15px 0;
        font-size: 16px;
        line-height: 1.6;
        ">
        <h3 style="margin-top: 0; color: #3498DB; font-weight: 600;">Market Basket Analysis</h3>
        <p style="color: black; margin-bottom: 10px;">Discover what products are frequently purchased together and explore association rules.</p>
    </div>
    """, unsafe_allow_html=True)

with feature_col2:
    st.markdown("""
    <div class='highlight' style="
        color: black;
        background: linear-gradient(to right, #F0F8FF, #F8F9F9);
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #3498DB;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 15px 0;
        font-size: 16px;
        line-height: 1.6;
        ">
        <h3 style="margin-top: 0; color: #3498DB; font-weight: 600;">User Segmentation</h3>
        <p style="color: black; margin-bottom: 10px;">Explore different customer segments based on their shopping behaviors.</p>
    </div>
    """, unsafe_allow_html=True)
    

with feature_col3:
    st.markdown("""
    <div class='highlight' style="
        color: black;
        background: linear-gradient(to right, #F0F8FF, #F8F9F9);
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #3498DB;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 15px 0;
        font-size: 16px;
        line-height: 1.6;
        ">
        <h3 style="margin-top: 0; color: #3498DB; font-weight: 600;">Personalized Recommendations</h3>
        <p style="color: black; margin-bottom: 10px;">Get product recommendations based on user purchase history.</p>
    </div>
    """, unsafe_allow_html=True)

