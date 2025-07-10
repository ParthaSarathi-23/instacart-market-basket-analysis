import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="About This Project", layout="wide")

st.markdown("<h1 class='main-header'>About This Project</h1>", unsafe_allow_html=True)

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
print(product_metrics["department"].unique)
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

st.markdown("""
<div class='info-box'>
<h2 class='sub-header'>Instacart Market Basket Analysis & Recommendation System</h2>
<p>This project analyzes customer purchasing patterns from Instacart data to understand shopping behaviors 
and build recommendation models.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("<h2 class='sub-header'>Key Components</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <h3 class='section-header'>Data Analysis</h3>
    <ul>
        <li>Exploratory analysis of order patterns and product popularity</li>
        <li>Time series analysis of order frequency by day and hour</li>
        <li>Product network analysis to discover related items</li>
    </ul>

    <h3 class='section-header'>Machine Learning Components</h3>
    <ul>
        <li>Association rule mining using Apriori algorithm</li>
        <li>Customer segmentation using K-means clustering</li>
        <li>Product recommendation system using Random Forest classifier</li>
    </ul>

    <h3 class='section-header'>Technologies Used</h3>
    <ul>
        <li>Python, Pandas, NumPy for data processing</li>
        <li>Scikit-learn for machine learning models</li>
        <li>Matplotlib, Seaborn and Plotly for visualization</li>
        <li>Streamlit for the interactive web application</li>
    </ul>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("<h2 class='sub-header'>Project Structure</h2>", unsafe_allow_html=True)

    st.code("""
    project/
    ├── data/
    │   ├── aisles.csv
    │   ├── departments.csv
    │   ├── orders.csv
    │   ├── order_products_*.csv
    │   └── products.csv
    ├── models/
    │   └── reorder_prediction_model.pkl
    ├── results/
    │   ├── product_metrics.csv
    │   ├── user_metrics.csv
    │   └── association_rules.csv
    ├── main.py
    ├── app.py (Streamlit app)
    └── README.md
    """)

    st.markdown("<h2 class='sub-header'>Usage</h2>", unsafe_allow_html=True)

    st.markdown("<p>To run the Streamlit app locally:</p>", unsafe_allow_html=True)
    st.code("streamlit run app.py")
    st.markdown("<p>Make sure you have the required data files in the correct location or use the simulated data mode.</p>", unsafe_allow_html=True)

# Usage instructions
st.markdown("<h2 class='sub-header'>How to Use This Dashboard</h2>", unsafe_allow_html=True)

st.markdown("""
<ol>
    <li>Navigate between pages using the sidebar menu</li>
    <li>Explore data insights on the Data Overview page</li>
    <li>Discover product relationships on the Market Basket Analysis page</li>
    <li>Learn about customer segments on the User Segmentation page</li>
    <li>Get personalized product recommendations on the Recommendations page</li>
</ol>
""", unsafe_allow_html=True)

# Upload section
st.markdown("<h2 class='sub-header'>Upload Your Own Data</h2>", unsafe_allow_html=True)

st.markdown("<p>You can upload your own Instacart data files to analyze with this app:</p>", unsafe_allow_html=True)

upload_col1, upload_col2 = st.columns(2)

with upload_col1:
    orders_file = st.file_uploader("Upload Orders CSV", type=['csv'])

with upload_col2:
    products_file = st.file_uploader("Upload Products CSV", type=['csv'])

if orders_file and products_file:
    st.success("Data files uploaded successfully! Click 'Process Data' to analyze.")

    if st.button("Process Data"):
        with st.spinner("Processing data..."):
            # Placeholder for real processing
            st.info("This is a demonstration - actual data processing would happen here.")
            st.success("Processing complete!")
