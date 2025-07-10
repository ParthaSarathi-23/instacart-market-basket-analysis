import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import pickle



st.set_page_config(page_title="Market Basket Analysis", layout="wide")
st.markdown("<h1 class='main-header'>Market Basket Analysis</h1>", unsafe_allow_html=True)

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

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<h2 class='sub-header'>Association Rules</h2>", unsafe_allow_html=True)

    if not association_rules.empty and 'antecedents' in association_rules.columns:
        min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.5, 0.01)
        min_lift = st.slider("Minimum Lift", 1.0, 10.0, 2.0, 0.1)
        filtered_rules = association_rules[
            (association_rules['confidence'] >= min_confidence) & 
            (association_rules['lift'] >= min_lift)
        ]
        
        if not filtered_rules.empty:
            st.dataframe(filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
        else:
            st.info("No rules match the selected criteria.")
    else:
        st.markdown("""
        <div class='highlight' style="
            color: black;
            background: linear-gradient(to right, #FCF3CF, #FDFEFE);
            padding: 20px;
            border-radius: 10px;
            border-left: 6px solid #F1C40F;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            margin: 15px 0;
            font-size: 16px;
            line-height: 1.6;
            font-family: sans-serif;
            ">
            <h3 style="margin-top: 0; color: #F1C40F; font-weight: 600;">Sample Association Rules</h3>
            <p style="color: black; margin-bottom: 0;">For demonstration purposes, here are some example rules discovered through market basket analysis.</p>
        </div>
        """, unsafe_allow_html=True)

        demo_rules = pd.DataFrame({
            'antecedents': ['Organic Banana, Strawberries', 'Organic Avocado, Tortilla Chips', 
                                 'Milk, Bread', 'Eggs, Bacon', 'Peanut Butter'],
            'consequents': ['Organic Blueberries', 'Lime', 'Butter', 'Orange Juice', 'Jelly'],
            'support': [0.12, 0.09, 0.15, 0.07, 0.06],
            'confidence': [0.68, 0.72, 0.60, 0.65, 0.82],
            'lift': [3.2, 4.5, 2.8, 3.8, 7.2]
        })

        min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.5, 0.01)
        min_lift = st.slider("Minimum Lift", 1.0, 10.0, 2.0, 0.1)
        filtered_rules = demo_rules[
            (demo_rules['confidence'] >= min_confidence) & 
            (demo_rules['lift'] >= min_lift)
        ]

        if not filtered_rules.empty:
            st.dataframe(filtered_rules)
        else:
            st.info("No rules match the selected criteria.")

with col2:
    st.markdown("<h2 class='sub-header'>Product Affinities</h2>", unsafe_allow_html=True)

    all_products = sorted(products_data['product_name'].dropna().unique().tolist())
    selected_product = st.selectbox("Select a product:", all_products)

    st.markdown("<h3 class='section-header'>Frequently Purchased With</h3>", unsafe_allow_html=True)

    if selected_product:
        np.random.seed(hash(selected_product) % 10000)
        related_products = np.random.choice(
            [p for p in all_products if p != selected_product], 
            size=5, 
            replace=False
        )
        related_scores = np.random.uniform(0.2, 0.8, size=5)
        related_df = pd.DataFrame({
            'Product': related_products,
            'Affinity Score': related_scores
        }).sort_values('Affinity Score', ascending=False)

        fig = px.bar(
            related_df,
            x='Affinity Score',
            y='Product',
            orientation='h',
            title=f'Products Frequently Purchased with {selected_product}',
            color='Affinity Score',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
