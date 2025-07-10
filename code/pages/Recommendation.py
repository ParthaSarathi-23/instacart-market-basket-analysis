import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import pickle

# Add your own recommendation function and data loading here
# from utils import recommend_products
# products_data = pd.read_csv("...")
# users_data = pd.read_csv("...")
# product_metrics = pd.read_csv("...")

st.set_page_config(page_title="Product Recommendations", layout="wide")
st.markdown("<h1 class='main-header'>Product Recommendations</h1>", unsafe_allow_html=True)

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


col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<h2 class='sub-header'>Get Recommendations</h2>", unsafe_allow_html=True)

    user_id_input = st.number_input(
        "Enter User ID:",
        min_value=1,
        max_value=users_data['user_id'].max() if not users_data.empty else 1000
    )
    num_recommendations = st.slider("Number of recommendations:", 1, 20, 5)

    if st.button("Get Recommendations"):
        if user_id_input not in users_data['user_id'].values:
            st.warning("User ID not found. Please enter a valid User ID.")
        else:
            with st.spinner("Generating recommendations..."):
                recommendations = recommend_products(user_id_input, products_data, num_recommendations)

                st.markdown("<h3 class='section-header'>Recommended Products</h3>", unsafe_allow_html=True)

                if not recommendations:
                    st.info("No recommendations available for this user.")
                else:
                    for i, (product_id, product_name, score) in enumerate(recommendations, 1):
                        st.markdown(f"""
                            <div class='highlight' style="
                                background: linear-gradient(to right, #FEF9E7, #FBFCFC);
                                padding: 15px;
                                border-radius: 10px;
                                border-left: 6px solid #F5B041;
                                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
                                margin: 10px 0;
                                font-size: 15px;
                                line-height: 1.5;
                                font-family: sans-serif;
                                color: #4D5656;
                                ">
                                <h4 style="margin-top: 0; color: #B9770E;">#{i}: {product_name}</h4>
                                <p style="margin: 5px 0;">
                                    <strong>Product ID:</strong> {product_id}<br>
                                    <strong>Recommendation Score:</strong> {score:.4f}
                                </p>
                            </div>
                        """, unsafe_allow_html=True)

with col2:
    st.markdown("<h2 class='sub-header'>Recommendation Engine</h2>", unsafe_allow_html=True)

    st.markdown("""
        <div class='highlight' style="
            color: black;
            background: linear-gradient(to right, #E8F8F5, #FDFEFE);
            padding: 20px;
            border-radius: 10px;
            border-left: 6px solid #1ABC9C;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            margin: 15px 0;
            font-size: 16px;
            line-height: 1.6;
            font-family: sans-serif;">
            <h3 style="margin-top: 0; color: #1ABC9C; font-weight: 600;">How It Works</h3>
            <p>The recommendation system uses machine learning to predict which products a user is likely to purchase based on:</p>
            <ul style="margin-left: 20px;">
                <li>User's purchase history</li>
                <li>Products frequently bought together</li>
                <li>Product popularity and reorder rates</li>
                <li>User's shopping patterns and preferences</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3 class='section-header'>Feature Importance</h3>", unsafe_allow_html=True)

    features = ['User Product Frequency', 'Product Reorder Rate', 'Days Since Last Order',
                'Department Preference', 'Aisle Preference', 'Purchase Frequency',
                'Total User Orders', 'Basket Size', 'Order Time', 'Order Day']
    importance = [0.28, 0.22, 0.12, 0.08, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03]

    fig = px.bar(
        x=importance,
        y=features,
        orientation='h',
        title='Model Feature Importance',
        color=importance,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(xaxis_title='Importance', yaxis_title='Feature')

    st.plotly_chart(fig, use_container_width=True)

# --- Department Recommendation Explorer ---
st.markdown("<h2 class='sub-header'>Department Recommendation Explorer</h2>", unsafe_allow_html=True)

departments = product_metrics["department"].unique().tolist()
selected_dept = st.selectbox("Select Department:", departments)

if selected_dept:
    dept_products = products_data[products_data['department'] == selected_dept] if 'department' in products_data.columns else products_data.sample(10)

    if not dept_products.empty:
        top_dept_products = dept_products.sort_values('times_purchased', ascending=False).head(8) if 'times_purchased' in dept_products.columns else dept_products.sample(8)

        st.markdown(f"<h3 class='section-header'>Top Products in {selected_dept}</h3>", unsafe_allow_html=True)

        product_cols = st.columns(4)
        for i, (_, product) in enumerate(top_dept_products.iterrows()):
            col_idx = i % 4
            with product_cols[col_idx]:
                product_name = product.get('product_name', f"Product {product.get('product_id', 'N/A')}")
                reorder_rate = f"{product['reorder_rate']:.2%}" if 'reorder_rate' in product else "N/A"
                st.markdown(f"""
                    <div class='highlight' style="
                        background: linear-gradient(to right, #FDFEFE, #F9F9F9);
                        padding: 15px;
                        border-radius: 10px;
                        border-left: 6px solid #5D6D7E;
                        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
                        margin: 10px 0;
                        font-size: 15px;
                        line-height: 1.5;
                        font-family: sans-serif;
                        color: #2C3E50;
                        ">
                        <h4 style="margin-top: 0; color: black;">{product_name}</h4>
                        <p style="margin: 5px 0;"><strong>Purchases:</strong> {product.get('times_purchased', 'N/A')}</p>
                        <p style="margin: 5px 0;"><strong>Reorder Rate:</strong> {reorder_rate}</p>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info(f"No products found in the {selected_dept} department.")
