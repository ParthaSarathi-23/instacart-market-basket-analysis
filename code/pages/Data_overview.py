import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import pickle

# Replace with your actual data loading logic


st.set_page_config(page_title="Data Overview", layout="wide")
st.markdown("<h1 class='main-header'>Data Overview</h1>", unsafe_allow_html=True)

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

tab1, tab2, tab3 = st.tabs(["Product Analysis", "User Analysis", "Time Series Analysis"])

with tab1:
    st.markdown("<h2 class='sub-header'>Product Analysis</h2>", unsafe_allow_html=True)

    st.markdown("<h3 class='section-header'>Top Products by Purchase Frequency</h3>", unsafe_allow_html=True)

    if 'product_name' in products_data.columns and 'times_purchased' in products_data.columns:
        top_products = products_data.sort_values('times_purchased', ascending=False).head(10)
        fig = px.bar(
            top_products,
            x='product_name',
            y='times_purchased',
            color='reorder_rate',
            color_continuous_scale='Viridis',
            title='Top 10 Products by Purchase Frequency',
            labels={'times_purchased': 'Purchase Count', 'product_name': 'Product', 'reorder_rate': 'Reorder Rate'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Product purchase data not available.")

    st.markdown("<h3 class='section-header'>Product Data</h3>", unsafe_allow_html=True)

    search_term = st.text_input("Search for products:")
    if search_term:
        filtered_data = products_data[products_data['product_name'].str.contains(search_term, case=False)]
        st.dataframe(filtered_data)
    else:
        st.dataframe(products_data.head(100))

with tab2:
    st.markdown("<h2 class='sub-header'>User Analysis</h2>", unsafe_allow_html=True)

    st.markdown("<h3 class='section-header'>User Shopping Behavior</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            users_data,
            x='total_orders',
            nbins=20,
            title='Distribution of Orders per User',
            labels={'total_orders': 'Number of Orders'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            users_data,
            x='avg_products_per_order',
            nbins=20,
            title='Distribution of Average Products per Order',
            labels={'avg_products_per_order': 'Average Products per Order'}
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<h3 class='section-header'>User Metrics</h3>", unsafe_allow_html=True)
    st.dataframe(users_data.describe())

    st.markdown("<h3 class='section-header'>User Lookup</h3>", unsafe_allow_html=True)
    user_id_input = st.number_input(
        "Enter User ID:", 
        min_value=1, 
        max_value=users_data['user_id'].max() if not users_data.empty else 100
    )

    if st.button("Look Up User"):
        user_data = users_data[users_data['user_id'] == user_id_input]
        if not user_data.empty:
            st.write(user_data)
        else:
            st.warning("User not found.")

with tab3:
    st.markdown("<h2 class='sub-header'>Time Series Analysis</h2>", unsafe_allow_html=True)

    st.markdown("<h3 class='section-header'>Order Patterns by Day of Week</h3>", unsafe_allow_html=True)

    days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    order_counts = [8500, 6200, 5800, 6300, 7100, 9200, 9800]

    fig = px.line(
        x=days,
        y=order_counts,
        markers=True,
        title='Order Volume by Day of Week',
        labels={'x': 'Day of Week', 'y': 'Number of Orders'}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<h3 class='section-header'>Order Patterns by Hour of Day</h3>", unsafe_allow_html=True)

    hours = list(range(24))
    hourly_orders = [120, 80, 50, 30, 20, 40, 150, 380, 650, 920, 1150, 1300, 
                     1400, 1380, 1250, 1100, 1000, 950, 850, 680, 520, 410, 300, 190]

    fig = px.line(
        x=hours,
        y=hourly_orders,
        markers=True,
        title='Order Volume by Hour of Day',
        labels={'x': 'Hour of Day', 'y': 'Number of Orders'}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<h3 class='section-header'>Order Heatmap</h3>", unsafe_allow_html=True)

    heatmap_data = np.zeros((7, 24))
    for d in range(7):
        for h in range(24):
            base = hourly_orders[h] * (0.7 + 0.3 * (order_counts[d] / max(order_counts)))
            heatmap_data[d, h] = base * (0.9 + 0.2 * np.random.random())

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=hours,
        y=days,
        colorscale='Viridis',
        colorbar=dict(title='Order Count')
    ))

    fig.update_layout(
        title='Order Heatmap: Day of Week vs Hour of Day',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week'
    )

    st.plotly_chart(fig, use_container_width=True)
