import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

st.set_page_config(page_title="Customer Segmentation & Anomaly Detection", layout="wide")
st.markdown("<h1 class='main-header'>Customer Segmentation & Anomaly Detection</h1>", unsafe_allow_html=True)

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
    .anomaly-high {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .anomaly-low {
        background-color: #E3F2FD;
        color: #0D47A1;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .anomaly-normal {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 2px 8px;
        border-radius: 4px;
    }
    .tab-content {
        padding: 16px;
        border: 1px solid #ddd;
        border-radius: 0 0 5px 5px;
    }
</style>
""", unsafe_allow_html=True)

# Define paths to data files
try:
    user_metrics = pd.read_csv('user_metrics.csv')
    product_data = pd.read_csv('product_metrics.csv')
except FileNotFoundError:
    st.warning("Data files not found. Using simulated data for demonstration.")
    user_metrics = pd.DataFrame()
    product_data = pd.DataFrame()

# Load data (for demonstration purposes, we'll simulate the data if files don't exist)
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

@st.cache_data
def load_product_data():
    if product_data.empty:
        # Simulate product data
        product_ids = range(1, 101)
        
        sim_product_data = pd.DataFrame({
            'product_id': product_ids,
            'times_purchased': np.random.randint(1, 200, size=100),
            'times_reordered': np.random.randint(0, 150, size=100),
            'product_name': [f"Product {i}" for i in product_ids],
            'department': np.random.choice(['grocery', 'produce', 'dairy eggs', 'snacks', 'beverages'], size=100),
            'aisle': np.random.choice(['water seltzer sparkling water', 'chips pretzels', 'yogurt', 'cookies cakes'], size=100)
        })
        
        sim_product_data['reorder_rate'] = sim_product_data['times_reordered'] / sim_product_data['times_purchased']
        return sim_product_data
    else:
        return product_data

# Enhance user data
def enhance_user_data(user_data):
    user_data['products_per_order_normalized'] = user_data['avg_products_per_order'] / user_data['avg_products_per_order'].max()
    user_data['order_frequency'] = user_data['total_orders'] / user_data['total_orders'].max()
    user_data['purchase_volume'] = user_data['total_products'] / user_data['total_products'].max()
    user_data['recency_proxy'] = 1 / (1 + user_data['total_orders'])
    return user_data

# Feature engineering
def prepare_features(user_data):
    features = user_data[['total_orders', 'avg_products_per_order', 'reorder_ratio', 'order_frequency', 'purchase_volume']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, features.columns

# Find optimal clusters
def find_optimal_clusters(scaled_features, max_clusters=10):
    wcss = []
    silhouette_scores = []
    for i in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)
        score = silhouette_score(scaled_features, kmeans.labels_)
        silhouette_scores.append(score)
    return wcss, silhouette_scores

# Perform clustering
def perform_clustering(scaled_features, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)
    return clusters, kmeans

# Visualize clusters
def visualize_clusters_pca(scaled_features, clusters):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = clusters
    fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', title='Customer Segments - PCA Visualization', color_continuous_scale=px.colors.qualitative.G10)
    return fig

# Analyze segments
def analyze_segments(user_data, clusters):
    user_data['Cluster'] = clusters
    segment_analysis = user_data.groupby('Cluster').agg({
        'total_orders': 'mean',
        'total_products': 'mean',
        'total_reorders': 'mean',
        'avg_products_per_order': 'mean',
        'reorder_ratio': 'mean',
        'user_id': 'count'
    }).reset_index()
    segment_analysis = segment_analysis.rename(columns={'user_id': 'count'})

    def create_segment_description(row):
        descriptions = {
            0: "High-Value Customers",
            1: "Occasional Buyers",
            2: "Bulk Purchasers",
            3: "New/Infrequent Customers"
        }
        return descriptions.get(row['Cluster'], f"Segment {row['Cluster']}")

    segment_analysis['Segment Description'] = segment_analysis.apply(create_segment_description, axis=1)
    return segment_analysis, user_data

# Create a radar chart
def create_radar_chart(segment_analysis):
    categories = ['total_orders', 'total_products', 'total_reorders', 'avg_products_per_order', 'reorder_ratio']
    normalized_data = segment_analysis[categories].copy()
    for cat in categories:
        normalized_data[cat] = normalized_data[cat] / normalized_data[cat].max()

    fig = go.Figure()
    for i, row in segment_analysis.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=normalized_data.iloc[i].values, 
            theta=categories, 
            fill='toself', 
            name=f"Segment {row['Cluster']}: {row['Segment Description']}"
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])), 
        title="Segment Comparison - Key Metrics", 
        showlegend=True
    )
    return fig

# ---- ANOMALY DETECTION FUNCTIONS ----

@st.cache_data
def detect_user_anomalies(user_data, contamination=0.05):
    """
    Detect anomalies in user behavior using different methods:
    1. Isolation Forest
    2. Local Outlier Factor
    3. One-Class SVM
    """
    # Select features for anomaly detection
    features = user_data[['total_orders', 'total_products', 'avg_products_per_order', 'reorder_ratio']]
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    user_data['anomaly_isolation_forest'] = iso_forest.fit_predict(scaled_features)
    
    # Convert to 0 (normal) and 1 (anomaly)
    user_data['anomaly_isolation_forest'] = user_data['anomaly_isolation_forest'].apply(lambda x: 1 if x == -1 else 0)
    
    # Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    user_data['anomaly_lof'] = lof.fit_predict(scaled_features)
    user_data['anomaly_lof'] = user_data['anomaly_lof'].apply(lambda x: 1 if x == -1 else 0)
    
    # One-Class SVM
    ocsvm = OneClassSVM(kernel="rbf", gamma=0.1, nu=contamination)
    user_data['anomaly_ocsvm'] = ocsvm.fit_predict(scaled_features)
    user_data['anomaly_ocsvm'] = user_data['anomaly_ocsvm'].apply(lambda x: 1 if x == -1 else 0)
    
    # Combined score (ensemble approach)
    user_data['anomaly_score'] = user_data['anomaly_isolation_forest'] + user_data['anomaly_lof'] + user_data['anomaly_ocsvm']
    
    # Add anomaly flag (if at least 2 methods detect an anomaly)
    user_data['is_anomaly'] = user_data['anomaly_score'] >= 2
    
    return user_data

def detect_product_anomalies(product_data):
    """
    Detect anomalies in product performance using statistical methods
    """
    # Calculate z-scores for key metrics
    product_data['purchase_zscore'] = (product_data['times_purchased'] - product_data['times_purchased'].mean()) / product_data['times_purchased'].std()
    product_data['reorder_zscore'] = (product_data['times_reordered'] - product_data['times_reordered'].mean()) / product_data['times_reordered'].std()
    product_data['reorder_rate_zscore'] = (product_data['reorder_rate'] - product_data['reorder_rate'].mean()) / product_data['reorder_rate'].std()
    
    # Flag anomalies (absolute z-score > 2)
    product_data['purchase_anomaly'] = np.abs(product_data['purchase_zscore']) > 2
    product_data['reorder_anomaly'] = np.abs(product_data['reorder_zscore']) > 2
    product_data['reorder_rate_anomaly'] = np.abs(product_data['reorder_rate_zscore']) > 2
    
    # Create anomaly direction indicators
    product_data['purchase_direction'] = np.where(product_data['purchase_zscore'] > 2, 'high', 
                                         np.where(product_data['purchase_zscore'] < -2, 'low', 'normal'))
    product_data['reorder_direction'] = np.where(product_data['reorder_zscore'] > 2, 'high', 
                                        np.where(product_data['reorder_zscore'] < -2, 'low', 'normal'))
    product_data['reorder_rate_direction'] = np.where(product_data['reorder_rate_zscore'] > 2, 'high', 
                                             np.where(product_data['reorder_rate_zscore'] < -2, 'low', 'normal'))
    
    # Combined anomaly flag
    product_data['is_anomaly'] = (product_data['purchase_anomaly'] | product_data['reorder_anomaly'] | product_data['reorder_rate_anomaly'])
    
    return product_data

def plot_anomaly_distribution(user_data):
    """Plot the distribution of anomaly scores in user data"""
    fig = px.histogram(
        user_data, 
        x='anomaly_score',
        color='is_anomaly',
        title='Distribution of Anomaly Scores',
        labels={'anomaly_score': 'Anomaly Score (# of methods that flagged as anomaly)', 'count': 'Number of Users'},
        color_discrete_map={True: '#FF6F61', False: '#3498DB'},
        nbins=4
    )
    
    fig.update_layout(
        xaxis=dict(tickmode='linear', tick0=0, dtick=1),
        bargap=0.2
    )
    
    return fig

def plot_anomaly_scatter(user_data):
    """Create scatter plot highlighting anomalies"""
    fig = px.scatter(
        user_data,
        x='total_orders',
        y='avg_products_per_order',
        color='is_anomaly',
        size='total_products',
        hover_data=['user_id', 'reorder_ratio', 'anomaly_score'],
        title='User Behavior Anomalies',
        color_discrete_map={True: '#FF6F61', False: '#3498DB'}
    )
    
    fig.update_layout(
        xaxis_title='Total Orders',
        yaxis_title='Avg Products per Order'
    )
    
    return fig

def plot_product_anomalies(product_data):
    """Plot product anomalies by department"""
    anomaly_by_dept = product_data.groupby('department')['is_anomaly'].sum().reset_index()
    anomaly_by_dept['total_products'] = product_data.groupby('department').size().values
    anomaly_by_dept['anomaly_rate'] = anomaly_by_dept['is_anomaly'] / anomaly_by_dept['total_products']
    
    fig = px.bar(
        anomaly_by_dept,
        x='department',
        y='anomaly_rate',
        color='anomaly_rate',
        title='Anomaly Rate by Department',
        labels={'department': 'Department', 'anomaly_rate': 'Anomaly Rate'},
        color_continuous_scale='Viridis'
    )
    
    return fig

def plot_anomaly_metrics(product_data):
    """Visualize key metrics for anomalous vs normal products"""
    metrics = ['times_purchased', 'times_reordered', 'reorder_rate']
    
    fig = go.Figure()
    
    for metric in metrics:
        fig.add_trace(go.Box(
            y=product_data[product_data['is_anomaly'] == True][metric],
            name=f"{metric} (Anomaly)",
            boxmean=True,
            marker_color='#FF6F61'
        ))
        
        fig.add_trace(go.Box(
            y=product_data[product_data['is_anomaly'] == False][metric],
            name=f"{metric} (Normal)",
            boxmean=True,
            marker_color='#3498DB'
        ))
    
    fig.update_layout(
        title='Distribution of Key Metrics: Anomalous vs Normal Products',
        yaxis_title='Value',
        boxmode='group'
    )
    
    return fig

# Load user data
users_data = load_user_data()
products_data = load_product_data()


st.markdown("<h2 class='sub-header'>Anomaly Detection Dashboard</h2>", unsafe_allow_html=True)

st.markdown("""
<div class='info-box' style="
    background: linear-gradient(to right, #EBF5FB, #F4F6F7);
    padding: 15px;
    border-radius: 10px;
    border-left: 6px solid #3498DB;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
    margin: 10px 0;
    font-size: 15px;
    line-height: 1.6;
    font-family: sans-serif;
    color: #21618C;
">
    <h4 style="margin-top: 0; color: #2980B9;">About Anomaly Detection</h4>
    <p style="margin: 5px 0;">
        Anomaly detection identifies unusual patterns that don't conform to expected behavior. 
        In customer analytics, anomalies can indicate:
    </p>
    <ul style="padding-left: 20px; margin: 10px 0;">
        <li>Potential fraud or unusual account activity</li>
        <li>Highly valuable customers with unique behaviors</li>
        <li>Data quality issues or input errors</li>
        <li>Emerging trends or changing customer preferences</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Tabs for different anomaly detection views
anomaly_type = st.radio(
    "Select Analysis Type",
    ["User Behavior Anomalies", "Product Performance Anomalies"],
    horizontal=True
)

# Process data for anomaly detection
with st.spinner("Detecting anomalies..."):
    # Detect user anomalies
    contamination_rate = st.slider(
        "Anomaly Threshold (% of data expected to be anomalous)", 
        0.01, 0.20, 0.05, 
        format="%.2f"
    )
    
    if anomaly_type == "User Behavior Anomalies":
        # User anomaly detection
        # User anomaly detection
        user_data_with_anomalies = detect_user_anomalies(users_data.copy(), contamination=contamination_rate)
        
        col_a1, col_a2 = st.columns([1, 1])
        
        with col_a1:
            st.markdown("<h3 class='section-header'>User Anomaly Distribution</h3>", unsafe_allow_html=True)
            anomaly_dist_fig = plot_anomaly_distribution(user_data_with_anomalies)
            st.plotly_chart(anomaly_dist_fig, use_container_width=True)
            
            # Display anomaly statistics
            total_users = len(user_data_with_anomalies)
            anomaly_count = user_data_with_anomalies['is_anomaly'].sum()
            anomaly_pct = (anomaly_count / total_users) * 100
            
            st.markdown(f"""
            <div class='info-box' style="
                background: linear-gradient(to right, #EBF5FB, #F4F6F7);
                padding: 15px;
                border-radius: 10px;
                border-left: 6px solid #3498DB;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
                margin: 10px 0;
                font-size: 15px;
                line-height: 1.6;
                font-family: sans-serif;
                color: #21618C;
            ">
                <h4 style="margin-top: 0; color: #2980B9;">Anomaly Summary</h4>
                <p style="margin: 5px 0;">
                    <strong>Total Users:</strong> {total_users}<br>
                    <strong>Anomalous Users:</strong> {anomaly_count} ({anomaly_pct:.2f}%)<br>
                    <strong>Detection Methods:</strong> Isolation Forest, Local Outlier Factor, One-Class SVM
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_a2:
            st.markdown("<h3 class='section-header'>User Behavior Anomalies</h3>", unsafe_allow_html=True)
            anomaly_scatter_fig = plot_anomaly_scatter(user_data_with_anomalies)
            st.plotly_chart(anomaly_scatter_fig, use_container_width=True)
            
        # Display anomalous users table
        st.markdown("<h3 class='section-header'>Anomalous Users</h3>", unsafe_allow_html=True)
        anomaly_users = user_data_with_anomalies[user_data_with_anomalies['is_anomaly'] == True].sort_values(by='anomaly_score', ascending=False)
        
        if len(anomaly_users) > 0:
            display_cols = ['user_id', 'total_orders', 'total_products', 'avg_products_per_order', 'reorder_ratio', 'anomaly_score']
            st.dataframe(anomaly_users[display_cols].reset_index(drop=True), use_container_width=True)
            
            # Individual anomaly investigation
            st.markdown("<h3 class='section-header'>Investigate Specific User</h3>", unsafe_allow_html=True)
            selected_anomaly_user = st.selectbox(
                "Select a user to investigate",
                options=anomaly_users['user_id'].tolist(),
                format_func=lambda x: f"User {x}"
            )
            
            user_detail = user_data_with_anomalies[user_data_with_anomalies['user_id'] == selected_anomaly_user].iloc[0]
            
            col_d1, col_d2, col_d3 = st.columns(3)
            
            with col_d1:
                st.metric("Total Orders", user_detail['total_orders'], 
                            delta=f"{user_detail['total_orders'] - users_data['total_orders'].mean():.1f} vs avg")
            with col_d2:
                st.metric("Products per Order", f"{user_detail['avg_products_per_order']:.2f}", 
                            delta=f"{user_detail['avg_products_per_order'] - users_data['avg_products_per_order'].mean():.2f} vs avg")
            with col_d3:
                st.metric("Reorder Ratio", f"{user_detail['reorder_ratio']:.2f}", 
                            delta=f"{user_detail['reorder_ratio'] - users_data['reorder_ratio'].mean():.2f} vs avg")
            
            # Anomaly explanation
            st.markdown("<h4>Why is this user anomalous?</h4>", unsafe_allow_html=True)
            
            explanation = []
            if user_detail['total_orders'] > users_data['total_orders'].quantile(0.95):
                explanation.append("• **Extremely high number of orders** (top 5% of all users)")
            elif user_detail['total_orders'] < users_data['total_orders'].quantile(0.05):
                explanation.append("• **Unusually low number of orders** (bottom 5% of all users)")
            
            if user_detail['avg_products_per_order'] > users_data['avg_products_per_order'].quantile(0.95):
                explanation.append("• **Very large basket size** (top 5% of all users)")
            elif user_detail['avg_products_per_order'] < users_data['avg_products_per_order'].quantile(0.05):
                explanation.append("• **Unusually small basket size** (bottom 5% of all users)")
            
            if user_detail['reorder_ratio'] > users_data['reorder_ratio'].quantile(0.95):
                explanation.append("• **Exceptionally high reorder ratio** (top 5% of all users)")
            elif user_detail['reorder_ratio'] < users_data['reorder_ratio'].quantile(0.05):
                explanation.append("• **Unusually low reorder ratio** (bottom 5% of all users)")
            
            if len(explanation) > 0:
                for exp in explanation:
                    st.markdown(exp)
            else:
                st.markdown("• This user has an unusual combination of behaviors that collectively make them an outlier.")
            
            # Recommendations
            st.markdown("<h4>Recommendations</h4>", unsafe_allow_html=True)
            
            if user_detail['total_orders'] > users_data['total_orders'].quantile(0.95):
                st.markdown("• Consider adding this high-frequency user to a VIP program")
                st.markdown("• Study this user's preferences to understand what drives their loyalty")
            
            if user_detail['avg_products_per_order'] > users_data['avg_products_per_order'].quantile(0.95):
                st.markdown("• Offer bundle discounts to capitalize on large basket purchases")
                st.markdown("• Analyze items commonly purchased together by this user for cross-sell opportunities")
            
            if user_detail['reorder_ratio'] < users_data['reorder_ratio'].quantile(0.05):
                st.markdown("• Implement win-back campaign to encourage repeat purchases")
                st.markdown("• Survey this user to understand why they aren't reordering items")
        else:
            st.write("No anomalous users detected with current threshold.")
    
    else:  # Product Performance Anomalies
        # Product anomaly detection
        product_data_with_anomalies = detect_product_anomalies(products_data.copy())
        
        col_p1, col_p2 = st.columns([1, 1])
        
        with col_p1:
            st.markdown("<h3 class='section-header'>Product Anomalies by Department</h3>", unsafe_allow_html=True)
            dept_anomaly_fig = plot_product_anomalies(product_data_with_anomalies)
            st.plotly_chart(dept_anomaly_fig, use_container_width=True)
        
        with col_p2:
            st.markdown("<h3 class='section-header'>Metric Distribution: Anomalous vs Normal</h3>", unsafe_allow_html=True)
            metrics_fig = plot_anomaly_metrics(product_data_with_anomalies)
            st.plotly_chart(metrics_fig, use_container_width=True)
        
        # Display anomalous products table
        st.markdown("<h3 class='section-header'>Anomalous Products</h3>", unsafe_allow_html=True)
        
        # Format the anomaly direction with appropriate styling
        def format_direction(value, direction):
            if direction == 'high':
                return f"<span class='anomaly-high'>{value:.2f} (HIGH)</span>"
            elif direction == 'low':
                return f"<span class='anomaly-low'>{value:.2f} (LOW)</span>"
            else:
                return f"<span class='anomaly-normal'>{value:.2f}</span>"
        
        # Create a formatted dataframe for display
        anomaly_products = product_data_with_anomalies[product_data_with_anomalies['is_anomaly'] == True].sort_values(
            by=['purchase_anomaly', 'reorder_anomaly', 'reorder_rate_anomaly'], 
            ascending=False
        )
        
        if len(anomaly_products) > 0:
            display_cols = ['product_id', 'product_name', 'department', 'aisle', 'times_purchased', 'times_reordered', 'reorder_rate']
            formatted_anomalies = anomaly_products[display_cols].copy()
            
            # Create custom formatter for each metric showing direction
            st.write("Products with unusually high or low metrics:")
            
            # We'll create a custom display with HTML formatting
            for i, row in formatted_anomalies.iterrows():
                purchase_direction = anomaly_products.loc[i, 'purchase_direction']
                reorder_direction = anomaly_products.loc[i, 'reorder_direction']
                reorder_rate_direction = anomaly_products.loc[i, 'reorder_rate_direction']
                
                st.markdown(f"""
                <div style="
                    background: white;
                    padding: 10px;
                    border-radius: 5px;
                    border: 1px solid #ddd;
                    margin-bottom: 10px;
                ">
                    <h4 style="margin: 0 0 5px 0;">{row['product_name']}</h4>
                    <p style="margin: 0 0 5px 0; color: #555; font-size: 14px;">
                        <strong>ID:</strong> {row['product_id']} | 
                        <strong>Department:</strong> {row['department']} | 
                        <strong>Aisle:</strong> {row['aisle']}
                    </p>
                    <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px;">
                        <div style="flex: 1; min-width: 150px;">
                            <div style="font-size: 12px; color: #777;">PURCHASES</div>
                            <div>
                                {format_direction(row['times_purchased'], purchase_direction)}
                            </div>
                        </div>
                        <div style="flex: 1; min-width: 150px;">
                            <div style="font-size: 12px; color: #777;">REORDERS</div>
                            <div>
                                {format_direction(row['times_reordered'], reorder_direction)}
                            </div>
                        </div>
                        <div style="flex: 1; min-width: 150px;">
                            <div style="font-size: 12px; color: #777;">REORDER RATE</div>
                            <div>
                                {format_direction(row['reorder_rate'], reorder_rate_direction)}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Product anomaly insights
            st.markdown("<h3 class='section-header'>Product Anomaly Insights</h3>", unsafe_allow_html=True)
            
            # High purchase, low reorder rate
            high_purchase_low_reorder = product_data_with_anomalies[
                (product_data_with_anomalies['purchase_direction'] == 'high') & 
                (product_data_with_anomalies['reorder_rate_direction'] == 'low')
            ]
            
            # Low purchase, high reorder rate
            low_purchase_high_reorder = product_data_with_anomalies[
                (product_data_with_anomalies['purchase_direction'] == 'low') & 
                (product_data_with_anomalies['reorder_rate_direction'] == 'high')
            ]
            
            col_i1, col_i2 = st.columns(2)
            
            with col_i1:
                st.markdown("""
                <div class='highlight' style="
                    background: linear-gradient(to right, #FDEDEC, #F9EBEA);
                    padding: 15px;
                    border-radius: 10px;
                    border-left: 6px solid #E74C3C;
                    height: 100%;
                    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
                    margin: 10px 0;
                    font-size: 15px;
                    line-height: 1.6;
                    font-family: sans-serif;
                    color: #922B21;
                ">
                    <h4 style="margin-top: 0; color: #C0392B;">Popular but Not Reordered</h4>
                    <p style="margin: 5px 0;">
                        Products with high purchase volume but low reorder rates may indicate:
                    </p>
                    <ul style="padding-left: 20px; margin: 10px 0;">
                        <li>Poor product quality or customer dissatisfaction</li>
                        <li>One-time seasonal or promotional items</li>
                        <li>Products that don't require frequent repurchasing</li>
                    </ul>
                    <p style="margin: 5px 0;">
                        <strong>Count:</strong> {len(high_purchase_low_reorder)}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_i2:
                st.markdown("""
                <div class='highlight' style="
                    background: linear-gradient(to right, #E8F8F5, #E8F6F3);
                    padding: 15px;
                    border-radius: 10px;
                    border-left: 6px solid #1ABC9C;
                    height: 100%;
                    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
                    margin: 10px 0;
                    font-size: 15px;
                    line-height: 1.6;
                    font-family: sans-serif;
                    color: #0E6251;
                ">
                    <h4 style="margin-top: 0; color: #16A085;">Hidden Gems</h4>
                    <p style="margin: 5px 0;">
                        Products with low purchase volume but high reorder rates may indicate:
                    </p>
                    <ul style="padding-left: 20px; margin: 10px 0;">
                        <li>Niche products with exceptionally loyal customers</li>
                        <li>Potential for increased marketing to expand customer base</li>
                        <li>Products that could benefit from better visibility</li>
                    </ul>
                    <p style="margin: 5px 0;">
                        <strong>Count:</strong> {len(low_purchase_high_reorder)}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.write("No anomalous products detected with current threshold.")
            
# Add download functionality
st.markdown("<h3 class='section-header'>Export Anomaly Data</h3>", unsafe_allow_html=True)

if anomaly_type == "User Behavior Anomalies" and 'user_data_with_anomalies' in locals():
    export_user_data = user_data_with_anomalies.copy()
    
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    
    csv = convert_df_to_csv(export_user_data)
    st.download_button(
        "Download User Anomaly Data (CSV)",
        csv,
        "user_anomalies.csv",
        "text/csv",
        key='download-user-anomalies'
    )

elif anomaly_type == "Product Performance Anomalies" and 'product_data_with_anomalies' in locals():
    export_product_data = product_data_with_anomalies.copy()
    
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    
    csv = convert_df_to_csv(export_product_data)
    st.download_button(
        "Download Product Anomaly Data (CSV)",
        csv,
        "product_anomalies.csv",
        "text/csv",
        key='download-product-anomalies'
    )
