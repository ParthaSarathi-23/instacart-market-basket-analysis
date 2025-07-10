import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.markdown("<h1 class='main-header'>Customer Segmentation</h1>", unsafe_allow_html=True)

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
try:
    user_metrics = pd.read_csv('user_metrics.csv')
except FileNotFoundError:
    st.warning("User metrics file not found. Some features may not work properly.")
    user_metrics = pd.DataFrame()

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
        fig.add_trace(go.Scatterpolar(r=normalized_data.iloc[i].values, theta=categories, fill='toself', name=f"Segment {row['Cluster']}: {row['Segment Description']}"))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title="Segment Comparison - Key Metrics", showlegend=True)
    return fig

# Load user data
users_data = load_user_data()

# Main content structure with similar styling to Market Basket Analysis
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<h2 class='sub-header'>Optimal Cluster Analysis</h2>", unsafe_allow_html=True)

    with st.spinner("Processing data..."):
        users_data = enhance_user_data(users_data)
        scaled_features, feature_names = prepare_features(users_data)
        wcss, silhouette_scores = find_optimal_clusters(scaled_features)
    
    tab1, tab2 = st.tabs(["Elbow Method", "Silhouette Score"])
    
    with tab1:
        fig_elbow = px.line(x=list(range(2, len(wcss) + 2)), y=wcss, labels={'x': 'Number of Clusters', 'y': 'WCSS'})
        fig_elbow.update_layout(title="Elbow Method")
        st.plotly_chart(fig_elbow, use_container_width=True)
    
    with tab2:
        fig_silhouette = px.line(x=list(range(2, len(silhouette_scores) + 2)), y=silhouette_scores, labels={'x': 'Number of Clusters', 'y': 'Silhouette Score'})
        fig_silhouette.update_layout(title="Silhouette Score")
        st.plotly_chart(fig_silhouette, use_container_width=True)
    
    recommended_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    st.write(f"Recommended clusters based on silhouette score: {recommended_clusters}")

with col2:
    st.markdown("<h2 class='sub-header'>Segment Settings</h2>", unsafe_allow_html=True)
    
    n_clusters = st.slider("Select Number of Clusters", 2, 8, recommended_clusters if 'recommended_clusters' in locals() else 4)
    
    st.markdown("""
    <div class='highlight' style="
        background: linear-gradient(to right, #FEF9E7, #FBFCFC);
        padding: 15px;
        border-radius: 10px;
        border-left: 6px solid #F5B041;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
        margin: 10px 0;
        font-size: 15px;
        line-height: 1.6;
        font-family: sans-serif;
        color: #4D5656;
    ">
        <h4 style="margin-top: 0; color: #B9770E;">Insight</h4>
        <p style="margin: 5px 0;">
            Adjust the number of clusters to find the optimal customer segmentation for your business needs.
        </p>
    </div>
""", unsafe_allow_html=True)

    st.markdown("<h3 class='section-header'>What is Customer Segmentation?</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='highlight' style="
        background: linear-gradient(to right, #FEF9E7, #FBFCFC);
        padding: 15px;
        border-radius: 10px;
        border-left: 6px solid #F5B041;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
        margin: 10px 0;
        font-size: 15px;
        line-height: 1.6;
        font-family: sans-serif;
        color: #4D5656;
    ">
        <h4 style="margin-top: 0; color: #B9770E;">Customer Segmentation Benefits</h4>
        <p style="margin: 5px 0;">
            Customer segmentation groups your customers based on common characteristics to help you:
        </p>
        <ul style="padding-left: 20px; margin: 10px 0;">
            <li>Target marketing efforts more effectively</li>
            <li>Personalize customer experiences</li>
            <li>Allocate resources more efficiently</li>
            <li>Identify high-value customer groups</li>
        </ul>
    </div>
""", unsafe_allow_html=True)


# Apply clustering based on user selection
clusters, kmeans = perform_clustering(scaled_features, n_clusters)
segment_analysis, user_data_with_clusters = analyze_segments(users_data, clusters)

# Second row with full width
st.markdown("<h2 class='sub-header'>Customer Segments Visualization</h2>", unsafe_allow_html=True)

col3, col4 = st.columns([3, 2])

with col3:
    pca_fig = visualize_clusters_pca(scaled_features, clusters)
    st.plotly_chart(pca_fig, use_container_width=True)

with col4:
    segment_dist = user_data_with_clusters['Cluster'].value_counts().reset_index()
    segment_dist.columns = ['Cluster', 'Count']
    segment_dist = segment_dist.merge(segment_analysis[['Cluster', 'Segment Description']], on='Cluster')
    
    fig_dist = px.pie(
        segment_dist, 
        values='Count', 
        names='Segment Description', 
        title="Customer Segment Distribution",
        color_discrete_sequence=px.colors.qualitative.G10
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# Third row
st.markdown("<h2 class='sub-header'>Segment Comparison</h2>", unsafe_allow_html=True)

col5, col6 = st.columns([1, 1])

with col5:
    radar_fig = create_radar_chart(segment_analysis)
    st.plotly_chart(radar_fig, use_container_width=True)

with col6:
    st.markdown("<h3 class='section-header'>Segment Analysis</h3>", unsafe_allow_html=True)
    formatted_segment_analysis = segment_analysis.copy()
    for col in ['total_orders', 'total_products', 'total_reorders', 'avg_products_per_order', 'reorder_ratio']:
        formatted_segment_analysis[col] = formatted_segment_analysis[col].round(2)
    st.dataframe(formatted_segment_analysis, use_container_width=True)

# Fourth row - Feature importance
st.markdown("<h2 class='sub-header'>Feature Importance</h2>", unsafe_allow_html=True)
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=feature_names)
normalized_centers = cluster_centers.copy()
for col in normalized_centers.columns:
    normalized_centers[col] = (normalized_centers[col] - normalized_centers[col].min()) / (normalized_centers[col].max() - normalized_centers[col].min())

fig_features = px.imshow(
    normalized_centers.T, 
    text_auto=True, 
    labels=dict(x="Cluster", y="Feature", color="Normalized Value"), 
    x=[f"Cluster {i}" for i in range(n_clusters)], 
    y=feature_names, 
    color_continuous_scale="Viridis"
)
fig_features.update_layout(title="Feature Importance by Cluster")
st.plotly_chart(fig_features, use_container_width=True)

# Fifth row - Individual analysis
st.markdown("<h2 class='sub-header'>Individual Customer Analysis</h2>", unsafe_allow_html=True)

col7, col8 = st.columns([1, 2])

with col7:
    user_id_input = st.selectbox("Select User ID", user_data_with_clusters['user_id'].unique())
    
    selected_user = user_data_with_clusters[user_data_with_clusters['user_id'] == user_id_input].iloc[0]
    selected_cluster = selected_user['Cluster']
    
    st.markdown(f"""
    <div class='highlight' style="
        background: linear-gradient(to right, #FEF9E7, #FBFCFC);
        padding: 15px;
        border-radius: 10px;
        border-left: 6px solid #F5B041;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
        margin: 10px 0;
        font-size: 15px;
        line-height: 1.6;
        font-family: sans-serif;
        color: #4D5656;
    ">
        <h4 style="margin-top: 0; color: #B9770E;">User Segmentation Info</h4>
        <p style="margin: 5px 0;">
            <strong>User ID:</strong> {user_id_input}<br>
            <strong>Cluster:</strong> {int(selected_cluster)}<br>
            <strong>Segment:</strong> {segment_analysis[segment_analysis['Cluster'] == selected_cluster]['Segment Description'].iloc[0]}
        </p>
    </div>
""", unsafe_allow_html=True)

    
    st.markdown("<h3 class='section-header'>Recommendations</h3>", unsafe_allow_html=True)
    recommendations = {
        0: [
            "Offer premium loyalty program",
            "Exclusive early access to new products",
            "Personalized recommendations based on purchase history"
        ],
        1: [
            "Incentivize more frequent purchases",
            "Targeted promotions for complementary products",
            "Re-engagement campaigns for favorite products"
        ],
        2: [
            "Volume discounts for bulk purchases",
            "Family-sized packaging options",
            "Subscription service for regularly purchased items"
        ],
        3: [
            "First-time buyer offers",
            "Educational content about product benefits",
            "Simplified shopping experience to reduce friction"
        ]
    }
    
    cluster_int = int(selected_cluster)
    if cluster_int in recommendations:
        for rec in recommendations[cluster_int]:
            st.write(f"• {rec}")
    else:
        st.write("No specific recommendations available for this segment.")

with col8:
    metrics_to_compare = ['total_orders', 'total_products', 'avg_products_per_order', 'reorder_ratio']
    segment_avg = segment_analysis[segment_analysis['Cluster'] == selected_cluster][metrics_to_compare].iloc[0]
    
    comparison_data = pd.DataFrame({
        'Metric': metrics_to_compare,
        'User Value': [selected_user[m] for m in metrics_to_compare],
        'Segment Average': [segment_avg[m] for m in metrics_to_compare]
    })
    
    fig_compare = px.bar(comparison_data, x='Metric', y=['User Value', 'Segment Average'], barmode='group', title="User vs. Segment Average", color_discrete_sequence=['#FF6F61', '#3498DB'])
    st.plotly_chart(fig_compare, use_container_width=True)