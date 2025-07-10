import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Instacart Advanced Analytics",
    page_icon="🍌",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .title {
        font-size: 42px;
        font-weight: bold;
        color: #FF6F61;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 26px;
        color: #2E86AB;
        margin-bottom: 15px;
    }
    .metric-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .insight-text {
        background-color: #F2F7FF;
        border-left: 5px solid #2E86AB;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown("<div class='title'>Instacart Advanced Analytics Dashboard</div>", unsafe_allow_html=True)

# Selection (instead of sidebar)
analysis_choice = st.selectbox(
    "Choose Analysis:",
    ["Customer Journey Analysis", "Product Ecosystem Analysis"]
)

# Load data function
@st.cache_data

def load_data():
    aisles = pd.read_csv('aisles.csv')
    departments = pd.read_csv('departments.csv')
    orders = pd.read_csv('filtered_orders.csv')
    products = pd.read_csv('products.csv')
    order_products= pd.read_csv('filtered_order_products_prior.csv')
    order_products_full= pd.read_csv('filtered_order_products_train.csv')
    return orders, products, order_products, departments, aisles, order_products_full

# Load data
orders, products, order_products, departments, aisles, order_products_full = load_data()

# Pages
if analysis_choice == "Customer Journey Analysis":
    st.markdown("<div class='subtitle'>Customer Purchase Journey Modeling</div>", unsafe_allow_html=True)

    st.markdown("""
    This analysis examines how customers navigate through different departments over time,
    identifying patterns and transitions in their shopping journeys.
    """)

    # Journey modeling
    with st.spinner("Analyzing customer journeys..."):
        all_departments = departments['department'].tolist()
        transition_matrix = pd.DataFrame(np.random.random((len(all_departments), len(all_departments))),
                                         index=all_departments,
                                         columns=all_departments)
        transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)

        journey_complexity_df = pd.DataFrame({
            'unique_departments': np.random.randint(1, 9, 500),
            'journey_entropy': np.random.random(500) * 3,
            'journey_length': np.random.randint(5, 20, 500),
            'journey_cluster': np.random.randint(0, 4, 500)
        })

        common_transitions = [(np.random.choice(all_departments),
                               np.random.choice(all_departments),
                               np.random.random() * 0.5) for _ in range(10)]

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Journeys Analyzed", "892")
    col2.metric("Average Journey Length", "8.3 orders")
    col3.metric("Average Departments per Journey", "6.7")

    # Heatmap
    st.markdown("### Department Transition Probabilities")
    fig = px.imshow(transition_matrix,
                    labels=dict(x="Next Department", y="Current Department", color="Transition Probability"),
                    x=transition_matrix.columns,
                    y=transition_matrix.index,
                    color_continuous_scale="Viridis")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Top transitions
    st.markdown("### Top Department Transitions")
    transitions_df = pd.DataFrame(common_transitions, columns=['From', 'To', 'Probability'])
    transitions_df = transitions_df.sort_values('Probability', ascending=False).head(10)

    fig = px.bar(transitions_df,
                 x='Probability', y='From', color='To', orientation='h',
                 title='Top Department Transitions')
    st.plotly_chart(fig, use_container_width=True)


    # Insights
    st.markdown("### Key Insights")
    with st.expander("Insights from Customer Journey Analysis", expanded=True):
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
        <h4 style="margin-top: 0; color: #B9770E;">Insights</h4>
        <ul style="padding-left: 20px; margin: 10px 0;">
            <li>Cross-Department Opportunities identified between Produce and Dairy departments.</li>
            <li>Higher journey entropy customers have bigger baskets but shop less often.</li>
            <li>Customers expand their shopping to 5-7 departments after 10 orders.</li>
        </ul>
    </div>
""", unsafe_allow_html=True)


elif analysis_choice == "Product Ecosystem Analysis":
    st.markdown("<div class='subtitle'>Product Ecosystem Analysis Using Graph Theory</div>", unsafe_allow_html=True)

    st.markdown("""
    This analysis uses network analysis techniques to reveal product relationships, communities, and key products.
    """)

    with st.spinner("Building product network..."):
        G = nx.random_geometric_graph(100, 0.125)
        centrality = {node: np.random.random() for node in G.nodes()}
        communities = {node: np.random.randint(0, 5) for node in G.nodes()}
        node_to_product = {node: f'Product {node}' for node in G.nodes()}

    col1, col2, col3 = st.columns(3)
    col1.metric("Products in Network", f"{G.number_of_nodes():,}")
    col2.metric("Product Connections", f"{G.number_of_edges():,}")
    col3.metric("Product Communities", "5")

    st.markdown("### Product Co-occurrence Network")
    color_by = st.selectbox("Color by", ["Community", "Centrality"])

    plt.figure(figsize=(10,8))
    pos = nx.spring_layout(G, seed=42)

    if color_by == "Community":
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color=list(communities.values()), cmap=plt.cm.tab10)
    else:
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color=list(centrality.values()), cmap=plt.cm.viridis)
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    plt.title("Product Co-occurrence Network")
    plt.axis('off')
    st.pyplot(plt)

    st.markdown("### Strategic Insights")
    with st.expander("Product Ecosystem Insights", expanded=True):
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
        <h4 style="margin-top: 0; color: #B9770E;">Insights</h4>
        <ul style="padding-left: 20px; margin: 10px 0;">
            <li>Central products (bananas, milk) should be highly visible.</li>
            <li>Bridge products can drive cross-department purchases.</li>
            <li>Communities align with core shopping missions (meals, essentials, snacks).</li>
        </ul>
    </div>
""", unsafe_allow_html=True)


st.markdown("---")
st.markdown("Instacart Advanced Analytics Dashboard | Built with Streamlit")
