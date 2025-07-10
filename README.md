# 🛒 Instacart Market Basket Analysis

A comprehensive data mining project that analyzes customer purchasing behavior using the Instacart dataset. This interactive web application provides insights into market basket patterns, customer segmentation, product recommendations, and anomaly detection.

## 🚀 Features

- **🏠 Interactive Dashboard**: Clean and intuitive Streamlit interface
- **📊 Data Overview**: Comprehensive dataset exploration and statistics
- **🛍️ Market Basket Analysis**: Association rules mining and frequent itemset analysis
- **👥 User Segmentation**: Customer clustering based on purchasing behavior
- **💡 Recommendation System**: Product recommendation engine
- **🔍 Anomaly Detection**: Identification of unusual purchasing patterns
- **📈 Advanced Analytics**: Statistical analysis and visualizations

## 📋 Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Features Overview](#features-overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/instacart-market-basket-analysis.git
   cd instacart-market-basket-analysis
   ```

2. **Install required packages**
   ```bash
   pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn wordcloud pillow
   ```

3. **Run the application**
   ```bash
   streamlit run code/Home.py
   ```

4. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`

## 🖥️ Usage

### Running the Application

```bash
cd code
streamlit run Home.py
```

### Navigation

The application consists of multiple pages accessible through the sidebar:

- **🏠 Home**: Main dashboard and project overview
- **📊 Data Overview**: Dataset exploration and basic statistics
- **📈 Analysis**: Detailed statistical analysis
- **🛍️ Market Basket Analysis**: Association rules and frequent patterns
- **👥 User Segmentation**: Customer clustering analysis
- **💡 Recommendation**: Product recommendation system
- **🔍 Anomaly Detection**: Unusual pattern identification
- **ℹ️ About**: Project information and documentation

## 📁 Project Structure

```
Data_Mining_Project/
├── code/
│   ├── Home.py                          # Main Streamlit application
│   ├── pages/                           # Multi-page application modules
│   │   ├── About.py                     # Project information
│   │   ├── Analysis.py                  # Statistical analysis
│   │   ├── anamoly_detection.py         # Anomaly detection algorithms
│   │   ├── Data_overview.py             # Dataset exploration
│   │   ├── Market_Basket_Analysis.py    # Association rules mining
│   │   ├── Recommendation.py            # Recommendation engine
│   │   └── User_Segmentation.py         # Customer clustering
│   ├── data/                            # Dataset files
│   │   ├── aisles.csv                   # Product aisle information
│   │   ├── departments.csv              # Department categorization
│   │   ├── products.csv                 # Product details
│   │   ├── filtered_orders.csv          # Processed order data
│   │   ├── filtered_order_products_*.csv # Order-product relationships
│   │   ├── association_rules.csv        # Pre-computed association rules
│   │   ├── product_metrics.csv          # Product performance metrics
│   │   └── user_metrics.csv             # Customer behavior metrics
│   └── models/
│       └── reorder_prediction_model.pkl # Trained machine learning model
└── README.md
```

## 🎯 Features Overview

### 1. Data Overview
- Dataset statistics and information
- Missing value analysis
- Data distribution visualizations
- Sample data exploration

### 2. Market Basket Analysis
- **Association Rules Mining**: Discover relationships between products
- **Frequent Itemsets**: Identify commonly purchased product combinations
- **Support, Confidence, Lift Metrics**: Measure rule strength and reliability
- **Interactive Visualizations**: Network graphs and heatmaps

### 3. User Segmentation
- **Customer Clustering**: Group customers based on purchasing behavior
- **RFM Analysis**: Recency, Frequency, Monetary value segmentation
- **Behavioral Patterns**: Identify distinct customer groups
- **Segment Profiling**: Detailed analysis of each customer segment

### 4. Recommendation System
- **Collaborative Filtering**: User-based recommendations
- **Content-Based Filtering**: Product similarity recommendations
- **Hybrid Approach**: Combined recommendation strategies
- **Personalized Suggestions**: Tailored product recommendations

### 5. Anomaly Detection
- **Statistical Methods**: Identify outliers in purchasing patterns
- **Machine Learning Approaches**: Unsupervised anomaly detection
- **Behavioral Analysis**: Detect unusual customer behavior
- **Fraud Detection**: Identify potentially fraudulent transactions

## 📊 Dataset

This project uses the **Instacart Market Basket Analysis** dataset, which contains:

- **Orders**: 3.4M+ orders from 200K+ users
- **Products**: 50K+ products across 21 departments
- **Order Products**: Detailed order-product relationships
- **Aisles & Departments**: Product categorization hierarchy

### Key Metrics Analyzed:
- Purchase frequency and timing
- Product reorder rates
- Customer lifetime value
- Seasonal purchasing patterns
- Cross-selling opportunities

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Scikit-learn
- **Association Rules**: MLxtend (if used)
- **Others**: WordCloud, PIL, Pickle

## 📈 Key Insights

- Identification of top product associations
- Customer segmentation into distinct behavioral groups
- Optimized product recommendation strategies
- Detection of unusual purchasing patterns
- Market basket size and composition analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## 🙏 Acknowledgments

- Instacart for providing the dataset
- Streamlit team for the amazing framework
- Open source community for various libraries used

---

⭐ If you found this project helpful, please give it a star!

## 🚀 Quick Start

```bash
# Clone and run in 3 simple steps
git clone https://github.com/yourusername/instacart-market-basket-analysis.git
cd instacart-market-basket-analysis/code
streamlit run Home.py
```