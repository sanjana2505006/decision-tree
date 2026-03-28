import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from decision_tree_core import DecisionTreeClassifier
from data_generation import generate_dataset, add_label_noise, get_dataset_info
from visualization import (
    create_scatter_plot, create_impurity_comparison_plot, create_tree_structure_plot,
    create_prediction_path_plot, create_overfitting_comparison_plot, create_noise_analysis_plot
)

# Set page configuration
st.set_page_config(
    page_title="Decision Tree Interactive Visualizer",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .explanation-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin-bottom: 1rem;
    }
    .key-takeaway {
        background-color: #e8f5e8;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 3px solid #27ae60;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown('<h1 class="main-header">🌳 Decision Tree Interactive Visualizer</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="explanation-box">
    <strong>What is a Decision Tree?</strong><br>
    A Decision Tree is a supervised learning algorithm that recursively splits data into subsets based on feature values to create pure, homogeneous regions. Each internal node represents a decision, and each leaf node represents a prediction.
</div>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.markdown("## 🎛️ Controls")

# Dataset selection
dataset_info = get_dataset_info()
dataset_type = st.sidebar.selectbox(
    "Select Dataset Type",
    options=list(dataset_info.keys()),
    format_func=lambda x: f"{dataset_info[x]['name']} - {dataset_info[x]['difficulty']}",
    help="Choose the type of dataset to explore different decision boundary complexities"
)

# Model parameters
st.sidebar.markdown("### Model Parameters")
max_depth = st.sidebar.slider(
    "Maximum Depth",
    min_value=1,
    max_value=10,
    value=5,
    help="Maximum number of levels in the tree. Higher values can lead to overfitting."
)

min_samples_split = st.sidebar.slider(
    "Minimum Samples per Split",
    min_value=2,
    max_value=20,
    value=2,
    help="Minimum number of samples required to split a node."
)

criterion = st.sidebar.selectbox(
    "Impurity Criterion",
    options=['gini', 'entropy'],
    help="Function to measure the quality of a split."
)

# Data parameters
st.sidebar.markdown("### Data Parameters")
noise_level = st.sidebar.slider(
    "Noise Level",
    min_value=0.0,
    max_value=0.5,
    value=0.1,
    step=0.05,
    help="Amount of random noise to add to the data."
)

label_noise_ratio = st.sidebar.slider(
    "Label Noise Ratio",
    min_value=0.0,
    max_value=0.3,
    value=0.0,
    step=0.05,
    help="Fraction of labels to randomly flip."
)

n_samples = st.sidebar.slider(
    "Number of Samples",
    min_value=100,
    max_value=1000,
    value=300,
    step=50,
    help="Total number of data points to generate."
)

# Generate data
@st.cache_data
def load_data(dataset_type, n_samples, noise_level, label_noise_ratio, random_state=42):
    X, y = generate_dataset(dataset_type, n_samples, noise_level, random_state)
    if label_noise_ratio > 0:
        y = add_label_noise(y, label_noise_ratio)
    return X, y

X, y = load_data(dataset_type, n_samples, noise_level, label_noise_ratio)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
@st.cache_data
def train_model(X_train, y_train, max_depth, min_samples_split, criterion):
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion
    )
    tree.fit(X_train, y_train, feature_names=['Feature 0', 'Feature 1'])
    return tree

tree = train_model(X_train, y_train, max_depth, min_samples_split, criterion)

# Get decision boundary
decision_boundary = tree.get_decision_regions(
    X_range=(X[:, 0].min()-0.1, X[:, 0].max()+0.1),
    y_range=(X[:, 1].min()-0.1, X[:, 1].max()+0.1)
)

# Main content with tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Data & Splits", "📏 Impurity Measures", "🎯 Split Selection", 
    "🌲 Tree Growth", "🛤️ Prediction Path", "📈 Overfitting & Depth", "🔧 Noise & Pruning"
])

with tab1:
    st.markdown('<h2 class="section-header">Data Partitioning & Feature Splits</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Main visualization
        fig = create_scatter_plot(
            X, y, 
            title=f"Dataset: {dataset_info[dataset_type]['name']}",
            decision_boundary=decision_boundary
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="explanation-box">
            <strong>How Decision Trees Split Data</strong><br>
            Decision trees recursively partition the feature space by creating axis-aligned boundaries. Each split tries to maximize the purity of the resulting subsets.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="key-takeaway">
            <strong>What to observe:</strong><br>
            • How the tree creates rectangular decision regions<br>
            • How deeper trees create more complex boundaries<br>
            • How different datasets require different split strategies
        </div>
        """, unsafe_allow_html=True)
        
        # Dataset statistics
        st.markdown("#### Dataset Statistics")
        st.write(f"**Total Samples:** {len(X)}")
        st.write(f"**Training Samples:** {len(X_train)}")
        st.write(f"**Test Samples:** {len(X_test)}")
        st.write(f"**Class Distribution:** {np.bincount(y)}")
        st.write(f"**Model Accuracy:** {accuracy_score(y_test, tree.predict(X_test)):.3f}")

with tab2:
    st.markdown('<h2 class="section-header">Impurity Measures (Gini / Entropy)</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Gini explanation
        st.markdown("""
        <div class="explanation-box">
            <strong>Gini Impurity</strong><br>
            Gini = 1 - Σ(pᵢ²)<br>
            Measures the probability of misclassifying a randomly chosen element.
        </div>
        """, unsafe_allow_html=True)
        
        # Entropy explanation
        st.markdown("""
        <div class="explanation-box">
            <strong>Entropy</strong><br>
            Entropy = -Σ(pᵢ × log₂(pᵢ))<br>
            Measures the amount of uncertainty or randomness in the data.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Comparison plot
        # Generate some example impurity values
        p_values = np.linspace(0.01, 0.99, 100)
        gini_values = [1 - (p**2 + (1-p)**2) for p in p_values]
        entropy_values = [-(p*np.log2(p) + (1-p)*np.log2(1-p)) for p in p_values]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p_values, y=gini_values, name='Gini', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=p_values, y=entropy_values, name='Entropy', line=dict(color='red')))
        
        fig.update_layout(
            title="Impurity Measures Comparison",
            xaxis_title="Class Probability (p)",
            yaxis_title="Impurity Value",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="key-takeaway">
        <strong>Key Insight:</strong><br>
        Both measures reach maximum impurity at p=0.5 (perfectly mixed) and minimum at p=0 or 1 (perfectly pure). Gini is computationally simpler, while Entropy has stronger theoretical foundations.
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown('<h2 class="section-header">Split Selection (Best Feature & Threshold)</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="explanation-box">
        <strong>How Splits Are Chosen</strong><br>
        The algorithm evaluates all possible splits and selects the one that maximizes information gain (impurity reduction). For each feature, it tries different threshold values and calculates the weighted average impurity of the resulting subsets.
    </div>
    """, unsafe_allow_html=True)
    
    # Simulate split candidates
    def get_split_candidates(X, y, tree):
        """Get example split candidates for visualization"""
        candidates = []
        
        # Generate some example splits
        for feature_idx in range(2):
            feature_values = np.unique(X[:, feature_idx])
            for i in range(0, len(feature_values), max(1, len(feature_values)//5)):
                threshold = feature_values[i]
                
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if len(left_mask) > 0 and len(right_mask) > 0:
                    # Calculate gain (simplified)
                    current_impurity = tree._calculate_impurity(y)
                    left_impurity = tree._calculate_impurity(y[left_mask])
                    right_impurity = tree._calculate_impurity(y[right_mask])
                    
                    n_left, n_right = len(left_mask), len(right_mask)
                    n_total = len(y)
                    
                    weighted_impurity = (n_left / n_total) * left_impurity + (n_right / n_total) * right_impurity
                    gain = current_impurity - weighted_impurity
                    
                    candidates.append({
                        'feature': f'Feature {feature_idx}',
                        'threshold': threshold,
                        'gain': gain
                    })
        
        return sorted(candidates, key=lambda x: x['gain'], reverse=True)[:5]
    
    split_candidates = get_split_candidates(X_train, y_train, tree)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Split candidates table
        st.markdown("#### Top Split Candidates")
        candidates_df = pd.DataFrame(split_candidates)
        candidates_df.columns = ['Feature', 'Threshold', 'Information Gain']
        st.dataframe(candidates_df, use_container_width=True)
    
    with col2:
        # Impurity comparison
        fig = create_impurity_comparison_plot(y_train, split_candidates, criterion)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="key-takeaway">
        <strong>What to observe:</strong><br>
        • Higher information gain indicates better splits<br>
        • The algorithm always chooses the split with maximum gain<br>
        • Different features may be optimal at different tree levels
    </div>
    """, unsafe_allow_html=True)

with tab4:
    st.markdown('<h2 class="section-header">Tree Structure & Growth</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="explanation-box">
        <strong>Tree Growth Process</strong><br>
        The tree grows recursively from the root node. Each internal node represents a decision based on a feature threshold. The growth stops when stopping criteria are met (max depth, min samples, or pure nodes).
    </div>
    """, unsafe_allow_html=True)
    
    # Get tree structure
    tree_structure = tree.get_tree_structure()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Tree visualization
        if tree_structure:
            fig = create_tree_structure_plot(tree_structure)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Tree structure not available")
    
    with col2:
        st.markdown("#### Tree Statistics")
        
        def count_nodes(node):
            if node is None:
                return 0, 0
            if node['is_leaf']:
                return 1, 1  # total nodes, leaf nodes
            left_total, left_leaves = count_nodes(node['left'])
            right_total, right_leaves = count_nodes(node['right'])
            return 1 + left_total + right_total, left_leaves + right_leaves
        
        if tree_structure:
            total_nodes, leaf_nodes = count_nodes(tree_structure)
            st.write(f"**Total Nodes:** {total_nodes}")
            st.write(f"**Leaf Nodes:** {leaf_nodes}")
            st.write(f"**Internal Nodes:** {total_nodes - leaf_nodes}")
            st.write(f"**Tree Depth:** {max_depth}")
            st.write(f"**Criterion:** {criterion.upper()}")
    
    st.markdown("""
    <div class="key-takeaway">
        <strong>Key Insight:</strong><br>
        The tree structure represents a hierarchical set of decision rules. Each path from root to leaf represents a unique combination of feature conditions leading to a prediction.
    </div>
    """, unsafe_allow_html=True)

with tab5:
    st.markdown('<h2 class="section-header">Prediction Path (Decision Rules)</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="explanation-box">
        <strong>How Predictions Are Made</strong><br>
        To make a prediction, we start at the root node and follow the decision rules based on the input features until we reach a leaf node. The leaf node contains the final prediction.
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive point selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Select a Point for Prediction")
        
        # Create sliders for point selection
        x_val = st.slider("Feature 0", float(X[:, 0].min()), float(X[:, 0].max()), 0.5)
        y_val = st.slider("Feature 1", float(X[:, 1].min()), float(X[:, 1].max()), 0.5)
        
        prediction_point = np.array([x_val, y_val])
        prediction = tree.predict([prediction_point])[0]
        path = tree.get_prediction_path(prediction_point)
        
        # Visualization with prediction point
        fig = create_prediction_path_plot(X, y, prediction_point, path, decision_boundary)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Prediction Path")
        st.write(f"**Final Prediction:** Class {prediction}")
        
        st.markdown("**Decision Rules:**")
        for i, step in enumerate(path[:-1]):  # Exclude final prediction
            st.write(f"{i+1}. {step['condition']}")
            st.write(f"   → {step['result']} (Value: {step['value']:.3f})")
            st.write(f"   Impurity: {step['impurity']:.3f}, Samples: {step['n_samples']}")
            st.write("")
        
        st.markdown(f"**Leaf Node:**")
        st.write(f"Prediction: Class {path[-1]['prediction']}")
        st.write(f"Impurity: {path[-1]['impurity']:.3f}")
        st.write(f"Samples: {path[-1]['n_samples']}")
    
    st.markdown("""
    <div class="key-takeaway">
        <strong>What to observe:</strong><br>
        • Each decision rule splits the feature space<br>
        • The path length depends on tree depth and point location<br>
        • Leaf node purity affects prediction confidence
    </div>
    """, unsafe_allow_html=True)

with tab6:
    st.markdown('<h2 class="section-header">Overfitting & Depth Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="explanation-box">
        <strong>Understanding Overfitting</strong><br>
        Overfitting occurs when the tree becomes too complex and learns noise in the training data. This results in high training accuracy but poor generalization to new data.
    </div>
    """, unsafe_allow_html=True)
    
    # Train shallow and deep trees for comparison
    shallow_tree = DecisionTreeClassifier(max_depth=2, min_samples_split=2, criterion=criterion)
    shallow_tree.fit(X_train, y_train, feature_names=['Feature 0', 'Feature 1'])
    
    deep_tree = DecisionTreeClassifier(max_depth=10, min_samples_split=2, criterion=criterion)
    deep_tree.fit(X_train, y_train, feature_names=['Feature 0', 'Feature 1'])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Performance Comparison")
        fig = create_overfitting_comparison_plot(shallow_tree, deep_tree, X_train, y_train, X_test, y_test)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Model Complexity Analysis")
        
        shallow_train_acc = accuracy_score(y_train, shallow_tree.predict(X_train))
        shallow_test_acc = accuracy_score(y_test, shallow_tree.predict(X_test))
        deep_train_acc = accuracy_score(y_train, deep_tree.predict(X_train))
        deep_test_acc = accuracy_score(y_test, deep_tree.predict(X_test))
        
        st.markdown("**Shallow Tree (Depth=2):**")
        st.write(f"Training Accuracy: {shallow_train_acc:.3f}")
        st.write(f"Test Accuracy: {shallow_test_acc:.3f}")
        st.write(f"Generalization Gap: {abs(shallow_train_acc - shallow_test_acc):.3f}")
        
        st.markdown("**Deep Tree (Depth=10):**")
        st.write(f"Training Accuracy: {deep_train_acc:.3f}")
        st.write(f"Test Accuracy: {deep_test_acc:.3f}")
        st.write(f"Generalization Gap: {abs(deep_train_acc - deep_test_acc):.3f}")
    
    st.markdown("""
    <div class="key-takeaway">
        <strong>Key Insight:</strong><br>
        A larger generalization gap (difference between training and test accuracy) indicates overfitting. Shallow trees typically generalize better but may underfit complex patterns.
    </div>
    """, unsafe_allow_html=True)

with tab7:
    st.markdown('<h2 class="section-header">Noise & Pruning Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="explanation-box">
        <strong>Effect of Noise and Pruning</strong><br>
        Real-world data contains noise that can cause trees to create spurious splits. Pruning techniques and proper stopping criteria help improve model robustness.
    </div>
    """, unsafe_allow_html=True)
    
    # Generate clean and noisy versions
    X_clean, y_clean = generate_dataset(dataset_type, n_samples, 0.0, 42)
    X_noisy, y_noisy = generate_dataset(dataset_type, n_samples, noise_level * 2, 43)
    
    # Train on clean and noisy data
    clean_tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion)
    clean_tree.fit(X_clean, y_clean, feature_names=['Feature 0', 'Feature 1'])
    
    noisy_tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion)
    noisy_tree.fit(X_noisy, y_noisy, feature_names=['Feature 0', 'Feature 1'])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Noise Robustness Analysis")
        fig = create_noise_analysis_plot(clean_tree, noisy_tree, X_clean, y_clean, X_noisy, y_noisy)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Pruning Effects")
        
        st.markdown("**Stopping Criteria Benefits:**")
        st.write("• Prevents overfitting to noise")
        st.write("• Reduces model complexity")
        st.write("• Improves interpretability")
        st.write("• Better generalization")
        
        st.markdown("**Current Settings:**")
        st.write(f"Max Depth: {max_depth}")
        st.write(f"Min Samples Split: {min_samples_split}")
        st.write(f"Noise Level: {noise_level:.2f}")
        st.write(f"Label Noise: {label_noise_ratio:.2f}")
    
    st.markdown("""
    <div class="key-takeaway">
        <strong>Important Observation:</strong><br>
        Trees trained on noisy data may perform poorly on clean data, indicating overfitting to noise. Proper regularization (max depth, min samples) helps create more robust models.
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d;">
    <p>🌳 Decision Tree Interactive Visualizer - Explore how decision trees learn and make predictions</p>
    <p>Adjust the sidebar parameters to see how different settings affect tree behavior and performance</p>
</div>
""", unsafe_allow_html=True)
