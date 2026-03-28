# 🌳 Decision Tree Interactive Visualizer

An interactive Streamlit application that visualizes how Decision Trees learn to split data into meaningful regions, grow, and make predictions.

## 🎯 Key Features

### 7 Core Conceptual Components

1. **Data Partitioning & Feature Splits** - Visualize how trees recursively partition the feature space
2. **Impurity Measures** - Compare Gini vs Entropy calculations
3. **Split Selection** - See how the algorithm chooses the best feature and threshold
4. **Tree Structure & Growth** - Watch the tree grow dynamically
5. **Stopping Criteria & Overfitting** - Understand the balance between complexity and generalization
6. **Prediction Path** - Follow the decision rules for any point
7. **Noise & Pruning Analysis** - Explore robustness to real-world data challenges

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone or download the project files
2. Navigate to the project directory
3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your web browser at `http://localhost:8501`

## 📊 Application Structure

### Sidebar Controls

- **Dataset Selection**: Choose from 5 different dataset types (Simple, Complex, Moons, Circles, XOR)
- **Model Parameters**: 
  - Maximum Depth (1-10)
  - Minimum Samples per Split (2-20)
  - Impurity Criterion (Gini/Entropy)
- **Data Parameters**:
  - Noise Level (0-0.5)
  - Label Noise Ratio (0-0.3)
  - Number of Samples (100-1000)

### Main Tabs

1. **📊 Data & Splits** - Scatter plot with decision boundaries
2. **📏 Impurity Measures** - Gini vs Entropy comparison
3. **🎯 Split Selection** - Best split visualization and information gain
4. **🌲 Tree Growth** - Interactive tree structure diagram
5. **🛤️ Prediction Path** - Follow decision rules for selected points
6. **📈 Overfitting & Depth** - Compare shallow vs deep trees
7. **🔧 Noise & Pruning** - Robustness analysis

## 🧮 Technical Implementation

### Core Components

- **`decision_tree_core.py`**: Custom Decision Tree implementation with:
  - Gini and Entropy impurity calculations
  - Recursive tree growth algorithm
  - Prediction path tracking
  - Tree structure extraction for visualization

- **`data_generation.py`**: Dataset utilities with:
  - 5 different dataset types
  - Noise injection capabilities
  - Label noise simulation

- **`visualization.py`**: Plotting functions using:
  - Plotly for interactive visualizations
  - Matplotlib for supporting plots
  - NetworkX for tree structure diagrams

- **`app.py`**: Main Streamlit application with:
  - Tabbed interface for organized exploration
  - Interactive controls and real-time updates
  - Educational explanations and key takeaways

## 🎓 Learning Objectives

After using this visualizer, you will understand:

- How Decision Trees recursively partition the feature space
- The role of impurity measures in split selection
- How stopping criteria prevent overfitting
- The trade-offs between model complexity and generalization
- How noise affects tree robustness
- How to interpret tree structures and decision paths

## 🔬 Dataset Types

1. **Simple Linear**: Linearly separable data with clear boundaries
2. **Complex Multi-cluster**: Multiple clusters with overlapping regions
3. **Half-moons**: Non-linear half-moon shaped patterns
4. **Concentric Circles**: Nested circular patterns
5. **XOR Pattern**: Classic XOR problem requiring non-linear boundaries

## 🎨 Features

- **Interactive Visualizations**: All plots are interactive with zoom, pan, and hover capabilities
- **Real-time Updates**: Changes to parameters immediately update all visualizations
- **Educational Content**: Each section includes explanations and key takeaways
- **Performance Metrics**: Accuracy measurements and generalization analysis
- **Comparative Analysis**: Side-by-side comparisons of different model configurations

## 🛠️ Customization

You can easily extend the application by:

1. Adding new dataset types in `data_generation.py`
2. Implementing additional impurity measures in `decision_tree_core.py`
3. Creating new visualization types in `visualization.py`
4. Adding new analysis tabs in `app.py`

## 📚 Educational Use

This visualizer is designed for:

- **Machine Learning Students**: Understanding Decision Tree fundamentals
- **Data Science Educators**: Teaching ML concepts interactively
- **Researchers**: Exploring tree behavior under different conditions
- **Practitioners**: Understanding model interpretability and robustness

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the visualizer!

## 📄 License

This project is open source and available under the MIT License.

---

**Happy Learning! 🌳**
