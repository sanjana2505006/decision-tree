import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from typing import List, Tuple, Dict, Any

def create_scatter_plot(X, y, title="Data Distribution", show_lines=False, 
                       split_lines=None, decision_boundary=None):
    """Create interactive scatter plot with optional split lines and decision boundaries"""
    
    fig = go.Figure()
    
    # Add decision boundary if provided
    if decision_boundary is not None:
        xx, yy, zz = decision_boundary
        fig.add_trace(go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=zz,
            showscale=False,
            opacity=0.3,
            colorscale='RdBu',
            contours=dict(
                start=0,
                end=1,
                size=1
            )
        ))
    
    # Add data points
    colors = ['red', 'blue']
    labels = ['Class 0', 'Class 1']
    
    for i in range(2):
        mask = y == i
        fig.add_trace(go.Scatter(
            x=X[mask, 0],
            y=X[mask, 1],
            mode='markers',
            name=labels[i],
            marker=dict(color=colors[i], size=8, opacity=0.7)
        ))
    
    # Add split lines if provided
    if show_lines and split_lines:
        for line in split_lines:
            if line['feature'] == 0:  # Vertical line
                fig.add_shape(
                    type="line",
                    x0=line['threshold'], y0=0,
                    x1=line['threshold'], y1=1,
                    line=dict(color="green", width=2, dash="dash")
                )
            else:  # Horizontal line
                fig.add_shape(
                    type="line",
                    x0=0, y0=line['threshold'],
                    x1=1, y1=line['threshold'],
                    line=dict(color="green", width=2, dash="dash")
                )
    
    fig.update_layout(
        title=title,
        xaxis_title="Feature 0",
        yaxis_title="Feature 1",
        showlegend=True,
        width=600,
        height=500
    )
    
    return fig

def create_impurity_comparison_plot(y_values, split_candidates, criterion='gini'):
    """Create plot showing impurity before and after splits"""
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Current Node Impurity', 'Best Split Candidates'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Current node class distribution
    unique, counts = np.unique(y_values, return_counts=True)
    colors = ['red', 'blue'][:len(unique)]
    
    fig.add_trace(
        go.Bar(x=[f'Class {i}' for i in unique], y=counts, 
               marker_color=colors, name='Current'),
        row=1, col=1
    )
    
    # Split candidates
    if split_candidates:
        gains = [s['gain'] for s in split_candidates]
        features = [f"{s['feature']} @ {s['threshold']:.2f}" for s in split_candidates]
        
        fig.add_trace(
            go.Bar(x=features, y=gains, marker_color='lightblue', name='Gain'),
            row=1, col=2
        )
    
    fig.update_layout(
        title=f"Impurity Analysis ({criterion.upper()})",
        showlegend=False,
        height=400
    )
    
    return fig

def create_tree_structure_plot(tree_structure):
    """Create interactive tree visualization"""
    
    if not tree_structure:
        return go.Figure()
    
    # Create networkx graph
    G = nx.DiGraph()
    
    def add_nodes(node):
        if node is None:
            return
            
        node_id = node['id']
        
        # Create node label
        if node['is_leaf']:
            label = f"Leaf\nClass: {node['value']}\nSamples: {node['n_samples']}\nImpurity: {node['impurity']:.3f}"
            color = 'lightgreen'
        else:
            label = f"{node['feature']} <= {node['threshold']:.3f}\nSamples: {node['n_samples']}\nImpurity: {node['impurity']:.3f}"
            color = 'lightblue'
        
        G.add_node(node_id, label=label, color=color)
        
        if not node['is_leaf']:
            G.add_edge(node_id, node['left_id'])
            G.add_edge(node_id, node['right_id'])
            add_nodes(node['left'])
            add_nodes(node['right'])
    
    add_nodes(tree_structure)
    
    # Position nodes using hierarchy
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
    
    # Extract positions and labels
    x_nodes = [pos[node][0] for node in G.nodes()]
    y_nodes = [pos[node][1] for node in G.nodes()]
    node_labels = [G.nodes[node]['label'] for node in G.nodes()]
    node_colors = [G.nodes[node]['color'] for node in G.nodes()]
    
    # Create edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create plot
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='gray'),
        hoverinfo='none',
        mode='lines'
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=x_nodes, y=y_nodes,
        mode='markers+text',
        marker=dict(size=20, color=node_colors),
        text=node_labels,
        textposition='middle center',
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title="Decision Tree Structure",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    
    return fig

def create_prediction_path_plot(X, y, prediction_point, path, decision_boundary=None):
    """Create plot showing prediction path for a specific point"""
    
    fig = go.Figure()
    
    # Add decision boundary if provided
    if decision_boundary is not None:
        xx, yy, zz = decision_boundary
        fig.add_trace(go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=zz,
            showscale=False,
            opacity=0.3,
            colorscale='RdBu'
        ))
    
    # Add data points
    colors = ['red', 'blue']
    labels = ['Class 0', 'Class 1']
    
    for i in range(2):
        mask = y == i
        fig.add_trace(go.Scatter(
            x=X[mask, 0],
            y=X[mask, 1],
            mode='markers',
            name=labels[i],
            marker=dict(color=colors[i], size=8, opacity=0.5)
        ))
    
    # Add prediction point
    fig.add_trace(go.Scatter(
        x=[prediction_point[0]],
        y=[prediction_point[1]],
        mode='markers',
        name='Prediction Point',
        marker=dict(color='black', size=15, symbol='star')
    ))
    
    fig.update_layout(
        title="Prediction Path Visualization",
        xaxis_title="Feature 0",
        yaxis_title="Feature 1",
        showlegend=True,
        width=600,
        height=500
    )
    
    return fig

def create_overfitting_comparison_plot(shallow_tree, deep_tree, X_train, y_train, X_test, y_test):
    """Compare shallow vs deep tree performance"""
    
    # Get predictions
    shallow_train_pred = shallow_tree.predict(X_train)
    shallow_test_pred = shallow_tree.predict(X_test)
    deep_train_pred = deep_tree.predict(X_train)
    deep_test_pred = deep_tree.predict(X_test)
    
    # Calculate accuracies
    from sklearn.metrics import accuracy_score
    
    shallow_train_acc = accuracy_score(y_train, shallow_train_pred)
    shallow_test_acc = accuracy_score(y_test, shallow_test_pred)
    deep_train_acc = accuracy_score(y_train, deep_train_pred)
    deep_test_acc = accuracy_score(y_test, deep_test_pred)
    
    # Create comparison plot
    fig = go.Figure()
    
    models = ['Shallow Tree', 'Deep Tree']
    train_accs = [shallow_train_acc, deep_train_acc]
    test_accs = [shallow_test_acc, deep_test_acc]
    
    fig.add_trace(go.Bar(
        name='Training Accuracy',
        x=models,
        y=train_accs,
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Test Accuracy',
        x=models,
        y=test_accs,
        marker_color='lightcoral'
    ))
    
    fig.update_layout(
        title='Overfitting Analysis: Shallow vs Deep Tree',
        xaxis_title='Model',
        yaxis_title='Accuracy',
        barmode='group',
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def create_noise_analysis_plot(clean_tree, noisy_tree, X_clean, y_clean, X_noisy, y_noisy):
    """Analyze effect of noise on tree performance"""
    
    from sklearn.metrics import accuracy_score
    
    # Performance on clean data
    clean_on_clean = accuracy_score(y_clean, clean_tree.predict(X_clean))
    noisy_on_clean = accuracy_score(y_clean, noisy_tree.predict(X_clean))
    
    # Performance on noisy data
    clean_on_noisy = accuracy_score(y_noisy, clean_tree.predict(X_noisy))
    noisy_on_noisy = accuracy_score(y_noisy, noisy_tree.predict(X_noisy))
    
    fig = go.Figure()
    
    scenarios = ['Clean Data', 'Noisy Data']
    clean_tree_acc = [clean_on_clean, clean_on_noisy]
    noisy_tree_acc = [noisy_on_clean, noisy_on_noisy]
    
    fig.add_trace(go.Bar(
        name='Tree on Clean Data',
        x=scenarios,
        y=clean_tree_acc,
        marker_color='lightgreen'
    ))
    
    fig.add_trace(go.Bar(
        name='Tree on Noisy Data',
        x=scenarios,
        y=noisy_tree_acc,
        marker_color='lightyellow'
    ))
    
    fig.update_layout(
        title='Noise Analysis: Model Robustness',
        xaxis_title='Test Scenario',
        yaxis_title='Accuracy',
        barmode='group',
        yaxis=dict(range=[0, 1])
    )
    
    return fig
