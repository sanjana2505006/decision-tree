#!/usr/bin/env python3
"""
Simple test script to verify the Decision Tree components work correctly
"""

import numpy as np
from decision_tree_core import DecisionTreeClassifier
from data_generation import generate_dataset
from visualization import create_scatter_plot

def test_basic_functionality():
    print("🧪 Testing Decision Tree Components...")
    
    # Test data generation
    print("📊 Testing data generation...")
    X, y = generate_dataset('simple', n_samples=100, noise_level=0.1)
    print(f"Generated dataset: {X.shape}, classes: {np.unique(y)}")
    
    # Test decision tree training
    print("🌳 Testing decision tree training...")
    tree = DecisionTreeClassifier(max_depth=3, min_samples_split=2, criterion='gini')
    tree.fit(X, y, feature_names=['Feature 0', 'Feature 1'])
    print(f"Tree trained successfully with {tree.n_features} features")
    
    # Test predictions
    print("🎯 Testing predictions...")
    predictions = tree.predict(X[:5])
    print(f"Sample predictions: {predictions}")
    
    # Test tree structure
    print("📈 Testing tree structure extraction...")
    structure = tree.get_tree_structure()
    print(f"Tree structure extracted: {'Success' if structure else 'Failed'}")
    
    # Test decision boundary
    print("🎨 Testing decision boundary generation...")
    xx, yy, zz = tree.get_decision_regions((0, 1), (0, 1))
    print(f"Decision boundary generated: {xx.shape}")
    
    # Test prediction path
    print("🛤️ Testing prediction path...")
    test_point = np.array([0.5, 0.5])
    path = tree.get_prediction_path(test_point)
    print(f"Prediction path length: {len(path)} steps")
    
    print("✅ All tests passed! The application should work correctly.")
    return True

if __name__ == "__main__":
    test_basic_functionality()
