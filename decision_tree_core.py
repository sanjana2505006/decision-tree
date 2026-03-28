import numpy as np
from collections import Counter
from typing import Tuple, List, Optional, Dict, Any

class DecisionTreeNode:
    def __init__(self, depth=0, max_depth=None):
        self.depth = depth
        self.max_depth = max_depth
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.impurity = None
        self.n_samples = None
        self.is_leaf = False
        
class DecisionTreeClassifier:
    def __init__(self, max_depth=5, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
        self.n_features = None
        self.n_classes = None
        self.feature_names = None
        
    def fit(self, X, y, feature_names=None):
        self.n_features = X.shape[1]
        self.n_classes = len(np.unique(y))
        self.feature_names = feature_names or [f'Feature {i}' for i in range(self.n_features)]
        self.root = self._grow_tree(X, y)
        
    def _grow_tree(self, X, y, depth=0):
        node = DecisionTreeNode(depth=depth, max_depth=self.max_depth)
        node.n_samples = len(y)
        node.impurity = self._calculate_impurity(y)
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            len(y) < self.min_samples_split or 
            len(np.unique(y)) == 1):
            node.value = self._most_common_label(y)
            node.is_leaf = True
            return node
            
        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)
        
        if best_gain == 0:
            node.value = self._most_common_label(y)
            node.is_leaf = True
            return node
            
        node.feature_index = best_feature
        node.threshold = best_threshold
        
        # Split data
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        
        node.left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        node.right = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        
        return node
    
    def _find_best_split(self, X, y):
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        current_impurity = self._calculate_impurity(y)
        
        for feature_idx in range(self.n_features):
            feature_values = np.unique(X[:, feature_idx])
            
            for threshold in feature_values:
                left_indices = X[:, feature_idx] <= threshold
                right_indices = X[:, feature_idx] > threshold
                
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                    
                left_impurity = self._calculate_impurity(y[left_indices])
                right_impurity = self._calculate_impurity(y[right_indices])
                
                n_left, n_right = len(left_indices), len(right_indices)
                n_total = len(y)
                
                weighted_impurity = (n_left / n_total) * left_impurity + (n_right / n_total) * right_impurity
                gain = current_impurity - weighted_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    
        return best_feature, best_threshold, best_gain
    
    def _calculate_impurity(self, y):
        if len(y) == 0:
            return 0
            
        counts = np.bincount(y)
        probabilities = counts / len(y)
        
        if self.criterion == 'gini':
            return 1 - np.sum(probabilities ** 2)
        elif self.criterion == 'entropy':
            return -np.sum(probabilities * np.log2(probabilities + 1e-10))
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def _most_common_label(self, y):
        if len(y) == 0:
            return 0
        return Counter(y).most_common(1)[0][0]
    
    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for x in X])
    
    def _predict_single(self, x, node):
        if node.is_leaf:
            return node.value
        
        if x[node.feature_index] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
    
    def get_tree_structure(self):
        """Extract tree structure for visualization"""
        def extract_node(node, node_id=0):
            if node is None:
                return None
                
            node_data = {
                'id': node_id,
                'feature': self.feature_names[node.feature_index] if node.feature_index is not None else None,
                'threshold': node.threshold,
                'impurity': node.impurity,
                'n_samples': node.n_samples,
                'value': node.value,
                'is_leaf': node.is_leaf,
                'depth': node.depth
            }
            
            if not node.is_leaf:
                left_id = node_id * 2 + 1
                right_id = node_id * 2 + 2
                node_data['left'] = extract_node(node.left, left_id)
                node_data['right'] = extract_node(node.right, right_id)
                node_data['left_id'] = left_id
                node_data['right_id'] = right_id
                
            return node_data
        
        return extract_node(self.root)
    
    def get_decision_regions(self, X_range, y_range, resolution=100):
        """Get decision boundaries for visualization"""
        x_min, x_max = X_range
        y_min, y_max = y_range
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )
        
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        predictions = self.predict(grid_points)
        
        return xx, yy, predictions.reshape(xx.shape)
    
    def get_prediction_path(self, x):
        """Get the path taken through the tree for a prediction"""
        path = []
        node = self.root
        
        while not node.is_leaf:
            path.append({
                'feature': self.feature_names[node.feature_index],
                'threshold': node.threshold,
                'value': x[node.feature_index],
                'condition': f"{self.feature_names[node.feature_index]} <= {node.threshold:.3f}",
                'result': x[node.feature_index] <= node.threshold,
                'impurity': node.impurity,
                'n_samples': node.n_samples
            })
            
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
                
        path.append({
            'prediction': node.value,
            'impurity': node.impurity,
            'n_samples': node.n_samples
        })
        
        return path
