import numpy as np
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split

def generate_dataset(dataset_type='simple', n_samples=300, noise_level=0.1, random_state=42):
    """Generate different types of 2D classification datasets"""
    
    np.random.seed(random_state)
    
    if dataset_type == 'simple':
        # Simple linearly separable dataset
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            random_state=random_state
        )
        
    elif dataset_type == 'complex':
        # More complex dataset with multiple clusters
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=2,
            random_state=random_state
        )
        
    elif dataset_type == 'moons':
        # Half-moon shapes
        X, y = make_moons(
            n_samples=n_samples,
            noise=noise_level,
            random_state=random_state
        )
        
    elif dataset_type == 'circles':
        # Concentric circles
        X, y = make_circles(
            n_samples=n_samples,
            noise=noise_level,
            factor=0.5,
            random_state=random_state
        )
        
    elif dataset_type == 'xor':
        # XOR pattern
        X = np.random.uniform(-2, 2, (n_samples, 2))
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
        
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Add noise if specified
    if noise_level > 0 and dataset_type not in ['moons', 'circles']:
        X += np.random.normal(0, noise_level, X.shape)
    
    # Normalize features to [0, 1] range for better visualization
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    
    return X, y

def add_label_noise(y, noise_ratio=0.1):
    """Add random label noise to simulate real-world scenarios"""
    n_samples = len(y)
    n_noisy = int(n_samples * noise_ratio)
    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
    
    y_noisy = y.copy()
    for idx in noisy_indices:
        # Flip the label
        y_noisy[idx] = 1 - y_noisy[idx]
    
    return y_noisy

def get_dataset_info():
    """Get information about available datasets"""
    return {
        'simple': {
            'name': 'Simple Linear',
            'description': 'Linearly separable data with clear boundaries',
            'difficulty': 'Easy'
        },
        'complex': {
            'name': 'Complex Multi-cluster',
            'description': 'Multiple clusters per class with overlapping regions',
            'difficulty': 'Medium'
        },
        'moons': {
            'name': 'Half-moons',
            'description': 'Non-linear half-moon shaped patterns',
            'difficulty': 'Medium'
        },
        'circles': {
            'name': 'Concentric Circles',
            'description': 'Nested circular patterns',
            'difficulty': 'Hard'
        },
        'xor': {
            'name': 'XOR Pattern',
            'description': 'Classic XOR problem requiring non-linear boundaries',
            'difficulty': 'Hard'
        }
    }
