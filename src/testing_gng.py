import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from utils.gng import GNG

# Generate a synthetic dataset from a mixture of Gaussians (4D to match cart-pole state space)
np.random.seed(11)
n_samples = 2000
n_dims = 4

cluster1 = np.random.randn(n_samples // 2, n_dims) * 0.5 + np.array([2.0, -1.0, 0.5, -0.5])
cluster2 = np.random.randn(n_samples // 2, n_dims) * 0.5 + np.array([-2.0, 1.0, -0.5, 0.5])
data = np.vstack([cluster1, cluster2])

# Create and train GNG
gng = GNG(data, epsilon_b=0.05, epsilon_n=0.001, lambda_=50, alpha=0.5, max_nodes=30)
gng.fit(data, max_iterations=5000)

print(f"Number of nodes: {len(gng.nodes)}")
print(f"Number of edges: {len(gng.edges)}")
print(f"Node indices: {list(gng.nodes.keys())}")

gng.visualize(title="GNG trained on 4D Gaussian mixture")
