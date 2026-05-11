import numpy as np

# Initialize arrays
a = np.array([[1, 2, 3, 4, 5]])
b = np.array([[10, 20, 30, 40, 50]])
matrix = np.arange(9).reshape(3, 3)

# Basic operations
print("a:", a)
print("Shape of a: ", a.shape)
print("b:", b)
print("Shape of b: ", b.shape)
print("a + b:", a + b)
print("a * b:", a * b)
print("dot product:", np.dot(a, b))
print("mean of a:", np.mean(a))
print("std of b:", np.std(b))

# Matrix operations
print("\nmatrix:\n", matrix)
print("matrix transposed:\n", matrix.T)
print("matrix sum along axis 0:", matrix.sum(axis=0))
print("matrix sum along axis 1:", matrix.sum(axis=1))

