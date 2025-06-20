import numpy as np
from scipy.linalg import inv, eig
import time

def generate_large_matrix(size):
    np.random.seed(42)
    return np.random.rand(size, size)

def matrix_operations(matrix):
    print("Starting matrix operations...")

    # Matrix multiplication
    start = time.time()
    result_mul = np.dot(matrix, matrix)
    print(f"Matrix multiplication done in {time.time() - start:.2f} seconds")

    # Matrix inversion
    start = time.time()
    result_inv = inv(matrix + np.eye(matrix.shape[0]) * 1e-5)  # add epsilon to avoid singular matrix
    print(f"Matrix inversion done in {time.time() - start:.2f} seconds")

    # Eigenvalue decomposition
    start = time.time()
    values, vectors = eig(matrix)
    print(f"Eigenvalue decomposition done in {time.time() - start:.2f} seconds")

    return result_mul, result_inv, values

if __name__ == "__main__":
    SIZE = 1000  # increase for more load (e.g., 2000+ for stress test)
    matrix = generate_large_matrix(SIZE)
    matrix_operations(matrix)
