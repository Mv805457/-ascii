"""Linear algebra operations module.

Implements SVD compression, matrix projection, and Sobel edge detection
using numpy for the Linear Algebra pipeline.
"""

import numpy as np


def svd_compress(matrix: np.ndarray, rank: int) -> tuple:
    """Perform SVD compression on matrix.

    Args:
        matrix: Input 2D matrix to compress.
        rank: Target rank for compression.

    Returns:
        Tuple of (compressed_matrix, U_reduced, S_reduced, Vt_reduced).
    """
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    rank = min(rank, len(S))

    U_reduced = U[:, :rank]
    S_reduced = S[:rank]
    Vt_reduced = Vt[:rank, :]

    compressed = U_reduced @ np.diag(S_reduced) @ Vt_reduced

    return compressed, U_reduced, S_reduced, Vt_reduced


def project_matrix(
    matrix: np.ndarray,
    U: np.ndarray,
    S: np.ndarray,
    Vt: np.ndarray
) -> tuple:
    """Project matrix onto reduced subspace from SVD.

    Args:
        matrix: Original matrix for error calculation.
        U: Left singular vectors.
        S: Singular values.
        Vt: Transposed right singular vectors.

    Returns:
        Tuple of (projection, reconstruction_error).
    """
    projection = U @ np.diag(S) @ Vt
    reconstruction_error = np.linalg.norm(matrix - projection, 'fro')

    return projection, reconstruction_error


def sobel_edges(grayscale: np.ndarray) -> np.ndarray:
    """Apply Sobel edge detection using manual convolution.

    Uses standard Sobel kernels:
        Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        Gy = [[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]]

    Args:
        grayscale: 2D grayscale image array.

    Returns:
        Edge magnitude array.
    """
    gx_kernel = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=np.float64)

    gy_kernel = np.array([[-1, -2, -1],
                          [ 0,  0,  0],
                          [ 1,  2,  1]], dtype=np.float64)

    height, width = grayscale.shape

    gx = np.zeros((height, width), dtype=np.float64)
    gy = np.zeros((height, width), dtype=np.float64)

    pad = 1
    padded = np.pad(grayscale, pad, mode='reflect')

    for i in range(height):
        for j in range(width):
            region = padded[i:i+3, j:j+3]
            gx[i, j] = np.sum(region * gx_kernel)
            gy[i, j] = np.sum(region * gy_kernel)

    magnitude = np.sqrt(gx**2 + gy**2)

    return magnitude


def normalize(matrix: np.ndarray) -> np.ndarray:
    """Normalize matrix to 0-255 range.

    Args:
        matrix: Input matrix to normalize.

    Returns:
        Normalized matrix with values in [0, 255].
    """
    min_val = np.min(matrix)
    max_val = np.max(matrix)

    if max_val - min_val < 1e-10:
        return np.zeros_like(matrix)

    normalized = (matrix - min_val) / (max_val - min_val) * 255.0

    return np.clip(normalized, 0, 255)


def combine_matrices(
    grayscale: np.ndarray,
    edges: np.ndarray,
    edge_weight: float = 0.3
) -> np.ndarray:
    """Combine grayscale and edge matrices.

    Formula: combined = (1 - edge_weight) * grayscale + edge_weight * edges

    Args:
        grayscale: Grayscale image matrix.
        edges: Edge magnitude matrix.
        edge_weight: Weight for edge contribution (0.0-1.0).

    Returns:
        Combined matrix.
    """
    gs_norm = normalize(grayscale)
    edges_norm = normalize(edges)

    combined = (1 - edge_weight) * gs_norm + edge_weight * edges_norm

    return np.clip(combined, 0, 255)
