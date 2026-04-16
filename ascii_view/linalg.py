"""Linear algebra operations module.

Implements SVD compression, matrix projection, and Sobel edge detection
using numpy for the Linear Algebra pipeline.

Performance notes
-----------------
* ``svd_compress``  – avoids constructing the full diagonal matrix by using
  element-wise broadcasting (``U * S``), cutting both memory and FLOP count.
* ``project_matrix`` – same broadcasting trick; Frobenius error via
  ``np.linalg.norm`` with no intermediate copy.
* ``sobel_edges``   – replaced the O(H·W) Python double-loop with a fully
  vectorised sliding-window built from ``np.lib.stride_tricks.as_strided``,
  giving a single C-level einsum over the unrolled windows.  Math is
  identical to the original manual convolution.
* ``normalize``     – uses in-place arithmetic to avoid extra allocations.
* ``combine_matrices`` – calls ``normalize`` once per input, then blends
  with a single fused expression.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import as_strided

# ---------------------------------------------------------------------------
# Module-level constants – Sobel kernels (read-only views)
# ---------------------------------------------------------------------------
_GX = np.array([[-1,  0,  1],
                [-2,  0,  2],
                [-1,  0,  1]], dtype=np.float64)
_GX.flags.writeable = False

_GY = np.array([[-1, -2, -1],
                [ 0,  0,  0],
                [ 1,  2,  1]], dtype=np.float64)
_GY.flags.writeable = False

# Flattened kernels used by the einsum convolution
_GX_FLAT = _GX.ravel()   # shape (9,)
_GY_FLAT = _GY.ravel()   # shape (9,)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _sliding_windows(padded: np.ndarray, kh: int = 3, kw: int = 3) -> np.ndarray:
    """Return a zero-copy view of all (kh, kw) sliding windows in *padded*.

    Parameters
    ----------
    padded:
        2-D array that has already been padded appropriately.
    kh, kw:
        Kernel height and width (default 3×3).

    Returns
    -------
    windows : ndarray, shape (out_h, out_w, kh*kw)
        Each ``windows[i, j]`` is the flattened patch centred at pixel
        ``(i, j)`` of the *unpadded* image.  This is a **view** – no data
        is copied.
    """
    ph, pw   = padded.shape
    out_h    = ph - kh + 1
    out_w    = pw - kw + 1
    sb, sr   = padded.strides          # byte strides of the padded array

    # Shape  : (out_h, out_w, kh, kw)
    # Strides: step one row → sb, step one col → sr, step kernel row → sb,
    #          step kernel col → sr
    windows_4d = as_strided(
        padded,
        shape   = (out_h, out_w, kh, kw),
        strides = (sb,    sr,    sb, sr),
    )
    # Reshape to (out_h, out_w, kh*kw) so we can do a single dot / einsum
    return windows_4d.reshape(out_h, out_w, kh * kw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def svd_compress(matrix: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform SVD compression on *matrix*.

    The reconstruction formula is the standard rank-*k* approximation:

        compressed = U_r · diag(S_r) · Vt_r

    Implemented without ``np.diag`` to avoid allocating an (r×r) matrix –
    instead we use broadcasting: ``(U_r * S_r) @ Vt_r``.

    Parameters
    ----------
    matrix:
        Input 2-D matrix to compress.
    rank:
        Target rank for the approximation.

    Returns
    -------
    compressed : ndarray
        Rank-*k* approximation of *matrix*.
    U_reduced : ndarray, shape (m, rank)
        Left singular vectors.
    S_reduced : ndarray, shape (rank,)
        Singular values.
    Vt_reduced : ndarray, shape (rank, n)
        Transposed right singular vectors.
    """
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    rank       = min(rank, S.shape[0])
    U_reduced  = U[:, :rank]           # (m, rank)
    S_reduced  = S[:rank]              # (rank,)
    Vt_reduced = Vt[:rank, :]          # (rank, n)

    # Broadcasting: multiply each column of U by the corresponding S,
    # then project into row-space via Vt – equivalent to U @ diag(S) @ Vt
    compressed = (U_reduced * S_reduced) @ Vt_reduced

    return compressed, U_reduced, S_reduced, Vt_reduced


def project_matrix(
    matrix: np.ndarray,
    U: np.ndarray,
    S: np.ndarray,
    Vt: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Project *matrix* onto the reduced subspace defined by the SVD factors.

    Reconstruction formula (identical to original):

        projection = U · diag(S) · Vt

    The Frobenius reconstruction error measures how well the low-rank
    subspace captures *matrix*.

    Parameters
    ----------
    matrix:
        Original matrix used only for the error calculation.
    U:
        Left singular vectors  (m × r).
    S:
        Singular values        (r,).
    Vt:
        Transposed right singular vectors  (r × n).

    Returns
    -------
    projection : ndarray
        Low-rank reconstruction.
    reconstruction_error : float
        Frobenius norm of ``matrix - projection``.
    """
    projection           = (U * S) @ Vt
    reconstruction_error = float(np.linalg.norm(matrix - projection, 'fro'))

    return projection, reconstruction_error


from typing import Union

def sobel_edges(grayscale: np.ndarray, return_dirs: bool = False) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Apply Sobel edge detection via fully-vectorised sliding-window convolution.

    Uses the standard Sobel kernels (math unchanged):

        Gx = [[-1, 0, 1],   Gy = [[-1,-2,-1],
               [-2, 0, 2],         [ 0, 0, 0],
               [-1, 0, 1]]         [ 1, 2, 1]]

    Edge magnitude:  ``M = sqrt(Gx² + Gy²)``

    The original O(H·W) Python double-loop is replaced by:

    1. ``np.pad``  – reflect-pad by 1 pixel on each side (same border
       handling as before).
    2. ``as_strided`` – build a zero-copy (out_h, out_w, 9) view of all
       3×3 patches.
    3. ``np.einsum`` – contract patches against both flattened kernels in
       one C-level call:  ``'ijk,k->ij'``.

    Parameters
    ----------
    grayscale:
        2-D grayscale image array (H × W).

    Returns
    -------
    magnitude : ndarray, shape (H, W)
        Edge magnitude at each pixel.
        If return_dirs is True, returns (magnitude, directions) where
        directions is an array of indices [0: '|', 1: '\\', 2: '_', 3: '/'].
    """
    padded  = np.pad(grayscale.astype(np.float64, copy=False), 1, mode='reflect')
    windows = _sliding_windows(padded)          # (H, W, 9) – zero-copy view

    gx = np.einsum('ijk,k->ij', windows, _GX_FLAT, optimize=True)
    gy = np.einsum('ijk,k->ij', windows, _GY_FLAT, optimize=True)

    magnitude = np.sqrt(gx * gx + gy * gy)
    
    if not return_dirs:
        return magnitude

    # Map angle to 4 direction bins: 0=vertical(|), 1=diagonal-down(\), 2=horizontal(_), 3=diagonal-up(/)
    angle = np.arctan2(gy, gx) % np.pi
    dirs = np.floor((angle + np.pi/8) / (np.pi/4)).astype(int) % 4

    return magnitude, dirs


def normalize(matrix: np.ndarray) -> np.ndarray:
    """Normalize *matrix* to the [0, 255] range.

    Formula (identical to original):

        normalized = (matrix - min) / (max - min) * 255

    Returns a zero-filled array of the same shape when the matrix is flat
    (``max - min < 1e-10``).

    Parameters
    ----------
    matrix:
        Input matrix to normalize.

    Returns
    -------
    normalized : ndarray
        Values clipped to ``[0, 255]``.
    """
    min_val = float(matrix.min())
    max_val = float(matrix.max())

    if max_val - min_val < 1e-10:
        return np.zeros_like(matrix, dtype=np.float64)

    # In-place to avoid an extra intermediate array
    result = (matrix - min_val) * (255.0 / (max_val - min_val))
    return np.clip(result, 0.0, 255.0, out=result)


def combine_matrices(
    grayscale: np.ndarray,
    edges: np.ndarray,
    edge_weight: float = 0.3,
) -> np.ndarray:
    """Blend normalised grayscale and edge matrices.

    Formula (identical to original):

        combined = (1 - edge_weight) · grayscale_norm + edge_weight · edges_norm

    Parameters
    ----------
    grayscale:
        Grayscale image matrix.
    edges:
        Edge magnitude matrix (same shape as *grayscale*).
    edge_weight:
        Scalar weight for the edge contribution, in ``[0.0, 1.0]``.

    Returns
    -------
    combined : ndarray
        Blended matrix clipped to ``[0, 255]``.
    """
    gs_norm    = normalize(grayscale)
    edges_norm = normalize(edges)

    combined = (1.0 - edge_weight) * gs_norm + edge_weight * edges_norm
    return np.clip(combined, 0.0, 255.0, out=combined)
