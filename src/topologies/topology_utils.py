"""
Utility functions for topology matrices.

This module contains general-purpose operations that are shared across
different topology generators, such as row normalisation and saving/loading
topology matrices.

Topology construction itself should remain in the individual generator
modules (random_network.py, small_world.py, scale_free.py, etc.).
"""

from pathlib import Path

import numpy as np


# =============================================================================
# Matrix Normalisation
# =============================================================================

def normalize_rows(
    P: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Normalise each row of a matrix so that it sums to one.

    In the model, the topology matrix P represents influence weights.
    Therefore, each row should form a probability-like distribution over
    the agents from whom a given agent receives influence.

    Parameters
    ----------
    P : np.ndarray
        Two-dimensional matrix containing non-negative influence weights.
    eps : float
        Numerical tolerance used to identify rows whose total weight is
        effectively zero.

    Returns
    -------
    np.ndarray
        Row-normalised matrix.

    Notes
    -----
    Rows whose sum is smaller than `eps` are left as zero rows rather than
    being divided by an extremely small number.
    """

    # Convert the input to a floating-point NumPy array.
    P = np.asarray(P, dtype=float)

    # Compute the total weight in each row.
    row_sums = P.sum(axis=1, keepdims=True)

    # Create a copy so that the original input matrix is not modified.
    P_normalized = P.copy()

    # Identify rows with a meaningful positive total weight.
    valid_rows = (row_sums[:, 0] > eps)

    # Normalise only valid rows.
    P_normalized[valid_rows] = (
        P_normalized[valid_rows]
        / row_sums[valid_rows]
    )

    # Explicitly keep effectively empty rows equal to zero.
    P_normalized[~valid_rows] = 0.0

    return P_normalized


# =============================================================================
# Saving Topologies
# =============================================================================

def save_topology(
    path: str | Path,
    P: np.ndarray,
) -> None:
    """
    Save a topology matrix to a compressed NumPy file.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination file path. The recommended extension is `.npz`.
    P : np.ndarray
        Topology/influence matrix to save.

    Notes
    -----
    The parent directory is created automatically if it does not already
    exist. The matrix is stored under the key `P`.
    """

    # Convert the path to a Path object for safer path handling.
    path = Path(path)

    # Create the destination directory when necessary.
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Store the matrix in compressed NumPy format.
    np.savez_compressed(
        path,
        P=np.asarray(P, dtype=float),
    )


# =============================================================================
# Loading Topologies
# =============================================================================

def load_topology(
    path: str | Path,
) -> np.ndarray:
    """
    Load a topology matrix previously saved with `save_topology`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the `.npz` topology file.

    Returns
    -------
    np.ndarray
        Loaded topology/influence matrix.

    Raises
    ------
    KeyError
        If the file does not contain a matrix stored under the key `P`.
    """

    # Convert the supplied path to a Path object.
    path = Path(path)

    # Open the compressed NumPy archive.
    with np.load(path) as data:

        # All topology files created by this project are expected to store
        # the influence matrix under the key "P".
        if "P" not in data:
            raise KeyError(
                f"Topology file '{path}' does not contain a 'P' matrix."
            )

        # Return an independent floating-point copy of the matrix.
        P = np.asarray(
            data["P"],
            dtype=float,
        ).copy()

    return P
