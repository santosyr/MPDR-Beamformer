"""
dsp.py
======
Algorithms for distributed signal processing and sensor-network modelling.

This module collects the building blocks used by the MPDR beamformer solvers
(``mpdr.py``).  It is intentionally kept self-contained so that each function
can also be studied and tested in isolation.

Contents
--------
Signal generation
    ``awgn``        — Additive White Gaussian Noise.
    ``npsk``        — Random n-PSK symbol sequences.

Network topology
    ``adjacency_matrix``   — Build an adjacency matrix from a dict definition.

Distributed consensus
    ``average_consensus``     — Finite-time exact average consensus (scalar).
"""

import numpy as np

# =============================================================================
# Signal generation
# =============================================================================

def awgn(
    shape: tuple,
    power: float,
    complex_noise: bool = True,
) -> np.ndarray:
    """
    Generate Additive White Gaussian Noise (AWGN).

    For **complex** noise the total variance σ² is split equally between
    the real and imaginary parts, so each part has variance σ²/2:

    .. math::

        n = \\sqrt{\\sigma^2/2}\\,(\\mathcal{N}(0,1) + j\\,\\mathcal{N}(0,1))

    This ensures that ``E[|n|²] = σ²`` regardless of the number of
    dimensions.

    For **real** noise:

    .. math::

        n = \\sqrt{\\sigma^2}\\,\\mathcal{N}(0,1)

    Parameters
    ----------
    shape : tuple
        Shape of the desired noise array (e.g. ``(N,)`` or ``(N, K)``).
    power : float
        Noise variance σ² (linear power).  Must be non-negative.
    complex_noise : bool, optional
        If ``True`` (default), generates complex-valued noise.
        If ``False``, generates real-valued noise.

    Returns
    -------
    np.ndarray
        Noise array of the requested ``shape`` and dtype
        (``complex128`` or ``float64``).

    Examples
    --------
    Real noise vector of length 5:

    >>> noise = awgn(shape=(5,), power=0.1, complex_noise=False)
    >>> noise.shape
    (5,)
    >>> noise.dtype
    dtype('float64')

    Complex noise matrix of shape (2, 3) with unit variance:

    >>> noise = awgn(shape=(2, 3), power=1.0)
    >>> noise.shape
    (2, 3)
    >>> np.iscomplexobj(noise)
    True
    """
    if complex_noise:
        noise = (
            np.sqrt(power / 2.0)
            * (np.random.randn(*shape) + 1j * np.random.randn(*shape))
        )
    else:
        noise = np.sqrt(power) * np.random.randn(*shape)

    return noise


def npsk(
    num_symbols: int,
    n: int = 4,
    rotation: float = np.pi / 4,
) -> np.ndarray:
    """
    Generate a random sequence of n-PSK symbols.

    Symbols are drawn uniformly at random from an n-point Phase Shift
    Keying (PSK) constellation:

    .. math::

        c_k = e^{j (2\\pi k / n \\;+\\; \\phi_0)},
        \\quad k = 0, 1, \\dots, n-1

    where φ₀ = ``rotation``.  The constellation is normalised to unit
    average power (‖c_k‖ = 1 for all k, so the normalisation factor
    is trivially 1; it is included explicitly for generality).

    Parameters
    ----------
    num_symbols : int
        Number of symbols to generate.
    n : int, optional
        Modulation order.  Must be a positive power of 2 (e.g. 2 for
        BPSK, 4 for QPSK, 8 for 8-PSK).  Default is 4 (QPSK).
    rotation : float, optional
        Constellation rotation in radians.  Default is π/4, which places
        QPSK symbols at ±45° and ±135°.

    Returns
    -------
    np.ndarray of shape (num_symbols,)
        Complex unit-magnitude PSK symbols.

    Raises
    ------
    ValueError
        If ``n`` is not a positive power of 2.

    Examples
    --------
    Generate 1000 QPSK symbols and verify unit average power:

    >>> symbols = npsk(1000, n=4)
    >>> symbols.shape
    (1000,)
    >>> np.isclose(np.mean(np.abs(symbols) ** 2), 1.0)
    True

    BPSK symbols (n=2):

    >>> symbols = npsk(500, n=2, rotation=0.0)
    >>> set(np.round(symbols.real, 6))  # only ±1
    {-1.0, 1.0}
    """
    if not (n > 0 and (n & (n - 1)) == 0):
        raise ValueError(
            f"Modulation order 'n' must be a positive power of 2, got {n}."
        )

    indices       = np.random.randint(0, n, num_symbols)
    angles        = 2.0 * np.pi * np.arange(n) / n + rotation
    constellation = np.exp(1j * angles)

    # Normalise to unit average power (redundant for pure PSK, kept for clarity)
    constellation /= np.sqrt(np.mean(np.abs(constellation) ** 2))

    return constellation[indices]

# =============================================================================
# Network topology
# =============================================================================

def adjacency_matrix(network: dict[str, dict[str, any]]) -> np.ndarray:
    """
    Build an adjacency matrix from a dictionary-based network definition.

    Constructs a symmetric binary adjacency matrix representing an
    undirected graph.  Each entry A[i, j] = 1 if and only if node i and
    node j are connected; diagonal entries are zero (no self-loops).

    Parameters
    ----------
    network : dict[str, dict]
        A dictionary whose keys are node names (strings) and whose values
        are dictionaries containing at least the key ``'connections'``
        with a list of neighbouring node names.

        Example layout::

            {
                "NodeA": {"connections": ["NodeB"]},
                "NodeB": {"connections": ["NodeA", "NodeC"]},
                "NodeC": {"connections": ["NodeB"]},
            }

    Returns
    -------
    np.ndarray of shape (P, P)
        Symmetric integer adjacency matrix, where P is the number of nodes.

    Raises
    ------
    ValueError
        If a connection references a node name that is not a key in
        ``network``.

    Examples
    --------
    Linear chain A — B — C:

    >>> net = {
    ...     "NodeA": {"connections": ["NodeB"]},
    ...     "NodeB": {"connections": ["NodeA", "NodeC"]},
    ...     "NodeC": {"connections": ["NodeB"]},
    ... }
    >>> adjacency_matrix(net)
    array([[0, 1, 0],
           [1, 0, 1],
           [0, 1, 0]])
    """
    nodes      = list(network.keys())
    P          = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
    A          = np.zeros((P, P), dtype=int)

    for node, info in network.items():
        i = node_index[node]
        for neighbour in info.get("connections", []):
            if neighbour not in node_index:
                raise ValueError(
                    f"Connection error: node '{neighbour}' not found in the network."
                )
            j       = node_index[neighbour]
            A[i, j] = 1
            A[j, i] = 1  # enforce symmetry (undirected graph)

    return A


# =============================================================================
# Distributed consensus
# =============================================================================

def average_consensus(
    A: np.ndarray,
    u0: np.ndarray,
    tol: float = 1e-8,
) -> float:
    """
    Compute the exact network average via finite-time Average Consensus.

    This algorithm allows all nodes of a connected graph to converge to
    the exact average of their initial values in a *finite* number of
    steps (at most R - 1), where R is the number of **distinct**
    eigenvalues of the graph Laplacian **L** = **D** - **A**.

    Algorithm
    ---------
    Let {λ₁ = 0, λ₂, …, λ_R} be the distinct eigenvalues of **L**
    in ascending order.  The weight matrices are:

    .. math::

        W^{(0)} = \\frac{(-1)^{R-1}}{\\lambda_2 \\cdots \\lambda_R}
                  (\\mathbf{L} - \\lambda_R \\mathbf{I})

        W^{(t)} = \\mathbf{L} - \\lambda_{t+1} \\mathbf{I},
        \\quad t = 1, \\dots, R-2

    Starting from **u**₀, the update rule is **u**_{t+1} = W^{(t)} **u**_t.
    After R - 1 steps every entry of **u** equals the global average
    (1/P) Σ_p u₀[p], provided the graph is connected.

    Parameters
    ----------
    A : np.ndarray of shape (P, P)
        Symmetric binary adjacency matrix of the graph.
    u0 : np.ndarray of shape (P,)
        Initial values at each of the P nodes.
    tol : float, optional
        Tolerance used to verify that the smallest Laplacian eigenvalue
        is (numerically) zero, i.e. that the graph is connected.
        Default is 1e-8.

    Returns
    -------
    float
        The consensus value, equal to ``np.mean(u0)`` for a connected
        graph.

    Notes
    -----
    After R - 1 steps, all entries of **u** converge to the same value.
    Returning ``u[0]`` is therefore equivalent to returning any ``u[p]``
    or ``np.mean(u)`` — only one scalar needs to be communicated after
    the last iteration.

    The algorithm has complexity O(R · P²) per call.  For dense graphs
    R is small (close to P), but a well-designed sparse topology can
    reduce R significantly.

    Examples
    --------
    Three-node linear graph 0 — 1 — 2:

    >>> A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> u0 = np.array([10.0, 20.0, 30.0])
    >>> val = average_consensus(A, u0)
    >>> np.isclose(val, np.mean(u0))
    True
    """
    P = A.shape[0]
    D = np.diag(A.sum(axis=1))
    L = D - A

    # Find distinct eigenvalues of L, sorted in ascending order
    eigvals        = np.linalg.eigvals(L).real
    distinct_eigs  = np.unique(np.round(eigvals, decimals=8))
    distinct_eigs  = np.sort(distinct_eigs)
    R              = len(distinct_eigs)

    if np.abs(distinct_eigs[0]) > tol:
        print(
            "Warning: graph may not be connected — smallest Laplacian "
            "eigenvalue is not zero."
        )

    I = np.eye(P)
    u = u0.copy().astype(float)

    for t in range(R - 1):
        if t == 0:
            # W⁽⁰⁾ = ((-1)^(R-1) / (λ₂ · λ₃ · … · λ_R)) (L − λ_R I)
            prod_eigs = np.prod(distinct_eigs[1:]) if R > 1 else 1.0
            W = ((-1) ** (R - 1) / prod_eigs) * (L - distinct_eigs[-1] * I)
        else:
            # W⁽ᵗ⁾ = L − λ_{t+1} I
            W = L - distinct_eigs[t] * I

        u = W @ u

    # After R-1 steps all entries are equal to the global average.
    return u[0]
