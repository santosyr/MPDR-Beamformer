"""
mpdr.py
=======
Minimum Power Distortionless Response (MPDR) / Capon beamformer solvers.

Three solvers are provided, each targeting a different operational context:

``theoretical``
    Uses the **analytical** covariance matrix built from known signal
    parameters.  Serves as the ideal performance baseline.

``estimated``
    Builds a **sample** covariance matrix from simulated snapshots and solves
    the beamforming system with the **Conjugate Gradient (CG)** algorithm,
    avoiding direct matrix inversion.

``distributed``
    Implements a fully **distributed** CG solver for multi-node sensor
    networks.  Nodes collaborate through the **Average Consensus** protocol
    without ever sharing raw data.

Mathematical background
-----------------------
The MPDR beamformer minimises the total output power subject to a
distortionless constraint in the direction of the Signal of Interest (SOI):

    minimise  wᴴ R_xx w
    subject to  wᴴ a(θ,φ) = 1

where **R_xx** ∈ ℂᴺˣᴺ is the array covariance matrix and **a**(θ,φ) ∈ ℂᴺ
is the steering vector toward the SOI.  The closed-form solution is:

    w_opt = R_xx⁻¹ a / (aᴴ R_xx⁻¹ a)

In practice the matrix inverse is replaced by solving the linear system
R_xx w = a using the Conjugate Gradient algorithm (O(N²) per iteration
instead of O(N³) for direct inversion).

References
----------
Van Trees, H. L. (2002). *Optimum Array Processing*. Wiley-Interscience.
Capon, J. (1969). High-resolution frequency-wavenumber spectrum analysis.
    *Proceedings of the IEEE*, 57(8), 1408–1418.
"""

import numpy as np

from . import dsp
from . import plotter
from .steering import steering_vector


# =============================================================================
# Theoretical solver
# =============================================================================

def theoretical(
    positions: np.ndarray,
    theta_soi: float,
    phi_soi: float,
    signal_power: float = 1.0,
    noise_power: float = 1.0,
    wavelength: float = 1.0,
    interferers_data: list[dict] = None,
    plots: bool = False,
    **plot_kwargs,
) -> np.ndarray:
    """
    Compute MPDR weights from the analytical covariance matrix.

    This solver constructs the theoretical covariance matrix **R_xx**
    directly from the known signal scenario.  It then solves the MPDR
    optimisation exactly, providing the ideal (oracle) beamformer weights
    that serve as an upper-bound reference for the estimated and
    distributed solvers.

    The covariance matrix is assembled as:

    .. math::

        \\mathbf{R}_{xx} = \\sigma_s^2 \\mathbf{a}_s \\mathbf{a}_s^H
          + \\sum_{i} \\sigma_i^2 \\mathbf{a}_i \\mathbf{a}_i^H
          + \\sigma_n^2 \\mathbf{I}

    where :math:`\\sigma_s^2` is the SOI power, :math:`\\sigma_i^2` and
    :math:`\\mathbf{a}_i` are the power and steering vector of the i-th
    interferer, and :math:`\\sigma_n^2` is the noise variance.

    The optimal weight vector is:

    .. math::

        \\mathbf{w} =
        \\frac{\\mathbf{R}_{xx}^{-1} \\mathbf{a}_s}
             {\\mathbf{a}_s^H \\mathbf{R}_{xx}^{-1} \\mathbf{a}_s}

    Note
    ----
    The matrix inverse is computed via ``numpy.linalg.solve``, which uses
    an LU decomposition (O(N³)).  This is acceptable for the theoretical
    solver because it runs once on a known, well-conditioned matrix.  For
    iterative or large-scale problems, prefer ``estimated`` or
    ``distributed``.

    Parameters
    ----------
    positions : np.ndarray of shape (N, 3)
        Cartesian (x, y, z) coordinates of the N sensor elements.
    theta_soi : float
        Elevation angle of the SOI in degrees.  See ``steering.py`` for
        the angle convention.
    phi_soi : float
        Azimuth angle of the SOI in degrees.
    signal_power : float, optional
        Linear power :math:`\\sigma_s^2` of the SOI.  Default is 1.0.
    noise_power : float, optional
        Noise variance :math:`\\sigma_n^2` (assumed spatially white and
        uncorrelated across sensors).  Default is 1.0.
    wavelength : float, optional
        Signal wavelength λ.  Must use the same units as ``positions``.
        Default is 1.0.
    interferers_data : list of dict, optional
        Each dictionary defines one interferer and must contain:

        * ``'theta'`` (float) — elevation angle in degrees.
        * ``'phi'``   (float) — azimuth angle in degrees.
        * ``'power'`` (float) — linear interference power.
        * ``'wavelength'`` (float) — interferer wavelength.

        Defaults to ``None`` (no interferers).
    plots : bool, optional
        If ``True``, generates a polar beam-pattern plot via
        ``plotter.polar``.  Default is ``False``.
    **plot_kwargs
        Extra keyword arguments forwarded to ``plotter.polar``.
        Useful keys: ``'weights_labels'`` (list[str]), ``'r_lim'``
        (tuple[float, float]).

    Returns
    -------
    np.ndarray of shape (N,)
        Complex optimal beamformer weight vector.

    Examples
    --------
    Minimal case — no interferers:

    >>> import numpy as np
    >>> positions = np.array([[0, 0, 0], [0.5, 0, 0], [1.0, 0, 0]])
    >>> w = theoretical(positions=positions, theta_soi=90.0, phi_soi=0.0)
    >>> w.shape
    (3,)
    >>> # The distortionless constraint must hold: wᴴ a ≈ 1
    >>> from .steering import steering_vector
    >>> a = steering_vector(90.0, 0.0, 1.0, positions)
    >>> np.isclose(np.vdot(a, w), 1.0)
    True

    With one interferer at 60° azimuth:

    >>> interferers = [{'theta': 90, 'phi': 60, 'power': 20.0, 'wavelength': 1.0}]
    >>> w_int = theoretical(
    ...     positions=positions,
    ...     theta_soi=90.0,
    ...     phi_soi=0.0,
    ...     interferers_data=interferers,
    ... )
    >>> w_int.shape
    (3,)
    """
    num_sensors = positions.shape[0]

    # ------------------------------------------------------------------
    # Build the analytical covariance matrix R_xx
    # ------------------------------------------------------------------

    # SOI contribution: σ_s² * a_s * a_sᴴ
    a_soi = steering_vector(theta_soi, phi_soi, wavelength, positions)
    R_xx  = signal_power * np.outer(a_soi, a_soi.conj())

    # Interferer contributions: Σ_i σ_i² * a_i * a_iᴴ
    if interferers_data is not None:
        for source in interferers_data:
            a_int = steering_vector(
                source['theta'], source['phi'], source['wavelength'], positions
            )
            R_xx += source['power'] * np.outer(a_int, a_int.conj())

    # Noise term: σ_n² * I  (spatially white, uncorrelated)
    R_xx += noise_power * np.eye(num_sensors)

    # ------------------------------------------------------------------
    # Solve for w_opt = R_xx⁻¹ a_s / (a_sᴴ R_xx⁻¹ a_s)
    # ------------------------------------------------------------------
    # numpy.linalg.solve(A, b) computes A⁻¹ b without forming the inverse.
    R_inv_a = np.linalg.solve(R_xx, a_soi)

    # Normalisation scalar: a_sᴴ R_xx⁻¹ a_s  (real-valued and positive)
    norm = np.vdot(a_soi, R_inv_a)

    # Guard against numerical zero (should not happen with noise floor)
    w = R_inv_a / (norm + np.finfo(float).eps)

    # ------------------------------------------------------------------
    # Optional plot
    # ------------------------------------------------------------------
    if plots:
        label = plot_kwargs.get('weights_labels', ['Theoretical'])[0]
        interferer_directions = [(i['theta'], i['phi']) for i in (interferers_data or [])]

        plotter.polar(
            positions      = positions,
            models_data    = [{'weights': w, 'label': label}],
            wavelength     = wavelength,
            theta_cut      = theta_soi,
            phi_cut        = phi_soi,
            soi_directions = [(theta_soi, phi_soi)],
            interferers    = interferer_directions,
            **plot_kwargs,
        )

    return w


# =============================================================================
# Estimated (sample covariance) solver
# =============================================================================

def estimated(
    positions: np.ndarray,
    theta_soi: float,
    phi_soi: float,
    signal_power: float = 1.0,
    noise_power: float = 1.0,
    wavelength: float = 1.0,
    interferers_data: list[dict] = None,
    snapshots: int = 1000,
    max_iter: int = None,
    epsilon: float = 1e-9,
    plots: bool = False,
    **plot_kwargs,
) -> np.ndarray:
    """
    Estimate MPDR weights from simulated snapshots using Conjugate Gradient.

    This solver simulates a realistic received signal, estimates the
    sample covariance matrix from the snapshots, and then applies the
    **Conjugate Gradient (CG)** algorithm to solve:

    .. math::

        \\hat{\\mathbf{R}}_{xx}\\, \\mathbf{h} = \\mathbf{a}_s

    without ever computing the matrix inverse explicitly.  The solution
    **h** is then normalised to enforce the distortionless constraint:

    .. math::

        \\mathbf{w} = \\frac{\\mathbf{h}}{\\mathbf{a}_s^H \\mathbf{h}}

    Signal model
    ------------
    The received signal matrix **X** ∈ ℂᴺˣᴷ (N sensors, K snapshots) is:

    .. math::

        \\mathbf{X} = \\mathbf{a}_s s[k]
                    + \\sum_i \\mathbf{a}_i s_i[k]
                    + \\mathbf{N}

    where s[k] and s_i[k] are i.i.d. n-PSK symbols and **N** is AWGN.
    The sample covariance matrix is:

    .. math::

        \\hat{\\mathbf{R}}_{xx} = \\frac{1}{K} \\mathbf{X} \\mathbf{X}^H

    Conjugate Gradient iteration
    ----------------------------
    Starting from **h**₀ = **0**, residual **r**₀ = **a**_s, and search
    direction **p**₀ = **r**₀, each CG step performs:

    1. α = (rᵢᴴ rᵢ) / (pᵢᴴ R̂ pᵢ)
    2. **h**ᵢ₊₁ = **h**ᵢ + α **p**ᵢ
    3. **r**ᵢ₊₁ = **r**ᵢ − α R̂ **p**ᵢ
    4. β = (rᵢ₊₁ᴴ rᵢ₊₁) / (rᵢᴴ rᵢ)
    5. **p**ᵢ₊₁ = **r**ᵢ₊₁ + β **p**ᵢ

    The iteration stops when ‖**r**ᵢ‖ < ``epsilon`` or after ``max_iter``
    steps (default: N, the number of sensors, which guarantees convergence
    for a positive-definite system in exact arithmetic).

    Parameters
    ----------
    positions : np.ndarray of shape (N, 3)
        Cartesian (x, y, z) coordinates of the N sensor elements.
    theta_soi : float
        Elevation angle of the SOI in degrees.
    phi_soi : float
        Azimuth angle of the SOI in degrees.
    signal_power : float, optional
        Linear power of the SOI signal.  Default is 1.0.
    noise_power : float, optional
        Noise variance σ² at each sensor.  Default is 1.0.
    wavelength : float, optional
        Signal wavelength λ.  Default is 1.0.
    interferers_data : list of dict, optional
        Same format as in ``theoretical``.  Defaults to ``None``.
    snapshots : int, optional
        Number of time samples K used to estimate R̂_xx.  Larger values
        give a better covariance estimate.  Default is 1000.
    max_iter : int, optional
        Maximum number of CG iterations.  Defaults to N (number of
        sensors), which is the theoretical maximum needed for exact
        convergence of CG on a positive-definite system.
    epsilon : float, optional
        Convergence threshold on the residual norm ‖**r**ᵢ‖.
        Default is 1e-9.
    plots : bool, optional
        If ``True``, generates a polar beam-pattern plot.  Default is
        ``False``.
    **plot_kwargs
        Extra keyword arguments forwarded to ``plotter.polar``.

    Returns
    -------
    np.ndarray of shape (N,)
        Estimated complex beamformer weight vector.

    Examples
    --------
    >>> import numpy as np
    >>> positions = np.array([[i * 0.5, 0, 0] for i in range(4)])
    >>> w = estimated(
    ...     positions=positions,
    ...     theta_soi=80.0,
    ...     phi_soi=20.0,
    ...     snapshots=500,
    ... )
    >>> w.shape
    (4,)
    """
    num_sensors = positions.shape[0]

    if max_iter is None:
        max_iter = num_sensors

    # ------------------------------------------------------------------
    # Simulate the received signal matrix X  (N × K)
    # ------------------------------------------------------------------
    a_soi = steering_vector(theta_soi, phi_soi, wavelength, positions)

    # SOI: unit-power n-PSK symbols scaled to sqrt(signal_power)
    s_soi    = dsp.npsk(snapshots, n=4) * np.sqrt(signal_power)
    x_received = a_soi[:, np.newaxis] * s_soi  # shape (N, K)

    # Interferer contributions
    if interferers_data is not None:
        for source in interferers_data:
            a_int = steering_vector(
                source['theta'], source['phi'], source['wavelength'], positions
            )
            s_int = dsp.npsk(snapshots, n=4) * np.sqrt(source['power'])
            x_received += a_int[:, np.newaxis] * s_int

    # Additive white Gaussian noise
    x_received += dsp.awgn((num_sensors, snapshots), power=noise_power)

    # ------------------------------------------------------------------
    # Sample covariance matrix  R̂_xx = (1/K) X Xᴴ
    # ------------------------------------------------------------------
    R_hat = (x_received @ x_received.conj().T) / snapshots

    # ------------------------------------------------------------------
    # Conjugate Gradient: solve  R̂_xx h = a_soi
    # ------------------------------------------------------------------
    h = np.zeros(num_sensors, dtype=complex)   # h₀ = 0
    r = a_soi.copy()                           # r₀ = a_soi − R̂ h₀ = a_soi
    p = r.copy()                               # p₀ = r₀

    r_norm_sq_old = np.vdot(r, r)  # rᴴ r  (a real, non-negative scalar)

    for _ in range(max_iter):
        Rp    = R_hat @ p
        alpha = r_norm_sq_old / (np.vdot(p, Rp) + np.finfo(float).eps)

        h += alpha * p
        r -= alpha * Rp

        r_norm_sq_new = np.vdot(r, r)

        if np.sqrt(r_norm_sq_new.real) < epsilon:
            break

        beta          = r_norm_sq_new / r_norm_sq_old
        p             = r + beta * p
        r_norm_sq_old = r_norm_sq_new

    # ------------------------------------------------------------------
    # Normalise to enforce distortionless constraint: aᴴ w = 1
    # ------------------------------------------------------------------
    w = h / (np.vdot(a_soi, h) + np.finfo(float).eps)

    # ------------------------------------------------------------------
    # Optional plot
    # ------------------------------------------------------------------
    if plots:
        label = plot_kwargs.get('weights_labels', ['Estimated (CG)'])[0]
        interferer_directions = [(i['theta'], i['phi']) for i in (interferers_data or [])]

        plotter.polar(
            positions      = positions,
            models_data    = [{'weights': w, 'label': label}],
            wavelength     = wavelength,
            theta_cut      = theta_soi,
            phi_cut        = phi_soi,
            soi_directions = [(theta_soi, phi_soi)],
            interferers    = interferer_directions,
            **plot_kwargs,
        )

    return w


# =============================================================================
# Distributed solver (Average Consensus + CG)
# =============================================================================

def distributed(
    positions: list[np.ndarray],
    theta_soi: float,
    phi_soi: float,
    signal_power: float = 1.0,
    noise_power: float = 1.0,
    wavelength: float = 1.0,
    interferers_data: list[dict] = None,
    snapshots: int = 1000,
    max_iter: int = None,
    epsilon: float = 1e-9,
    adjacency_matrix: np.ndarray = None,
    plots: bool = False,
    **plot_kwargs,
) -> np.ndarray:
    """
    Distributed MPDR beamformer via Average Consensus and Conjugate Gradient.

    This solver targets a network of P sensor nodes that cannot share raw
    data.  Each node p holds its local received signal **X**_p ∈ ℂᴷᵖˣᴷ
    and local steering vector **a**_p ∈ ℂᴷᵖ (where K_p is the number of
    sensors at node p).  The global CG iteration is executed in a
    *distributed* fashion: the scalar quantities that CG requires are
    computed as network averages via the **Average Consensus** protocol.

    Distributed CG formulation
    --------------------------
    The global system to be solved is R̂_xx **h** = **a**_s, where:

    .. math::

        \\hat{\\mathbf{R}}_{xx} = \\frac{1}{K} \\mathbf{X} \\mathbf{X}^H,
        \\quad
        \\mathbf{X} = \\begin{bmatrix} \\mathbf{X}_1 \\\\ \\vdots \\\\
        \\mathbf{X}_P \\end{bmatrix}

    The key observation is that the matrix-vector product R̂ **p** can be
    decomposed as:

    .. math::

        \\mathbf{q}_p = \\frac{1}{K} \\mathbf{X}_p
            \\underbrace{\\left( \\mathbf{X}^H \\mathbf{p}
            \\right)}_{\\mathbf{t}},
        \\qquad
        \\mathbf{t}[k] = \\sum_{p=1}^{P}
            \\mathbf{x}_p[k]^H \\mathbf{p}_p

    The global vector **t** ∈ ℂᴷ (one scalar per snapshot) is computed
    by running Average Consensus on the local inner products
    {**x**_p[k]ᴴ **p**_p} at each snapshot k, then scaling by P.

    The remaining CG scalars (αᵢ numerator, pᵢᴴ Rp denominator, βᵢ) are
    likewise obtained by consensus on locally computable quantities, so
    every node executes identical CG steps using only its own data and
    the consensus outputs.

    Network topology
    ----------------
    The ``adjacency_matrix`` defines which nodes can communicate.  The
    Average Consensus algorithm (``dsp.average_consensus``) converges in
    exactly R − 1 steps, where R is the number of distinct eigenvalues of
    the graph Laplacian.  A denser graph (more edges) reduces R and
    therefore speeds up consensus at the cost of communication overhead.

    Parameters
    ----------
    positions : list of np.ndarray
        List of P arrays, each of shape (K_p, 3), containing the Cartesian
        sensor positions for node p.
    theta_soi : float
        Elevation angle of the SOI in degrees.
    phi_soi : float
        Azimuth angle of the SOI in degrees.
    signal_power : float, optional
        Linear power of the SOI signal.  Default is 1.0.
    noise_power : float, optional
        Noise variance at each sensor.  Default is 1.0.
    wavelength : float, optional
        Signal wavelength λ.  Default is 1.0.
    interferers_data : list of dict, optional
        Same format as in ``theoretical``.  Defaults to ``None``.
    snapshots : int, optional
        Number of time samples K.  Default is 1000.
    max_iter : int, optional
        Maximum number of distributed CG iterations.  Defaults to the
        total number of sensors N = Σ K_p.
    epsilon : float, optional
        Convergence threshold on the global residual energy ‖**r**‖².
        Default is 1e-9.
    adjacency_matrix : np.ndarray of shape (P, P)
        Symmetric binary adjacency matrix of the network.
        ``adjacency_matrix[i, j] = 1`` if nodes i and j can communicate.
    plots : bool, optional
        If ``True``, generates polar and 3-D spatial beam-pattern plots.
        Default is ``False``.
    **plot_kwargs
        Extra keyword arguments forwarded to ``plotter.polar``.

    Returns
    -------
    np.ndarray of shape (N_total,)
        Normalised global weight vector, formed by concatenating the local
        weight vectors of all P nodes (N_total = Σ K_p).

    Examples
    --------
    Two-node network, 2 sensors per node:

    >>> import numpy as np
    >>> node1 = np.array([[0.0, 0, 0], [0.5, 0, 0]])
    >>> node2 = np.array([[1.0, 0, 0], [1.5, 0, 0]])
    >>> A = np.array([[0, 1], [1, 0]])
    >>> w = distributed(
    ...     positions=[node1, node2],
    ...     adjacency_matrix=A,
    ...     theta_soi=90.0,
    ...     phi_soi=0.0,
    ... )
    >>> w.shape   # total sensors = 2 + 2 = 4
    (4,)
    """
    P    = len(positions)                          # number of nodes
    K_p  = [pos.shape[0] for pos in positions]    # sensors per node
    K    = sum(K_p)                                # total sensors

    if max_iter is None:
        max_iter = K

    # ------------------------------------------------------------------
    # Local steering vectors
    # ------------------------------------------------------------------
    a_p_soi = [
        steering_vector(theta_soi, phi_soi, wavelength, p_local)
        for p_local in positions
    ]

    # ------------------------------------------------------------------
    # Simulate local received signals  X_p  (K_p × snapshots)
    # ------------------------------------------------------------------
    # All nodes observe the same SOI and interferer symbol sequences —
    # only the steering vectors (and hence the phase shifts) differ.
    s_soi = dsp.npsk(snapshots, n=4) * np.sqrt(signal_power)

    if interferers_data:
        s_interferers = [
            dsp.npsk(snapshots, n=4) * np.sqrt(src['power'])
            for src in interferers_data
        ]

    x_p_list = []
    for p_idx, p_pos in enumerate(positions):
        x_local = a_p_soi[p_idx][:, np.newaxis] * s_soi  # SOI component

        if interferers_data:
            for i, s_int in enumerate(s_interferers):
                src   = interferers_data[i]
                a_int = steering_vector(
                    src['theta'], src['phi'], src['wavelength'], p_pos
                )
                x_local += a_int[:, np.newaxis] * s_int

        x_local += dsp.awgn((K_p[p_idx], snapshots), power=noise_power)
        x_p_list.append(x_local)

    # ------------------------------------------------------------------
    # Distributed CG initialisation
    # ------------------------------------------------------------------
    # Each node p maintains local copies of the CG variables:
    #   w_p  — local weight vector (contributes to the global w)
    #   r_p  — local residual slice
    #   p_p  — local search direction slice

    w_p = [np.zeros(k, dtype=complex) for k in K_p]
    r_p = [a_local.copy()             for a_local in a_p_soi]
    p_p = [r_local.copy()             for r_local in r_p]

    # Global residual energy: rᴴ r = P * AC( {rₚᴴ rₚ} )
    r_norm_sq_local = np.array([np.vdot(r, r).real for r in r_p])
    r_norm_sq_old   = P * dsp.average_consensus(adjacency_matrix, r_norm_sq_local)

    # ------------------------------------------------------------------
    # Distributed CG iterations
    # ------------------------------------------------------------------
    for _ in range(max_iter):

        # --- Step 1: compute global projection vector t ∈ ℂᴷ ---
        # t[k] = P * AC( {x_p[k]ᴴ p_p} )   for each snapshot k
        t = np.zeros(snapshots, dtype=complex)
        for k in range(snapshots):
            local_ip = np.array([
                np.vdot(x_p[:, k], pp)
                for x_p, pp in zip(x_p_list, p_p)
            ])
            t[k] = P * dsp.average_consensus(adjacency_matrix, local_ip)

        # --- Step 2: local matrix-vector product q_p = (1/K) X_p t ---
        q_p = [(x_p @ t) / snapshots for x_p in x_p_list]

        # --- Step 3: denominator for α — pᴴ R̂ p via consensus ---
        pq_local  = np.array([np.vdot(pp, qp).real for pp, qp in zip(p_p, q_p)])
        pq_global = P * dsp.average_consensus(adjacency_matrix, pq_local)

        alpha = r_norm_sq_old / (pq_global + np.finfo(float).eps)

        # --- Step 4: update local weights and residuals ---
        for i in range(P):
            w_p[i] += alpha * p_p[i]
            r_p[i] -= alpha * q_p[i]

        # --- Step 5: new global residual energy ---
        r_norm_sq_local = np.array([np.vdot(r, r).real for r in r_p])
        r_norm_sq_new   = P * dsp.average_consensus(adjacency_matrix, r_norm_sq_local)

        if r_norm_sq_new < epsilon:
            break

        # --- Step 6: update search directions ---
        beta = r_norm_sq_new / r_norm_sq_old
        for i in range(P):
            p_p[i] = r_p[i] + beta * p_p[i]

        r_norm_sq_old = r_norm_sq_new

    # ------------------------------------------------------------------
    # Assemble and normalise the global weight vector
    # ------------------------------------------------------------------
    w_global = np.concatenate(w_p)
    a_global = np.concatenate(a_p_soi)

    # Enforce distortionless constraint: aᴴ w = 1
    w_normalised = w_global / (np.vdot(a_global, w_global) + np.finfo(float).eps)

    # ------------------------------------------------------------------
    # Optional plots
    # ------------------------------------------------------------------
    if plots:
        label = plot_kwargs.get('weights_labels', ['Distributed'])[0]
        all_positions    = np.vstack(positions)
        interferer_dirs  = [(i['theta'], i['phi']) for i in (interferers_data or [])]

        plotter.polar(
            positions      = all_positions,
            models_data    = [{'weights': w_normalised, 'label': label}],
            wavelength     = wavelength,
            theta_cut      = theta_soi,
            phi_cut        = phi_soi,
            soi_directions = [(theta_soi, phi_soi)],
            interferers    = interferer_dirs,
            **plot_kwargs,
        )

    return w_normalised
