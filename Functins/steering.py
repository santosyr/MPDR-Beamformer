"""
steering.py
===========
Steering vector computation for arbitrary 3-D sensor array geometries.

Background
----------
A steering vector **a**(θ, φ) encodes the phase shift experienced at each
sensor of an array when a far-field, narrowband plane wave arrives from
direction (θ, φ).  For sensor at position **p**_n the phase shift is:

    ψ_n = exp(-j **k** · **p**_n)

where **k** is the wave vector:

    **k** = (2π/λ) * [sin θ cos φ,  sin θ sin φ,  cos θ]ᵀ

and λ is the signal wavelength.

Angle convention
----------------
This module uses the **physics / ISO 80000-2** spherical coordinate system:

    θ  (theta) — polar / elevation angle, measured from the +z axis.
                 Range [0°, 180°].  θ = 0° → boresight along +z.
    φ  (phi)   — azimuth angle, measured from the +x axis toward +y.
                 Range [0°, 360°).

Sign convention
---------------
The phase factor uses **exp(-j ψ)**, i.e. a wave arriving at sensor n
*leads* the reference at the origin by ψ_n > 0.  This is consistent with
the convention used throughout the MPDR solver (``mpdr.py``).
"""

import numpy as np

def steering_vector(
    theta_soi: float | np.ndarray,
    phi_soi: float | np.ndarray,
    wavelength: float,
    positions: np.ndarray,
) -> np.ndarray:
    """
    Compute the steering vector for an arbitrary 3-D sensor array.

    The steering vector **a**(θ, φ) ∈ ℂᴺ is defined as::

        a_n(θ, φ) = exp(-j (2π/λ) p_n · û)

    where **p**_n is the position of the n-th sensor, λ is the wavelength,
    and **û** = [sin θ cos φ, sin θ sin φ, cos θ]ᵀ is the unit vector
    pointing in the direction of arrival (DOA).

    Parameters
    ----------
    theta_soi : float or array-like of shape (M,)
        Elevation angle(s) in degrees.  Measured from the +z axis (zenith).
        Scalar or 1-D array.  Valid range: [0°, 180°].
    phi_soi : float or array-like of shape (L,)
        Azimuth angle(s) in degrees.  Measured from the +x axis toward +y.
        Scalar or 1-D array.  Valid range: [0°, 360°).
    wavelength : float
        Wavelength λ of the incoming signal.  Must be positive and in the
        same unit system as ``positions``.
    positions : np.ndarray of shape (N, 3)
        Cartesian coordinates (x, y, z) of the N sensor elements.

    Returns
    -------
    np.ndarray
        Steering vector(s).  Shape depends on the input angles:

        * **Scalar** ``theta_soi`` and ``phi_soi`` → shape **(N,)**.
        * **Array** inputs of shapes (M,) and (L,) → shape **(N, L, M)**.
          Axis 0 indexes sensors, axis 1 indexes φ values, axis 2
          indexes θ values.

    Notes
    -----
    * For a Uniform Linear Array (ULA) along x with half-wavelength
      spacing (d = λ/2), the classical steering vector reduces to::

          a_n = exp(-j π n sin θ cos φ)

    * The phase convention exp(-j ψ) means that a sensor further from
      the origin in the DOA direction has a *smaller* (more negative)
      phase — i.e. the wave arrives earlier there.

    * When ``theta_soi`` and ``phi_soi`` are arrays, the function builds
      a meshgrid internally (φ along axis 0 of the grid, θ along axis 1)
      to enable fully vectorised computation without Python loops.

    Examples
    --------
    Single direction (scalar inputs) — returns shape (N,):

    >>> import numpy as np
    >>> positions = np.array([[0.0, 0, 0], [0.5, 0, 0], [1.0, 0, 0]])
    >>> a = steering_vector(theta_soi=90.0, phi_soi=0.0,
    ...                     wavelength=1.0, positions=positions)
    >>> a.shape
    (3,)
    >>> np.abs(a)          # unit-magnitude entries
    array([1., 1., 1.])

    Grid of angles — returns shape (N, L, M):

    >>> theta_angles = np.array([0.0, 45.0, 90.0])   # M = 3
    >>> phi_angles   = np.array([0.0, 90.0])          # L = 2
    >>> A = steering_vector(theta_soi=theta_angles, phi_soi=phi_angles,
    ...                     wavelength=1.0, positions=positions)
    >>> A.shape
    (3, 2, 3)

    Verifying boresight (θ = 0°): all sensors have zero phase shift
    because the wave travels along z, perpendicular to the ULA:

    >>> a_bore = steering_vector(0.0, 0.0, 1.0, positions)
    >>> np.allclose(a_bore, np.ones(3))
    True
    """
    # ------------------------------------------------------------------
    # Wave number magnitude
    # ------------------------------------------------------------------
    k = 2.0 * np.pi / wavelength

    # ------------------------------------------------------------------
    # Scalar case — both angles are single values
    # ------------------------------------------------------------------
    if np.isscalar(theta_soi) and np.isscalar(phi_soi):
        theta_rad = np.radians(theta_soi)
        phi_rad   = np.radians(phi_soi)

        # Unit direction-of-arrival vector û (Cartesian)
        k_vector = k * np.array([
            np.sin(theta_rad) * np.cos(phi_rad),   # x-component
            np.sin(theta_rad) * np.sin(phi_rad),   # y-component
            np.cos(theta_rad),                     # z-component
        ])

        # Phase shifts: ψ_n = k_vector · p_n,  shape (N,)
        phase_shifts = positions @ k_vector  # equivalent to np.dot(positions, k_vector)

    # ------------------------------------------------------------------
    # Vectorised case — at least one angle is an array
    # ------------------------------------------------------------------
    else:
        theta_rad = np.asarray(np.radians(theta_soi))   # shape (M,) or scalar
        phi_rad   = np.asarray(np.radians(phi_soi))     # shape (L,) or scalar

        # Build a 2-D angular grid: phi along axis 0, theta along axis 1.
        # theta_grid and phi_grid each have shape (L, M).
        theta_grid, phi_grid = np.meshgrid(theta_rad, phi_rad)

        # Wave-vector field: shape (L, M, 3)
        k_vector = np.stack([
            k * np.sin(theta_grid) * np.cos(phi_grid),  # k_x
            k * np.sin(theta_grid) * np.sin(phi_grid),  # k_y
            k * np.cos(theta_grid),                     # k_z
        ], axis=-1)

        # Phase shifts via tensor contraction:
        #   positions: (N, 3),  k_vector: (L, M, 3)
        #   result: (N, L, M)
        phase_shifts = np.tensordot(positions, k_vector, axes=([1], [-1]))

    return np.exp(-1j * phase_shifts)
