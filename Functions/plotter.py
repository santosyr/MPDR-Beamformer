"""
plotter.py
==========
Visualisation utilities for MPDR beamformer results.

Four plot types are provided:

``topology``
    3-D scatter plot of sensor positions, with an optional SOI direction vector.

``linear``
    Cartesian (x-y) beam pattern: normalised gain in dB vs. scan angle.

``polar``
    Polar beam pattern: normalised gain in dB plotted on a polar axis.
    Supports single-beam and multi-beam weight matrices.

``spatial``
    3-D surface plot of the full spherical beam pattern.

All functions display plots interactively via Matplotlib and optionally
export the computed pattern data to a tab-separated text file.

Notes
-----
The module sets the Matplotlib backend to ``QtAgg`` at import time.
If Qt is not available in your environment, change the backend at the top
of this file (e.g. ``matplotlib.use('TkAgg')`` or ``'Agg'`` for
non-interactive use).
"""

import matplotlib

def _set_best_backend() -> None:
    """
    Select the best available interactive Matplotlib backend automatically.

    The function tries backends in preference order and silently falls back
    to the next candidate if a binding is missing.  ``'Agg'`` (non-
    interactive, renders to memory) is the final fallback so that the
    module always imports cleanly even in headless environments.

    Preference order
    ----------------
    1. ``QtAgg``   — requires PyQt6, PySide6, PyQt5, or PySide2.
    2. ``TkAgg``   — requires Tkinter (ships with most Python installers).
    3. ``WXAgg``   — requires wxPython.
    4. ``Agg``     — non-interactive; ``plt.show()`` is a no-op.
    """
    for backend in ('QtAgg', 'TkAgg', 'WXAgg', 'Agg'):
        try:
            matplotlib.use(backend)
            # Force Matplotlib to actually load the backend now so that
            # any ImportError surfaces here rather than at plot time.
            import matplotlib.pyplot as _plt  # noqa: F401
            break
        except Exception:
            continue

_set_best_backend()

import numpy             as np
import pandas            as pd
import matplotlib.cm     as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .steering import steering_vector


# =============================================================================
# Array topology
# =============================================================================

def topology(
    positions: np.ndarray,
    title: str = "Array Topology",
    soi_direction: tuple[float, float] | None = None,
) -> None:
    """
    Visualise the 3-D geometry of a sensor array.

    Plots each sensor as a point in 3-D space.  Optionally overlays
    coordinate-axis arrows and a unit vector pointing toward the
    Signal of Interest (SOI).

    Parameters
    ----------
    positions : np.ndarray of shape (N, 3)
        Cartesian (x, y, z) coordinates of the N sensor elements.
    title : str, optional
        Figure title.  Default is ``"Array Topology"``.
    soi_direction : tuple[float, float] or None, optional
        If provided as ``(theta, phi)`` in degrees, draws a red arrow from
        the origin toward the specified direction.  Uses the same
        angle convention as ``steering.py`` (θ from +z, φ from +x).
        Default is ``None``.

    Returns
    -------
    None
        Displays the plot interactively.  Call ``plt.show()`` afterward
        if running outside an interactive session.

    Examples
    --------
    >>> import numpy as np
    >>> positions = np.array([[i * 0.5, 0, 0] for i in range(6)])
    >>> topology(positions, soi_direction=(90.0, 45.0))
    """
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')

    ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        c='b', marker='o', s=50, label="Array Elements",
    )

    if soi_direction is not None:
        theta_soi, phi_soi = np.deg2rad(soi_direction)

        # Scale the SOI arrow to 80 % of the largest array dimension
        max_dim = np.ptp(positions, axis=0).max()
        r       = max_dim * 0.8

        x_soi = r * np.sin(theta_soi) * np.cos(phi_soi)
        y_soi = r * np.sin(theta_soi) * np.sin(phi_soi)
        z_soi = r * np.cos(theta_soi)

        ax.quiver(
            0, 0, 0, x_soi, y_soi, z_soi,
            color='r', lw=2, arrow_length_ratio=0.1, label="SOI Direction",
        )

        # Coordinate-axis reference arrows
        L = max_dim * 0.8
        for vec, lbl in zip([(L,0,0), (0,L,0), (0,0,L)], ['x', 'y', 'z']):
            ax.quiver(0, 0, 0, *vec, arrow_length_ratio=0.05, color='k')
            ax.text(*vec, lbl)

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal')
    fig.tight_layout()


# =============================================================================
# Cartesian (linear) beam pattern
# =============================================================================

def linear(
    positions: np.ndarray,
    models_data: list[dict],
    wavelength: float,
    theta_cut: float,
    phi_cut: float,
    soi_directions: list[tuple[float, float]] | None = None,
    resolution: int = 3600,
    cut_plane: str = 'phi',
    xlim: tuple[float, float] = (0, 360),
    ylim: tuple[float, float] = (-50, 0),
    interferers: list[tuple[float, float]] | None = None,
    export_filepath: str | None = None,
    figsize: tuple[float, float] = (12, 7),
) -> None:
    """
    Plot one or more beam patterns on a Cartesian (linear) chart.

    Displays the normalised beam response in dB as a function of the
    scan angle for a fixed cut plane.  Multiple weight vectors can be
    overlaid for direct comparison.

    Cut-plane convention
    --------------------
    * ``cut_plane='phi'``   — fixes φ = ``phi_cut`` and sweeps θ ∈ [0°, 360°].
    * ``cut_plane='theta'`` — fixes θ = ``theta_cut`` and sweeps φ ∈ [0°, 360°].

    Parameters
    ----------
    positions : np.ndarray of shape (N, 3)
        Cartesian sensor coordinates.
    models_data : list of dict
        Each dictionary defines one beamformer to plot.  Required keys:

        * ``'weights'`` (np.ndarray of shape (N,)) — complex weight vector.
        * ``'label'``   (str) — legend entry.

        Optional keys: ``'color'`` (str), ``'linestyle'`` (str).
    wavelength : float
        Signal wavelength λ.
    theta_cut : float
        Elevation angle (degrees) used as the fixed value when
        ``cut_plane='theta'``, and as the SOI marker position when
        ``cut_plane='phi'``.
    phi_cut : float
        Azimuth angle (degrees) used as the fixed value when
        ``cut_plane='phi'``, and as the SOI marker position when
        ``cut_plane='theta'``.
    soi_directions : list of (theta, phi), optional
        SOI directions to mark with a vertical dashed line.  Only
        directions whose fixed-plane angle matches ``theta_cut`` or
        ``phi_cut`` (within floating-point tolerance) are drawn.
    resolution : int, optional
        Number of angular points in the sweep.  Default is 3600.
    cut_plane : {'phi', 'theta'}, optional
        Which angle to hold fixed.  Default is ``'phi'``.
    xlim : tuple[float, float], optional
        x-axis limits in degrees.  Default is ``(0, 360)``.
    ylim : tuple[float, float], optional
        y-axis limits in dB.  Default is ``(-50, 0)``.
    interferers : list of (theta, phi), optional
        Interferer directions to mark with vertical red dashed lines.
    export_filepath : str or None, optional
        If provided, saves the angle and dB arrays to a tab-separated
        file at this path.  Default is ``None``.
    figsize : tuple[float, float], optional
        Matplotlib figure size.  Default is ``(12, 7)``.

    Returns
    -------
    None
        Displays the plot interactively.

    Examples
    --------
    >>> import numpy as np
    >>> positions = np.array([[i * 0.5, 0, 0] for i in range(8)])
    >>> w = np.ones(8) / 8  # simple uniform weights
    >>> linear(
    ...     positions=positions,
    ...     models_data=[{'weights': w, 'label': 'Uniform'}],
    ...     wavelength=1.0,
    ...     theta_cut=90.0,
    ...     phi_cut=0.0,
    ... )
    """
    if cut_plane == 'phi':
        angles_deg = np.linspace(0, 360, resolution)
        x_label    = rf"Elevation angle θ (°)   [slice at φ = {phi_cut:.1f}°]"
    elif cut_plane == 'theta':
        angles_deg = np.linspace(0, 360, resolution)
        x_label    = rf"Azimuth angle φ (°)   [slice at θ = {theta_cut:.1f}°]"
    else:
        raise ValueError("cut_plane must be 'phi' or 'theta'.")

    # Compute vectorised steering matrix for the full angular sweep
    if cut_plane == 'phi':
        S = np.squeeze(steering_vector(angles_deg, phi_cut,   wavelength, positions))
    else:
        S = np.squeeze(steering_vector(theta_cut,  angles_deg, wavelength, positions))
    # S has shape (N, resolution)

    export_data = {"angle_deg": angles_deg}

    fig, ax = plt.subplots(figsize=figsize)

    for model in models_data:
        w   = model['weights']
        lbl = model['label']
        clr = model.get('color')
        ls  = model.get('linestyle', '-')

        resp = np.abs(w.conj() @ S)
        resp /= np.max(resp) + 1e-12
        db   = 20.0 * np.log10(resp + 1e-12)

        ax.plot(angles_deg, db, label=lbl, color=clr, linestyle=ls, linewidth=2)

        safe_lbl          = lbl.replace(" ", "_").replace("$", "").replace("\\", "")
        export_data[safe_lbl] = db

    # SOI markers
    plotted_soi = False
    if soi_directions:
        for th_s, ph_s in soi_directions:
            in_plane = (
                (cut_plane == 'phi'   and np.isclose(ph_s, phi_cut,   atol=1e-8))
                or (cut_plane == 'theta' and np.isclose(th_s, theta_cut, atol=1e-8))
            )
            if in_plane:
                x_val = th_s if cut_plane == 'phi' else ph_s
                lbl   = 'SOI' if not plotted_soi else '_nolegend_'
                ax.axvline(x_val, color='darkgreen', linestyle='-.', lw=2, label=lbl)
                plotted_soi = True

    # Interferer markers
    plotted_int = False
    if interferers:
        for th_i, ph_i in interferers:
            in_plane = (
                (cut_plane == 'phi'   and np.isclose(ph_i, phi_cut,   atol=1e-8))
                or (cut_plane == 'theta' and np.isclose(th_i, theta_cut, atol=1e-8))
            )
            if in_plane:
                x_val = th_i if cut_plane == 'phi' else ph_i
                lbl   = 'Interferer' if not plotted_int else '_nolegend_'
                ax.axvline(x_val, color='darkred', linestyle='--', lw=2, label=lbl)
                plotted_int = True

    ax.set_xlabel(x_label)
    ax.set_ylabel("Normalised Magnitude (dB)")
    ax.set_title("Cartesian Beam Pattern")
    ax.grid(True, linestyle=':')

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ncol = (
        len(models_data)
        + (1 if plotted_soi else 0)
        + (1 if plotted_int else 0)
    )
    if ncol > 0:
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.15),
            ncol=ncol,
            frameon=False,
        )

    if export_filepath:
        pd.DataFrame(export_data).to_csv(
            export_filepath, sep='\t', index=False, float_format="%.6f"
        )
        print(f"Linear pattern data exported to: {export_filepath}")


# =============================================================================
# Polar beam pattern
# =============================================================================

def polar(
    positions: np.ndarray | list[np.ndarray],
    models_data: list[dict],
    wavelength: float,
    theta_cut: float,
    phi_cut: float,
    soi_directions: list[tuple[float, float]] | None = None,
    resolution: int = 1080,
    cut_plane: str = 'phi',
    r_lim: tuple[float, float] | None = None,
    interferers: list[tuple[float, float]] | None = None,
    export_filepath: str | None = None,
    figsize: tuple[float, float] = (12, 8),
) -> None:
    """
    Plot polar beam patterns for one or more beamformer weight vectors.

    Displays normalised gain (dB) on a polar axis as the scan angle sweeps
    360°.  Supports single-beam (shape (N,)) and multi-beam (shape (N, B))
    weight matrices, and accepts different sensor position arrays for each
    model (e.g. when comparing centralised vs. distributed arrays).

    Cut-plane convention
    --------------------
    Same as ``linear``:

    * ``cut_plane='phi'``   — fixes φ = ``phi_cut``, sweeps θ.
    * ``cut_plane='theta'`` — fixes θ = ``theta_cut``, sweeps φ.

    Parameters
    ----------
    positions : np.ndarray of shape (N, 3) or list of np.ndarray
        Sensor positions.  If a single array is provided, it is used for
        all entries in ``models_data``.  If a list is provided, its length
        must equal ``len(models_data)``.
    models_data : list of dict
        Each dictionary defines one beamformer.  Required keys:

        * ``'weights'`` (np.ndarray, shape (N,) or (N, B)) — weight vector
          or matrix (B beams).
        * ``'label'``   (str) — base legend label.

        Optional keys: ``'color'`` (str, used for single-beam only),
        ``'linestyle'`` (str).
    wavelength : float
        Signal wavelength λ.
    theta_cut : float
        Elevation cut angle in degrees.
    phi_cut : float
        Azimuth cut angle in degrees.
    soi_directions : list of (theta, phi), optional
        SOI directions to mark with radial dashed lines.
    resolution : int, optional
        Number of angular points.  Default is 1080.
    cut_plane : {'phi', 'theta'}, optional
        Fixed-angle plane.  Default is ``'phi'``.
    r_lim : tuple[float, float] or None, optional
        Radial (dB) axis limits ``(r_min, r_max)``.  If ``None``, the
        limits are set automatically (r_max = 0 dB, r_min = max of −60 dB
        and the global minimum response).
    interferers : list of (theta, phi), optional
        Interferer directions to mark with radial red dashed lines.
    export_filepath : str or None, optional
        If provided, saves pattern data as a tab-separated file.
    figsize : tuple[float, float], optional
        Matplotlib figure size.  Default is ``(12, 8)``.

    Returns
    -------
    None
        Displays the plot interactively.

    Examples
    --------
    >>> import numpy as np
    >>> positions = np.array([[i * 0.5, 0, 0] for i in range(8)])
    >>> w = np.ones(8) / 8
    >>> polar(
    ...     positions=positions,
    ...     models_data=[{'weights': w, 'label': 'Uniform'}],
    ...     wavelength=1.0,
    ...     theta_cut=90.0,
    ...     phi_cut=0.0,
    ... )
    """
    # --- Normalise positions input ----------------------------------------
    if not isinstance(positions, list):
        positions_list = [positions] * len(models_data)
    else:
        positions_list = positions
        if len(positions_list) != len(models_data):
            raise ValueError(
                "When 'positions' is a list, its length must equal "
                "len(models_data)."
            )

    # --- Angular sweep ----------------------------------------------------
    ang_deg = np.linspace(0, 360, resolution, endpoint=True)
    ang_rad = np.radians(ang_deg)

    export_data   = {"angle_deg": ang_deg}
    fig, ax       = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=figsize)
    beam_cmap     = plt.get_cmap("tab10")
    global_min_db = 0.0

    # --- Main loop: one entry per model ----------------------------------
    for pos, model in zip(positions_list, models_data):
        w_input    = model['weights']
        base_label = model['label']
        base_color = model.get('color')
        base_ls    = model.get('linestyle', '-')

        # Standardise weights to 2-D: (N, B)
        w_matrix = w_input[:, np.newaxis] if w_input.ndim == 1 else w_input
        num_sensors, num_beams = w_matrix.shape

        # Steering matrix for this model's sensor positions
        if cut_plane == 'phi':
            sv = np.squeeze(steering_vector(ang_deg, phi_cut,   wavelength, pos))
        elif cut_plane == 'theta':
            sv = np.squeeze(steering_vector(theta_cut, ang_deg, wavelength, pos))
        else:
            raise ValueError("cut_plane must be 'phi' or 'theta'.")

        # Inner loop: one curve per beam
        for b in range(num_beams):
            w_vec = w_matrix[:, b]

            resp    = np.abs(w_vec.conj() @ sv)
            max_val = np.max(resp) if np.max(resp) > 0 else 1e-12
            db      = 20.0 * np.log10(resp / max_val + 1e-12)

            global_min_db = min(global_min_db, np.min(db))

            # Style: multi-beam uses colour map; single-beam respects dict colour
            if num_beams > 1:
                final_label = f"{base_label} — Beam {b + 1}"
                color       = beam_cmap(b % 10)
                ls          = base_ls
            else:
                final_label = base_label
                color       = base_color
                ls          = base_ls

            ax.plot(ang_rad, db, color=color, linestyle=ls, linewidth=2, label=final_label)

            safe_lbl              = final_label.replace(" ", "_").replace("$", "").replace("\\", "")
            export_data[safe_lbl] = db

    # --- Axis configuration ----------------------------------------------
    if r_lim is not None:
        rmin, rmax = r_lim
    else:
        rmax = 0.0
        rmin = max(-60.0, global_min_db)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim((rmin, rmax))
    ax.set_rlabel_position(135)

    plane_sym = r'\phi'   if cut_plane == 'phi'   else r'\theta'
    plane_val = phi_cut   if cut_plane == 'phi'   else theta_cut
    ax.set_title(
        f"Beam Pattern  (cut at ${plane_sym} = {plane_val:.1f}^\\circ$)",
        va='bottom',
    )

    # --- Radial marker helper -------------------------------------------
    def _radial_line(angle_deg, color, ls, label_text, already_plotted):
        ang   = np.radians(angle_deg)
        label = label_text if not already_plotted else "_nolegend_"
        ax.plot(
            [ang, ang], [rmin, rmax],
            color=color, linestyle=ls, lw=1.5, alpha=0.7, label=label,
        )
        return True

    # SOI markers
    if soi_directions:
        plotted_soi = False
        for th_s, ph_s in soi_directions:
            match = (
                (cut_plane == 'phi'   and np.isclose(ph_s, phi_cut,   atol=1e-5))
                or (cut_plane == 'theta' and np.isclose(th_s, theta_cut, atol=1e-5))
            )
            if match:
                m_ang       = th_s if cut_plane == 'phi' else ph_s
                plotted_soi = _radial_line(m_ang, 'darkgreen', '-.', 'SOI', plotted_soi)

    # Interferer markers
    if interferers:
        plotted_int = False
        for th_i, ph_i in interferers:
            match = (
                (cut_plane == 'phi'   and np.isclose(ph_i, phi_cut,   atol=1e-5))
                or (cut_plane == 'theta' and np.isclose(th_i, theta_cut, atol=1e-5))
            )
            if match:
                m_ang       = th_i if cut_plane == 'phi' else ph_i
                plotted_int = _radial_line(m_ang, 'darkred', '--', 'Interferer', plotted_int)

    # --- Legend and layout -----------------------------------------------
    handles, labs = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labs, loc='center left', bbox_to_anchor=(0.82, 0.5), frameon=False)
        plt.subplots_adjust(right=0.8)

    plt.tight_layout(rect=[0, 0, 0.8, 1])

    fig.patch.set_facecolor('none')
    ax.patch.set_facecolor('none')

    # --- Export ----------------------------------------------------------
    if export_filepath:
        pd.DataFrame(export_data).to_csv(
            export_filepath, sep='\t', index=False, float_format="%.6f"
        )
        print(f"Polar pattern data exported to: {export_filepath}")


# =============================================================================
# 3-D spatial beam pattern
# =============================================================================

def spatial(
    positions: np.ndarray,
    weights: np.ndarray,
    wavelength: float,
    soi_directions: list[tuple[float, float]] | None = None,
    interferers: list[tuple[float, float]] | None = None,
    resolution: int = 1080,
) -> None:
    """
    Visualise the full 3-D spherical beam pattern.

    Computes the beamformer response over a (θ, φ) grid covering the full
    sphere and maps it onto a 3-D surface, where the radial distance from
    the origin equals the normalised linear gain at each direction.

    Parameters
    ----------
    positions : np.ndarray of shape (N, 3)
        Cartesian sensor coordinates.
    weights : np.ndarray of shape (N,)
        Complex beamformer weight vector.
    wavelength : float
        Signal wavelength λ.
    soi_directions : list of (theta, phi) or None, optional
        SOI directions marked with a dark-green line from the origin.
        Default is ``None``.
    interferers : list of (theta, phi) or None, optional
        Interferer directions marked with a red dashed line from the origin.
        Default is ``None``.
    resolution : int, optional
        Number of points along each angular axis.  Higher values give
        smoother surfaces at the cost of computation time.
        Default is 1080.

    Returns
    -------
    None
        Displays the plot interactively.

    Notes
    -----
    The surface colour encodes the linear (not dB) gain, normalised so
    that the peak response equals 1.  A Jet colormap and a colourbar are
    used to indicate relative gain levels.

    The steering matrix is computed via:

    .. math::

        r(\\theta, \\phi) = |\\mathbf{w}^H \\mathbf{a}(\\theta, \\phi)|

    and mapped to Cartesian coordinates by:

    .. math::

        (X, Y, Z) = r \\cdot (\\sin\\theta\\cos\\phi,\\;
                              \\sin\\theta\\sin\\phi,\\;
                              \\cos\\theta)

    Examples
    --------
    >>> import numpy as np
    >>> positions = np.array([[i * 0.5, 0, 0] for i in range(6)])
    >>> w = np.ones(6) / 6
    >>> spatial(positions=positions, weights=w, wavelength=1.0, resolution=90)
    """
    theta = np.linspace(0, np.pi,       resolution)       # elevation [0, π]
    phi   = np.linspace(0, 2 * np.pi,   2 * resolution)   # azimuth   [0, 2π]

    # Steering matrix over the full angular grid — shape (N, 2R, R)
    steering_mat = steering_vector(
        np.rad2deg(theta), np.rad2deg(phi), wavelength, positions
    )

    # Beamformer response: einsum over sensor index n
    # 'n,nlm->lm'  (n = sensor, l = phi index, m = theta index)
    response     = np.einsum('n,nlm->lm', weights.conj(), steering_mat)
    beam_pattern = np.abs(response)
    beam_pattern /= np.max(beam_pattern)   # normalise to [0, 1]

    # Map to Cartesian coordinates for 3-D surface plot
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    X = beam_pattern * np.sin(theta_grid) * np.cos(phi_grid)
    Y = beam_pattern * np.sin(theta_grid) * np.sin(phi_grid)
    Z = beam_pattern * np.cos(theta_grid)

    fig = plt.figure(figsize=(12, 10))
    ax  = fig.add_subplot(111, projection='3d')

    norm = mcolors.Normalize(vmin=beam_pattern.min(), vmax=beam_pattern.max())
    ax.plot_surface(
        X, Y, Z,
        facecolors  = cm.jet(norm(beam_pattern)),
        rstride     = 9,
        cstride     = 9,
        antialiased = True,
        alpha       = 0.8,
        linewidth   = 0.5,
        edgecolor   = (0, 0, 0, 0.2),
    )

    legend_handles = []

    # SOI direction lines
    if soi_directions:
        for i, (th_deg, ph_deg) in enumerate(soi_directions):
            th, ph = np.deg2rad(th_deg), np.deg2rad(ph_deg)
            x = 1.2 * np.sin(th) * np.cos(ph)
            y = 1.2 * np.sin(th) * np.sin(ph)
            z = 1.2 * np.cos(th)
            label = "SOI Direction" if i == 0 else None
            line, = ax.plot([0, x], [0, y], [0, z], color='darkgreen', lw=4, label=label)
            if i == 0:
                legend_handles.append(line)

    # Interferer direction lines
    if interferers:
        for i, (th_deg, ph_deg) in enumerate(interferers):
            th, ph = np.deg2rad(th_deg), np.deg2rad(ph_deg)
            x = 1.2 * np.sin(th) * np.cos(ph)
            y = 1.2 * np.sin(th) * np.sin(ph)
            z = 1.2 * np.cos(th)
            label = "Interferer Direction" if i == 0 else None
            line, = ax.plot([0, x], [0, y], [0, z], color='darkred', linestyle='--', lw=3, label=label)
            if i == 0:
                legend_handles.append(line)

    if legend_handles:
        ax.legend(handles=legend_handles)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3-D Beam Pattern")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_aspect('equal')

    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cm.jet),
        ax=ax, shrink=0.6, label="Relative Gain",
    )
