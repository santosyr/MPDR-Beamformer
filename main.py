"""
main.py
=======
Simulation script for comparing the three MPDR beamformer solvers.

This script sets up a scenario with a 12-element Uniform Linear Array
(ULA), two interferers, and high background noise, then computes the
optimal weight vector using each of the three solvers:

    1. Theoretical  — closed-form solution from the analytical covariance matrix.
    2. Estimated    — sample covariance + Conjugate Gradient (centralised).
    3. Distributed  — Average Consensus + Conjugate Gradient (4-node network).

Finally, the three beam patterns are overlaid on a single polar plot for
visual comparison.

Usage
-----
Run from the repository root:

    python main.py

Dependencies: numpy, matplotlib, and the ``Functions`` package in the same
directory.  No Qt installation is required; the plotter selects the best
available Matplotlib backend automatically.
"""

import numpy             as np
import matplotlib.pyplot as plt

from Functions import plotter
import Functions.mpdr as mpdr

# =============================================================================
# Plot colour palette
# =============================================================================
C_THEORETICAL = "#12487E"   # navy blue
C_ESTIMATED   = "#AF570B"   # burnt orange
C_DISTRIBUTED = "#114426"   # dark green

# =============================================================================
# Array geometry — Uniform Linear Array (ULA)
# =============================================================================
# 12 sensors spaced λ/2 apart along the x-axis (half-wavelength spacing
# when wavelength = 1.0).
N_SENSORS = 12
pos_full  = np.array([[i * 0.5, 0.0, 0.0] for i in range(N_SENSORS)])

# =============================================================================
# Scenario parameters
# =============================================================================
theta_soi  = 45.0    # SOI elevation angle (degrees)
phi_soi    = 0.0     # SOI azimuth angle (degrees)
wavelength = 1.0     # normalised wavelength

# Two narrowband interferers in the same cut plane (phi = 0)
interferers = [
    {'theta': 80.0, 'phi': 0.0, 'power': 10.0, 'wavelength': 1.0},
    {'theta':  0.0, 'phi': 0.0, 'power': 10.0, 'wavelength': 1.0},
]

snapshots   = 100000      # number of snapshots for sample covariance estimation
noise_power = 1e-3        # noise variance σ² (low SNR environment)

# =============================================================================
# 1 — Theoretical solver
# =============================================================================
# Uses the exact, analytically constructed covariance matrix.
# This is the ideal reference: it requires perfect knowledge of all
# signal powers and directions, which is unavailable in practice.
print("\n[1/3] Computing Theoretical weights ...")

w_theo = mpdr.theoretical(
    positions        = pos_full,
    theta_soi        = theta_soi,
    phi_soi          = phi_soi,
    wavelength       = wavelength,
    interferers_data = interferers,
    noise_power      = noise_power,
    plots            = False,
)

# =============================================================================
# 2 — Estimated solver (centralised Conjugate Gradient)
# =============================================================================
# Simulates the received signal, estimates R̂_xx from snapshots, and
# solves R̂_xx h = a via CG — no direct matrix inversion.
print("\n[2/3] Computing Estimated weights (centralised CG) ...")

w_est = mpdr.estimated(
    positions        = pos_full,
    theta_soi        = theta_soi,
    phi_soi          = phi_soi,
    wavelength       = wavelength,
    interferers_data = interferers,
    noise_power      = noise_power,
    snapshots        = snapshots,
    max_iter         = None,    # defaults to N_SENSORS iterations
    epsilon          = 1e-9,
    plots            = False,
)

# =============================================================================
# 3 — Distributed solver (Average Consensus + CG)
# =============================================================================
# The 12-sensor array is partitioned into 4 nodes of 3 sensors each,
# arranged in a linear chain: Node 1 — Node 2 — Node 3 — Node 4.
# Nodes share only consensus scalars, never raw data.
print("\n[3/3] Computing Distributed weights (4-node network, consensus CG) ...")

node_positions = [
    pos_full[0:3],    # Node 1: sensors 0–2
    pos_full[3:6],    # Node 2: sensors 3–5
    pos_full[6:9],    # Node 3: sensors 6–8
    pos_full[9:12],   # Node 4: sensors 9–11
]

# Linear-chain topology: each node talks only to its immediate neighbours.
#   Node 1 — Node 2 — Node 3 — Node 4
adj_matrix = np.array([
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
])

w_dist = mpdr.distributed(
    positions        = node_positions,
    adjacency_matrix = adj_matrix,
    theta_soi        = theta_soi,
    phi_soi          = phi_soi,
    wavelength       = wavelength,
    interferers_data = interferers,
    noise_power      = noise_power,
    snapshots        = snapshots,
    max_iter         = None,    # defaults to total number of sensors
    epsilon          = 1e-9,
    plots            = False,
)

# =============================================================================
# Comparison plot — polar beam patterns
# =============================================================================
print("\nGenerating comparison polar plot ...")

models = [
    {'weights': w_theo, 'label': 'Theoretical',    'color': C_THEORETICAL, 'linestyle': '-'},
    {'weights': w_est,  'label': 'Estimated (CG)', 'color': C_ESTIMATED,   'linestyle': '--'},
    {'weights': w_dist, 'label': 'Distributed',    'color': C_DISTRIBUTED, 'linestyle': ':'},
]

interferer_dirs = [(src['theta'], src['phi']) for src in interferers]

plotter.polar(
    positions      = pos_full,
    models_data    = models,
    wavelength     = wavelength,
    theta_cut      = theta_soi,
    phi_cut        = phi_soi,
    soi_directions = [(theta_soi, phi_soi)],
    interferers    = interferer_dirs,
    cut_plane      = 'phi',
    r_lim          = (-60, 0),
)

plt.show()
print("\nDone.")
