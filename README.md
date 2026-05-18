# MPDR Beamformer

A research-oriented Python implementation of the **Minimum Power Distortionless Response (MPDR)** beamformer — also known as the **Capon** beamformer — for arbitrary 3-D sensor array geometries.

Three solvers are provided, each targeting a different operational context: a theoretical baseline, a centralised sample-based estimator, and a fully distributed network solver. All solvers share the same steering-vector engine and produce directly comparable weight vectors.

---

## Table of Contents

1. [Background](#1-background)
2. [Repository Structure](#2-repository-structure)
3. [Installation](#3-installation)
4. [Quick Start](#4-quick-start)
5. [Module Reference](#5-module-reference)
   - [steering.py](#51-steeringpy)
   - [mpdr.py](#52-mpdrpy)
   - [dsp.py](#53-dsppy)
   - [plotter.py](#54-plotterpy)
6. [Worked Example](#6-worked-example)
7. [Mathematical Derivation](#7-mathematical-derivation)
   - [Signal Model](#71-signal-model)
   - [MPDR Optimisation Problem](#72-mpdr-optimisation-problem)
   - [Closed-Form Solution](#73-closed-form-solution)
   - [Conjugate Gradient Solver](#74-conjugate-gradient-solver)
   - [Distributed CG via Average Consensus](#75-distributed-cg-via-average-consensus)
8. [Angle Convention](#8-angle-convention)
9. [Extending the Code](#9-extending-the-code)
10. [Citation](#10-citation)
11. [References](#11-references)

---

## 1. Background

A **beamformer** is a spatial filter applied to the outputs of a sensor array. Given N sensors and a complex weight vector **w** ∈ ℂᴺ, the array output at time k is:

```
y[k] = wᴴ x[k]
```

where **x**[k] ∈ ℂᴺ is the received signal vector. The goal is to choose **w** so that signals from a desired direction (the *Signal of Interest*, SOI) pass through undistorted, while signals from other directions (interferers, noise) are suppressed.

The **MPDR** criterion formalises this as a constrained optimisation: minimise the total output power (suppressing everything) subject to maintaining unity gain toward the SOI. This produces the optimal spatial filter under the assumption of a known SOI direction and a known (or estimated) covariance matrix.

Unlike the simpler delay-and-sum beamformer, MPDR adapts its nulls to the actual interference directions, achieving much deeper suppression.

---

## 2. Repository Structure

```
mpdr-beamformer/
│
├── Functions/                  # Core package
│   ├── __init__.py             # Public API exports
│   ├── steering.py             # Steering vector computation
│   ├── mpdr.py                 # MPDR beamformer solvers
│   ├── dsp.py                  # Signal generation and consensus algorithms
│   └── plotter.py              # Beam pattern visualisation
│
├── main.py                     # Example simulation script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

### Role of each module

| Module | Responsibility |
|---|---|
| `steering.py` | Converts a direction (θ, φ) and sensor positions into a complex phase-shift vector. All solvers and the plotter depend on this. |
| `mpdr.py` | The three beamformer solvers: `theoretical`, `estimated`, `distributed`. |
| `dsp.py` | Signal simulation (AWGN, n-PSK), adaptive filtering (NLMS), and the Average Consensus algorithm for distributed computing. |
| `plotter.py` | Polar, Cartesian, 3-D spatial, and array topology plots. Backend-agnostic: selects Qt, Tk, Wx, or Agg automatically. |

---

## 3. Installation

**Python 3.10 or later** is required (the code uses `match`-free type union syntax `A | B` from PEP 604).

### 1. Clone the repository

```bash
git clone https://github.com/your-org/mpdr-beamformer.git
cd mpdr-beamformer
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
numpy>=1.26
scipy>=1.12
matplotlib>=3.8
pandas>=2.0
```

> **No Qt required.** The plotter detects the best available Matplotlib backend automatically (Qt → Tk → Wx → Agg). If you want interactive plots, install either `PyQt6` or `PySide6`:
> ```bash
> pip install PyQt6
> ```

---

## 4. Quick Start

```python
import numpy as np
import matplotlib.pyplot as plt
import Functions.mpdr as mpdr

# 1. Define a 6-element ULA with half-wavelength spacing
positions = np.array([[i * 0.5, 0.0, 0.0] for i in range(6)])

# 2. Define the scenario
theta_soi  = 90.0   # SOI at broadside (elevation 90° = perpendicular to the array)
phi_soi    = 0.0
wavelength = 1.0
interferers = [
    {'theta': 60.0, 'phi': 0.0, 'power': 10.0, 'wavelength': 1.0},
]

# 3. Compute weights with the theoretical solver
w = mpdr.theoretical(
    positions        = positions,
    theta_soi        = theta_soi,
    phi_soi          = phi_soi,
    wavelength       = wavelength,
    interferers_data = interferers,
    plots            = True,        # generate a polar beam pattern
)

plt.show()
print("Weight vector:", w)
```

Running `main.py` executes a more complete simulation comparing all three solvers side by side:

```bash
python main.py
```

---

## 5. Module Reference

### 5.1 `steering.py`

#### `steering_vector(theta_soi, phi_soi, wavelength, positions)`

Computes the steering vector **a**(θ, φ) ∈ ℂᴺ:

```
a_n(θ, φ) = exp(−j (2π/λ) pₙ · û)
```

where **û** = [sin θ cos φ, sin θ sin φ, cos θ]ᵀ is the unit direction-of-arrival vector.

| Parameter | Type | Description |
|---|---|---|
| `theta_soi` | `float` or `ndarray (M,)` | Elevation angle(s) in degrees |
| `phi_soi` | `float` or `ndarray (L,)` | Azimuth angle(s) in degrees |
| `wavelength` | `float` | Signal wavelength λ |
| `positions` | `ndarray (N, 3)` | Sensor Cartesian coordinates |

**Returns:** `ndarray (N,)` for scalar inputs; `ndarray (N, L, M)` for array inputs.

```python
from Functions.steering import steering_vector
import numpy as np

positions = np.array([[0, 0, 0], [0.5, 0, 0], [1.0, 0, 0]])

# Single direction
a = steering_vector(90.0, 0.0, 1.0, positions)   # shape (3,)

# Full angular grid
theta = np.linspace(0, 180, 181)
phi   = np.linspace(0, 360, 361)
A = steering_vector(theta, phi, 1.0, positions)   # shape (3, 361, 181)
```

---

### 5.2 `mpdr.py`

All three solvers accept the same core parameters and return a weight vector of shape **(N,)**.

#### `mpdr.theoretical(...)`

Builds the **analytical covariance matrix** from known signal parameters and solves the MPDR system via LU decomposition (`numpy.linalg.solve`). Use this as the ideal performance upper bound.

```python
w = mpdr.theoretical(
    positions        = positions,   # (N, 3)
    theta_soi        = 90.0,
    phi_soi          = 0.0,
    signal_power     = 1.0,
    noise_power      = 1e-3,
    wavelength       = 1.0,
    interferers_data = interferers, # list of dicts, or None
    plots            = False,
)
```

#### `mpdr.estimated(...)`

Simulates received snapshots, estimates the **sample covariance matrix** R̂_xx, and solves R̂_xx **h** = **a** using the **Conjugate Gradient** algorithm — no matrix inversion.

```python
w = mpdr.estimated(
    positions        = positions,
    theta_soi        = 90.0,
    phi_soi          = 0.0,
    noise_power      = 1e-3,
    wavelength       = 1.0,
    interferers_data = interferers,
    snapshots        = 1000,        # more snapshots → better R̂_xx estimate
    max_iter         = None,        # defaults to N (number of sensors)
    epsilon          = 1e-9,        # CG convergence threshold
    plots            = False,
)
```

#### `mpdr.distributed(...)`

Runs a fully **distributed** CG solver across a network of P sensor nodes. Nodes share only consensus scalars — never raw data. Requires an adjacency matrix describing the network topology.

```python
node_positions = [pos_node1, pos_node2, pos_node3, pos_node4]  # list of (K_p, 3) arrays

adj = np.array([        # linear chain: 1—2—3—4
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
])

w = mpdr.distributed(
    positions        = node_positions,
    adjacency_matrix = adj,
    theta_soi        = 90.0,
    phi_soi          = 0.0,
    noise_power      = 1e-3,
    wavelength       = 1.0,
    interferers_data = interferers,
    snapshots        = 1000,
    max_iter         = None,
    epsilon          = 1e-9,
    plots            = False,
)
# w has shape (N_total,) where N_total = sum of sensors across all nodes
```

**Interferer dictionary format** (same for all three solvers):

```python
interferers = [
    {'theta': 60.0, 'phi': 0.0, 'power': 10.0, 'wavelength': 1.0},
    {'theta': 120.0, 'phi': 0.0, 'power': 5.0, 'wavelength': 1.0},
    # add as many as needed
]
```

---

### 5.3 `dsp.py`

Utility functions for signal simulation and distributed computing.

| Function | Description |
|---|---|
| `awgn(shape, power)` | Complex or real AWGN with specified variance σ². |
| `npsk(num_symbols, n)` | Random n-PSK symbol sequence (unit average power). |
| `nlms(x_k, s_k, mu)` | Normalised LMS adaptive filter. |
| `adjacency_matrix(network)` | Build an adjacency matrix from a dictionary network definition. |
| `connectivity_graph(centers, method, k)` | Build an adjacency matrix from node positions using KNN, MST, or hybrid strategies. |
| `average_consensus(A, u0)` | Finite-time exact average consensus on a connected graph. |
| `elementwise_consensus(A, local_matrices)` | Element-wise consensus for aggregating complex matrices. |

#### Building a network topology programmatically

```python
from Functions import dsp
import numpy as np

# Option A: dictionary definition (small, hand-crafted networks)
network = {
    "Node1": {"connections": ["Node2"]},
    "Node2": {"connections": ["Node1", "Node3"]},
    "Node3": {"connections": ["Node2"]},
}
A = dsp.adjacency_matrix(network)

# Option B: position-based (large or randomised networks)
centers = np.array([[0,0,0], [5,0,0], [10,0,0], [5,5,0]], dtype=float)
A = dsp.connectivity_graph(centers, method='mst_knn', k=2)
```

---

### 5.4 `plotter.py`

All plot functions display results interactively and optionally export the computed pattern data to a tab-separated file.

| Function | Output |
|---|---|
| `topology(positions, soi_direction)` | 3-D scatter of sensor positions with optional SOI arrow. |
| `linear(...)` | Cartesian beam pattern: gain (dB) vs. scan angle. |
| `polar(...)` | Polar beam pattern. Supports multi-model overlay and multi-beam weight matrices. |
| `spatial(...)` | Full 3-D spherical surface beam pattern. |

```python
from Functions import plotter
import matplotlib.pyplot as plt

# Overlay three models on one polar plot
models = [
    {'weights': w_theo, 'label': 'Theoretical',    'color': '#12487E', 'linestyle': '-'},
    {'weights': w_est,  'label': 'Estimated (CG)', 'color': '#AF570B', 'linestyle': '--'},
    {'weights': w_dist, 'label': 'Distributed',    'color': '#114426', 'linestyle': ':'},
]

plotter.polar(
    positions      = positions,
    models_data    = models,
    wavelength     = 1.0,
    theta_cut      = 90.0,
    phi_cut        = 0.0,
    soi_directions = [(90.0, 0.0)],
    interferers    = [(60.0, 0.0), (120.0, 0.0)],
    cut_plane      = 'phi',
    r_lim          = (-60, 0),
    export_filepath = 'results/pattern.tsv',  # optional
)
plt.show()
```

---

## 6. Worked Example

The script `main.py` runs the following scenario end to end.

**Setup:**

- **Array:** 12-element ULA, half-wavelength spacing along x.
- **SOI:** θ = 0°, φ = 0° (along the z-axis / array boresight).
- **Interferer 1:** θ = 80°, φ = 0°, power = 10 × SOI.
- **Interferer 2:** θ = 45°, φ = 0°, power = 10 × SOI.
- **Network:** 4 nodes of 3 sensors each in a linear chain.
- **Snapshots:** 1000 (for the estimated and distributed solvers).

**Expected output:**

All three polar patterns should show:
- Unity gain (0 dB) at θ = 0°.
- Deep nulls (≪ −20 dB) at θ = 80° and θ = 45°.
- The `Theoretical` pattern is the sharpest reference; `Estimated` and `Distributed` converge toward it as the number of snapshots increases.

```bash
python main.py

# [1/3] Computing Theoretical weights ...
#       Done.  Weight vector norm: 0.2887
# [2/3] Computing Estimated weights (centralised CG) ...
#       Done.  Weight vector norm: 0.2881
# [3/3] Computing Distributed weights (4-node network, consensus CG) ...
#       Done.  Weight vector norm: 0.2884
# Generating comparison polar plot ...
```

---

## 7. Mathematical Derivation

### 7.1 Signal Model

Consider N sensors receiving a superposition of one SOI, Q interferers, and spatially white noise. The received signal vector at snapshot k is:

```
x[k] = a_s · s[k]  +  Σᵢ aᵢ · sᵢ[k]  +  n[k]
```

where:

- **a**_s ∈ ℂᴺ — steering vector toward the SOI.
- s[k] — SOI symbol (zero-mean, variance σ²_s).
- **a**ᵢ — steering vector toward the i-th interferer.
- sᵢ[k] — i-th interferer symbol (variance σ²ᵢ).
- **n**[k] ~ CN(**0**, σ²_n **I**) — spatially white Gaussian noise.

The theoretical array covariance matrix is:

```
R_xx = E[x[k] x[k]ᴴ]
     = σ²_s a_s a_sᴴ  +  Σᵢ σ²ᵢ aᵢ aᵢᴴ  +  σ²_n I
```

### 7.2 MPDR Optimisation Problem

Choose the weight vector **w** ∈ ℂᴺ to:

```
minimise    wᴴ R_xx w          (minimise total output power)
subject to  wᴴ a_s = 1         (preserve SOI with unity gain)
```

The constraint ensures that the SOI component of the output is not distorted, while the minimisation suppresses everything else (interferers + noise).

### 7.3 Closed-Form Solution

Using the method of Lagrange multipliers, the unique solution is:

```
w_opt = R_xx⁻¹ a_s / (a_sᴴ R_xx⁻¹ a_s)
```

In practice the inverse is computed by solving the linear system:

```
R_xx h = a_s   →   w = h / (a_sᴴ h)
```

This avoids forming R_xx⁻¹ explicitly, which is numerically preferable and cheaper for large N.

### 7.4 Conjugate Gradient Solver

The `estimated` solver replaces R_xx with the **sample covariance matrix**:

```
R̂_xx = (1/K) X Xᴴ
```

where **X** ∈ ℂᴺˣᴷ is the matrix of K received snapshots, and then solves R̂_xx **h** = **a**_s iteratively using the **Conjugate Gradient (CG)** algorithm.

Starting from **h**₀ = **0**:

```
r₀ = a_s,   p₀ = r₀

For i = 0, 1, 2, ... :

    αᵢ  =  (rᵢᴴ rᵢ) / (pᵢᴴ R̂ pᵢ)

    hᵢ₊₁  =  hᵢ + αᵢ pᵢ
    rᵢ₊₁  =  rᵢ − αᵢ R̂ pᵢ

    if ‖rᵢ₊₁‖ < ε : break

    βᵢ  =  (rᵢ₊₁ᴴ rᵢ₊₁) / (rᵢᴴ rᵢ)
    pᵢ₊₁  =  rᵢ₊₁ + βᵢ pᵢ
```

CG converges in at most N steps for a positive-definite system. Each iteration costs one matrix-vector product R̂ **p** (O(N²)), giving total complexity O(N³) in the worst case — the same order as direct inversion, but with much better practical convergence in few iterations when R̂ is well-conditioned.

### 7.5 Distributed CG via Average Consensus

In the `distributed` solver, the global array of N sensors is partitioned into P nodes. Node p holds K_p sensors (N = Σ K_p) and the local data matrix **X**_p ∈ ℂᴷᵖˣᴷ. No node has access to the full **X**.

The key insight is that the matrix-vector product R̂ **p** can be decomposed as:

```
(R̂ p)ₗₒ꜀ₐₗ,p = (1/K) Xₚ · t

where  t[k] = Σₚ xₚ[k]ᴴ pₚ   (inner products, one per snapshot)
```

The global vector **t** is computed snapshot by snapshot using **Average Consensus**: each node computes its local inner product xₚ[k]ᴴ **p**ₚ and the network converges to the global sum via:

```
t[k] = P · AC( { xₚ[k]ᴴ pₚ }ₚ₌₁ᴾ )
```

where AC denotes the Average Consensus operator and the factor P converts the average to a sum. The remaining CG scalars (αᵢ numerator, pᵢᴴ R̂ **p** denominator, βᵢ) are obtained similarly. Every node runs identical CG update steps using only its own data plus the consensus outputs — no raw data is ever transmitted.

**Average Consensus** on a connected graph with P nodes and R distinct Laplacian eigenvalues converges in exactly R − 1 steps to the exact global average. Sparser graphs have smaller R and converge faster; denser graphs have higher R but provide communication redundancy.

---

## 8. Angle Convention

This repository uses the **physics / ISO 80000-2** spherical coordinate system throughout:

```
θ (theta) — polar / elevation angle
            Measured from the +z axis (zenith).
            Range: [0°, 180°].
            θ = 0°  →  direction along +z (array boresight for a z-axis array).
            θ = 90° →  horizontal plane (broadside for a ULA along x or y).

φ (phi)   — azimuth angle
            Measured from the +x axis, rotating toward +y.
            Range: [0°, 360°).
```

The phase convention is **exp(−j ψ)**, so a sensor further from the origin in the direction-of-arrival has a more negative (leading) phase.

```
     z  (θ = 0°)
     │
     │
     │______ x  (φ = 0°)
    /
   / y  (φ = 90°)
```

---

## 9. Extending the Code

### Adding a new solver

Create a new function in `mpdr.py` following the same signature pattern:

```python
def my_solver(
    positions: np.ndarray,
    theta_soi: float,
    phi_soi: float,
    ...
) -> np.ndarray:
    """Docstring following the NumPy format."""
    a_soi = steering_vector(theta_soi, phi_soi, wavelength, positions)
    # ... your solver logic ...
    return w   # shape (N,), normalised so that aᴴ w = 1
```

Then export it from `Functions/__init__.py` and add it to the comparison loop in `main.py`.

### Changing the array geometry

Any positions array of shape (N, 3) is accepted. Examples:

```python
# Uniform Circular Array (UCA), radius r, in the xy-plane
N, r = 8, 1.0
angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
pos_uca = np.column_stack([r * np.cos(angles), r * np.sin(angles), np.zeros(N)])

# Random 3-D array
pos_random = np.random.uniform(-5, 5, size=(16, 3))

# L-shaped array
pos_L = np.vstack([
    np.array([[i * 0.5, 0, 0] for i in range(6)]),   # horizontal arm
    np.array([[0, i * 0.5, 0] for i in range(1, 6)]), # vertical arm
])
```

### Changing the network topology

Use `dsp.connectivity_graph` to build the adjacency matrix automatically from node positions:

```python
from Functions import dsp

centers = np.array([node.mean(axis=0) for node in node_positions])  # node centroids
adj = dsp.connectivity_graph(centers, method='mst_knn', k=2)
```

---

## 10. Citation

If you use this code in academic work, please cite:

```bibtex
@software{mpdr_beamformer,
  author  = {Your Name},
  title   = {{MPDR Beamformer}: A Research Implementation of the Capon/MPDR
             Beamformer with Distributed Consensus},
  year    = {2025},
  url     = {https://github.com/your-org/mpdr-beamformer},
}
```

---

## 11. References

- Capon, J. (1969). High-resolution frequency-wavenumber spectrum analysis. *Proceedings of the IEEE*, 57(8), 1408–1418.
- Van Trees, H. L. (2002). *Optimum Array Processing: Part IV of Detection, Estimation, and Modulation Theory*. Wiley-Interscience.
- Haykin, S. (2002). *Adaptive Filter Theory* (4th ed.). Prentice Hall.
- Xiao, L., & Boyd, S. (2004). Fast linear iterations for distributed averaging. *Systems & Control Letters*, 53(1), 65–78.
- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.
