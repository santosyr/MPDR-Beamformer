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

A **beamformer** is a spatial filter applied to the outputs of a sensor array. Given $N$ sensors and a complex weight vector $\mathbf{w} \in \mathbb{C}^{N}$, the scalar array output at snapshot $k$ is:

$$y[k] = \mathbf{w}^{H} \mathbf{x}[k]$$

where $\mathbf{x}[k] \in \mathbb{C}^{N}$ is the received signal vector and $(\cdot)^{H}$ denotes the Hermitian (conjugate) transpose. The goal is to choose $\mathbf{w}$ so that signals arriving from a desired direction — the *Signal of Interest* (SOI) — pass through undistorted, while signals from all other directions (interferers and noise) are suppressed.

The **MPDR** criterion formalises this as a constrained optimisation: minimise the total output power $\mathbf{w}^{H}\mathbf{R}_{xx}\mathbf{w}$ subject to maintaining unity gain toward the SOI direction. This produces the optimal adaptive spatial filter under the assumption of a known SOI steering vector and a known (or estimated) array covariance matrix.

Unlike the simpler delay-and-sum beamformer — which applies uniform, non-adaptive weights — MPDR places its nulls precisely at the interference directions, achieving far deeper suppression.

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
| `steering.py` | Converts a direction $(\theta, \varphi)$ and sensor positions into a complex phase-shift vector. All solvers and the plotter depend on this. |
| `mpdr.py` | The three beamformer solvers: `theoretical`, `estimated`, `distributed`. |
| `dsp.py` | Signal simulation (AWGN, $n$-PSK), and the Average Consensus algorithm for distributed computing. |
| `plotter.py` | Polar, Cartesian, 3-D spatial, and array-topology plots. Backend-agnostic: selects Qt, Tk, Wx, or Agg automatically. |

---

## 3. Installation

**Python 3.10 or later** is required (the code uses the type-union syntax `A | B` introduced in PEP 604).

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

> **No Qt required.** The plotter detects the best available Matplotlib backend automatically (Qt → Tk → Wx → Agg). For interactive plots, optionally install `PyQt6` or `PySide6`:
> ```bash
> pip install PyQt6
> ```

---

## 4. Quick Start

```python
import numpy as np
import matplotlib.pyplot as plt
import Functions.mpdr as mpdr

# 1. Define a 6-element ULA with half-wavelength spacing (d = lambda/2 = 0.5)
positions = np.array([[i * 0.5, 0.0, 0.0] for i in range(6)])

# 2. Define the scenario
theta_soi  = 90.0   # broadside: theta = 90 deg is perpendicular to the array axis
phi_soi    = 0.0
wavelength = 1.0
interferers = [
    {'theta': 60.0, 'phi': 0.0, 'power': 10.0, 'wavelength': 1.0},
]

# 3. Compute weights and plot the beam pattern
w = mpdr.theoretical(
    positions        = positions,
    theta_soi        = theta_soi,
    phi_soi          = phi_soi,
    wavelength       = wavelength,
    interferers_data = interferers,
    plots            = True,
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

Computes the steering vector $\mathbf{a}(\theta, \varphi) \in \mathbb{C}^{N}$, where the $n$-th entry encodes the phase shift at sensor $n$:

$$a_n(\theta, \varphi) = \exp\!\left( -j \frac{2\pi}{\lambda}\, \mathbf{p}_n \cdot \hat{\mathbf{u}} \right)$$

with $\mathbf{p}_n \in \mathbb{R}^3$ the position of sensor $n$ and $\hat{\mathbf{u}}$ the unit direction-of-arrival (DOA) vector:

$$\hat{\mathbf{u}}(\theta,\varphi) = \begin{bmatrix} \sin\theta\cos\varphi \\ \sin\theta\sin\varphi \\ \cos\theta \end{bmatrix}$$

| Parameter | Type | Description |
|---|---|---|
| `theta_soi` | `float` or `ndarray` $(M,)$ | Elevation angle(s) $\theta$ in degrees |
| `phi_soi` | `float` or `ndarray` $(L,)$ | Azimuth angle(s) $\varphi$ in degrees |
| `wavelength` | `float` | Signal wavelength $\lambda$ |
| `positions` | `ndarray` $(N, 3)$ | Sensor Cartesian coordinates |

**Returns:** `ndarray` $(N,)$ for scalar inputs; `ndarray` $(N, L, M)$ for array inputs (axis 0 = sensors, axis 1 = $\varphi$ values, axis 2 = $\theta$ values).

```python
from Functions.steering import steering_vector
import numpy as np

positions = np.array([[0, 0, 0], [0.5, 0, 0], [1.0, 0, 0]])

# Single direction -> shape (3,)
a = steering_vector(90.0, 0.0, 1.0, positions)

# Full angular grid -> shape (3, 361, 181)
theta = np.linspace(0, 180, 181)
phi   = np.linspace(0, 360, 361)
A = steering_vector(theta, phi, 1.0, positions)
```

---

### 5.2 `mpdr.py`

All three solvers accept the same core parameters and return a weight vector of shape $(N,)$, normalised so that $\mathbf{a}_s^H \mathbf{w} = 1$.

#### `mpdr.theoretical(...)`

Builds the **analytical covariance matrix** from known signal parameters and solves the MPDR linear system via LU decomposition (`numpy.linalg.solve`). Use this as the ideal performance upper bound.

```python
w = mpdr.theoretical(
    positions        = positions,   # ndarray (N, 3)
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

Simulates $K$ received snapshots, estimates the **sample covariance matrix** $\hat{\mathbf{R}}_{xx}$, and solves $\hat{\mathbf{R}}_{xx} \mathbf{h} = \mathbf{a}_s$ using the **Conjugate Gradient** algorithm — no matrix inversion required.

```python
w = mpdr.estimated(
    positions        = positions,
    theta_soi        = 90.0,
    phi_soi          = 0.0,
    noise_power      = 1e-3,
    wavelength       = 1.0,
    interferers_data = interferers,
    snapshots        = 1000,    # larger K -> better covariance estimate
    max_iter         = None,    # defaults to N (number of sensors)
    epsilon          = 1e-9,    # CG convergence threshold on ||r_i||
    plots            = False,
)
```

#### `mpdr.distributed(...)`

Runs a fully **distributed** CG solver across a network of $P$ sensor nodes. Nodes exchange only consensus scalars — never raw data. An adjacency matrix describes the network topology.

```python
node_positions = [pos_node1, pos_node2, pos_node3, pos_node4]  # list of (K_p, 3) arrays

adj = np.array([        # linear chain: Node 1 — Node 2 — Node 3 — Node 4
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
    max_iter         = None,    # defaults to N_total = sum of all K_p
    epsilon          = 1e-9,
    plots            = False,
)
# w has shape (N_total,), where N_total = sum of sensors across all nodes
```

**Interferer dictionary format** (identical for all three solvers):

```python
interferers = [
    {'theta': 60.0, 'phi': 0.0, 'power': 10.0, 'wavelength': 1.0},
    {'theta': 120.0, 'phi': 0.0, 'power':  5.0, 'wavelength': 1.0},
    # add as many entries as needed
]
```

---

### 5.3 `dsp.py`

Utility functions for signal simulation and distributed computing.

| Function | Description |
|---|---|
| `awgn(shape, power)` | Complex or real AWGN with variance $\sigma^2 =$ `power`. |
| `npsk(num_symbols, n)` | Random $n$-PSK symbol sequence with unit average power. |
| `adjacency_matrix(network)` | Build an adjacency matrix from a dictionary network definition. |
| `average_consensus(A, u0)` | Finite-time exact average consensus on a connected graph. |

#### Building a network topology

```python
from Functions import dsp
import numpy as np

# hand-crafted dictionary
network = {
    "Node1": {"connections": ["Node2"]},
    "Node2": {"connections": ["Node1", "Node3"]},
    "Node3": {"connections": ["Node2"]},
}
A = dsp.adjacency_matrix(network)

```

---

### 5.4 `plotter.py`

All plot functions display results interactively and optionally export computed pattern data to a tab-separated file.

| Function | Output |
|---|---|
| `topology(positions, soi_direction)` | 3-D scatter of sensor positions with optional SOI arrow. |
| `linear(...)` | Cartesian beam pattern: gain (dB) vs. scan angle. |
| `polar(...)` | Polar beam pattern. Supports multi-model overlay and multi-beam weight matrices. |
| `spatial(...)` | Full 3-D spherical surface beam pattern coloured by relative gain. |

```python
from Functions import plotter
import matplotlib.pyplot as plt

models = [
    {'weights': w_theo, 'label': 'Theoretical',    'color': '#12487E', 'linestyle': '-'},
    {'weights': w_est,  'label': 'Estimated (CG)', 'color': '#AF570B', 'linestyle': '--'},
    {'weights': w_dist, 'label': 'Distributed',    'color': '#114426', 'linestyle': ':'},
]

plotter.polar(
    positions       = positions,
    models_data     = models,
    wavelength      = 1.0,
    theta_cut       = 90.0,
    phi_cut         = 0.0,
    soi_directions  = [(90.0, 0.0)],
    interferers     = [(60.0, 0.0), (120.0, 0.0)],
    cut_plane       = 'phi',
    r_lim           = (-60, 0),
    export_filepath = 'results/pattern.tsv',  # optional
)
plt.show()
```

---

## 6. Worked Example

The script `main.py` runs the following scenario end to end.

**Setup:**

- **Array:** 12-element ULA, spacing $d = \lambda/2 = 0.5$ along the $x$-axis.
- **SOI:** $\theta = 0°$, $\varphi = 0°$ (along the $+z$-axis, end-fire).
- **Interferer 1:** $\theta = 80°$, $\varphi = 0°$, power $= 10\,\sigma_s^2$.
- **Interferer 2:** $\theta = 45°$, $\varphi = 0°$, power $= 10\,\sigma_s^2$.
- **Network (distributed solver):** $P = 4$ nodes of $K_p = 3$ sensors each in a linear chain.
- **Snapshots:** $K = 1\,000$ (for the estimated and distributed solvers).

**Expected output:**

All three polar patterns should show:

- Unity gain ($0\,\text{dB}$) at $\theta = 0°$.
- Deep nulls ($\ll -20\,\text{dB}$) at $\theta = 80°$ and $\theta = 45°$.
- The `Theoretical` curve is the sharpest reference; `Estimated` and `Distributed` converge toward it as $K$ increases.

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

Consider $N$ sensors receiving a superposition of one SOI, $Q$ interferers, and spatially white noise. The received signal vector at snapshot $k$ is:

$$\mathbf{x}[k] = \mathbf{a}_s\, s[k] + \sum_{i=1}^{Q} \mathbf{a}_i\, s_i[k] + \mathbf{n}[k]$$

where:

- $\mathbf{a}_s \in \mathbb{C}^{N}$ — steering vector toward the SOI direction $(\theta_s, \varphi_s)$.
- $s[k]$ — SOI symbol, zero-mean with power $\sigma_s^2 = \mathbb{E}[|s[k]|^2]$.
- $\mathbf{a}_i \in \mathbb{C}^{N}$ — steering vector toward the $i$-th interferer.
- $s_i[k]$ — $i$-th interferer symbol with power $\sigma_i^2$.
- $\mathbf{n}[k] \sim \mathcal{CN}(\mathbf{0},\, \sigma_n^2 \mathbf{I})$ — spatially white complex Gaussian noise.

All signals are assumed mutually uncorrelated. Under this assumption, the theoretical array covariance matrix $\mathbf{R}_{xx} = \mathbb{E}[\mathbf{x}[k]\mathbf{x}[k]^H]$ decomposes as:

$$\mathbf{R}_{xx} = \sigma_s^2\, \mathbf{a}_s \mathbf{a}_s^{H} + \sum_{i=1}^{Q} \sigma_i^2\, \mathbf{a}_i \mathbf{a}_i^{H} + \sigma_n^2\, \mathbf{I}$$

### 7.2 MPDR Optimisation Problem

Choose $\mathbf{w} \in \mathbb{C}^{N}$ to solve:

$$\begin{aligned} \underset{\mathbf{w}}{\text{minimise}} \quad & \mathbf{w}^{H} \mathbf{R}_{xx}\, \mathbf{w} \\ \text{subject to} \quad & \mathbf{w}^{H} \mathbf{a}_s = 1 \end{aligned}$$

The objective $\mathbf{w}^{H}\mathbf{R}_{xx}\mathbf{w} = \mathbb{E}[|y[k]|^2]$ is the total output power. Minimising it drives the beamformer to suppress all interference and noise, while the linear constraint $\mathbf{w}^{H}\mathbf{a}_s = 1$ ensures the SOI is passed with unit gain and zero phase distortion.

### 7.3 Closed-Form Solution

Applying the method of Lagrange multipliers to the problem in §7.2, the unique global minimiser is:

$$\boxed{\mathbf{w}_{\mathrm{opt}} = \frac{\mathbf{R}_{xx}^{-1}\, \mathbf{a}_s}{\mathbf{a}_s^{H}\, \mathbf{R}_{xx}^{-1}\, \mathbf{a}_s}}$$

Forming $\mathbf{R}_{xx}^{-1}$ explicitly is both numerically fragile and costly $\bigl(\mathcal{O}(N^3)\bigr)$. Instead, the numerator is obtained by solving the equivalent linear system:

$$\mathbf{R}_{xx}\, \mathbf{h} = \mathbf{a}_s \quad \Longrightarrow \quad \mathbf{h} = \mathbf{R}_{xx}^{-1}\mathbf{a}_s$$

and the final weight vector is recovered by the normalisation:

$$\mathbf{w} = \frac{\mathbf{h}}{\mathbf{a}_s^{H}\, \mathbf{h}}$$

### 7.4 Conjugate Gradient Solver

The `estimated` solver replaces $\mathbf{R}_{xx}$ with the **sample covariance matrix** built from $K$ snapshots:

$$\hat{\mathbf{R}}_{xx} = \frac{1}{K} \mathbf{X}\mathbf{X}^{H}, \qquad \mathbf{X} \in \mathbb{C}^{N \times K}$$

and solves $\hat{\mathbf{R}}_{xx}\,\mathbf{h} = \mathbf{a}_s$ iteratively with the **Conjugate Gradient (CG)** algorithm. Starting from $\mathbf{h}_0 = \mathbf{0}$, $\mathbf{r}_0 = \mathbf{a}_s$, $\mathbf{p}_0 = \mathbf{r}_0$, each CG iteration performs:

$$\alpha_i = \frac{\mathbf{r}_i^{H}\mathbf{r}_i}{\mathbf{p}_i^{H}\,\hat{\mathbf{R}}_{xx}\,\mathbf{p}_i}$$

$$\mathbf{h}_{i+1} = \mathbf{h}_i + \alpha_i\,\mathbf{p}_i, \qquad \mathbf{r}_{i+1} = \mathbf{r}_i - \alpha_i\,\hat{\mathbf{R}}_{xx}\,\mathbf{p}_i$$

$$\beta_i = \frac{\mathbf{r}_{i+1}^{H}\mathbf{r}_{i+1}}{\mathbf{r}_i^{H}\mathbf{r}_i}, \qquad \mathbf{p}_{i+1} = \mathbf{r}_{i+1} + \beta_i\,\mathbf{p}_i$$

The iteration stops when $|\mathbf{r}_{i+1}| < \varepsilon$ or after `max_iter` steps. For a positive-definite system, CG converges in at most $N$ steps; in practice it reaches machine precision in far fewer iterations when $\hat{\mathbf{R}}_{xx}$ is well-conditioned. The dominant cost per iteration is one matrix-vector product $\hat{\mathbf{R}}_{xx}\mathbf{p}_i \in \mathcal{O}(N^2)$.

### 7.5 Distributed CG via Average Consensus

In the `distributed` solver, the $N$ sensors are partitioned among $P$ nodes. Node $p$ holds $K_p$ sensors $\bigl(N = \sum_{p=1}^{P} K_p\bigr)$, with local data matrix $\mathbf{X}_p \in \mathbb{C}^{K_p \times K}$ and local steering vector $\mathbf{a}_p \in \mathbb{C}^{K_p}$. The global quantities are the vertical concatenations:

$$\mathbf{X} = \begin{bmatrix}\mathbf{X}_1 \\ \vdots \\ \mathbf{X}_P\end{bmatrix} \in \mathbb{C}^{N \times K}, \qquad \mathbf{a}_s = \begin{bmatrix}\mathbf{a}_1 \\ \vdots \\ \mathbf{a}_P\end{bmatrix} \in \mathbb{C}^{N}$$

Writing the CG search direction as $\mathbf{p} = [\mathbf{p}_1^T \cdots \mathbf{p}_P^T]^T$, the global matrix-vector product decomposes as:

$$\hat{\mathbf{R}}_{xx}\,\mathbf{p} = \frac{1}{K}\mathbf{X}\underbrace{\mathbf{X}^{H}\mathbf{p}}_{\mathbf{t} \,\in\, \mathbb{C}^{K}}$$

where the $k$-th component of $\mathbf{t}$ is:

$$t[k] = \mathbf{x}[k]^{H}\mathbf{p} = \sum_{p=1}^{P} \mathbf{x}_p[k]^{H}\mathbf{p}_p$$

Each node knows only its own local term $\mathbf{x}_p[k]^{H}\mathbf{p}_p$. The global sum is recovered via **Average Consensus** (AC):

$$t[k] = P \cdot \mathrm{AC}\!\left(\left\{\mathbf{x}_p[k]^{H}\mathbf{p}_p\right\}_{p=1}^{P}\right)$$

where the factor $P$ converts the network average into a sum. Each node then forms its local block of the product $\hat{\mathbf{R}}_{xx}\mathbf{p}$:

$$\mathbf{q}_p = \frac{1}{K}\,\mathbf{X}_p\,\mathbf{t}$$

The remaining CG scalars — the denominator $\mathbf{p}^H\hat{\mathbf{R}}_{xx}\mathbf{p} = \sum_p \mathbf{p}_p^H\mathbf{q}_p$ and the residual energy $\|\mathbf{r}_{i+1}\|^2 = \sum_p \|\mathbf{r}_p\|^2$ — are computed by the same consensus mechanism. Every node executes identical CG update steps using only its own data and the scalar consensus outputs. No raw sensor data is ever transmitted across the network.

**Average Consensus** on a connected graph with $P$ nodes and $R$ distinct Laplacian eigenvalues $\{0 = \lambda_1 < \lambda_2 \leq \cdots \leq \lambda_R\}$ converges to the exact global average in exactly $R - 1$ steps, using the sequence of weight matrices:

$$W^{(0)} = \frac{(-1)^{R-1}}{\lambda_2 \cdots \lambda_R}\left(\mathbf{L} - \lambda_R \mathbf{I}\right), \qquad W^{(t)} = \mathbf{L} - \lambda_{t+1}\mathbf{I}, \quad t = 1,\dots, R-2$$

where $\mathbf{L} = \mathbf{D} - \mathbf{A}$ is the graph Laplacian ($\mathbf{D}$ is the degree matrix, $\mathbf{A}$ the adjacency matrix). Sparser topologies have fewer distinct eigenvalues (smaller $R$) and therefore converge in fewer communication rounds; denser topologies increase $R$ but provide redundancy against link failures.

---

## 8. Angle Convention

This repository uses the **physics / ISO 80000-2** spherical coordinate system throughout:

| Symbol | Name | Measured from | Range |
|---|---|---|---|
| $\theta$ | Polar / elevation angle | $+z$ axis (zenith) | $[0°,\ 180°]$ |
| $\varphi$ | Azimuth angle | $+x$ axis, rotating toward $+y$ | $[0°,\ 360°)$ |

Key reference directions:

- $\theta = 0°$ — along $+z$. For a ULA along $x$, this is the *end-fire* direction.
- $\theta = 90°$ — the horizontal plane. For a ULA along $x$, this is the *broadside* plane.
- $\varphi = 0°$ — along $+x$ within the horizontal plane.
- $\varphi = 90°$ — along $+y$ within the horizontal plane.

The **phase sign convention** is $\exp(-j\psi_n)$, so a sensor displaced further from the origin in the DOA direction accumulates a more negative (leading) phase. The unit DOA vector in Cartesian coordinates is:

$$\hat{\mathbf{u}}(\theta,\varphi) = \begin{bmatrix} \sin\theta\cos\varphi \\ \sin\theta\sin\varphi \\ \cos\theta \end{bmatrix}$$

```
     z  (theta = 0 deg)
     |
     |
     |_______ x  (phi = 0 deg)
    /
   /  y  (phi = 90 deg)
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
    wavelength: float = 1.0,
    ...
) -> np.ndarray:
    """
    Short description.

    Parameters
    ----------
    ...

    Returns
    -------
    np.ndarray of shape (N,)
        Weight vector normalised so that a_s^H w = 1.
    """
    a_soi = steering_vector(theta_soi, phi_soi, wavelength, positions)
    # ... solver logic ...
    w = h / (np.vdot(a_soi, h) + np.finfo(float).eps)
    return w
```

Then export it from `Functions/__init__.py` and add it to the `models` list in `main.py`.

### Changing the array geometry

Any positions array of shape $(N, 3)$ is accepted. Some common geometries:

```python
import numpy as np

# Uniform Circular Array (UCA), radius r, in the xy-plane
N, r = 8, 1.0
phi_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
pos_uca = np.column_stack([r*np.cos(phi_angles), r*np.sin(phi_angles), np.zeros(N)])

# Random 3-D deployment
pos_random = np.random.uniform(-5, 5, size=(16, 3))

# L-shaped planar array
pos_L = np.vstack([
    np.array([[i * 0.5, 0.0, 0.0] for i in range(6)]),    # horizontal arm
    np.array([[0.0, i * 0.5, 0.0] for i in range(1, 6)]), # vertical arm
])
```

### Changing the network topology

Use `dsp.connectivity_graph` to generate the adjacency matrix automatically from node centroid positions:

```python
from Functions import dsp
import numpy as np

# Compute the centroid of each node's sensor cluster
centers = np.array([pos.mean(axis=0) for pos in node_positions])

# Hybrid MST + 2-NN topology (recommended for consensus algorithms)
adj = dsp.connectivity_graph(centers, method='mst_knn', k=2)
```

---

## 10. Citation

If you use this code in academic work, please cite:

```bibtex
@software{mpdr_beamformer,
  author  = {Yuri Ribeiro dos Santos},
  title   = {{MPDR Beamformer}: A Research Implementation of the {Capon/MPDR}
             Beamformer with Distributed Consensus},
  year    = {2025},
  url     = {https://github.com/santosyr/MPDR-Beamformer},
}
```

---

## 11. References

- Capon, J. (1969). High-resolution frequency-wavenumber spectrum analysis. *Proceedings of the IEEE*, 57(8), 1408–1418.
- Van Trees, H. L. (2002). *Optimum Array Processing: Part IV of Detection, Estimation, and Modulation Theory*. Wiley-Interscience.
- Haykin, S. (2002). *Adaptive Filter Theory* (4th ed.). Prentice Hall.
- Xiao, L., & Boyd, S. (2004). Fast linear iterations for distributed averaging. *Systems & Control Letters*, 53(1), 65–78.
- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.
