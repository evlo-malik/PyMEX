"""Liu et al., PRB 88, 085433, corrected arXiv v4: Eqs. 4–11, Table II (GGA)."""
import numpy as np

material = "MoS2"  # MoS2, MoSe2, WS2 or WSe2

# a in angstrom; all the other parameters in eV.
parameters = {
    "MoS2":  dict(a=3.190, epsilon1=1.046, epsilon2=2.104, t0=-0.184,
                  t1=0.401, t2=0.507, t11=0.218, t12=0.338, t22=0.057),
    "MoSe2": dict(a=3.326, epsilon1=0.919, epsilon2=2.065, t0=-0.188,
                  t1=0.317, t2=0.456, t11=0.211, t12=0.290, t22=0.130),
    "WS2":   dict(a=3.191, epsilon1=1.130, epsilon2=2.275, t0=-0.206,
                  t1=0.567, t2=0.536, t11=0.286, t12=0.384, t22=-0.061),
    "WSe2":  dict(a=3.325, epsilon1=0.943, epsilon2=2.179, t0=-0.207,
                  t1=0.457, t2=0.486, t11=0.263, t12=0.329, t22=0.034),
}


def hamiltonian(k):
    k_x, k_y = k[:2]
    p = parameters[material]
    a = p["a"]
    epsilon1, epsilon2 = p["epsilon1"], p["epsilon2"]
    t0, t1, t2 = p["t0"], p["t1"], p["t2"]
    t11, t12, t22 = p["t11"], p["t12"], p["t22"]

    # Angles for the nearest-neighbour hopping phases.
    alpha = k_x * a / 2
    beta = np.sqrt(3) * k_y * a / 2

    # Diagonal entries: on-site energy plus hopping to neighbouring cells.
    h0 = epsilon1 + 2 * t0 * (np.cos(2 * alpha) + 2 * np.cos(alpha) * np.cos(beta))
    h11 = (
        epsilon2 + 2 * t11 * np.cos(2 * alpha)
        + (t11 + 3 * t22) * np.cos(alpha) * np.cos(beta)
    )
    h22 = (
        epsilon2 + 2 * t22 * np.cos(2 * alpha)
        + (3 * t11 + t22) * np.cos(alpha) * np.cos(beta)
    )

    # Off-diagonal entries: keep the real and imaginary contributions separate.
    h1_real = -2 * np.sqrt(3) * t2 * np.sin(alpha) * np.sin(beta)
    h1_imag = 2 * t1 * (np.sin(2 * alpha) + np.sin(alpha) * np.cos(beta))
    h1 = h1_real + 1j * h1_imag

    h2_real = 2 * t2 * (np.cos(2 * alpha) - np.cos(alpha) * np.cos(beta))
    h2_imag = 2 * np.sqrt(3) * t1 * np.cos(alpha) * np.sin(beta)
    h2 = h2_real + 1j * h2_imag

    h12_real = np.sqrt(3) * (t22 - t11) * np.sin(alpha) * np.sin(beta)
    h12_imag = 4 * t12 * np.sin(alpha) * (np.cos(alpha) - np.cos(beta))
    h12 = h12_real + 1j * h12_imag

    # Basis order: d_z2, d_xy, d_x2-y2. The lower half is the conjugate of the upper.
    matrix = np.array([
        [h0,          h1,           h2],
        [np.conj(h1), h11,          h12],
        [np.conj(h2), np.conj(h12), h22],
    ], dtype=complex)
    return matrix


a = parameters[material]["a"]
lattice = np.array([
    [a,     0.0,                0.0],
    [a / 2, np.sqrt(3) * a / 2, 0.0],
    [0.0,   0.0,               20.0],
])

# The three metal d orbitals share one site, in Hamiltonian row order.
centres = np.array([
    [0.0, 0.0, 0.0],  # d_z2
    [0.0, 0.0, 0.0],  # d_xy
    [0.0, 0.0, 0.0],  # d_x2-y2
])
