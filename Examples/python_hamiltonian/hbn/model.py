"""Nearest-neighbour hBN model. Lengths are in angstroms and energies in eV."""
import numpy as np

# Model parameters: lattice constant, onsite energy and nearest-neighbour hopping.
a = 2.50
onsite = 3.625
t = 2.30
material = "hBN"

# Lattice: each row says where the next equivalent unit cell starts.
a1 = np.array([a, 0.0, 0.0])
a2 = np.array([a / 2, np.sqrt(3) * a / 2, 0.0])
a3 = np.array([0.0, 0.0, 20.0])
lattice = np.array([a1, a2, a3])

centres = np.array([[0.0, 0.0, 0.0], [0.0, a / np.sqrt(3), 0.0]])


def hamiltonian(k):
    """Return the periodic 2 × 2 electronic Hamiltonian at Cartesian k (1/Å)."""
    k_xy = np.asarray(k[:2])

    f = (1 + np.exp(-1j * (k_xy @ a2[:2])) + np.exp(1j * (k_xy @ (a1 - a2)[:2])))

    # Diagonal entries are the N and B onsite energies. The other entries are hoppings.
    return np.array([[-onsite, -t * f],
                     [-t * f.conjugate(), onsite]])



# def hamiltonian(k):
#    k = np.asarray(k[:2])
#    tau1 = np.array([0, a / np.sqrt(3)])
#    tau2 = np.array([a / 2, -a / (2 * np.sqrt(3))])
#    tau3 = np.array([-a / 2, -a / (2 * np.sqrt(3))])
#    f = np.exp(1j * (k @ tau1)) + np.exp(1j * (k @ tau2)) + np.exp(1j * (k @ tau3))
