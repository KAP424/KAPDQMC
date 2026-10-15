"""Independent Fock-space tests of Wick contractions and complex conventions.

Run: python -m unittest discover -s test/Free_WaveFunction -p test_wavef_observables.py
"""
import unittest
from itertools import combinations
import numpy as np
from wavef_observables import (
    OPERATORS, local_mean, connected, density_connected, lattice_geometry,
)


def annihilator(n, i):
    result = np.zeros((2**n, 2**n), dtype=complex)
    for state in range(2**n):
        if state & (1 << i):
            parity = bin(state & ((1 << i) - 1)).count("1")
            result[state ^ (1 << i), state] = (-1)**parity
    return result


class GaussianObservableTests(unittest.TestCase):
    def test_complex_slater_against_fock_space(self):
        rng = np.random.default_rng(481)
        orbitals, _ = np.linalg.qr(rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2)))
        state = np.zeros(16, dtype=complex)
        for occupied in combinations(range(4), 2):
            state[sum(1 << i for i in occupied)] = np.linalg.det(orbitals[list(occupied), :])
        c = [annihilator(4, i) for i in range(4)]
        P = orbitals @ orbitals.conj().T
        actual_P = np.array([[np.vdot(state, c[j].conj().T @ c[i] @ state)
                              for j in range(4)] for i in range(4)])
        np.testing.assert_allclose(P, actual_P, atol=1e-12)
        operators = []
        for indices in ([0, 1], [2, 3]):
            for op in OPERATORS.values():
                fock = sum(op[a, b] * c[i].conj().T @ c[j]
                           for a, i in enumerate(indices) for b, j in enumerate(indices))
                mean = np.vdot(state, fock @ state)
                np.testing.assert_allclose(local_mean(P, indices, op), mean, atol=1e-12)
                operators.append((indices, op, fock, mean))
        # Includes coincident cells, different cells, and noncommuting Sx/Sy.
        for ia, a, fa, ma in operators:
            for ib, b, fb, mb in operators:
                exact = np.vdot(state, fa @ fb @ state) - ma * mb
                np.testing.assert_allclose(connected(P, ia, a, ib, b), exact, atol=1e-12)
        for i in range(4):
            for j in range(4):
                ni, nj = c[i].conj().T @ c[i], c[j].conj().T @ c[j]
                exact = np.vdot(state, ni @ nj @ state) - P[i, i] * P[j, j]
                np.testing.assert_allclose(density_connected(P)[i, j], exact, atol=1e-12)

    def test_sy_sign_and_mixed_zero_modes(self):
        spinor = np.array([1, 1j]) / np.sqrt(2)
        P = np.outer(spinor, spinor.conj())
        self.assertAlmostEqual(local_mean(P, [0, 1], OPERATORS["Sy"]).real, 0.5)
        mixed = np.eye(2) / 2
        # Independent half-occupied modes have Var(N)=1/2, unlike a fixed-N mixture.
        self.assertAlmostEqual(connected(mixed, [0, 1], OPERATORS["N"],
                                        [0, 1], OPERATORS["N"]).real, 0.5)

    def test_rectangular_geometry_and_reference(self):
        for lx, ly in ((12, 12), (5, 4), (3, 5)):
            meta = {"Lx": lx, "Ly": ly, "a1": [np.sqrt(3), 0],
                    "a2": [-np.sqrt(3)/2, 1.5], "delta_AB": [-np.sqrt(3)/2, 0.5]}
            for ref in ((1, 1), (lx, ly)):
                geo = lattice_geometry(meta, ref)
                self.assertEqual(len(np.unique(geo["displacements"], axis=0)), lx * ly)
                np.testing.assert_allclose(geo["centers"][geo["reference_cell"]], [0, 0])
                self.assertEqual(len(geo["bond_pairs"]), 3 * lx * ly)
                self.assertEqual(len(np.unique(geo["bond_pairs"], axis=0)), 3 * lx * ly)
                degree = np.bincount(geo["bond_pairs"].ravel())
                np.testing.assert_array_equal(degree, np.full(2 * lx * ly, 3))
                # Every displayed or wrapped neighbour has a nearest image at distance 1.
                a1, a2 = geo["basis"]
                translations = np.array([u * lx * a1 + v * ly * a2
                                         for u in (-1, 0, 1) for v in (-1, 0, 1)])
                for i, j in geo["bond_pairs"]:
                    diff = geo["positions"][j] - geo["positions"][i]
                    self.assertAlmostEqual(np.linalg.norm(diff + translations, axis=1).min(), 1.0)


if __name__ == "__main__":
    unittest.main()
