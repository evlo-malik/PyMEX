"""Check model contracts and unsupported inputs without running a BSE solver."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from python_tb import PythonTB
from python_tb_config import prepare_python_config, uses_python_model, validate_band_windows


class InputContractTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / "Examples/python_hamiltonian/hbn/pymex_tb.yaml"
        self.config = yaml.safe_load(path.read_text())
        # PyMEX normalizes keys and non-filename strings when it reads YAML.
        self.config["material"] = self.config["material"].lower()
        self.config["bse"]["qpoints"] = self.config["bse"].pop("Qpoints")
        self.config["absorption"] = self.config.pop("Absorption")
        self.config["eh_interaction"]["real"]["potential"] = "keldysh"

    def test_wannier_configuration_is_left_unchanged(self):
        legacy = {"bse": {"method": "tb", "wannier_io": {"win_file": "model.win"},
                          "tb": {"bands": {"engine": "wannier90"}}}}
        original = deepcopy(legacy)
        self.assertFalse(uses_python_model(legacy))
        prepare_python_config(legacy)
        self.assertEqual(legacy, original)

    def test_energy_and_vector_windows_are_identical_independent_copies(self):
        prepare_python_config(self.config)
        tb = self.config["bse"]["tb"]
        self.assertEqual(tb["bands"], tb["cnmk"])
        self.assertIsNot(tb["bands"], tb["cnmk"])

    def test_inconsistent_energy_and_vector_windows_are_rejected(self):
        self.config["bse"]["tb"]["cnmk"] = {"vb_idx": [0, 1], "cb_idx": [1, 3]}
        with self.assertRaisesRegex(ValueError, "same bands"):
            prepare_python_config(self.config)

    def test_finite_q_is_rejected(self):
        self.config["bse"]["qpoints"]["hspoints"] = [[0.1, 0.0, 0.0]]
        with self.assertRaisesRegex(ValueError, "Q ="):
            prepare_python_config(self.config)

    def test_unsupported_modes_are_rejected(self):
        for key, value in (("system", "gpu"), ("restart", "restart"), ("material", "3d")):
            config = deepcopy(self.config)
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                prepare_python_config(config)

    def test_screened_exchange_requires_direct(self):
        self.config["eh_interaction"]["include"].update(exchange=True, screen_exchange=True)
        with self.assertRaisesRegex(ValueError, "requires direct"):
            prepare_python_config(self.config)

    def test_invalid_band_windows_are_rejected(self):
        for bands in ({"vb_idx": [0, 2], "cb_idx": [1, 3]},
                      {"vb_idx": [-1, 0], "cb_idx": [1, 2]},
                      {"vb_idx": [0, 1], "cb_idx": [1, 4]}):
            with self.subTest(bands=bands), self.assertRaises(ValueError):
                validate_band_windows(bands, 3)

    def test_two_model_examples_satisfy_the_input_contract(self):
        for name, orbitals in (("hbn", 2), ("tmd", 3)):
            with self.subTest(model=name):
                reader = PythonTB(ROOT / "Examples/python_hamiltonian" / name / "model.py", [6, 6, 1])
                k = np.array([0.137, 0.293, 0.0]) @ reader.get_reciprocal()
                matrix, gradient = reader.get_Hk_and_grad_Hk(kpt=k)
                self.assertEqual(matrix.shape, (orbitals, orbitals, 1))
                self.assertEqual(gradient.shape, (orbitals, orbitals, 1, 3))
                np.testing.assert_allclose(matrix[:, :, 0], matrix[:, :, 0].conj().T, atol=1e-12)
                np.testing.assert_array_equal(gradient[:, :, 0, 2], 0)


if __name__ == "__main__":
    unittest.main()
