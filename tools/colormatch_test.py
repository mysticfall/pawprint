# SPDX-License-Identifier: GPL-3.0-or-later
"""Run with system Python: boundary statistics and exact preservation checks."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('colormatch', Path(__file__).resolve().parents[1] / 'pawprint/colormatch.py')
match = importlib.util.module_from_spec(spec)
spec.loader.exec_module(match)


class BoundaryMatchTest(unittest.TestCase):
    def fixture(self):
        y, x = np.mgrid[:128, :128]
        lab = np.stack((45 + x * .05, 8 + y * .02, 5 + x * .03), axis=-1)
        reference = np.ones((128, 128, 4), dtype=np.float32)
        reference[:, :, :3] = match._rgb(lab)
        weights = np.zeros((128, 128), dtype=np.float32)
        weights[40:88, 40:88] = 1
        weights[39, 40:88] = .5
        metadata = dict(albedo_reference=reference, weights=weights, bounds=(24, 24, 104, 104))
        source = reference.copy()
        source[:, :, :3] = match._rgb(lab + (5, 3, -4))
        source[:, :, 3] = .75
        return source, reference, metadata

    def test_corrects_shift_preserves_alpha_and_outside(self):
        source, reference, metadata = self.fixture()
        result = match.match(source, metadata)
        edit = metadata['weights'] > 0
        np.testing.assert_allclose(result[:, :, :3][edit], reference[:, :, :3][edit], atol=2e-5)
        self.assertTrue(np.array_equal(result[~edit], source[~edit]))
        self.assertTrue(np.array_equal(result[:, :, 3], source[:, :, 3]))

    def test_interior_feature_does_not_train_correction(self):
        source, reference, metadata = self.fixture()
        desired = match._rgb(np.array([55., -8., 20.]))
        source[50:70, 50:70, :3] = match._rgb(np.array([60., -5., 16.]))
        result = match.match(source, metadata)
        np.testing.assert_allclose(result[60, 60, :3], desired, atol=2e-5)

    def test_untrusted_background_does_not_bias_statistics(self):
        source, reference, metadata = self.fixture()
        reference[:, :45, 3] = 0
        reference[:, :45, :3] = 0
        source[:, :45, :3] = 1
        result = match.match(source, metadata)
        np.testing.assert_allclose(result[60, 60, :3], reference[60, 60, :3], atol=2e-5)

    def test_missing_empty_and_full_frame_reference_skip(self):
        source, reference, metadata = self.fixture()
        self.assertIs(match.match(source, {'weights': metadata['weights']}), source)
        reference[:, :, 3] = 0
        self.assertIs(match.match(source, metadata), source)
        reference[:, :, 3] = 1
        metadata['weights'][:] = 1
        self.assertIs(match.match(source, metadata), source)

    def test_uniform_known_colour_uses_offset_not_unstable_gain(self):
        source, reference, metadata = self.fixture()
        reference[:, :, :3] = (.4, .3, .2)
        for shift in ((5, 3, -4), (25, 3, -4)):
            with self.subTest(shift=shift):
                source[:, :, :3] = match._rgb(match._lab(np.array([.4, .3, .2])) + shift)
                result = match.match(source, metadata)
                np.testing.assert_allclose(result[60, 60, :3], (.4, .3, .2), atol=2e-5)


if __name__ == '__main__':
    unittest.main()
