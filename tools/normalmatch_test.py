# SPDX-License-Identifier: GPL-3.0-or-later
"""System Python tests for experimental mesh-flow removal (no Blender needed)."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('normalmatch', Path(__file__).resolve().parents[1] / 'pawprint/normalmatch.py')
normalmatch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(normalmatch)


class NormalMatchTest(unittest.TestCase):
    def fixture(self, detail=None):
        y, x = np.mgrid[-1:1:256j, -1:1:256j]
        mesh = normalmatch._exp(np.stack((.45 * x, np.zeros_like(x)), -1))
        if detail is None:
            detail = np.zeros((*x.shape, 2))
        # Angular inputs let these tests isolate the regression exactly.
        estimate = normalmatch._exp(normalmatch._log(mesh) + detail)
        rgba = np.concatenate((estimate * .5 + .5, np.ones((*x.shape, 1))), -1)
        reference = np.concatenate((mesh, np.ones((*x.shape, 1))), -1)
        return rgba, dict(mesh_normals=reference, weights=np.ones(x.shape)), detail

    def test_mesh_only_becomes_neutral(self):
        rgba, metadata, _ = self.fixture()
        out = normalmatch.match(rgba, metadata)
        np.testing.assert_allclose(out[..., :3], np.broadcast_to((.5, .5, 1), out[..., :3].shape), atol=1e-10)

    def test_unmatched_fine_and_broad_detail(self):
        y, x = np.mgrid[-1:1:256j, -1:1:256j]
        for detail in (np.stack((.025 * np.sin(24 * np.pi * y), .025 * np.cos(24 * np.pi * x)), -1),
                       np.stack((.18 * np.sin(np.pi * y), np.zeros_like(x)), -1)):
            rgba, metadata, _ = self.fixture(detail)
            out = normalmatch.match(rgba, metadata)
            error = normalmatch._log(out[..., :3] * 2 - 1) - detail
            self.assertLess(np.degrees(np.max(np.linalg.norm(error, axis=-1))), .1)

    def test_flat_mesh_preserves_relief(self):
        rgba, metadata, _ = self.fixture()
        metadata['mesh_normals'][..., :3] = (0, 0, 1)
        np.testing.assert_allclose(normalmatch.match(rgba, metadata), rgba, atol=1e-12)

    def test_alpha_outside_occlusion_and_feather(self):
        rgba, metadata, _ = self.fixture()
        metadata['weights'][:20] = 0
        metadata['weights'][20:30] = .5
        metadata['mesh_normals'][..., 3][:, :20] = 0
        rgba[..., 3][:, 20:30] = .5
        original = rgba.copy()
        out = normalmatch.match(rgba, metadata)
        np.testing.assert_array_equal(out[..., 3], original[..., 3])
        np.testing.assert_array_equal(out[:20], original[:20])
        np.testing.assert_array_equal(out[:, :20], original[:, :20])
        np.testing.assert_array_equal(rgba, original)
        np.testing.assert_allclose(out[20:30, 30:, :3], np.broadcast_to((.5, .5, 1), out[20:30, 30:, :3].shape), atol=1e-10)

    def test_missing_or_empty_reference(self):
        rgba, metadata, _ = self.fixture()
        self.assertIs(normalmatch.match(rgba, {}), rgba)
        metadata['weights'][:] = .5
        self.assertIs(normalmatch.match(rgba, metadata), rgba)

    def test_coincident_relief_is_aggressively_removed(self):
        # Explicit limitation: this strategy cannot distinguish relief from
        # curvature when their spatial/angular fields coincide.
        rgba, metadata, _ = self.fixture()
        detail = normalmatch._log(metadata['mesh_normals'][..., :3]) * .2
        rgba, metadata, _ = self.fixture(detail)
        out = normalmatch.match(rgba, metadata)
        np.testing.assert_allclose(out[..., :3], np.broadcast_to((.5, .5, 1), out[..., :3].shape), atol=1e-10)


if __name__ == '__main__':
    unittest.main()
