# SPDX-License-Identifier: GPL-3.0-or-later
"""Experimental full-strength, multiscale removal of inferred mesh normal flow.

This is angular regression, not a calibrated Chord camera model. Correlated
relief can be removed along with curvature; there are deliberately no confidence
or strength gates in this experimental strategy.
"""
import numpy as np


def _unit(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-10)


def _log(values):
    values = _unit(values)
    length = np.linalg.norm(values[..., :2], axis=-1, keepdims=True)
    theta = np.arctan2(length, values[..., 2:3])
    return values[..., :2] * theta / np.maximum(length, 1e-10)


def _exp(values):
    theta = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.concatenate((values * np.sin(theta) / np.maximum(theta, 1e-10),
                           np.cos(theta)), axis=-1)


def _smooth(values, mask, sigma):
    height, width = mask.shape
    kernel = np.exp(-2 * np.pi**2 * sigma**2 *
                    (np.fft.fftfreq(height)[:, None]**2 +
                     np.fft.rfftfreq(width)[None, :]**2))
    weight = np.fft.irfft2(np.fft.rfft2(mask) * kernel, s=mask.shape)
    blurred = np.fft.irfft2(np.fft.rfft2(values * mask[..., None], axes=(0, 1)) *
                           kernel[..., None], s=mask.shape, axes=(0, 1))
    return blurred / np.maximum(weight[..., None], 1e-8)


def match(rgba, metadata):
    """Correct placed saved-frame normals; preserve alpha and uneditable pixels.

    Mesh reference RGB is signed camera right/down/toward; alpha is target-slot
    coverage. The fit uses opaque editable interior, while correction extends
    through feathered coverage. Missing reference/interior leaves pixels intact.
    FFT smoothing retains the trial's periodic-boundary behavior.
    """
    reference = metadata.get('mesh_normals')
    if reference is None:
        return rgba
    weights = metadata['weights']
    finite = np.all(np.isfinite(reference), axis=-1) & np.all(np.isfinite(rgba), axis=-1)
    valid = (finite & (weights > 0) & (rgba[..., 3] > 0) &
             (reference[..., 3] > 1e-5) &
             (np.linalg.norm(reference[..., :3], axis=-1) > .5))
    fit = valid & (weights > .95) & (rgba[..., 3] > .95) & (reference[..., 3] > .95)
    if not np.any(fit):
        return rgba
    normal = _log(np.where(valid[..., None], rgba[..., :3] * 2 - 1, (0, 0, 1)))
    geometry = _log(np.where(valid[..., None], reference[..., :3], (0, 0, 1)))
    # Same proportional scales as the 1024px trial; no protected fine band.
    scales = np.array((2, 8, 32, 128)) * max(weights.shape) / 1024
    predicted = np.zeros_like(normal)
    previous_normal, previous_geometry = normal, geometry

    def accumulate(mesh_band, normal_band):
        mesh_fit, normal_fit = mesh_band[fit], normal_band[fit]
        energy = np.sum(mesh_fit * mesh_fit, axis=0)
        gain = np.divide(np.sum(mesh_fit * normal_fit, axis=0), energy,
                         out=np.zeros(2), where=energy > 1e-12)
        return mesh_band * gain

    for sigma in scales:
        smooth_normal = _smooth(normal, valid, sigma)
        smooth_geometry = _smooth(geometry, valid, sigma)
        predicted += accumulate(previous_geometry - smooth_geometry,
                                previous_normal - smooth_normal)
        previous_normal, previous_geometry = smooth_normal, smooth_geometry
    predicted += accumulate(previous_geometry, previous_normal)
    residual = _exp(normal - predicted)
    output = rgba.copy()
    output[valid, :3] = residual[valid] * .5 + .5
    return output
