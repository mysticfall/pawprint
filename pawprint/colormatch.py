# SPDX-License-Identifier: GPL-3.0-or-later
"""Boundary-derived albedo colour matching, independent of Blender and lighting."""
import numpy as np


def _lab(rgb):
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    xyz = linear @ np.array([[.4124564, .2126729, .0193339],
                             [.3575761, .7151522, .1191920],
                             [.1804375, .0721750, .9503041]])
    xyz /= (.95047, 1, 1.08883)
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack((116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])), axis=-1)


def _rgb(lab):
    y = (lab[..., 0] + 16) / 116
    f = np.stack((y + lab[..., 1] / 500, y, y - lab[..., 2] / 200), axis=-1)
    xyz = np.where(f > 6 / 29, f ** 3, 3 * (6 / 29) ** 2 * (f - 4 / 29))
    xyz *= (.95047, 1, 1.08883)
    linear = xyz @ np.array([[3.2404542, -.9692660, .0556434],
                             [-1.5371385, 1.8760108, -.2040259],
                             [-.4985314, .0415560, 1.0572252]])
    linear = np.clip(linear, 0, 1)
    return np.where(linear <= .0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - .055)


def _dilate(mask, radius):
    padded = np.pad(mask.astype(np.int32), radius)
    integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)
    width = 2 * radius + 1
    return (integral[width:, width:] - integral[:-width, width:]
            - integral[width:, :-width] + integral[:-width, :-width]) > 0


def match(rgba, metadata):
    """Correct only editable RGB; return unchanged if no trusted boundary exists.

    The reference is a fresh emission capture of the existing target-slot stack.
    Its alpha is trusted texture coverage, not the generation selection. Statistics
    genuinely omit excluded pixels, unlike zero-filling a masked LAB image.
    """
    reference = metadata.get('albedo_reference')
    if reference is None:
        return rgba
    weights = metadata['weights']
    if reference.shape != rgba.shape or weights.shape != rgba.shape[:2]:
        raise ValueError('Albedo matching reference and selection must align with the saved frame')
    edit = weights > 0
    x0, y0, x1, y1 = metadata['bounds']
    available = np.zeros_like(edit)
    available[y0:y1, x0:x1] = True
    radius = min(24, max(4, round(min(x1 - x0, y1 - y0) * .02)))
    ring = _dilate(edit, radius) & ~edit & available
    confidence = reference[:, :, 3] ** 2 * rgba[:, :, 3]
    ring &= confidence > .05
    if np.count_nonzero(ring) < 64:
        return rgba
    source, target = _lab(rgba[:, :, :3][ring]), _lab(reference[:, :, :3][ring])
    sample_weights = confidence[ring].astype(np.float64)
    # Discard inconsistent pairs (eg. a generated feature drifting into the band).
    residual = target - source
    distance = np.linalg.norm(residual - np.median(residual, axis=0), axis=1)
    keep = distance <= np.percentile(distance, 90)
    source, target, sample_weights = source[keep], target[keep], sample_weights[keep]
    if (sample_weights.sum() < 16
            or sample_weights.sum() ** 2 / np.square(sample_weights).sum() < 64):
        return rgba

    def moments(values):
        mean = np.average(values, axis=0, weights=sample_weights)
        std = np.sqrt(np.average((values - mean) ** 2, axis=0, weights=sample_weights))
        return mean, std

    source_mean, source_std = moments(source)
    target_mean, target_std = moments(target)
    gain = np.ones(3)
    stable = (source_std > 1) & (target_std > 1)
    gain[stable] = np.clip(target_std[stable] / source_std[stable], .5, 2)
    offset = target_mean - source_mean
    result = rgba.copy()
    result[:, :, :3][edit] = _rgb((_lab(rgba[:, :, :3][edit]) - source_mean) * gain
                                 + source_mean + offset)
    return result
