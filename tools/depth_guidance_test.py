# SPDX-License-Identifier: GPL-3.0-or-later
"""Target-focused depth checks in an isolated factory Blender; run with Python."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile


def worker():
    import bpy
    import numpy as np
    from types import SimpleNamespace
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from projection_test import load_extension

    ext = load_extension()
    normalize = ext.capture.normalize_geometry_depth
    # Rare same-object outliers must not compress the main surface's contrast.
    target = np.linspace(4, 6, 10000, dtype=np.float32).reshape(100, 100)
    target[0, 0], target[-1, -1] = 1, 1000
    result = normalize(target, target)
    assert result[0, 0] == 1 and result[-1, -1] == 0
    assert result.std() > .28
    assert np.all(np.diff(result.ravel()[1:-1]) <= 0)
    assert np.all(normalize(np.full((8, 8), 5.), np.full((8, 8), 5.)) == 1)
    try:
        normalize(np.zeros((8, 8)), np.zeros((8, 8)))
    except ValueError:
        pass
    else:
        raise AssertionError('Empty geometry must report an error')

    scene = bpy.data.scenes.new('Depth guidance fixture')
    bpy.context.window.scene = scene
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, location=(-.7, 0, 0))
    target_object = bpy.context.object
    target_object.name = 'Depth target'
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -40))
    background = bpy.context.object
    background.name = 'Unrelated distant background'
    # Sloped background gives the no-target fallback real variation.
    background.rotation_euler.y = .2
    camera = bpy.data.objects.new('Depth test eye', bpy.data.cameras.new('Depth test eye'))
    scene.collection.objects.link(camera)
    camera.location = (0, 0, 5)
    camera.data.lens = 35
    bpy.context.view_layer.update()
    view = camera.matrix_world.inverted()
    window = camera.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=96, y=96)
    frame = SimpleNamespace(width=96, height=96, view_matrix=ext.projection.flattened(view),
                            projection=ext.projection.flattened(window @ view),
                            clip_start=.1, clip_end=100)
    layer = SimpleNamespace(**vars(frame), image=SimpleNamespace(size=(96, 96)))
    scene_depth = ext.projection.frame_depths(bpy.context, frame)
    target_depth = ext.projection.frame_depths(bpy.context, frame, only_original=target_object)
    visible = target_depth > 0
    assert visible.any() and (~visible).any()
    focused = normalize(target_depth, scene_depth)
    old = (scene_depth.max() - scene_depth) / (scene_depth.max() - scene_depth.min())
    assert focused[visible].std() > old[visible].std() * 10
    assert np.all(focused[~visible] == 0)

    # Snapshot values must still be exactly the unnormalized full-scene depths.
    snapshot = ext.projection.depth_image(bpy.context, frame)
    assert np.array_equal(ext.capture.pixels(snapshot)[:, :, 0], scene_depth)
    before = ext.capture.pixels(snapshot).copy()
    with tempfile.TemporaryDirectory(prefix='pawprint-depth-', dir='/tmp/opencode') as temp:
        directory = Path(temp)
        def output(bounds):
            size = (bounds[2] - bounds[0], bounds[3] - bounds[1])
            ext.capture.geometry_depth(bpy.context, layer, directory,
                                       dict(bounds=bounds, request_size=size), target_object)
            image = bpy.data.images.load(str(directory / 'depth.png'), check_existing=False)
            image.colorspace_settings.name = 'Non-Color'
            pixels = ext.capture.pixels(image).copy()
            bpy.data.images.remove(image)
            assert np.all(pixels[:, :, 3] == 1)
            return pixels[:, :, 0]

        bounds = (0, 0, 96, 96)
        full = output(bounds)
        assert np.allclose(full, focused, atol=1/255)
        assert np.array_equal(full, output(bounds)), 'Shared preview/generate path is deterministic'
        # Crop stats exclude other target depths as well as background distances.
        crop = (15, 35, 35, 55)
        x0, y0, x1, y1 = crop
        expected = normalize(target_depth[y0:y1, x0:x1], scene_depth[y0:y1, x0:x1])
        assert np.allclose(output(crop), expected, atol=1/255)
        assert not np.allclose(expected, focused[y0:y1, x0:x1], atol=.01)
        # Rightmost crop sees only surrounding geometry: fallback stays usable.
        crop = (80, 0, 96, 96)
        assert not np.any(target_depth[:, 80:])
        fallback = output(crop)
        assert fallback.std() > .25 and fallback.min() == 0 and fallback.max() == 1
        # An unrelated foreground occluder must not reveal target depth behind it.
        bpy.ops.mesh.primitive_plane_add(size=.8, location=(-.7, 0, 2))
        bpy.context.view_layer.update()
        occluded_scene = ext.projection.frame_depths(bpy.context, frame)
        blocked = visible & (occluded_scene < target_depth - .01)
        assert blocked.any()
        assert np.all(output(bounds)[blocked] == 0)
        assert np.array_equal(before, ext.capture.pixels(snapshot))
    bpy.data.images.remove(snapshot)
    ext.unregister()
    print('Depth guidance checks passed')


if __name__ == '__main__':
    if '--worker' in sys.argv:
        worker()
    else:
        subprocess.run([os.environ.get('BLENDER', 'blender'), '--background', '--factory-startup',
                        '--python-exit-code', '1', '--python', str(Path(__file__).resolve()),
                        '--', '--worker'], check=True)
