# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender factory/background test; optional -- --eevee."""
from pathlib import Path
import sys
import tempfile

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from projection_test import load_extension, setup_fixture, create_projection


def main():
    ext = load_extension()
    scene, obj, original = setup_fixture(bpy)
    scene.camera.data.clip_end = 100
    if '--eevee' in sys.argv:
        scene.render.engine = 'BLENDER_EEVEE'
    layer = create_projection(bpy, ext, obj, scene.camera)
    values = np.ones((128, 128, 4), dtype=np.float32)
    values[:, :, :3] = (.4, .25, .2)
    layer.image.pixels.foreach_set(values.ravel())
    layer.image.update()
    # A no-material object exercises temporary slot creation/removal.
    bpy.ops.mesh.primitive_cube_add(size=.5, location=(.8, 0, .5))
    cube = bpy.context.object
    bpy.context.view_layer.objects.active = obj
    background = scene.world.node_tree.nodes['Background']
    background.inputs['Color'].default_value = (.1, .4, 1, 1)
    background.inputs['Strength'].default_value = 5
    materials = [(slot.link, slot.material) for slot in obj.material_slots]
    before = ext.capture.pixels(layer.image).copy()
    counts = (len(bpy.data.materials), len(bpy.data.scenes), len(bpy.data.images))
    with tempfile.TemporaryDirectory(dir='/tmp/opencode', prefix='pawprint-albedo-reference-') as directory:
        directory = Path(directory)
        reference = ext.capture.albedo_reference(bpy.context, layer, directory)
        np.testing.assert_allclose(reference[64, 40, :3], (.4, .25, .2), atol=2 / 255)
        assert reference[64, 40, 3] > .99
        assert reference[64, 100, 3] == 0, 'Other slot/occluder contributed matching samples'
        assert [(slot.link, slot.material) for slot in obj.material_slots] == materials
        assert not len(cube.data.materials)
        assert np.array_equal(ext.capture.pixels(layer.image), before)
        assert counts == (len(bpy.data.materials), len(bpy.data.scenes), len(bpy.data.images))
        # Changed lighting cannot affect the reference.
        background.inputs['Strength'].default_value = .01
        again = ext.capture.albedo_reference(bpy.context, layer, directory)
        np.testing.assert_allclose(again, reference, atol=1 / 255)
        # The untouched initial gray base is not a colour-matching reference.
        values[:, :, 3] = 0
        layer.image.pixels.foreach_set(values.ravel())
        layer.image.update()
        empty = ext.capture.albedo_reference(bpy.context, layer, directory)
        assert empty[64, 40, 3] == 0
        # Explicit gray paint remains valid reference texture.
        values[:] = (.18, .18, .18, 1)
        layer.image.pixels.foreach_set(values.ravel())
        layer.image.update()
        gray = ext.capture.albedo_reference(bpy.context, layer, directory)
        assert gray[64, 40, 3] > .99
        before = ext.capture.pixels(layer.image).copy()
        render = ext.capture.render_frame
        def fail(*args, **kwargs):
            raise RuntimeError('Injected albedo reference capture failure')
        ext.capture.render_frame = fail
        try:
            try:
                ext.capture.albedo_reference(bpy.context, layer, directory)
                raise AssertionError('Expected reference capture failure')
            except RuntimeError as exc:
                assert 'Injected' in str(exc)
        finally:
            ext.capture.render_frame = render
        assert [(slot.link, slot.material) for slot in obj.material_slots] == materials
        assert not len(cube.data.materials)
        assert np.array_equal(ext.capture.pixels(layer.image), before)
        assert counts == (len(bpy.data.materials), len(bpy.data.scenes), len(bpy.data.images))
    print({'unlit_target_albedo': True, 'trusted_coverage': True, 'reference_cleanup': True})


main()
