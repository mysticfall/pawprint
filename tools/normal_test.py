# SPDX-License-Identifier: GPL-3.0-or-later
"""Production normal channel calibration, UV commit and persistence (factory Blender)."""
from pathlib import Path
import sys
import tempfile

import bpy
import numpy as np
from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from projection_test import load_extension, setup_fixture, create_projection


def main():
    ext = load_extension()
    scene, obj, _ = setup_fixture(bpy)
    scene.camera.data.clip_end = 100
    if '--eevee' in sys.argv:
        scene.render.engine = 'BLENDER_EEVEE'
    layer = create_projection(bpy, ext, obj, scene.camera)
    material = obj.active_material
    stack = material.pawprint
    scene.pawprint_base_width = scene.pawprint_base_height = 128
    for face in obj.data.polygons:
        left, right = (.1, .45) if face.index == 0 else (.55, .9)
        for loop, co in zip(face.loop_indices, [(left,.1),(right,.1),(right,.9),(left,.9)]):
            obj.data.uv_layers.active.data[loop].uv = co
    values = np.ones((128, 128, 4), dtype=np.float32)
    layer.image.pixels.foreach_set(values.ravel())
    layer.image.update()
    layer.image.pack()
    source_view = tuple(layer.view_matrix)
    with tempfile.TemporaryDirectory(prefix='pawprint-normal-test-', dir='/tmp/opencode') as directory:
        directory = Path(directory)
        def measure(rgb, alpha=1, visible=True, mirror='NONE', rotation=0):
            values[:, :, :3], values[:, :, 3] = rgb, alpha
            layer.normal = ext.normals.image(values)
            layer.visible = visible
            layer.mirror = mirror
            view = Matrix.Rotation(rotation, 4, 'Z') @ ext.projection.matrix(source_view)
            layer.view_matrix = ext.projection.flattened(view)
            # Compile basis while rotated; capture remains at original saved frame.
            ext.projection.build_material(material, emission=True, normal_bake=True)
            layer.view_matrix = source_view
            result = ext.capture.render_frame(bpy.context, layer, directory / 'normal.exr', scene_linear=True)
            return result[64, 40, :3] * 2 - 1
        def unit(v):
            return np.asarray(v) / np.linalg.norm(v)
        for rgb, expected in [((.5,.5,1),(0,0,1)), ((.8,.5,.9),(.6,0,.8)),
                              ((.5,.8,.9),(0,-.6,.8))]:
            assert np.allclose(measure(rgb), unit(expected), atol=.015), (rgb, measure(rgb), expected)
        partial = measure((.8,.5,.9), alpha=.5)
        assert np.allclose(partial, unit((.3,0,.9)), atol=.015), partial
        assert np.allclose(measure((.8,.5,.9), visible=False), (0,0,1), atol=.015)
        # Left half samples the kept positive-X half, with the vector reflected.
        assert np.allclose(measure((.8,.5,.9), mirror='X_PLUS'), unit((-.6,0,.8)), atol=.015)
        assert np.allclose(measure((.8,.5,.9), rotation=np.pi/2), unit((0,-.6,.8)), atol=.015)
        measure((.8,.5,.9))
        upper = stack.layers.add()
        for key in ('projection', 'view_matrix', 'view_rotation', 'view_location', 'view_distance',
                    'lens', 'clip_start', 'clip_end', 'width', 'height'):
            setattr(upper, key, getattr(layer, key))
        upper.depth = layer.depth
        upper.image = layer.image
        upper_values = values.copy()
        upper_values[:, :, :3], upper_values[:, :, 3] = (.5,.8,.9), .5
        upper.normal = ext.normals.image(upper_values)
        ext.projection.build_material(material, emission=True, normal_bake=True)
        composed = ext.capture.render_frame(bpy.context, layer, directory / 'composed.exr', scene_linear=True)
        assert np.allclose(composed[64,40,:3]*2-1, unit((.3,-.3,.8)), atol=.015)
        # Full coverage replaces lower detail; identical overlapping estimates
        # are idempotent rather than increasing their tilt at each layer.
        upper_values[:, :, 3] = 1
        upper.normal = ext.normals.image(upper_values)
        ext.projection.build_material(material, emission=True, normal_bake=True)
        replaced = ext.capture.render_frame(bpy.context, layer, directory / 'replaced.exr', scene_linear=True)
        assert np.allclose(replaced[64,40,:3]*2-1, unit((0,-.6,.8)), atol=.015)
        upper_values[:, :, :3] = (.8,.5,.9)
        upper.normal = ext.normals.image(upper_values)
        ext.projection.build_material(material, emission=True, normal_bake=True)
        repeated = ext.capture.render_frame(bpy.context, layer, directory / 'repeated.exr', scene_linear=True)
        assert np.allclose(repeated[64,40,:3]*2-1, unit((.6,0,.8)), atol=.015)
        stack.layers.remove(1)
        layer = stack.layers[0]
        ext.projection.build_material(material)
        before = ext.capture.pixels(layer.normal).copy()
        weights = np.zeros((128,128), np.float32)
        weights[20:80,20:80] = .4
        patch = values.copy()
        patch[:,:,:3] = (.5,.8,.9)
        mixed = ext.normals.merged(layer.normal, patch, weights)
        assert np.array_equal(mixed[weights == 0], before[weights == 0])
        assert np.allclose(np.linalg.norm(mixed[weights > 0,:3]*2-1, axis=1), 1)
        # UV handedness, not the saved image's red axis, determines export RGB.
        uv_data = obj.data.uv_layers[stack.uv_name].data
        for item in uv_data:
            item.uv.x = 1 - item.uv.x
        mirrored = ext.baking._bake(bpy.context, stack, 0, 'Mirrored UV Normal Test', normal=True)
        assert np.allclose(ext.capture.pixels(mirrored)[64, 96, :3]*2-1, (-.6,0,.8), atol=.03)
        for item in uv_data:
            item.uv.x = 1 - item.uv.x
        bpy.ops.ed.undo_push(message='Before paired UV bake')
        material_name = material.name
        assert bpy.ops.pawprint.commit_base('EXEC_DEFAULT', True) == {'FINISHED'}
        assert not stack.layers and stack.base_normal and stack.base_normal.packed_file
        decoded = ext.capture.pixels(stack.base_normal)[64, 32, :3] * 2 - 1
        assert np.allclose(decoded, (.6,0,.8), atol=.03), decoded
        normal_name = stack.base_normal.name
        bpy.ops.ed.undo()
        stack = bpy.data.materials[material_name].pawprint
        assert len(stack.layers) == 1 and stack.base_normal is None
        bpy.ops.ed.redo()
        stack = bpy.data.materials[material_name].pawprint
        assert not stack.layers and stack.base_normal.name == normal_name
        path = directory / 'normal.blend'
        bpy.ops.wm.save_as_mainfile(filepath=str(path))
        bpy.ops.wm.open_mainfile(filepath=str(path))
        stack = bpy.data.materials[material_name].pawprint
        assert stack.base_normal.is_float and stack.base_normal.packed_file
        assert stack.base_normal.colorspace_settings.name == 'Non-Color'
        assert stack.id_data.node_tree.nodes['Diffuse Albedo'].inputs['Normal'].is_linked
    # A curved, smooth mesh must not bake its broad shape into a neutral map.
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, location=(0, 0, 2))
    sphere = bpy.context.object
    sphere.rotation_euler = (.2, .4, .3)
    sphere.scale = (1, .7, 1.2)
    sphere.data.materials.append(bpy.data.materials.new('Sphere original material'))
    for polygon in sphere.data.polygons:
        polygon.use_smooth = True
    neutral_layer = create_projection(bpy, ext, sphere, bpy.context.scene.camera)
    sphere_stack = sphere.active_material.pawprint
    neutral = np.ones((128, 128, 4), dtype=np.float32)
    neutral[:, :, :3] = (.5, .5, 1)
    neutral_layer.normal = ext.normals.image(neutral)
    baked = ext.baking._bake(bpy.context, sphere_stack, 0, 'Curved Neutral Test', normal=True)
    pixels = ext.capture.pixels(baked)
    bare = ext.baking._bake(bpy.context, sphere_stack, -1, 'Bare Geometry Test', normal=True)
    assert np.allclose(pixels[:, :, :3], ext.capture.pixels(bare)[:, :, :3], atol=.015)
    assert np.max(np.abs(pixels[:, :, :3] - (.5, .5, 1))) < .02, (
        'Mesh curvature leaked into tangent normal map', pixels[:, :, :3].min(axis=(0, 1)),
        pixels[:, :, :3].max(axis=(0, 1)))
    assert baked['pawprint_normal_space'] == 'TANGENT'
    # Opaque neutral replacement clears lower texture detail, while zero normal
    # coverage preserves it. The replacement also applies over a committed base.
    detail = neutral.copy()
    detail[:, :, :3] = (.7, .5, .5 + np.sqrt(.84) / 2)
    sphere_stack.base_normal = ext.normals.image(detail)
    with_layer = ext.baking._bake(bpy.context, sphere_stack, 0, 'Replaced Detail Test', normal=True)
    without_layer = ext.baking._bake(bpy.context, sphere_stack, -1, 'Base Detail Test', normal=True)
    replaced_pixels = ext.capture.pixels(with_layer)[:, :, :3]
    base_pixels = ext.capture.pixels(without_layer)[:, :, :3]
    covered = np.max(np.abs(replaced_pixels - base_pixels), axis=2) > .05
    assert np.count_nonzero(covered) > 100
    # UV bake filtering includes partially covered boundary texels; require a
    # neutral opaque interior without mistaking that transition for full coverage.
    assert np.count_nonzero(np.max(np.abs(replaced_pixels - (.5,.5,1)), axis=2) < .02) > 100
    assert np.max(np.abs(replaced_pixels - (.5,.5,1))) <= np.max(np.abs(base_pixels - (.5,.5,1))) + .02
    assert np.max(np.abs(ext.capture.pixels(without_layer)[:, :, :3] - detail[:, :, :3])) < .03
    neutral[:, :, 3] = 0
    neutral_layer.normal = ext.normals.image(neutral)
    transparent = ext.baking._bake(bpy.context, sphere_stack, 0, 'Transparent Detail Test', normal=True)
    assert np.allclose(ext.capture.pixels(transparent), ext.capture.pixels(without_layer), atol=.015)
    print('Production normal axes, coverage, mirror, basis, replacement, idempotence, merge, tangent bake, curved neutrality, undo and persistence passed')


main()
