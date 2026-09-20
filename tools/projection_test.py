# SPDX-License-Identifier: GPL-3.0-or-later
"""Isolated projection render/persistence checks; run with system Python."""

import argparse
from array import array
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def load_extension():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import pawprint
    pawprint.register()
    return pawprint


def setup_fixture(bpy):
    scene = bpy.data.scenes.new("Pawprint Projection Test")
    bpy.context.window.scene = scene
    mesh = bpy.data.meshes.new("Target surfaces")
    # Left/right target slots and a rear face hidden from the saved eye.
    mesh.from_pydata(
        [(-2, -2, 0), (0, -2, 0), (0, 2, 0), (-2, 2, 0),
         (2, -2, 0), (2, 2, 0),
         (-2, -2, -1), (0, -2, -1), (0, 2, -1), (-2, 2, -1)],
        [], [(0, 1, 2, 3), (1, 4, 5, 2), (6, 7, 8, 9)],
    )
    mesh.uv_layers.new(name="UVMap")
    obj = bpy.data.objects.new("Pawprint Test Target", mesh)
    scene.collection.objects.link(obj)
    material = bpy.data.materials.new("Original red slot")
    material.use_nodes = True
    material.node_tree.nodes.clear()
    output = material.node_tree.nodes.new('ShaderNodeOutputMaterial')
    emission = material.node_tree.nodes.new('ShaderNodeEmission')
    emission.inputs[0].default_value = (1, 0, 0, 1)
    material.node_tree.links.new(emission.outputs[0], output.inputs['Surface'])
    obj.data.materials.append(material)
    obj.data.materials.append(material)
    obj.data.polygons[1].material_index = 1
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    camera_data = bpy.data.cameras.new("Saved Eye")
    camera = bpy.data.objects.new("Saved Eye", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0, 0, 5)
    camera_data.lens = 50
    scene.camera = camera
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 1
    scene.render.resolution_x = scene.render.resolution_y = 128
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    bpy.context.view_layer.update()
    return scene, obj, material


def create_projection(bpy, extension, obj, camera):
    from mathutils import Matrix
    assert bpy.ops.pawprint.create_stack() == {'FINISHED'}
    stack = obj.active_material.pawprint
    stack.width = stack.height = 128
    view = camera.matrix_world.inverted()
    window = camera.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=128, y=128)
    stack.view_matrix = extension.projection.flattened(view)
    stack.projection = extension.projection.flattened(window @ view)
    stack.clip_start, stack.clip_end = 0.1, 100
    stack.view_rotation = camera.matrix_world.to_quaternion()
    stack.view_location = (0, 0, 0)
    stack.view_distance = 5
    stack.lens = 50
    stack.depth = extension.projection.depth_image(bpy.context, stack)
    stack.image = extension.projection.new_image("Editable test RGBA", 128, 128, pattern=True)
    stack.has_view = True
    extension.model.migrate(obj.active_material)
    stack = extension.model.active_layer(stack)
    extension.projection.build_material(obj.active_material)
    assert extension.projection.matrix(stack.projection) != Matrix.Identity(4)
    return stack


def render_pixels(bpy, path):
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(str(path), check_existing=False)
    pixels = list(image.pixels[:])
    bpy.data.images.remove(image)
    return pixels


def worker(directory, engine='CYCLES'):
    import bpy
    from mathutils import Vector
    extension = load_extension()
    scene, obj, original = setup_fixture(bpy)
    scene.render.engine = engine
    stack = create_projection(bpy, extension, obj, scene.camera)
    assert obj.material_slots[1].material == original
    base = stack.id_data.pawprint.base
    assert stack.id_data.pawprint.original_material == original
    assert base.source == 'GENERATED' or base.packed_file
    assert stack.image.packed_file and stack.depth.packed_file

    def pixel(pixels, x, y, width=128):
        return pixels[(y * width + x) * 4:(y * width + x) * 4 + 3]

    def linear(value):
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    base_color = linear(base.pixels[0])

    pixels = render_pixels(bpy, directory / "front.exr")
    # Far from silhouettes/alpha boundaries: low-poly perspective must map 1:1.
    actual = pixel(pixels, 40, 40)
    source = [linear(v) for v in pixel(list(stack.image.pixels[:]), 40, 40)]
    assert max(abs(a - b) for a, b in zip(actual, source)) < 0.025, (actual, source)
    assert pixel(pixels, 96, 32)[0] > 0.99 and pixel(pixels, 96, 32)[1] < 0.01
    assert max(abs(v - base_color) for v in pixel(pixels, 56, 64)) < 0.02
    half_pixel = stack.image.pixels[(40 * 128 + 10) * 4:(40 * 128 + 10) * 4 + 4]
    blended = [linear(v) * half_pixel[3] + base_color * (1 - half_pixel[3]) for v in half_pixel[:3]]
    assert max(abs(a - b) for a, b in zip(pixel(pixels, 10, 40), blended)) < 0.025, pixel(pixels, 10, 40)

    stack.visible = False
    hidden = render_pixels(bpy, directory / "hidden.exr")
    assert max(abs(v - base_color) for v in pixel(hidden, 40, 40)) < 0.02
    stack.visible = True

    # Put an occluder over the tested pixel, capture it, then hide it for inspection.
    # The hidden target region must remain unprojected from the original snapshot.
    bpy.ops.mesh.primitive_plane_add(size=0.6, location=(-0.65, -0.65, 1))
    occluder = bpy.context.active_object
    stack.depth = extension.projection.depth_image(bpy.context, stack)
    extension.projection.build_material(obj.active_material)
    occluder.hide_render = True
    occluded = render_pixels(bpy, directory / "occluded.exr")
    assert max(abs(v - base_color) for v in pixel(occluded, 40, 40)) < 0.02
    # Viewport-hidden geometry must not enter the capture.
    occluder.hide_set(True)
    bpy.context.view_layer.update()
    stack.depth = extension.projection.depth_image(bpy.context, stack)
    extension.projection.build_material(obj.active_material)
    unoccluded = render_pixels(bpy, directory / "unoccluded.exr")
    assert max(abs(a - b) for a, b in zip(pixel(unoccluded, 40, 40), source)) < 0.025

    # Tight tolerance must still cover a sloping, unsplit quad between depth samples.
    for vertex in obj.data.vertices:
        vertex.co.z += vertex.co.x * 0.9
    bpy.context.view_layer.update()
    stack.depth_tolerance = 0.001
    stack.depth = extension.projection.depth_image(bpy.context, stack)
    extension.projection.build_material(obj.active_material)
    scene.render.resolution_x = scene.render.resolution_y = 256
    sloped = render_pixels(bpy, directory / "sloped.exr")
    assert max(abs(a - b) for a, b in zip(pixel(sloped, 80, 80, 256), source)) < 0.025, pixel(sloped, 80, 80, 256)
    scene.render.resolution_x = scene.render.resolution_y = 128
    for vertex in obj.data.vertices:
        vertex.co.z -= vertex.co.x * 0.9
    bpy.context.view_layer.update()
    stack.depth = extension.projection.depth_image(bpy.context, stack)
    extension.projection.build_material(obj.active_material)

    # Inspect the rear face through the gap from the side, without recapturing.
    scene.camera.location = (-5, 0, -0.5)
    rear_point = Vector((-1.8, 0, -1))
    scene.camera.rotation_euler = (rear_point - scene.camera.location).to_track_quat('-Z', 'Y').to_euler()
    rear = render_pixels(bpy, directory / "side.exr")
    assert max(abs(v - base_color) for v in pixel(rear, 64, 64)) < 0.02, pixel(rear, 64, 64)
    assert abs(stack.depth.pixels[(32 * 128 + 32) * 4] - 5) < 0.001

    # A surface whose winding faces away from the saved eye gets no layer.
    scene.camera.location = (0, 0, 5)
    scene.camera.rotation_euler = (0, 0, 0)
    obj.data.polygons[0].flip()
    obj.data.update()
    backfacing = render_pixels(bpy, directory / "backfacing.exr")
    assert max(abs(v - base_color) for v in pixel(backfacing, 40, 40)) < 0.02, pixel(backfacing, 40, 40)
    obj.data.polygons[0].flip()
    obj.data.update()

    # A native image-buffer edit propagates to the shader without rebuilding it.
    stack.image.pixels.foreach_set(array('f', (0.1, 0.7, 0.2, 1)) * (128 * 128))
    stack.image.update()
    stack.image.pack()
    scene.camera.location = (0, 0, 5)
    scene.camera.rotation_euler = (0, 0, 0)
    edited = render_pixels(bpy, directory / "edited.exr")
    assert max(abs(a - linear(b)) for a, b in zip(pixel(edited, 40, 40), (0.1, 0.7, 0.2))) < 0.02
    bpy.context.view_layer.objects.active = obj

    # Mirror folds the projection across the object's local axis: the kept half
    # passes through, the discarded side samples the mirrored kept half, and the
    # image content of the discarded half (here: right-half blue for X+) is
    # never used. The render (40, 40) normally samples image pixel (40, 40).
    # Image pixels fill row-major from the bottom-left (index = y * width + x),
    # so select the half by the pixel's column: x < 64 keeps red, x >= 64 carries
    # the blue that X+ must ignore.
    asymmetric = array('f')
    for index in range(128 * 128):
        asymmetric.extend((0.9, 0.2, 0.1, 1) if index % 128 < 64 else (0.1, 0.2, 0.9, 1))
    stack.image.pixels.foreach_set(asymmetric)
    stack.image.update()
    stack.image.pack()
    stack.mirror = 'X_PLUS'
    mirrored = render_pixels(bpy, directory / "mirror-xplus.exr")
    assert max(abs(a - linear(b)) for a, b in zip(pixel(mirrored, 40, 40), (0.1, 0.2, 0.9))) < 0.025, pixel(mirrored, 40, 40)
    stack.mirror = 'X_MINUS'
    kept = render_pixels(bpy, directory / "mirror-xminus.exr")
    assert max(abs(a - linear(b)) for a, b in zip(pixel(kept, 40, 40), (0.9, 0.2, 0.1))) < 0.025, pixel(kept, 40, 40)
    stack.mirror = 'NONE'
    stack.image.pixels.foreach_set(array('f', (0.1, 0.7, 0.2, 1)) * (128 * 128))
    stack.image.update()
    stack.image.pack()

    # Two saved viewpoints, independently editable images and bottom-to-top order.
    owner = obj.active_material.pawprint
    upper = owner.layers.add()
    upper.name = 'Upper angled patch'
    upper.width = upper.height = 128
    scene.camera.location = (0.3, 0, 5)
    scene.camera.rotation_euler = (Vector((0, 0, 0)) - scene.camera.location).to_track_quat('-Z', 'Y').to_euler()
    bpy.context.view_layer.update()
    view = scene.camera.matrix_world.inverted()
    window = scene.camera.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=128, y=128)
    upper.view_matrix = extension.projection.flattened(view)
    upper.projection = extension.projection.flattened(window @ view)
    upper.clip_start, upper.clip_end = 0.1, 100
    upper.depth = extension.projection.depth_image(bpy.context, upper)
    upper.image = extension.projection.new_image('Upper alpha patch', 128, 128)
    upper.image.pixels.foreach_set(array('f', (0.8, 0.1, 0.1, 0.5)) * (128 * 128))
    upper.image.update()
    upper.image.pack()
    owner.active_index = 0
    assert all(layer.visible for layer in owner.layers), 'Selection must not change visibility'
    scene.camera.location = (0, 0, 5)
    scene.camera.rotation_euler = (0, 0, 0)
    mixed = render_pixels(bpy, directory / 'layers.exr')
    expected = [(linear(a) + linear(b)) / 2 for a, b in zip((0.1, 0.7, 0.2), (0.8, 0.1, 0.1))]
    assert max(abs(a - b) for a, b in zip(pixel(mixed, 40, 40), expected)) < 0.025
    assert bpy.ops.pawprint.move_layer(direction=1) == {'FINISHED'}
    reordered = render_pixels(bpy, directory / 'reordered.exr')
    assert max(abs(a - b) for a, b in zip(pixel(reordered, 40, 40), pixel(edited, 40, 40))) < 0.025
    assert bpy.ops.pawprint.move_layer(direction=-1) == {'FINISHED'}
    upper = owner.layers[1]
    upper.visible = False
    hidden_upper = render_pixels(bpy, directory / 'hidden-upper.exr')
    assert max(abs(a - b) for a, b in zip(pixel(hidden_upper, 40, 40), pixel(edited, 40, 40))) < 0.025
    upper.visible = True
    image = upper.image
    upper.image = None
    missing = render_pixels(bpy, directory / 'missing-image.exr')
    assert max(abs(a - b) for a, b in zip(pixel(missing, 40, 40), pixel(edited, 40, 40))) < 0.025
    upper.image = image
    doomed = image.copy()
    upper.image = doomed
    bpy.data.images.remove(doomed)
    bpy.context.view_layer.update()
    deleted = render_pixels(bpy, directory / 'deleted-image.exr')
    assert upper.image is None
    assert max(abs(a - b) for a, b in zip(pixel(deleted, 40, 40), pixel(edited, 40, 40))) < 0.025
    upper.image = image
    # A transparent upper image reveals later changes to the lower image dynamically.
    image.pixels.foreach_set(array('f', (0.8, 0.1, 0.1, 0)) * (128 * 128))
    image.update()
    image.pack()
    erased = render_pixels(bpy, directory / 'erased-upper.exr')
    assert max(abs(a - b) for a, b in zip(pixel(erased, 40, 40), pixel(edited, 40, 40))) < 0.025
    # Structural undo restores the layer and its image reference, without pixel writes.
    owner.active_index = 1
    bpy.ops.ed.undo_push(message='Two layers')
    assert bpy.ops.pawprint.remove_layer() == {'FINISHED'}
    bpy.ops.ed.undo_push(message='Remove upper')
    assert len(obj.active_material.pawprint.layers) == 1
    bpy.ops.ed.undo()
    obj = bpy.data.objects['Pawprint Test Target']
    assert len(obj.active_material.pawprint.layers) == 2
    assert obj.active_material.pawprint.layers[1].image
    bpy.ops.ed.redo()
    obj = bpy.data.objects['Pawprint Test Target']
    assert len(obj.active_material.pawprint.layers) == 1
    bpy.ops.ed.undo()
    obj = bpy.data.objects['Pawprint Test Target']
    # Occlusion z-test: a grazing shelf behind the captured surface would pass
    # the plane test (its offset is nearly perpendicular to its own normal) and
    # the winding-facing test, so only the axial comparison can reject it. The
    # right quad is removed for this render so the shelf is actually visible.
    import bmesh
    stack_layer = obj.active_material.pawprint.layers[0]
    saved_tolerance = stack_layer.depth_tolerance
    stack_layer['depth_tolerance'] = 2.0
    right_indices = list(obj.data.polygons[1].vertices)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.faces.remove(bm.faces[1])
    shelf = [bm.verts.new(co) for co in ((0.5, -0.5, -2), (0.5, -0.5, -1),
                                         (1.5, -0.5, -1), (1.5, -0.5, -2))]
    face = bm.faces.new(shelf)
    face.material_index = 0
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    bpy.context.view_layer.update()
    occluded = render_pixels(bpy, directory / 'occlusion.exr')
    expected_base = linear(obj.active_material.pawprint.base.pixels[(50 * 128 + 91) * 4])
    assert max(abs(a - b) for a, b in zip(pixel(occluded, 91, 50), (expected_base,) * 3)) < 0.025, \
        ('Far-side surface leaked through the plane test', pixel(occluded, 91, 50))
    assert max(abs(a - b) for a, b in zip(pixel(occluded, 40, 40), pixel(edited, 40, 40))) < 0.025, \
        'Occlusion test cut visible coverage'
    stack_layer['depth_tolerance'] = saved_tolerance
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    shelf_face = bm.faces[-1]
    doomed = list(shelf_face.verts)
    bm.faces.remove(shelf_face)
    for v in doomed:
        bm.verts.remove(v)
    bm.verts.ensure_lookup_table()
    restored = bm.faces.new([bm.verts[i] for i in right_indices])
    restored.material_index = 1
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    bpy.context.view_layer.update()

    # Mirror folding must survive the axial occlusion gate: folded fragments
    # reuse the kept half's visibility snapshot, so an occluder in front of
    # the kept twin must not erase the mirrored half.
    bpy.context.scene.camera.location = (0, 0, 5)
    bpy.context.scene.camera.rotation_euler = (0, 0, 0)
    occluder = bpy.data.objects.new('Mirror occluder', bpy.data.meshes.new('Mirror occluder'))
    occluder.data.from_pydata([(0.3, -2, 0.8), (2.5, -2, 0.8), (2.5, 2, 0.8), (0.3, 2, 0.8)],
                              [], [(0, 1, 2, 3)])
    bpy.context.scene.collection.objects.link(occluder)
    bpy.context.view_layer.update()
    stack_layer.depth = extension.projection.depth_image(bpy.context, stack_layer)
    stack_layer['depth_tolerance'] = 2.0
    stack_layer.mirror = 'X_PLUS'
    extension.projection.build_material(obj.active_material)
    occluder.hide_render = True
    stack_layer.image.pixels.foreach_set(array('f', (0.1, 0.1, 0.9, 1)) * (128 * 128))
    stack_layer.image.update()
    mirrored = render_pixels(bpy, directory / "mirrored.exr")
    # The right quad keeps the original red slot material in this fixture, so
    # the check is the folded half: both sample points fold onto twin regions
    # covered by the occluder in the snapshot yet must stay painted.
    assert pixel(mirrored, 32, 64)[2] > 0.5 and pixel(mirrored, 32, 64)[0] < 0.05, pixel(mirrored, 32, 64)
    assert pixel(mirrored, 48, 64)[2] > 0.5, pixel(mirrored, 48, 64)
    stack_layer.mirror = 'NONE'
    stack_layer['depth_tolerance'] = 0.01
    stack_layer.image.pixels.foreach_set(array('f', (0.1, 0.7, 0.2, 1)) * (128 * 128))
    stack_layer.image.update()
    occluder.hide_render = False
    stack_layer.depth = extension.projection.depth_image(bpy.context, stack_layer)
    extension.projection.build_material(obj.active_material)
    bpy.data.objects.remove(occluder, do_unlink=True)
    bpy.data.meshes.remove(bpy.data.meshes['Mirror occluder'])
    bpy.context.view_layer.update()
    bpy.ops.wm.save_as_mainfile(filepath=str(directory / "projection.blend"))
    print(f"PROJECTION RENDER CHECKS PASSED ({engine})")


def reload_worker(directory):
    import bpy
    load_extension()
    bpy.ops.wm.open_mainfile(filepath=str(directory / "projection.blend"))
    obj = bpy.data.objects['Pawprint Test Target']
    stack = obj.active_material.pawprint
    assert stack.enabled and len(stack.layers) == 2 and stack.owner == obj
    assert stack.layers[1].name == 'Upper angled patch'
    assert stack.layers[1].image.pixels[3] == 0
    assert stack.layers[0].projection[:] != stack.layers[1].projection[:]
    assert stack.base
    stack = stack.layers[0]
    assert stack.depth and stack.image
    assert stack.image.packed_file and stack.depth.packed_file
    assert abs(stack.image.pixels[1] - 0.7) < 0.01
    assert stack.view_distance == 5 and stack.width == 128
    assert obj.active_material.node_tree.nodes['Pawprint Layer'].image == stack.image
    print("PROJECTION SAVE/REOPEN CHECKS PASSED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blender', default='blender')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='pawprint-projection-') as temp:
        env = os.environ.copy()
        env['BLENDER_USER_RESOURCES'] = str(Path(temp) / 'config')
        for suffix in ('CONFIG', 'SCRIPTS', 'DATAFILES', 'EXTENSIONS'):
            env[f'BLENDER_USER_{suffix}'] = str(Path(temp) / suffix.lower())
        for stage in ('--worker', '--reload-worker', '--eevee-worker'):
            subprocess.run([args.blender, '--background', '--factory-startup', '--python-exit-code', '1',
                            '--python', str(Path(__file__).resolve()), '--', stage, temp], env=env, check=True)


if __name__ == '__main__':
    if '--worker' in sys.argv:
        worker(Path(sys.argv[-1]))
    elif '--eevee-worker' in sys.argv:
        worker(Path(sys.argv[-1]), 'BLENDER_EEVEE')
    elif '--reload-worker' in sys.argv:
        reload_worker(Path(sys.argv[-1]))
    else:
        main()
