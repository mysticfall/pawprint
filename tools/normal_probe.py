# SPDX-License-Identifier: GPL-3.0-or-later
"""Single-view Chord normal experiment; run in factory Blender, never a live scene.

blender -b --factory-startup --python-exit-code 1 --python tools/normal_probe.py -- \
    --snapshot scene.blend --source original.npy --output /tmp/opencode/normals

The snapshot must contain one Pawprint material with one non-mirrored layer.
Source is full-frame bottom-up RGBA, as exported from generation review.
No production channel persistence, painting or normal baking is implemented here.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pawprint
from pawprint import backend, capture, projection


def normal_shader(material, layer, image):
    """Attach experimental normal wiring to a freshly built single-layer material.

    Chord src/util.py:get_positions increases first world coordinate down rows;
    src/module/chord.py:compute_render swaps decoded R/G into that row/column
    basis. Thus the exported map is right/down/toward-viewer, not OpenGL +Y-up.
    This wiring is independent of the current observer camera.
    """
    tree = material.node_tree
    def node(kind, name):
        item = tree.nodes.new(kind)
        item.name = item.label = name
        return item

    def assign(socket, value):
        if isinstance(value, (tuple, list, Vector, float, int)):
            socket.default_value = value
        else:
            tree.links.new(value, socket)

    def vector(operation, a, b=(0, 0, 0)):
        item = node('ShaderNodeVectorMath', 'Normal ' + operation)
        item.operation = operation
        assign(item.inputs[0], a)
        assign(item.inputs[1], b)
        return item.outputs['Value' if operation == 'DOT_PRODUCT' else 'Vector']

    def blend(a, b, factor, name):
        item = node('ShaderNodeMixRGB', name)
        assign(item.inputs[0], factor)
        assign(item.inputs[1], a)
        assign(item.inputs[2], b)
        return item.outputs[0]

    if image.colorspace_settings.name != 'Non-Color':
        image.colorspace_settings.name = 'Non-Color'
    texture = node('ShaderNodeTexImage', 'Chord Normal (Non-Color)')
    texture.image = image
    texture.extension = 'CLIP'
    uv = tree.nodes['Pawprint Layer'].inputs['Vector'].links[0].from_socket
    tree.links.new(uv, texture.inputs['Vector'])
    decoded = vector('SUBTRACT', vector('MULTIPLY', texture.outputs['Color'], (2, 2, 2)), (1, 1, 1))
    components = node('ShaderNodeSeparateXYZ', 'Chord Right Down Toward')
    tree.links.new(decoded, components.inputs[0])
    rotation = projection.matrix(layer.view_matrix).inverted().to_3x3()
    right, up, toward = (rotation.col[i].normalized() for i in range(3))
    def transform(x, y, z):
        terms = [vector('MULTIPLY', axis, components.outputs[i]) for i, axis in enumerate((x, y, z))]
        return vector('NORMALIZE', vector('ADD', vector('ADD', terms[0], terms[1]), terms[2]))

    absolute = transform(right, -up, toward)
    surface = tree.nodes['World Surface'].outputs['Normal']
    # Camera-right projected onto the surface supplies a UV-independent tangent.
    # At grazing degeneracy use camera-up to construct a fallback right vector.
    tangent_raw = vector('SUBTRACT', right, vector('MULTIPLY', surface, vector('DOT_PRODUCT', surface, right)))
    length = vector('DOT_PRODUCT', tangent_raw, tangent_raw)
    degenerate = node('ShaderNodeMath', 'Grazing Tangent Fallback')
    degenerate.operation = 'LESS_THAN'
    assign(degenerate.inputs[0], length)
    degenerate.inputs[1].default_value = 1e-8
    tangent = vector('NORMALIZE', blend(tangent_raw, vector('CROSS_PRODUCT', up, surface),
                                        degenerate.outputs[0], 'Tangent Fallback'))
    bitangent = vector('NORMALIZE', vector('CROSS_PRODUCT', surface, tangent))
    detail = transform(tangent, vector('MULTIPLY', bitangent, (-1, -1, -1)), surface)
    mode = node('ShaderNodeValue', 'Normal Interpretation (0 Camera, 1 Surface Detail)')
    mode.outputs[0].default_value = 1
    strength = node('ShaderNodeValue', 'Normal Strength')
    strength.outputs[0].default_value = 1
    estimated = blend(absolute, detail, mode.outputs[0], 'Normal Interpretation')
    coverage = tree.nodes['Layer Over Base'].inputs[0].links[0].from_socket
    weight = node('ShaderNodeMath', 'Normal Coverage')
    weight.operation = 'MULTIPLY'
    tree.links.new(coverage, weight.inputs[0])
    tree.links.new(strength.outputs[0], weight.inputs[1])
    result = vector('NORMALIZE', blend(surface, estimated, weight.outputs[0], 'Normal Over Geometry'))
    tree.links.new(result, tree.nodes['Diffuse Albedo'].inputs['Normal'])
    for index, item in enumerate(list(tree.nodes)[-40:]):
        item.location = (1600 + (index % 8) * 200, -800 - (index // 8) * 240)
    mode.location, strength.location, texture.location = (1600, 400), (1600, 200), (1600, 0)
    return strength, mode, result


def self_test(directory):
    """Render the actual shader, checking signs, saved-view rotation and alpha gates."""
    from projection_test import setup_fixture, create_projection
    scene, obj, _ = setup_fixture(bpy)
    scene.camera.data.clip_end = 100
    layer = create_projection(bpy, pawprint, obj, scene.camera)
    material = obj.active_material
    rgba = np.ones((128, 128, 4), dtype=np.float32)
    layer.image.pixels.foreach_set(rgba.ravel())
    image = bpy.data.images.new('Normal calibration', 128, 128, float_buffer=True)
    image.colorspace_settings.name = 'Non-Color'
    image.alpha_mode = 'STRAIGHT'
    strength, mode, output = normal_shader(material, layer, image)
    # Encode world-space output into emission to inspect the shader without light.
    tree = material.node_tree
    scale = tree.nodes.new('ShaderNodeVectorMath')
    scale.operation = 'MULTIPLY_ADD'
    tree.links.new(output, scale.inputs[0])
    scale.inputs[1].default_value = (.5, .5, .5)
    scale.inputs[2].default_value = (.5, .5, .5)
    emission = tree.nodes.new('ShaderNodeEmission')
    tree.links.new(scale.outputs[0], emission.inputs[0])
    tree.links.new(emission.outputs[0], tree.nodes['Material Output'].inputs['Surface'])
    results = {}
    for name, value, expected in (
        ('flat', (.5, .5, 1), (0, 0, 1)),
        ('right', (.8, .5, .9), (.6, 0, .8)),
        ('down', (.5, .8, .9), (0, -.6, .8)),
    ):
        rgba[:, :, :3] = value
        image.pixels.foreach_set(rgba.ravel())
        image.update()
        values = capture.render_frame(bpy.context, layer, directory / (name + '.exr'), scene_linear=True)
        measured = values[40, 40, :3] * 2 - 1
        assert np.max(abs(measured - expected)) < .01, (name, measured, expected)
        results[name] = measured.tolist()
    strength.outputs[0].default_value = 0
    values = capture.render_frame(bpy.context, layer, directory / 'disabled.exr', scene_linear=True)
    assert np.max(abs(values[40, 40, :3] * 2 - 1 - (0, 0, 1))) < .01
    strength.outputs[0].default_value = 1
    tree.nodes['Pawprint Visibility'].outputs[0].default_value = 0
    values = capture.render_frame(bpy.context, layer, directory / 'hidden.exr', scene_linear=True)
    assert np.max(abs(values[40, 40, :3] * 2 - 1 - (0, 0, 1))) < .01
    tree.nodes['Pawprint Visibility'].outputs[0].default_value = 1
    # Fractional image coverage blends vectors before normalization.
    alpha_pixels = capture.pixels(layer.image)
    alpha_pixels[:, :, 3] = .5
    layer.image.pixels.foreach_set(alpha_pixels.ravel())
    layer.image.update()
    values = capture.render_frame(bpy.context, layer, directory / 'alpha.exr', scene_linear=True)
    expected = Vector((0, -.3, .9)).normalized()
    assert np.max(abs(values[40, 40, :3] * 2 - 1 - expected)) < .01, (values[40, 40, :3] * 2 - 1, tuple(expected))
    # A rotated saved camera basis must rotate absolute normals with the view.
    from mathutils import Matrix
    rotation = Matrix.Rotation(.65, 4, 'Z')
    alpha_pixels[:, :, 3] = 1
    layer.image.pixels.foreach_set(alpha_pixels.ravel())
    layer.image.update()
    saved_view = tuple(layer.view_matrix)
    layer.view_matrix = projection.flattened((rotation @ scene.camera.matrix_world).inverted())
    # Existing visibility mapping stays fixed for this diagnostic; only rebuild
    # normal wiring, allowing us to measure its saved basis independently.
    _, mode, rotated = normal_shader(material, layer, image)
    layer.view_matrix = saved_view
    mode.outputs[0].default_value = 0
    tree.links.new(rotated, scale.inputs[0])
    values = capture.render_frame(bpy.context, layer, directory / 'rotated.exr', scene_linear=True)
    expected = rotation.to_3x3() @ Vector((0, -.6, .8))
    assert np.max(abs(values[40, 40, :3] * 2 - 1 - expected)) < .01, (values[40, 40, :3] * 2 - 1, tuple(expected))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--environment', type=Path)
    parser.add_argument('--asset-base', type=Path, help='Original .blend directory for snapshot-relative image paths')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--server', default='http://127.0.0.1:8188')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--reuse-maps', action='store_true', help='Reuse maps already estimated in the output directory')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    args.output.mkdir(parents=True, exist_ok=True)
    pawprint.register()
    if args.self_test:
        print(json.dumps(self_test(args.output)))
        return
    if args.snapshot is None or args.source is None:
        parser.error('--snapshot and --source are required for the scene experiment')
    with bpy.data.libraries.load(str(args.snapshot.resolve())) as (source, destination):
        destination.scenes = source.scenes
    scenes = [scene for scene in destination.scenes if any(
        slot.material and slot.material.pawprint.enabled for obj in scene.objects for slot in obj.material_slots)]
    if len(scenes) != 1:
        raise ValueError('Expected one scene containing the Pawprint stack')
    scene = scenes[0]
    bpy.context.window.scene = scene
    materials = {slot.material for obj in scene.objects for slot in obj.material_slots
                 if slot.material and slot.material.pawprint.enabled}
    if len(materials) != 1:
        raise ValueError('Expected one Pawprint material')
    material = materials.pop()
    if len(material.pawprint.layers) != 1:
        raise ValueError('Experiment requires one layer')
    layer = material.pawprint.layers[0]
    if layer.mirror != 'NONE':
        raise ValueError('Mirrored normals are outside this experiment')
    if args.environment:
        environment = scene.world.node_tree.nodes['Environment Texture'].image
        environment.filepath = str(args.environment.resolve())
        environment.reload()
    if args.asset_base:
        for image in bpy.data.images:
            if image.source == 'FILE' and not image.packed_file:
                path = Path(bpy.path.abspath(image.filepath))
                if not path.is_file():
                    relative = os.path.relpath(path, args.snapshot.resolve().parent)
                    path = (args.asset_base / relative).resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                image.filepath = str(path)
                image.reload()
    values = np.load(args.source)
    if values.dtype == np.uint8:
        values = values.astype(np.float32) / 255
    if values.shape != (layer.height, layer.width, 4):
        raise ValueError('Source must match the full saved frame')
    capture.save_input(args.output / 'input.png', values, (layer.width, layer.height))
    if not args.reuse_maps:
        caps = backend.capabilities(json.loads(backend.Client(args.server).get('/object_info')))
        backend.publish(args.output, 'request.json', dict(operation='estimate', server=args.server,
                        settings=dict(chord=caps['chord'], tile=1024, overlap=128)))
        backend.run(args.output)
    if (args.output / 'error.json').exists():
        raise RuntimeError((args.output / 'error.json').read_text())
    if not (args.output / 'done.json').exists():
        raise RuntimeError('Estimation did not complete')
    loaded = bpy.data.images.load(str(args.output / backend.CHORD_OUTPUTS['10']), check_existing=False)
    pixels = capture.pixels(loaded)
    from pawprint import selection
    pixels[:, :, 3] = values[:, :, 3] * selection.footprint(layer, generation=True)
    albedo = bpy.data.images.new('Normal experiment albedo', layer.width, layer.height, alpha=True)
    albedo.pixels.foreach_set(pixels.ravel())
    albedo.pack()
    bpy.data.images.remove(loaded)
    layer.image = albedo
    normal = bpy.data.images.load(str(args.output / backend.CHORD_OUTPUTS['11']), check_existing=False)
    normal.name = 'Chord normal experiment'
    normal.colorspace_settings.name = 'Non-Color'
    normal.pack()
    projection.build_material(material)
    strength, mode, _ = normal_shader(material, layer, normal)
    world = scene.world
    test_world = bpy.data.worlds.new('Normal experiment dark world')
    test_world.use_nodes = True
    test_world.node_tree.nodes['Background'].inputs['Strength'].default_value = .05
    scene.world = test_world
    light = bpy.data.lights.new('Normal experiment sun', 'SUN')
    light.energy = 2
    sun = bpy.data.objects.new('Normal experiment sun', light)
    scene.collection.objects.link(sun)
    basis = projection.matrix(layer.view_matrix).inverted().to_3x3()
    for direction, local in [('left', (-1, 0, 1)), ('right', (1, 0, 1)), ('top', (0, 1, 1)), ('bottom', (0, -1, 1))]:
        sun.rotation_euler = (basis @ Vector(local)).to_track_quat('Z', 'Y').to_euler()
        bpy.context.view_layer.update()
        for version, enabled, detail in [('geometry', 0, 0), ('camera', 1, 0), ('detail', 1, 1)]:
            strength.outputs[0].default_value = enabled
            mode.outputs[0].default_value = detail
            capture.render_frame(bpy.context, layer, args.output / f'{direction}-{version}.png')
    scene.world = world
    sun.hide_render = True
    sun.hide_set(True)
    strength.outputs[0].default_value = 1
    mode.outputs[0].default_value = 1
    # Make the standalone artifact open at the saved view, with its controls easy
    # to find in the Shader Editor. Pack source textures to avoid snapshot paths.
    from mathutils import Quaternion
    for image in bpy.data.images:
        if image.source == 'FILE' and not image.packed_file:
            image.pack()
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    owner = material.pawprint.owner
    owner.select_set(True)
    bpy.context.view_layer.objects.active = owner
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            space = area.spaces.active
            space.shading.type = 'RENDERED'
            space.lens = layer.lens
            space.clip_start, space.clip_end = layer.clip_start, layer.clip_end
            space.region_3d.view_rotation = Quaternion(layer.view_rotation)
            space.region_3d.view_location = layer.view_location
            space.region_3d.view_distance = layer.view_distance
            space.region_3d.view_perspective = 'PERSP'
    text = bpy.data.texts.new('NORMAL EXPERIMENT - Read me')
    text.write('Single-view experiment. Material: ' + material.name + '\n'
               'Shader Editor: Normal Strength = 0 for geometry only, 1 for estimated normals.\n'
               'Normal Interpretation = 0 for saved-camera absolute, 1 for surface-relative detail.\n'
               'Use Rendered shading and orbit/change lighting. No mesh displacement.\n'
               'Pawprint edits rebuild this experimental wiring. Normal layer persistence,\n'
               'painting, mirror composition and baking are not implemented in this probe.\n')
    bpy.ops.wm.save_as_mainfile(filepath=str((args.output / 'normal-experiment.blend').resolve()))
    print(json.dumps(dict(saved=str(args.output / 'normal-experiment.blend'), normal=normal.name)))


if __name__ == '__main__':
    main()
