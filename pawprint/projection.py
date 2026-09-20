# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-fragment perspective projection with a saved scene-depth visibility image.

No mesh UV projection: homogeneous division in the shader preserves perspective
on large, low-poly faces. The existing UV map is used only by the committed base.
"""

from array import array

import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree


# Local-axis mirror choices: the selected direction names the half that is kept;
# fragments on the opposite side of the object's local plane sample the mirrored
# kept half, so image content belonging to the discarded side is never used.
MIRROR_AXES = {
    'X_PLUS': (1, 0, 0), 'X_MINUS': (-1, 0, 0),
    'Y_PLUS': (0, 1, 0), 'Y_MINUS': (0, -1, 0),
    'Z_PLUS': (0, 0, 1), 'Z_MINUS': (0, 0, -1),
}


def matrix(values):
    return Matrix([values[i:i + 4] for i in range(0, 16, 4)])


def flattened(value):
    return tuple(v for row in value for v in row)


def frame_size(region, aspect):
    width = min(region.width, region.height * aspect) * 0.85
    return width, width / aspect


def capture_view(stack, space, region):
    view = space.region_3d
    view.update()
    width, height = frame_size(region, stack.width / stack.height)
    crop = Matrix.Diagonal((region.width / width, region.height / height, 1, 1))
    stack.projection = flattened(crop @ view.perspective_matrix)
    stack.view_matrix = flattened(view.view_matrix)
    stack.view_rotation = view.view_rotation
    stack.view_location = view.view_location
    stack.view_distance = view.view_distance
    stack.lens = space.lens
    stack.clip_start = space.clip_start
    stack.clip_end = space.clip_end


def visible_geometry(context, space=None):
    """Snapshot evaluated, viewport-visible surface geometry, including instances."""
    depsgraph = context.evaluated_depsgraph_get()
    vertices, triangles = [], []
    for instance in depsgraph.object_instances:
        obj = instance.object
        visibility_owner = instance.parent if instance.is_instance else obj
        original = visibility_owner.original
        if not instance.show_self or not original.visible_get(view_layer=context.view_layer, viewport=space):
            continue
        if obj.type not in {'MESH', 'CURVE', 'SURFACE', 'FONT', 'META'}:
            continue
        mesh = obj.to_mesh()
        if mesh is None:
            continue
        try:
            mesh.calc_loop_triangles()
            offset = len(vertices)
            transform = instance.matrix_world
            vertices.extend(transform @ v.co for v in mesh.vertices)
            triangles.extend(tuple(offset + i for i in t.vertices) for t in mesh.loop_triangles)
        finally:
            obj.to_mesh_clear()
    return BVHTree.FromPolygons(vertices, triangles, all_triangles=True) if triangles else None


def depth_image(context, stack, space=None):
    """Store positive view-space depth at pixel centers; zero denotes empty space."""
    tree = visible_geometry(context, space)
    view = matrix(stack.view_matrix)
    inverse = matrix(stack.projection).inverted()
    eye = view.inverted().translation
    pixels = array('f', [0.0]) * (stack.width * stack.height * 4)
    for y in range(stack.height):
        for x in range(stack.width):
            point = inverse @ Vector((2 * (x + 0.5) / stack.width - 1,
                                      2 * (y + 0.5) / stack.height - 1, 0, 1))
            direction = (point.xyz / point.w - eye).normalized()
            forward = -(view.to_3x3() @ direction).z
            origin = eye + direction * (stack.clip_start / forward)
            hit = tree.ray_cast(origin, direction, (stack.clip_end - stack.clip_start) / forward)[0] if tree else None
            depth = -(view @ hit).z if hit is not None else 0.0
            index = (y * stack.width + x) * 4
            pixels[index:index + 4] = array('f', (depth, depth, depth, 1))
    image = bpy.data.images.new("Pawprint Visibility Depth", stack.width, stack.height, float_buffer=True)
    image.colorspace_settings.name = 'Non-Color'
    image.pixels.foreach_set(pixels)
    image.update()
    image.pack()
    return image


def new_image(name, width, height, pattern=False, transparent=False):
    image = bpy.data.images.new(name, width, height, alpha=True)
    image.generated_color = (0, 0, 0, 0) if pattern or transparent else (0.18, 0.18, 0.18, 1)
    if pattern:
        pixels = array('f')
        for y in range(height):
            v = (y + 0.5) / height
            for x in range(width):
                u = (x + 0.5) / width
                # Different corners, a transparent window, and a half-alpha band.
                alpha = 0 if 0.38 < u < 0.62 and 0.38 < v < 0.62 else (0.5 if u < 0.15 else 1)
                color = (u, v, 0.15 if int(u * 8) % 2 else 0.8, alpha)
                pixels.extend(color)
        image.pixels.foreach_set(pixels)
    image.update()
    image.pack()
    return image


def build_material(material):
    stack = material.pawprint
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()

    def node(kind, name, x, y):
        n = tree.nodes.new(kind)
        n.name = n.label = name
        n.location = (x, y)
        return n

    def assign(socket, value):
        if isinstance(value, (int, float, tuple, list, Vector)):
            socket.default_value = value
        else:
            tree.links.new(value, socket)

    def math(operation, a, b=0.0):
        n = node('ShaderNodeMath', operation, 0, -400)
        n.operation = operation
        assign(n.inputs[0], a)
        assign(n.inputs[1], b)
        return n.outputs[0]

    def vector_math(operation, a, b):
        n = node('ShaderNodeVectorMath', operation, 200, -600)
        n.operation = operation
        assign(n.inputs[0], a)
        assign(n.inputs[1], b)
        return n.outputs['Value' if operation == 'DOT_PRODUCT' else 'Vector']

    geometry = node('ShaderNodeNewGeometry', 'World Surface', -1200, 0)

    def row_dot(row, position):
        n = node('ShaderNodeVectorMath', 'Projection Row', -1000, -200)
        n.operation = 'DOT_PRODUCT'
        tree.links.new(position, n.inputs[0])
        n.inputs[1].default_value = tuple(row[:3])
        return math('ADD', n.outputs['Value'], row[3])

    uv = node('ShaderNodeUVMap', 'Base UV', -400, 400)
    uv.uv_map = stack.uv_name
    base = node('ShaderNodeTexImage', 'Pawprint Base', -150, 400)
    base.image = stack.base
    tree.links.new(uv.outputs['UV'], base.inputs['Vector'])

    owner = stack.owner
    color = base.outputs['Color']
    for stack in [layer for layer in stack.layers if layer.image and layer.depth]:
        transform = matrix(stack.projection)
        position = geometry.outputs['Position']
        true_normal = geometry.outputs['True Normal']
        fold = None
        if stack.mirror != 'NONE' and owner is not None:
            # Fold fragments across the object's local mirror plane before
            # projecting: the kept half maps straight through and the discarded
            # side samples the mirrored kept half, so the image content beyond
            # it is never used. Plane constants are baked here, matching the
            # project-wide assumption of stable object transforms.
            world = owner.matrix_world
            axis = (world.to_3x3() @ Vector(MIRROR_AXES[stack.mirror])).normalized()
            axis_node = node('ShaderNodeCombineXYZ', 'Mirror Axis', -1150, -300)
            for i in range(3):
                axis_node.inputs[i].default_value = axis[i]
            origin_node = node('ShaderNodeCombineXYZ', 'Mirror Plane Origin', -1150, -450)
            for i in range(3):
                origin_node.inputs[i].default_value = world.translation[i]
            side = vector_math('DOT_PRODUCT',
                               vector_math('SUBTRACT', position, origin_node.outputs[0]),
                               axis_node.outputs[0])
            fold = math('LESS_THAN', side, 0)
            offset = vector_math('MULTIPLY', axis_node.outputs[0],
                                 math('MULTIPLY', fold, math('MULTIPLY', side, 2)))
            position = vector_math('SUBTRACT', position, offset)
            # Gate the normal fold by the fragment's side, not by which way the
            # normal points: kept-side fragments may still face across the plane.
            normal_offset = vector_math('MULTIPLY', axis_node.outputs[0],
                                        math('MULTIPLY', fold, math('MULTIPLY',
                                             vector_math('DOT_PRODUCT', true_normal, axis_node.outputs[0]), 2)))
            true_normal = vector_math('SUBTRACT', true_normal, normal_offset)
        w = row_dot(transform[3], position)
        coordinates = node('ShaderNodeCombineXYZ', 'Perspective Image Coordinates', 200, 0)
        image_coordinates = []
        for i in range(2):
            coordinate = math('ADD', math('MULTIPLY', math('DIVIDE', row_dot(transform[i], position), w), 0.5), 0.5)
            image_coordinates.append(coordinate)
            tree.links.new(coordinate, coordinates.inputs[i])
        layer = node('ShaderNodeTexImage', 'Pawprint Layer', 450, 200)
        layer.image = stack.image
        layer.extension = 'CLIP'
        tree.links.new(coordinates.outputs[0], layer.inputs['Vector'])
        depth = node('ShaderNodeTexImage', 'Captured Scene Depth', 450, -200)
        depth.image = stack.depth
        depth.extension = 'CLIP'
        depth.interpolation = 'Closest'
        tree.links.new(coordinates.outputs[0], depth.inputs['Vector'])
        separate = node('ShaderNodeSeparateColor', 'Depth Channel', 700, -200)
        tree.links.new(depth.outputs['Color'], separate.inputs[0])
        tolerance = node('ShaderNodeValue', 'Pawprint Tolerance', 700, -400)
        tolerance.outputs[0].default_value = stack.depth_tolerance
        # Compare the captured point to the fragment's geometric plane, rather
        # than raw Z. Raw nearest-neighbor Z creates holes on sloping polygons
        # when the viewport and visibility image have different resolutions.
        inverse_view = matrix(stack.view_matrix).inverted()
        window = transform @ inverse_view
        ray = -inverse_view.col[2].xyz
        for i, size in enumerate(stack.depth.size):
            center = math('DIVIDE', math('ADD', math('FLOOR', math('MULTIPLY', image_coordinates[i], size)), 0.5), size)
            ndc = math('SUBTRACT', math('MULTIPLY', center, 2), 1)
            component = math('DIVIDE', math('ADD', ndc, window[i][2]), window[i][i])
            ray = vector_math('ADD', ray, vector_math('MULTIPLY', inverse_view.col[i].xyz, component))
        eye = inverse_view.translation
        captured_point = vector_math('ADD', eye, vector_math('MULTIPLY', ray, separate.outputs[0]))
        delta = vector_math('SUBTRACT', captured_point, position)
        difference = math('ABSOLUTE', vector_math('DOT_PRODUCT', delta, true_normal))
        coverage = math('LESS_THAN', difference, tolerance.outputs[0])
        coverage = math('MULTIPLY', coverage, math('GREATER_THAN', separate.outputs[0], 0))
        coverage = math('MULTIPLY', coverage, math('GREATER_THAN', w, stack.clip_start))
        toward_eye = node('ShaderNodeVectorMath', 'Toward Saved Eye', -800, -600)
        toward_eye.operation = 'SUBTRACT'
        toward_eye.inputs[0].default_value = eye
        tree.links.new(position, toward_eye.inputs[1])
        facing = node('ShaderNodeVectorMath', 'Saved View Facing', -600, -600)
        facing.operation = 'DOT_PRODUCT'
        # Renderers orient True Normal toward the current shading ray. Undo that
        # flip before testing winding against the saved eye, even while orbiting.
        winding_normal = vector_math('MULTIPLY', true_normal,
                                     math('SUBTRACT', 1, math('MULTIPLY', geometry.outputs['Backfacing'], 2)))
        tree.links.new(winding_normal, facing.inputs[0])
        tree.links.new(toward_eye.outputs[0], facing.inputs[1])
        coverage = math('MULTIPLY', coverage, math('GREATER_THAN', facing.outputs['Value'], 0))
        # Winding-independent axial occlusion: reject fragments clearly behind
        # the captured surface along the saved view ray. The plane test above is
        # blind along the ray (grazing silhouettes) and the facing test depends
        # on winding, so far-side surfaces can slip through both. The slack grows
        # with the grazing angle, disabling the test where texel-sized axial
        # error dominates; sub-texel-thin gaps still pass by design.
        # World-space size of one depth texel at unit axial distance, baked as a
        # constant: the snapshot's pixel-center sampling can differ from the
        # fragment's continuous projection by up to a texel, which becomes axial
        # error on tilted surfaces. Two texels of margin absorb that snap error
        # while staying far below genuine occlusion gaps.
        # Extract the scale from the pure window matrix: the combined
        # projection's diagonal is polluted by the camera rotation, which
        # collapses it toward zero for tilted views.
        texel_margin = 2.0 * max(2.0 / (window[0][0] * stack.depth.size[0]),
                                 2.0 / (window[1][1] * stack.depth.size[1]))
        axial_gap = math('SUBTRACT', w, separate.outputs[0])
        behind = math('GREATER_THAN', axial_gap, math('MULTIPLY', w, texel_margin))
        if fold is not None:
            # Mirror-folded fragments reuse the kept half's visibility snapshot
            # by design: the discarded side was never captured, so occluders in
            # front of the kept twin must not erase the mirrored half. Only the
            # kept side keeps the full axial occlusion comparison.
            behind = math('MULTIPLY', behind, math('SUBTRACT', 1, fold))
        coverage = math('MULTIPLY', coverage, math('SUBTRACT', 1, behind))
        visible = node('ShaderNodeValue', 'Pawprint Visibility', 700, 0)
        visible.outputs[0].default_value = float(stack.visible)
        alpha = math('MULTIPLY', layer.outputs['Alpha'], math('MULTIPLY', coverage, visible.outputs[0]))
        mix = node('ShaderNodeMixRGB', 'Layer Over Base', 1000, 300)
        tree.links.new(alpha, mix.inputs[0])
        tree.links.new(color, mix.inputs[1])
        tree.links.new(layer.outputs['Color'], mix.inputs[2])
        color = mix.outputs[0]
        tree.nodes.active = layer
        layer.select = True

    emission = node('ShaderNodeEmission', 'Unlit Appearance', 1250, 300)
    tree.links.new(color, emission.inputs['Color'])
    output = node('ShaderNodeOutputMaterial', 'Material Output', 1500, 300)
    tree.links.new(emission.outputs[0], output.inputs['Surface'])
