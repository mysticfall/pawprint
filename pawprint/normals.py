# SPDX-License-Identifier: GPL-3.0-or-later
"""Mesh-relative Chord normals: packed images and normalized vector replacement."""
import bpy
import numpy as np

from . import capture, projection


def image(values, name='Pawprint Normal'):
    result = bpy.data.images.new(name, width=values.shape[1], height=values.shape[0], alpha=True,
                                float_buffer=True)
    result.colorspace_settings.name = 'Non-Color'
    # RGB encodes independent vector data; alpha is coverage, not transparency.
    # In particular Eevee must not unpremultiply float RGB at feathered edges.
    result.alpha_mode = 'CHANNEL_PACKED'
    result.pixels.foreach_set(values.astype(np.float32).ravel())
    result.update()
    result.pack()
    return result


def merged(previous, values, weights):
    """Prepare an immutable replacement; preserve every untouched normal texel."""
    old = capture.pixels(previous) if previous else np.zeros_like(values)
    if old.shape != values.shape:
        raise ValueError('Normal channel dimensions do not match the layer')
    a = np.clip(weights, 0, 1)[:, :, None]
    old_weight = old[:, :, 3:4] * (1 - a)
    alpha = old_weight + a
    vector = (old[:, :, :3] * 2 - 1) * old_weight + (values[:, :, :3] * 2 - 1) * a
    length = np.linalg.norm(vector, axis=2, keepdims=True)
    vector = np.divide(vector, length, out=np.broadcast_to((0., 0., 1.), vector.shape).copy(), where=length > 1e-8)
    result = np.concatenate((vector * .5 + .5, alpha), axis=2)
    result[weights == 0] = old[weights == 0]
    return result


def surface_nodes(tree, layer, coordinates, surface, fold=None, axis=None):
    """Return a world-space normal and texture node at full strength.

    Chord is right/down/toward: negate the bitangent, not the stored green.
    The surface is the mesh shading normal, never the lower texture composite.
    Evaluate in the kept mirror half, then reflect the resulting vector back.
    """
    def node(kind, name):
        item = tree.nodes.new(kind)
        item.name = item.label = name
        return item

    def assign(socket, value):
        if isinstance(value, (tuple, list, float, int)) or type(value).__name__ == 'Vector':
            socket.default_value = value
        else:
            tree.links.new(value, socket)

    def vector(operation, a, b=(0, 0, 0)):
        item = node('ShaderNodeVectorMath', 'Normal ' + operation)
        item.operation = operation
        assign(item.inputs[0], a)
        assign(item.inputs[1], b)
        return item.outputs['Value' if operation == 'DOT_PRODUCT' else 'Vector']

    def reflect(value):
        if fold is None:
            return value
        offset = vector('MULTIPLY', axis, vector('DOT_PRODUCT', value, axis))
        return vector('SUBTRACT', value, vector('MULTIPLY', vector('MULTIPLY', offset, (2, 2, 2)), fold))

    surface = reflect(surface)
    rotation = projection.matrix(layer.view_matrix).inverted().to_3x3()
    right, up = rotation.col[0].normalized(), rotation.col[1].normalized()
    raw = vector('SUBTRACT', right, vector('MULTIPLY', surface, vector('DOT_PRODUCT', surface, right)))
    gate = node('ShaderNodeMath', 'Normal Tangent Fallback')
    gate.operation = 'LESS_THAN'
    assign(gate.inputs[0], vector('DOT_PRODUCT', raw, raw))
    gate.inputs[1].default_value = 1e-8
    mix = node('ShaderNodeMixRGB', 'Normal Tangent')
    assign(mix.inputs[0], gate.outputs[0])
    assign(mix.inputs[1], raw)
    assign(mix.inputs[2], vector('CROSS_PRODUCT', up, surface))
    tangent = vector('NORMALIZE', mix.outputs[0])
    down = vector('MULTIPLY', vector('NORMALIZE', vector('CROSS_PRODUCT', surface, tangent)), (-1, -1, -1))
    texture = node('ShaderNodeTexImage', 'Pawprint Normal')
    texture.image = layer.normal
    # RGB vectors must not interpolate toward black at the frame border.
    # Saved-frame coverage and clipped albedo alpha already bound visibility.
    texture.extension = 'EXTEND'
    assign(texture.inputs['Vector'], coordinates)
    decoded = vector('SUBTRACT', vector('MULTIPLY', texture.outputs['Color'], (2, 2, 2)), (1, 1, 1))
    components = node('ShaderNodeSeparateXYZ', 'Normal Components')
    assign(components.inputs[0], decoded)
    terms = [vector('MULTIPLY', axis, components.outputs[i]) for i, axis in enumerate((tangent, down, surface))]
    normal = vector('NORMALIZE', vector('ADD', vector('ADD', terms[0], terms[1]), terms[2]))
    return reflect(normal), texture
