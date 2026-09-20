# SPDX-License-Identifier: GPL-3.0-or-later

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from .model import active_stack, active_layer
from .projection import matrix, frame_size

_handle = None


def frame_corners(stack, region, view):
    if not stack.has_view or view.view_perspective != 'PERSP':
        return None
    saved_view = matrix(stack.view_matrix)
    if max(abs(a - b) for row_a, row_b in zip(saved_view, view.view_matrix) for a, b in zip(row_a, row_b)) > 0.0001:
        return None
    transform = view.perspective_matrix @ matrix(stack.projection).inverted()
    points = []
    for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)):
        p = transform @ Vector((x, y, 0, 1))
        if p.w <= 0:
            return None
        points.append(((p.x / p.w + 1) * region.width / 2, (p.y / p.w + 1) * region.height / 2))
    return points


def draw_frame():
    context = bpy.context
    if not context.region_data or not context.space_data.overlay.show_overlays:
        return
    stack = active_stack(context)
    if stack is None:
        return
    if context.space_data.region_quadviews:
        return
    if stack.preview and context.region_data.view_perspective == 'PERSP':
        width, height = frame_size(context.region, stack.width / stack.height)
        x, y = (context.region.width - width) / 2, (context.region.height - height) / 2
        points = [(x, y), (x + width, y), (x + width, y + height), (x, y + height), (x, y)]
    else:
        layer = active_layer(stack)
        points = frame_corners(layer, context.region, context.region_data) if layer else None
        if points:
            from . import selection
            selection.draw(layer, points)
    if not points:
        return
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'LINE_STRIP', {'pos': points})
    shader.bind()
    shader.uniform_float('color', (1.0, 0.65, 0.1, 1.0))
    batch.draw(shader)


def register():
    global _handle
    _handle = bpy.types.SpaceView3D.draw_handler_add(draw_frame, (), 'WINDOW', 'POST_PIXEL')


def unregister():
    global _handle
    if _handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, 'WINDOW')
        _handle = None
