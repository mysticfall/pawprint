# SPDX-License-Identifier: GPL-3.0-or-later
"""Image-space edit footprints, independent of projected surface coverage."""

import json
import math

import bpy
import numpy as np

from . import model

_cache = {}
_texture = None
_gesture = None


def dimensions(layer):
    return tuple(layer.image.size) if layer.image else (layer.width, layer.height)


def rasterize(paths, width, height, feather=0):
    """Pixel-center even/odd fill, ordered union/subtraction, finite box feather."""
    mask = np.zeros((height, width), dtype=np.float32)
    for operation, points in paths:
        inside = np.zeros((height, width), dtype=bool)
        if len(points) < 3:
            continue
        vertices = np.asarray(points, dtype=np.float64)
        ax, ay = vertices.T
        bx, by = np.roll(vertices, -1, axis=0).T
        for row in range(max(0, math.ceil(ay.min() * height - 0.5)),
                         min(height, math.ceil(ay.max() * height - 0.5))):
            y = (row + 0.5) / height
            crossing = (ay > y) != (by > y)
            intersections = np.sort(ax[crossing] + (bx[crossing] - ax[crossing])
                                    * (y - ay[crossing]) / (by[crossing] - ay[crossing]))
            columns = np.clip(np.ceil(intersections * width - 0.5), 0, width).astype(int)
            for left, right in zip(columns[::2], columns[1::2]):
                inside[row, left:right] = True
        if operation == 'REPLACE':
            mask[:] = inside
        elif operation == 'ADD':
            mask[inside] = 1
        else:
            mask[inside] = 0
    radius = int(feather)
    if radius:
        # Extend frame-edge values: selecting the full image must stay full at
        # its border. Feather has finite support, so unselected pixels stay exact.
        for axis in (0, 1):
            pads = [(0, 0), (0, 0)]
            pads[axis] = (radius, radius)
            padded = np.pad(mask, pads, mode='edge')
            integral = np.cumsum(padded, axis=axis, dtype=np.float64)
            pads[axis] = (1, 0)
            integral = np.pad(integral, pads)
            length = mask.shape[axis]
            mask = ((np.take(integral, np.arange(length) + 2 * radius + 1, axis=axis)
                     - np.take(integral, np.arange(length), axis=axis)) / (2 * radius + 1)).astype(np.float32)
    return mask


def footprint(layer, generation=False):
    global _cache, _texture
    width, height = dimensions(layer)
    radius = layer.generation_feather if generation else 0
    key = (layer.selection_paths, width, height, radius)
    if generation not in _cache or _cache[generation][0] != key:
        paths = json.loads(layer.selection_paths)
        mask = (np.ones((height, width), dtype=np.float32) if generation and not paths
                else rasterize(paths, width, height, radius))
        xs = np.flatnonzero(np.any(mask > 0, axis=0))
        ys = np.flatnonzero(np.any(mask > 0, axis=1))
        bounds = (int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1) if len(xs) else None
        _cache[generation] = key, mask, bounds
        if not generation:
            _texture = None
    return _cache[generation][1]


def context_bounds(layer):
    """Generation context; no selection means unrestricted/full-frame context."""
    width, height = dimensions(layer)
    if layer.generation_context == 'FRAME' or layer.selection_paths == '[]':
        return (0, 0, width, height)
    footprint(layer, generation=True)
    bounds = _cache[True][2]
    if bounds is None:
        return None
    a, b, c, d = bounds
    p = layer.generation_padding
    return (max(0, a - p), max(0, b - p), min(width, c + p), min(height, d + p))


def add_path(layer, points, operation):
    paths = [] if operation == 'REPLACE' else json.loads(layer.selection_paths)
    paths.append((operation, points))
    layer.selection_paths = json.dumps(paths, separators=(',', ':'))
    if not footprint(layer).any():
        layer.selection_paths = '[]'


def editable(context):
    from . import painting
    from .operators import viewport
    layer = model.active_layer(model.active_stack(context))
    return bool(not painting.active() and layer and layer.image and min(layer.image.size) > 0
                and context.object and context.object.mode == 'OBJECT'
                and viewport(context)[0])


class PAWPRINT_OT_select_lasso(bpy.types.Operator):
    bl_idname = 'pawprint.select_lasso'
    bl_label = 'Lasso Selection'
    bl_description = 'Drag in the saved image frame; Shift adds, Ctrl subtracts; Escape cancels'
    bl_options = {'UNDO'}
    operation: bpy.props.EnumProperty(items=[(v, v.title(), '') for v in ('REPLACE', 'ADD', 'SUBTRACT')], default='REPLACE')
    shape: bpy.props.EnumProperty(items=[('LASSO', 'Lasso', ''), ('BOX', 'Box', '')], default='LASSO')

    @classmethod
    def poll(cls, context):
        return editable(context)

    def invoke(self, context, event):
        global _gesture
        if _gesture:
            _gesture.finish()
        stack = model.active_stack(context)
        stack.preview = False
        bpy.ops.pawprint.restore_view()
        self._area = context.area
        self._identity = (stack.id_data.name, stack.active_index, model.active_layer(stack).image.name)
        self._points = []
        self._drawing = False
        self._operation = self.operation
        _gesture = self
        self._area.header_text_set(f'{self.shape.title()} · {self.operation.title()} · Shift: add · Ctrl: subtract · Escape: cancel')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        return self.invoke(context, None)

    def finish(self):
        global _gesture
        _gesture = None
        self._area.header_text_set(None)
        self._area.tag_redraw()

    def modal(self, context, event):
        from .overlay import frame_corners
        if _gesture is None or _gesture.as_pointer() != self.as_pointer():
            return {'CANCELLED'}
        stack = model.active_stack(context)
        layer = model.active_layer(stack)
        if (context.area != self._area or not layer or not layer.image
                or (stack.id_data.name, stack.active_index, layer.image.name) != self._identity):
            self.finish()
            return {'CANCELLED'}
        if event.type in {'ESC', 'RIGHTMOUSE'}:
            self.finish()
            return {'CANCELLED'}
        region = next(r for r in self._area.regions if r.type == 'WINDOW')
        corners = frame_corners(layer, region, self._area.spaces.active.region_3d)
        if not corners:
            self.finish()
            return {'CANCELLED'}
        x, y = event.mouse_x - region.x, event.mouse_y - region.y
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if not (0 <= x < region.width and 0 <= y < region.height):
                return {'RUNNING_MODAL'}
            self._drawing = True
            if event.ctrl:
                self._operation = 'SUBTRACT'
            elif event.shift:
                self._operation = 'ADD'
        if self._drawing and event.type in {'LEFTMOUSE', 'MOUSEMOVE'}:
            # Keep coordinates outside [0,1]; rasterization clips the polygon,
            # rather than distorting crossings by clamping individual vertices.
            point = ((x - corners[0][0]) / (corners[1][0] - corners[0][0]),
                     (y - corners[0][1]) / (corners[3][1] - corners[0][1]))
            if self.shape == 'BOX':
                if not self._points:
                    self._points = [point] * 4
                a, b = self._points[0]
                u, v = point
                self._points = [(a, b), (u, b), (u, v), (a, v)]
            elif not self._points or math.dist(point, self._points[-1]) > 0.25 / max(dimensions(layer)):
                self._points.append(point)
            self._area.tag_redraw()
        if self._drawing and event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            if self.shape == 'BOX' and (self._points[0][0] == self._points[2][0]
                                       or self._points[0][1] == self._points[2][1]):
                self.finish()
                return {'CANCELLED'}
            if len(self._points) >= 3:
                add_path(layer, self._points, self._operation)
                self.finish()
                return {'FINISHED'}
            self.finish()
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        self.finish()


class PAWPRINT_OT_selection_clear(bpy.types.Operator):
    bl_idname = 'pawprint.selection_clear'
    bl_label = 'Clear Selection'
    bl_options = {'UNDO'}

    @classmethod
    def poll(cls, context):
        return editable(context)

    def execute(self, context):
        layer = model.active_layer(model.active_stack(context))
        layer.selection_paths = '[]'
        return {'FINISHED'}


def draw(layer, corners):
    global _texture
    import gpu
    from gpu_extras.batch import batch_for_shader
    if min(dimensions(layer)) <= 0:
        return
    if not layer.selection_paths or layer.selection_paths == '[]':
        mask = None
    else:
        mask = footprint(layer)
    x, y = corners[0]
    w, h = corners[1][0] - x, corners[3][1] - y

    def line(points, color):
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        shader.bind()
        shader.uniform_float('color', color)
        batch_for_shader(shader, 'LINE_STRIP', {'pos': points}).draw(shader)

    gpu.state.blend_set('ALPHA')
    try:
        if mask is not None:
            if _texture is None:
                rgba = np.empty((*mask.shape, 4), dtype=np.float32)
                rgba[:, :, :3] = (0.1, 0.65, 1.0)
                rgba[:, :, 3] = mask * 0.3
                _texture = gpu.types.GPUTexture((mask.shape[1], mask.shape[0]), format='RGBA32F',
                    data=gpu.types.Buffer('FLOAT', rgba.size, rgba.ravel()))
            shader = gpu.shader.from_builtin('IMAGE')
            shader.bind()
            shader.uniform_sampler('image', _texture)
            batch_for_shader(shader, 'TRI_FAN', {'pos': corners[:4],
                'texCoord': [(0, 0), (1, 0), (1, 1), (0, 1)]}).draw(shader)
        if layer.generation_preview:
            bounds = context_bounds(layer)
            if bounds:
                width, height = dimensions(layer)
                a, b, c, d = bounds
                points = [(x + u / width * w, y + v / height * h)
                          for u, v in ((a, b), (c, b), (c, d), (a, d), (a, b))]
                line(points, (0.1, 1.0, 0.5, 1.0))
        if _gesture and _gesture._area == bpy.context.area and len(_gesture._points) > 1:
            points = [(x + u * w, y + v * h) for u, v in _gesture._points]
            line(points + points[:1], (0.2, 0.8, 1.0, 1.0))
    finally:
        gpu.state.blend_set('NONE')


CLASSES = (PAWPRINT_OT_select_lasso, PAWPRINT_OT_selection_clear)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    global _cache, _texture
    if _gesture:
        _gesture.finish()
    _cache = {}
    _texture = None
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
