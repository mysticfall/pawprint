# SPDX-License-Identifier: GPL-3.0-or-later
"""Managed native image painting through a temporary saved-frame canvas."""

import bpy
import json
from mathutils import Vector

from . import model, projection

_session = None
_TAG = '_pawprint_paint_proxy'


def active():
    return _session is not None


def aligned(state, space):
    """True when the viewport eye pose matches the saved view.

    view_matrix is the world-to-view pose: it changes with rotation, eye
    position and distance but not with lens/framing, so a viewport resized
    since capture stays aligned after the saved frame is refitted. With a
    matching pose every cursor ray originates at the saved eye, where ray
    position and displayed layer pixel coincide exactly.
    """
    view = space.region_3d
    if view.view_perspective != 'PERSP':
        return False
    current = projection.flattened(view.view_matrix)
    return all(abs(a - b) <= 1e-4 for a, b in zip(current, state['view_matrix']))


def paint_header(area, text=None):
    try:
        area.header_text_set(text)
    except ReferenceError:
        pass


def view_state(space):
    view = space.region_3d
    return (tuple(view.view_rotation), tuple(view.view_location), view.view_distance,
            view.view_perspective, space.lens, space.clip_start, space.clip_end)


def session_stack():
    if _session:
        material = bpy.data.materials.get(_session['material'])
        if material:
            return material.pawprint
    return None


def restore_paint_settings(scene, values):
    if scene:
        for key, value in values.items():
            if key in {'canvas', 'stencil_image'}:
                value = bpy.data.images.get(value) if value else None
            setattr(scene.tool_settings.image_paint, key, value)


def remove_proxy():
    for obj in list(bpy.data.objects):
        if obj.get(_TAG):
            was_active = bpy.context.view_layer.objects.active == obj
            target = bpy.context.view_layer.objects.get(obj.get('_pawprint_target', ''))
            if was_active and obj.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            restore_paint_settings(bpy.data.scenes.get(obj.get('_pawprint_scene', '')),
                                   json.loads(obj.get('_pawprint_paint_settings', '{}')))
            mesh = obj.data
            materials = list(mesh.materials)
            bpy.data.objects.remove(obj, do_unlink=True)
            if was_active and target:
                target.select_set(True)
                bpy.context.view_layer.objects.active = target
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
            for material in materials:
                if material and material.users == 0:
                    bpy.data.materials.remove(material)
    for image in list(bpy.data.images):
        if image.get(_TAG):
            bpy.data.images.remove(image)


def finish():
    global _session
    state, _session = _session, None
    if state is None:
        return
    window = state['window']
    area = state['area']
    scene = bpy.data.scenes.get(state['scene'])
    view_layer = scene.view_layers.get(state['view_layer']) if scene else None
    previous_scene = window.scene
    try:
        # Texture-paint teardown uses the window's evaluated scene internally,
        # even with a Python context override. Restore it for mode exit, then
        # leave the user's newly selected scene intact.
        if scene and window.scene != scene:
            window.scene = scene
        with bpy.context.temp_override(window=window, scene=scene or window.scene,
                                       view_layer=view_layer or window.view_layer):
            obj = bpy.context.active_object
            if obj and obj.get(_TAG) and obj.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            remove_proxy()
            for obj in bpy.context.selected_objects:
                obj.select_set(False)
            for name in state['selection']:
                obj = bpy.context.view_layer.objects.get(name)
                if obj:
                    obj.select_set(True)
            bpy.context.view_layer.objects.active = bpy.context.view_layer.objects.get(state['target'])
        restore_paint_settings(scene, state['paint'])
        for name, blend in state['blends'].items():
            brush = bpy.data.brushes.get(name)
            if brush:
                brush.blend = blend
        if area in window.screen.areas[:] and area.type == 'VIEW_3D':
            space = area.spaces.active
            space.shading.type = state['shading']
            space.show_gizmo = state['gizmo']
            paint_header(area, None)
            area.tag_redraw()
    finally:
        if window.scene != previous_scene:
            window.scene = previous_scene
        bpy.context.window_manager.event_timer_remove(state['timer'])


@bpy.app.handlers.persistent
def stop_before_file_change(*args):
    finish()


@bpy.app.handlers.persistent
def reconcile_undo(*args):
    # Memfile undo may remove or resurrect session-only objects. Native image
    # undo within an intact session needs no intervention.
    if _session:
        proxy = bpy.data.objects.get(_session['proxy'])
        if not proxy or proxy.mode != 'TEXTURE_PAINT':
            finish()
    else:
        remove_proxy()


class PAWPRINT_OT_paint_layer(bpy.types.Operator):
    bl_idname = 'pawprint.paint_layer'
    bl_label = 'Paint Layer'
    bl_description = 'Paint the active image in its saved view; navigation is free and strokes snap back to it; Escape exits'

    @classmethod
    def poll(cls, context):
        from .operators import viewport
        from . import generation
        layer = model.active_layer(model.active_stack(context))
        return bool(not active() and not generation.review_active()
                    and layer and layer.image and layer.depth
                    and min(layer.image.size) > 0
                    and context.object.mode == 'OBJECT' and viewport(context)[0])

    def execute(self, context):
        global _session
        from .operators import viewport
        stack = model.active_stack(context)
        layer = model.active_layer(stack)
        space, region = viewport(context)
        paint = context.scene.tool_settings.image_paint
        state = dict(window=context.window, area=context.area, scene=context.scene.name,
                     view_layer=context.view_layer.name,
                     target=context.object.name, material=stack.id_data.name,
                     image=layer.image.name, index=stack.active_index,
                     selection=[o.name for o in context.selected_objects],
                     shading=space.shading.type, gizmo=space.show_gizmo, blends={},
                      paint={key: getattr(paint, key) for key in
                             ('mode', 'use_normal_falloff', 'seam_bleed',
                              'use_stencil_layer', 'invert_stencil')})
        state['paint']['canvas'] = paint.canvas.name if paint.canvas else None
        state['paint']['stencil_image'] = paint.stencil_image.name if paint.stencil_image else None
        state['timer'] = context.window_manager.event_timer_add(0.1, window=context.window)
        _session = state
        try:
            bpy.ops.pawprint.restore_view()
            # The pose snapshot after the initial restore defines stroke
            # alignment; see aligned().
            state['view_matrix'] = projection.flattened(space.region_3d.view_matrix)
            inverse = projection.matrix(layer.projection).inverted()
            corners = ((-1, -1), (1, -1), (1, 1), (-1, 1))
            vertices = []
            for x, y in corners:
                point = inverse @ Vector((x, y, 0, 1))
                vertices.append(point.xyz / point.w)
            mesh = bpy.data.meshes.new('Pawprint temporary canvas')
            mesh.from_pydata(vertices, [], [(0, 1, 2, 3)])
            uv = mesh.uv_layers.new()
            for loop, (x, y) in zip(uv.data, corners):
                loop.uv = ((x + 1) / 2, (y + 1) / 2)
            proxy = bpy.data.objects.new('Pawprint temporary paint proxy', mesh)
            proxy[_TAG] = True
            proxy['_pawprint_target'] = state['target']
            proxy['_pawprint_scene'] = state['scene']
            proxy['_pawprint_paint_settings'] = json.dumps(state['paint'])
            context.scene.collection.objects.link(proxy)
            state['proxy'] = proxy.name
            proxy.hide_render = True
            material = bpy.data.materials.new('Pawprint transparent canvas')
            material.use_nodes = True
            nodes = material.node_tree.nodes
            nodes.clear()
            transparent = nodes.new('ShaderNodeBsdfTransparent')
            output = nodes.new('ShaderNodeOutputMaterial')
            material.node_tree.links.new(transparent.outputs[0], output.inputs['Surface'])
            mesh.materials.append(material)
            for obj in context.selected_objects:
                obj.select_set(False)
            proxy.select_set(True)
            context.view_layer.objects.active = proxy
            paint.mode, paint.canvas = 'IMAGE', layer.image
            paint.use_normal_falloff, paint.seam_bleed = False, 0
            paint.use_stencil_layer = False
            if layer.selection_paths != '[]':
                from . import selection
                import numpy as np
                mask = selection.footprint(layer)
                # Native stencil sampling is bilinear. Duplicate texels so tiny
                # projection roundoff at a canvas pixel center cannot interpolate
                # across a hard selection boundary (even a 1/255 color change).
                mask = np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1)
                stencil = bpy.data.images.new('Pawprint temporary selection',
                    width=mask.shape[1], height=mask.shape[0], float_buffer=False)
                stencil[_TAG] = True
                stencil.colorspace_settings.name = 'Non-Color'
                rgba = np.ones((*mask.shape, 4), dtype=np.float32)
                rgba[:, :, :3] = mask[:, :, None]
                stencil.pixels.foreach_set(rgba.ravel())
                stencil.update()
                paint.stencil_image = stencil
                paint.invert_stencil = True
                paint.use_stencil_layer = True
            space.shading.type, space.show_gizmo = 'MATERIAL', False
            with context.temp_override(region=region):
                bpy.ops.object.mode_set(mode='TEXTURE_PAINT')
            # The native header (including the brush color picker) must stay
            # visible during the session; header_text_set replaces its whole
            # contents, so only the stroke gate below may set a transient hint.
            context.window_manager.modal_handler_add(self)
        except Exception as error:
            finish()
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        self._state = state
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        state = _session
        if state is not getattr(self, '_state', None) or state is None:
            return {'FINISHED'}
        area = state['area']
        if event.type == 'TIMER':
            stack = session_stack()
            proxy = bpy.data.objects.get(state['proxy'])
            layer = model.active_layer(stack)
            if (context.window.scene.name != state['scene']
                    or area not in state['window'].screen.areas[:] or area.type != 'VIEW_3D'
                    or not proxy or context.view_layer.objects.active != proxy
                    or proxy.mode != 'TEXTURE_PAINT' or not layer
                    or not stack.owner
                    or stack.active_index != state['index'] or not layer.image
                    or layer.image.name != state['image']):
                finish()
                return {'FINISHED'}
            # Navigation is intentionally free between strokes; strokes
            # re-align through the gate below.
            return {'PASS_THROUGH'}
        if event.type == 'ESC' and event.value == 'PRESS':
            finish()
            return {'FINISHED'}
        region = next((r for r in area.regions if r.type == 'WINDOW'), None)
        inside = region and (region.x <= event.mouse_x < region.x + region.width
                             and region.y <= event.mouse_y < region.y + region.height)
        if inside and event.type == 'DEL' and event.value == 'PRESS':
            if bpy.ops.pawprint.delete_selected_pixels.poll():
                with context.temp_override(area=area, region=region):
                    bpy.ops.pawprint.delete_selected_pixels()
            return {'RUNNING_MODAL'}
        navigation = (event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
                                     'TRACKPADPAN', 'TRACKPADZOOM', 'MOUSEROTATE',
                                     'NDOF_MOTION', 'TAB', 'SPACE', 'HOME', 'ACCENT_GRAVE'}
                      or event.type.startswith('NUMPAD'))
        if state.get('stroking'):
            # A running stroke must keep its rays; navigation resumes on release.
            if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
                state['stroking'] = False
                return {'PASS_THROUGH'}
            if navigation:
                return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}
        if (inside and event.type == 'LEFTMOUSE' and event.value == 'PRESS'
                and not aligned(state, area.spaces.active)):
            # From any other eye the dab would land offset from the cursor
            # (parallax): restore the saved pose and drop this press instead.
            with context.temp_override(area=area, region=region):
                bpy.ops.pawprint.restore_view()
            paint_header(area, 'Saved view restored — start the stroke again. '
                               'Escape finishes.')
            return {'RUNNING_MODAL'}
        if inside and event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            state['stroking'] = True
            # An aligned press ends any transient gate hint, restoring the
            # native header (and its color picker) while the stroke runs.
            paint_header(area, None)
        return {'PASS_THROUGH'}

    def cancel(self, context):
        if _session is getattr(self, '_state', None):
            finish()


class PAWPRINT_OT_finish_paint(bpy.types.Operator):
    bl_idname = 'pawprint.finish_paint'
    bl_label = 'Finish Painting'

    @classmethod
    def poll(cls, context):
        return active()

    def execute(self, context):
        finish()
        return {'FINISHED'}


class PAWPRINT_OT_paint_blend(bpy.types.Operator):
    bl_idname = 'pawprint.paint_blend'
    bl_label = 'Paint Tool'
    erase: bpy.props.BoolProperty(default=False)

    @classmethod
    def poll(cls, context):
        return active() and context.scene.tool_settings.image_paint.brush is not None

    def execute(self, context):
        brush = context.scene.tool_settings.image_paint.brush
        _session['blends'].setdefault(brush.name, brush.blend)
        brush.blend = 'ERASE_ALPHA' if self.erase else 'MIX'
        return {'FINISHED'}


class PAWPRINT_OT_delete_selected_pixels(bpy.types.Operator):
    bl_idname = 'pawprint.delete_selected_pixels'
    bl_label = 'Delete Selected Pixels'
    bl_description = 'Clear selected pixels with native image undo; generation feather does not affect deletion'

    @classmethod
    def poll(cls, context):
        layer = model.active_layer(session_stack())
        return bool(active() and layer and layer.selection_paths != '[]'
                    and context.scene.tool_settings.image_paint.brush)

    def execute(self, context):
        from math import ceil, hypot
        from .overlay import frame_corners
        state = _session
        area = state['area']
        region = next(r for r in area.regions if r.type == 'WINDOW')
        layer = model.active_layer(session_stack())
        with context.temp_override(area=area, region=region):
            bpy.ops.pawprint.restore_view()
            corners = frame_corners(layer, region, area.spaces.active.region_3d)
            width, height = corners[1][0] - corners[0][0], corners[3][1] - corners[0][1]
            radius = ceil(hypot(width, height) / 2) + 2
            paint = context.scene.tool_settings.image_paint
            brush, unified = paint.brush, paint.unified_paint_settings
            if radius > unified.bl_rna.properties['size'].hard_max:
                self.report({'ERROR'}, 'Viewport too large for a single native erase; reduce its size')
                return {'CANCELLED'}
            settings = {'image_brush_type': 'DRAW', 'blend': 'ERASE_ALPHA',
                        'stroke_method': 'DOTS', 'curve_distance_falloff_preset': 'CONSTANT',
                        'texture': None, 'mask_texture': None, 'use_pressure_size': False,
                        'use_pressure_strength': False, 'use_accumulate': False,
                        'jitter': 0.0, 'use_locked_size': 'VIEW'}
            unified_settings = {'use_unified_size': True, 'use_unified_strength': True,
                                'size': radius, 'strength': 1.0, 'use_locked_size': 'VIEW'}
            old = {key: getattr(brush, key) for key in settings}
            old_unified = {key: getattr(unified, key) for key in unified_settings}
            try:
                for key, value in settings.items():
                    setattr(brush, key, value)
                for key, value in unified_settings.items():
                    setattr(unified, key, value)
                position = (corners[0][0] + width / 2, corners[0][1] + height / 2)
                stroke = [dict(name='Delete selection', mouse=position, mouse_event=position,
                               pressure=1, size=radius, time=0, is_start=True,
                               x_tilt=0, y_tilt=0, location=(0, 0, 0))]
                return bpy.ops.paint.image_paint('EXEC_DEFAULT', True, stroke=stroke)
            finally:
                for key, value in old.items():
                    setattr(brush, key, value)
                for key, value in old_unified.items():
                    setattr(unified, key, value)


CLASSES = (PAWPRINT_OT_paint_layer, PAWPRINT_OT_finish_paint, PAWPRINT_OT_paint_blend,
           PAWPRINT_OT_delete_selected_pixels)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    for handlers in (bpy.app.handlers.save_pre, bpy.app.handlers.load_pre):
        handlers.append(stop_before_file_change)
    for handlers in (bpy.app.handlers.undo_post, bpy.app.handlers.redo_post):
        handlers.append(reconcile_undo)


def unregister():
    finish()
    for handlers in (bpy.app.handlers.save_pre, bpy.app.handlers.load_pre):
        handlers.remove(stop_before_file_change)
    for handlers in (bpy.app.handlers.undo_post, bpy.app.handlers.redo_post):
        handlers.remove(reconcile_undo)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
