# SPDX-License-Identifier: GPL-3.0-or-later
"""Apply a generated patch through one native image-undo stroke."""
from math import ceil, hypot

import bpy
import numpy as np

from . import model, overlay, painting


def validate_image(image):
    if (not image or image.is_float or image.channels != 4
            or image.source not in {'FILE', 'GENERATED'} or min(image.size) <= 0
            or image.colorspace_settings.name != 'sRGB' or image.alpha_mode != 'STRAIGHT'):
        raise ValueError('Generation requires a byte RGBA, straight-alpha sRGB layer image')
    if max(image.size) > 2048:
        raise ValueError('This generation slice supports layer images up to 2048 pixels per axis')


def apply(context, rgba, weights):
    layer = model.active_layer(model.active_stack(context))
    validate_image(layer.image)
    if painting.active():
        raise ValueError('Finish painting before applying generation')
    region = next(r for r in context.area.regions if r.type == 'WINDOW')
    images = []
    old_brush, old_unified, old_paint = {}, {}, {}
    brush = unified = paint = None
    try:
        if bpy.ops.pawprint.paint_layer() != {'RUNNING_MODAL'}:
            raise RuntimeError('Could not enter the saved-frame painting session')
        with context.temp_override(region=region):
            corners = overlay.frame_corners(layer, region, context.space_data.region_3d)
            width, height = corners[1][0]-corners[0][0], corners[3][1]-corners[0][1]
            radius = ceil(hypot(width, height)/2) + 2
            paint = context.scene.tool_settings.image_paint
            brush, unified = paint.brush, paint.unified_paint_settings
            if not brush:
                raise ValueError('Select a native Texture Paint brush first')
            if radius > unified.bl_rna.properties['size'].hard_max:
                raise ValueError('Viewport is too large for result application; reduce its size')
            source_pixels = rgba.copy()
            source_pixels[:, :, 3] = 1
            for name, values, is_mask in (
                    ('Pawprint generated source', source_pixels, False),
                    ('Pawprint generated feather', np.repeat(weights[:, :, None], 4, axis=2), True)):
                values = np.repeat(np.repeat(values, 2, axis=0), 2, axis=1)
                values[:, :, 3] = 1
                image = bpy.data.images.new(name, width=values.shape[1], height=values.shape[0], float_buffer=is_mask)
                images.append(image.name)
                image[painting._TAG] = True
                if is_mask:
                    image.colorspace_settings.name = 'Non-Color'
                image.pixels.foreach_set(values.ravel())
                image.update()
            settings = dict(image_brush_type='CLONE', blend='MIX', stroke_method='DOTS',
                            curve_distance_falloff_preset='CONSTANT', texture=None, mask_texture=None,
                            use_pressure_size=False, use_pressure_strength=False, use_accumulate=False,
                            jitter=0.0, use_locked_size='VIEW', use_alpha=True)
            unified_settings = dict(use_unified_size=True, use_unified_strength=True,
                                    size=radius, strength=1.0, use_locked_size='VIEW')
            paint_settings = dict(clone_image=bpy.data.images[images[0]], use_clone_layer=True,
                                  clone_offset=(0, 0), clone_alpha=1,
                                  stencil_image=bpy.data.images[images[1]], use_stencil_layer=True,
                                  invert_stencil=True, dither=0, use_symmetry_x=False,
                                  use_symmetry_y=False, use_symmetry_z=False, use_cavity=False)
            old_brush = {key: getattr(brush, key) for key in settings}
            old_unified = {key: getattr(unified, key) for key in unified_settings}
            old_paint = {key: (tuple(getattr(paint, key)) if key == 'clone_offset' else getattr(paint, key))
                         for key in paint_settings}
            for owner, overrides in ((brush, settings), (unified, unified_settings), (paint, paint_settings)):
                for key, value in overrides.items():
                    setattr(owner, key, value)
            context.object.data.uv_layers[0].active_clone = True
            position = (corners[0][0]+width/2, corners[0][1]+height/2)
            stroke = [dict(name='Apply Pawprint generation', mouse=position, mouse_event=position,
                           pressure=1, size=radius, time=0, is_start=True,
                           x_tilt=0, y_tilt=0, location=(0, 0, 0))]
            if bpy.ops.paint.image_paint('EXEC_DEFAULT', True, stroke=stroke) != {'FINISHED'}:
                raise RuntimeError('Native result application was cancelled')
    finally:
        try:
            for owner, saved in ((brush, old_brush), (unified, old_unified), (paint, old_paint)):
                for key, value in saved.items():
                    setattr(owner, key, value)
        finally:
            try:
                painting.finish()
            finally:
                for name in images:
                    image = bpy.data.images.get(name)
                    if image:
                        bpy.data.images.remove(image)
