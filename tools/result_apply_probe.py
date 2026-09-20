# SPDX-License-Identifier: GPL-3.0-or-later
"""Isolated feasibility probe: native clone dab as generated-result image undo."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def setup():
    import bpy
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from locked_paint_probe import setup as fixture
    extension, area, region, target, _, image_name = fixture()
    if extension.painting.active():
        # Navigation is free during sessions now; realign so this probe's
        # full-frame EXEC dab maps through the saved view.
        with bpy.context.temp_override(area=area, region=region):
            assert bpy.ops.pawprint.restore_view() == {'FINISHED'}
    return extension, area, image_name


def check(state, generated=None, weights=None):
    import bpy
    import numpy as np
    extension, area, name = state
    region = next(r for r in area.regions if r.type == 'WINDOW')
    with bpy.context.temp_override(area=area, region=region):
        image = bpy.data.images[name]
        width, height = image.size
        before = np.array(image.pixels[:]).reshape(height, width, 4)
        others = {layer.image.name: np.array(layer.image.pixels[:])
                  for layer in extension.painting.session_stack().layers
                  if layer.image and layer.image.name != name}
        source = bpy.data.images.new('Temporary generated patch', width=width*2, height=height*2)
        rgba = np.zeros((height, width, 4), dtype=np.float32)
        y, x = np.mgrid[:height, :width]
        rgba[:, :, 0] = (x % 7) / 6
        rgba[:, :, 1] = (y % 11) / 10
        rgba[:, :, 2] = 0.6
        rgba[20:80, 30:110, 3] = 1
        rgba[80:90, 30:110, 3] = 0.5
        if generated is not None:
            rgba[:, :, :3] = generated[:, :, :3]
            rgba[:, :, 3] = weights
        weight = rgba[:, :, 3:4].copy()
        rgba[:, :, 3] = 1
        source.pixels.foreach_set(np.repeat(np.repeat(rgba, 2, axis=0), 2, axis=1).ravel())
        source.update()
        # Byte-source quantization is part of the expected native input.
        rgba = np.array(source.pixels[:]).reshape(height*2, width*2, 4)[::2, ::2]
        paint = bpy.context.scene.tool_settings.image_paint
        brush = paint.brush
        brush.image_brush_type = 'CLONE'
        paint.clone_image = source
        paint.use_clone_layer = True
        bpy.context.object.data.uv_layers[0].active_clone = True
        paint.clone_offset = (0, 0)
        paint.clone_alpha = 1
        stencil = bpy.data.images.new('Temporary feather', width=width*2, height=height*2, float_buffer=True)
        stencil.colorspace_settings.name = 'Non-Color'
        stencil_pixels = np.ones((height*2, width*2, 4), dtype=np.float32)
        stencil_pixels[:, :, :3] = np.repeat(np.repeat(weight, 2, axis=0), 2, axis=1)
        stencil.pixels.foreach_set(stencil_pixels.ravel())
        stencil.update()
        paint.stencil_image, paint.use_stencil_layer, paint.invert_stencil = stencil, True, True
        temporary_names = (source.name, stencil.name)
        brush.blend = 'MIX'
        brush.stroke_method = 'DOTS'
        brush.curve_distance_falloff_preset = 'CONSTANT'
        brush.texture = brush.mask_texture = None
        brush.use_pressure_size = brush.use_pressure_strength = False
        brush.use_accumulate = False
        unified = paint.unified_paint_settings
        unified.use_unified_size = unified.use_unified_strength = True
        unified.size, unified.strength = 2000, 1
        cx, cy = region.width / 2, region.height / 2
        stroke = [dict(name='Apply result', mouse=(cx, cy), mouse_event=(cx, cy), pressure=1,
                       size=2000, time=0, is_start=True, x_tilt=0, y_tilt=0, location=(0, 0, 0))]
        bpy.ops.ed.undo_push(message='Before result')
        assert bpy.ops.paint.image_paint('EXEC_DEFAULT', True, stroke=stroke) == {'FINISHED'}
        def pixels():
            return np.array(bpy.data.images[name].pixels[:]).reshape(height, width, 4)
        after = pixels()
        assert np.array_equal(after[weight[:, :, 0] == 0], before[weight[:, :, 0] == 0]), 'Outside changed'
        error = float(np.abs(after[weight[:, :, 0] == 1] - rgba[weight[:, :, 0] == 1]).max())
        print({'clone_max_error': error, 'changed_pixels': int(np.any(after != before, axis=2).sum())}, flush=True)
        assert error < 2 / 255, 'Clone mapping/color is not exact enough'
        assert np.any(after != before), 'Application changed nothing'
        partial = (weight[:, :, 0] > 0) & (weight[:, :, 0] < 1) & (before[:, :, 3] > 0)
        assert np.any(partial), 'Fixture must exercise feathered pixels'
        assert np.all(after[partial, 3] >= before[partial, 3]), 'Feather alpha unexpectedly decreased'
        expected_alpha = weight[:, :, 0] + before[:, :, 3] * (1 - weight[:, :, 0])
        assert np.abs(after[:, :, 3] - expected_alpha).max() <= 2/255, 'Feather alpha mismatch'
        expected_rgb = np.divide(rgba[:, :, :3] * weight + before[:, :, :3] * before[:, :, 3:4] * (1 - weight),
                                 expected_alpha[:, :, None], out=before[:, :, :3].copy(),
                                 where=expected_alpha[:, :, None] > 0)
        assert np.abs(after[partial, :3] - expected_rgb[partial]).max() <= 3/255, 'Feather color mismatch'
        bpy.ops.ed.undo()
        assert np.array_equal(pixels(), before), 'Result undo'
        bpy.ops.ed.redo()
        assert np.array_equal(pixels(), after), 'Result redo'
        brush.image_brush_type = 'DRAW'
        brush.blend = 'ERASE_ALPHA'
        assert bpy.ops.paint.image_paint('EXEC_DEFAULT', True, stroke=stroke) == {'FINISHED'}
        erased = pixels()
        bpy.ops.ed.undo()
        assert np.array_equal(pixels(), after), 'Native stroke undo after result'
        bpy.ops.ed.undo()
        assert np.array_equal(pixels(), before), 'Interleaved result undo'
        bpy.ops.ed.redo()
        assert np.array_equal(pixels(), after)
        bpy.ops.ed.redo()
        assert np.array_equal(pixels(), erased)
        bpy.ops.pawprint.finish_paint()
        for temporary_name in temporary_names:
            temporary = bpy.data.images.get(temporary_name)
            if temporary:
                bpy.data.images.remove(temporary)
        bpy.ops.ed.undo()
        assert np.array_equal(pixels(), after), 'Stroke undo after source cleanup'
        bpy.ops.ed.undo()
        assert np.array_equal(pixels(), before), 'Result undo after source cleanup'
        bpy.ops.ed.redo()
        assert np.array_equal(pixels(), after), 'Result redo must not need the source image'
        for other_name, original in others.items():
            assert np.array_equal(np.array(bpy.data.images[other_name].pixels[:]), original), 'Another layer changed'
        print({'native_result_application_undo': True, 'interleaved_native_strokes': True}, flush=True)


if __name__ == '__main__':
    if '--worker' in sys.argv:
        import bpy
        bpy.context.preferences.view.show_splash = False
        state = []
        def run():
            try:
                if not state:
                    state.append(setup())
                    return 1
                check(state[0])
                bpy.ops.wm.quit_blender()
            except Exception:
                import traceback
                traceback.print_exc()
                os._exit(1)
        bpy.app.timers.register(run, first_interval=2)
    else:
        with tempfile.TemporaryDirectory(prefix='pawprint-result-') as temp:
            env = os.environ.copy()
            env['PAWPRINT_PROBE_MANAGED'] = '1'
            for suffix in ('RESOURCES', 'CONFIG', 'SCRIPTS', 'DATAFILES', 'EXTENSIONS'):
                env[f'BLENDER_USER_{suffix}'] = str(Path(temp) / suffix.lower())
            subprocess.run(['blender', '--factory-startup', '--python-exit-code', '1',
                            '--python', str(Path(__file__).resolve()), '--', '--worker'],
                           env=env, check=True, timeout=120)
