# SPDX-License-Identifier: GPL-3.0-or-later
"""Isolated image-space selection, stencil painting and undo verification."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile


def check():
    import bpy
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from locked_paint_probe import setup
    extension, area, region, target_name, _, image_name = setup()
    selection = extension.selection
    rect = [(0.1, 0.2), (0.4, 0.2), (0.4, 0.6), (0.1, 0.6)]
    remote = [(0.8, 0.8), (0.9, 0.8), (0.9, 0.9), (0.8, 0.9)]
    mask = selection.rasterize([('REPLACE', rect), ('ADD', remote)], 100, 80)
    assert mask[20, 20] == 1 and mask[68, 85] == 1 and mask[40, 60] == 0
    cut = selection.rasterize([('REPLACE', rect), ('SUBTRACT', rect)], 100, 80)
    assert not cut.any()
    feather = selection.rasterize([('REPLACE', rect)], 100, 80, 4)
    assert 0 < feather[20, 9] < 1 and feather[20, 4] == 0 and feather[25, 25] == 1
    assert selection.rasterize([('REPLACE', [(0, 0), (1, 0), (1, 1), (0, 1)])], 100, 80, 4).min() == 1

    with bpy.context.temp_override(area=area, region=region):
        bpy.ops.pawprint.finish_paint()
        layer = bpy.data.objects[target_name].active_material.pawprint.layers[1]
        # Drive the real modal operator through press/move/release. Coordinates
        # come from the restored rectangular saved frame, not the target UVs.
        from types import SimpleNamespace
        assert bpy.ops.pawprint.select_lasso('INVOKE_DEFAULT', operation='REPLACE') == {'RUNNING_MODAL'}
        lasso = selection._gesture
        corners = extension.overlay.frame_corners(layer, region, area.spaces.active.region_3d)
        for index, (u, v) in enumerate(rect + rect[:1]):
            event = SimpleNamespace(type='LEFTMOUSE' if index in (0, 4) else 'MOUSEMOVE',
                value='RELEASE' if index == 4 else 'PRESS', ctrl=False, shift=False,
                mouse_x=region.x + corners[0][0] + u * (corners[1][0] - corners[0][0]),
                mouse_y=region.y + corners[0][1] + v * (corners[3][1] - corners[0][1]))
            result = lasso.modal(bpy.context, event)
        assert result == {'FINISHED'}
        assert np.array_equal(selection.footprint(layer), selection.rasterize([('REPLACE', rect)], 192, 128))
        selection.add_path(layer, remote, 'ADD')
        layer.generation_padding = 0
        assert selection.context_bounds(layer) == (19, 26, 173, 115), selection.context_bounds(layer)
        layer.generation_padding = 200
        assert selection.context_bounds(layer) == (0, 0, 192, 128)
        # Persisted selection metadata has native memfile undo independently of pixels.
        bpy.ops.ed.undo_push(message='Selection baseline')
        bpy.ops.pawprint.selection_clear('EXEC_DEFAULT', True)
        assert bpy.data.objects[target_name].active_material.pawprint.layers[1].selection_paths == '[]'
        bpy.ops.ed.undo()
        layer = bpy.data.objects[target_name].active_material.pawprint.layers[1]
        assert layer.selection_paths != '[]'
        before_paths = layer.selection_paths
        layer.generation_padding = 3
        layer.generation_feather = 4
        layer.generation_preview = True
        hard_mask = selection.footprint(layer).copy()
        generation_mask = selection.footprint(layer, generation=True)
        assert np.any((generation_mask > 0) & (generation_mask < 1))
        assert np.all((hard_mask == 0) | (hard_mask == 1))
        layer.generation_feather = 12
        assert np.array_equal(selection.footprint(layer), hard_mask)
        layer.generation_feather = 4
        # Exercise the actual GPU overlay, not only numeric rasterization.
        corners = extension.overlay.frame_corners(layer, region, area.spaces.active.region_3d)
        selection.draw(layer, corners)
        old_stencil = bpy.data.images.new('Existing native stencil', width=8, height=8)
        paint = bpy.context.scene.tool_settings.image_paint
        paint.stencil_image, paint.use_stencil_layer, paint.invert_stencil = old_stencil, True, False
        lower = bpy.data.objects[target_name].active_material.pawprint.layers[0].image
        lower_name, lower_pixels = lower.name, tuple(lower.pixels)
        assert bpy.ops.pawprint.paint_layer() == {'RUNNING_MODAL'}
        image = bpy.data.images[image_name]
        def pixels():
            return np.array(bpy.data.images[image_name].pixels[:]).reshape((128, 192, 4))
        before = pixels()
        paint = bpy.context.scene.tool_settings.image_paint
        brush = paint.brush
        brush.blend = 'ERASE_ALPHA'
        unified = paint.unified_paint_settings
        unified.use_unified_size = unified.use_unified_strength = True
        unified.size, unified.strength = 500, 1
        brush.use_pressure_strength = brush.use_pressure_size = False
        width, height = extension.projection.frame_size(region, 1.5)
        x, y = region.width / 2, region.height / 2
        stroke = [dict(name='selection probe', mouse=(x, y), mouse_event=(x, y), pressure=1,
                       size=500, time=0, is_start=True, x_tilt=0, y_tilt=0, location=(0, 0, 0))]
        mask = selection.footprint(layer)
        brush.blend = 'MIX'
        bpy.ops.ed.undo_push(message='Before selected color')
        bpy.ops.paint.image_paint(stroke=stroke)
        colored = pixels()
        assert np.array_equal(before[mask == 0], colored[mask == 0]), 'Color outside selection'
        assert not np.array_equal(before, colored), 'Selected color stroke changed nothing'
        bpy.ops.ed.undo()
        assert np.array_equal(pixels(), before)
        bpy.ops.ed.redo()
        assert np.array_equal(pixels(), colored)
        before = colored
        brush.blend = 'ERASE_ALPHA'
        bpy.ops.ed.undo_push(message='Before selected erase')
        assert bpy.ops.paint.image_paint(stroke=stroke) == {'FINISHED'}
        erased = pixels()
        mask = selection.footprint(layer)
        assert np.array_equal(before[mask == 0], erased[mask == 0]), 'Stencil touched unselected pixels'
        assert np.any(erased[mask == 1, 3] < before[mask == 1, 3]), 'Stencil did not erase selected pixels'
        bpy.ops.ed.undo()
        assert np.array_equal(pixels(), before), 'Selected erase undo'
        bpy.ops.ed.redo()
        assert np.array_equal(pixels(), erased), 'Selected erase redo'
        # A full-frame native dab must clear all fully selected pixels, not just
        # reduce alpha, while remaining one undo step interleaved with strokes.
        old_brush = (brush.blend, brush.curve_distance_falloff_preset, unified.size)
        assert bpy.ops.pawprint.delete_selected_pixels() == {'FINISHED'}
        deleted = pixels()
        assert np.array_equal(deleted[mask == 0], before[mask == 0]), 'Delete outside selection'
        assert deleted[mask == 1, 3].max() == 0, 'Delete did not clear fully selected pixels'
        assert (brush.blend, brush.curve_distance_falloff_preset, unified.size) == old_brush
        bpy.ops.ed.undo()
        assert np.array_equal(pixels(), erased), 'Delete single-step undo'
        bpy.ops.ed.undo()
        assert np.array_equal(pixels(), before), 'Interleaved erase/delete undo'
        bpy.ops.ed.redo()
        assert np.array_equal(pixels(), erased)
        bpy.ops.ed.redo()
        assert np.array_equal(pixels(), deleted)
        prior_erase = erased
        erased = deleted
        bpy.ops.pawprint.finish_paint()
        assert not any(i.get('_pawprint_paint_proxy') for i in bpy.data.images)
        paint = bpy.context.scene.tool_settings.image_paint
        assert paint.stencil_image == old_stencil and paint.use_stencil_layer and not paint.invert_stencil
        assert tuple(bpy.data.images[lower_name].pixels) == lower_pixels
        bpy.ops.ed.undo()
        assert np.array_equal(pixels(), prior_erase), 'Delete undo after session exit'
        bpy.ops.ed.redo()
        assert np.array_equal(pixels(), deleted), 'Delete redo after session exit'
        assert not any(i.get('_pawprint_paint_proxy') for i in bpy.data.images)
        # Repeat with a hard boundary: native stencil filtering must not change
        # any pixel outside the binary image-space footprint.
        layer = bpy.data.objects[target_name].active_material.pawprint.layers[1]
        layer.generation_feather = 0
        bpy.ops.pawprint.paint_layer()
        hard_before = pixels()
        hard_mask = selection.footprint(layer)
        assert bpy.ops.pawprint.delete_selected_pixels() == {'FINISHED'}
        assert np.array_equal(pixels()[hard_mask == 0], hard_before[hard_mask == 0])
        assert pixels()[hard_mask == 1, 3].max() == 0
        bpy.ops.pawprint.finish_paint()
        erased = pixels()
        layer = bpy.data.objects[target_name].active_material.pawprint.layers[1]
        layer.generation_feather = 4
        with tempfile.TemporaryDirectory(prefix='pawprint-selection-save-') as temp:
            path = str(Path(temp) / 'selection.blend')
            bpy.data.images[image_name].pack()
            bpy.ops.wm.save_as_mainfile(filepath=path)
            bpy.ops.wm.open_mainfile(filepath=path, load_ui=False)
            layer = bpy.data.objects[target_name].active_material.pawprint.layers[1]
            assert layer.selection_paths == before_paths and layer.generation_feather == 4
            assert np.array_equal(pixels(), erased)
            assert not any(i.get('_pawprint_paint_proxy') for i in bpy.data.images)
    print({'selection_math': True, 'lasso_mapping': True, 'selection_undo': True,
           'masked_native_color_erase_undo': True, 'delete_interleaved_undo': True,
           'selection_persistence': True}, flush=True)


if __name__ == '__main__':
    if '--worker' in sys.argv:
        import bpy
        bpy.context.preferences.view.show_splash = False
        def run():
            try:
                check()
                bpy.ops.wm.quit_blender()
            except Exception:
                import traceback
                traceback.print_exc()
                os._exit(1)
        bpy.app.timers.register(run, first_interval=2)
    else:
        with tempfile.TemporaryDirectory(prefix='pawprint-selection-') as temp:
            env = os.environ.copy()
            env['PAWPRINT_PROBE_MANAGED'] = '1'
            for suffix in ('RESOURCES', 'CONFIG', 'SCRIPTS', 'DATAFILES', 'EXTENSIONS'):
                env[f'BLENDER_USER_{suffix}'] = str(Path(temp) / suffix.lower())
            subprocess.run(['blender', '--factory-startup', '--python-exit-code', '1',
                            '--python', str(Path(__file__).resolve()), '--', '--worker'],
                           env=env, check=True, timeout=120)
