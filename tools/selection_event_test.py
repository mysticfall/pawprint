# SPDX-License-Identifier: GPL-3.0-or-later
"""Real Blender event-loop regression check for lasso/box selection modes."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def worker():
    import bpy
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from locked_paint_probe import setup
    extension, area, region, target, _, _ = setup()
    with bpy.context.temp_override(area=area, region=region):
        bpy.ops.pawprint.finish_paint()
    window = bpy.context.window
    selection = extension.selection
    rect = [(0.1, 0.2), (0.4, 0.2), (0.4, 0.6), (0.1, 0.6)]
    island = [(0.7, 0.7), (0.8, 0.7), (0.8, 0.8), (0.7, 0.8)]
    cut = [(0.2, 0.3), (0.3, 0.3), (0.3, 0.5), (0.2, 0.5)]
    cases = [('LASSO', 'REPLACE', rect, ''), ('LASSO', 'ADD', island, ''),
             ('LASSO', 'SUBTRACT', cut, ''), ('BOX', 'REPLACE', rect, ''),
             ('BOX', 'ADD', island, ''), ('BOX', 'SUBTRACT', cut, ''),
             ('LASSO', 'REPLACE', cut, 'shift'), ('BOX', 'REPLACE', island, 'ctrl'),
             ('BOX', 'REPLACE', island, 'cancel')]
    state = {'index': 0, 'expected': None, 'paths': []}

    def tick():
        try:
            with bpy.context.temp_override(area=area, region=region):
                layer = bpy.data.objects[target].active_material.pawprint.layers[1]
                if state['expected'] is not None:
                    assert selection._gesture is None, 'Modal gesture did not complete'
                    actual = selection.footprint(layer)
                    assert np.array_equal(actual, state['expected']), f"Event case {state['index']} failed"
                if state['index'] == len(cases):
                    bpy.ops.pawprint.selection_clear()
                    assert layer.selection_paths == '[]' and not selection.footprint(layer).any()
                    assert selection.footprint(layer, generation=True).min() == 1
                    assert selection.context_bounds(layer) == (0, 0, 192, 128)
                    selection.add_path(layer, rect, 'REPLACE')
                    selection.add_path(layer, rect, 'SUBTRACT')
                    assert layer.selection_paths == '[]', 'Subtracting everything must clear the selection'
                    print({'real_selection_events': True, 'lasso_box_modes': True,
                           'modifiers_cancel_clear': True}, flush=True)
                    bpy.ops.wm.quit_blender()
                    return None
                shape, operation, points, modifier = cases[state['index']]
                assert bpy.ops.pawprint.select_lasso('INVOKE_DEFAULT', shape=shape,
                    operation=operation) == {'RUNNING_MODAL'}
                corners = extension.overlay.frame_corners(layer, region, area.spaces.active.region_3d)
                x, y = corners[0]
                w, h = corners[1][0] - x, corners[3][1] - y
                coords = [(round(region.x + x + u * w), round(region.y + y + v * h)) for u, v in points]
                if shape == 'BOX':
                    coords = [coords[2], coords[0]]  # Reverse drag also supported.
                    a, b = coords[0]
                    c, d = coords[1]
                    polygon = [(a, b), (c, b), (c, d), (a, d)]
                else:
                    polygon = coords
                normalized = [((a - region.x - x) / w, (b - region.y - y) / h) for a, b in polygon]
                if modifier != 'cancel':
                    effective = {'shift': 'ADD', 'ctrl': 'SUBTRACT'}.get(modifier, operation)
                    if effective == 'REPLACE':
                        state['paths'] = []
                    state['paths'].append((effective, normalized))
                state['expected'] = selection.rasterize(state['paths'], 192, 128)
                kwargs = {'shift': modifier == 'shift', 'ctrl': modifier == 'ctrl'}
                window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=coords[0][0], y=coords[0][1])
                window.event_simulate(type='LEFTMOUSE', value='PRESS', x=coords[0][0], y=coords[0][1], **kwargs)
                for a, b in coords[1:]:
                    window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=a, y=b, **kwargs)
                if modifier == 'cancel':
                    window.event_simulate(type='ESC', value='PRESS', x=coords[-1][0], y=coords[-1][1])
                else:
                    window.event_simulate(type='LEFTMOUSE', value='RELEASE', x=coords[-1][0], y=coords[-1][1], **kwargs)
                state['index'] += 1
            return 0.3
        except Exception:
            import traceback
            traceback.print_exc()
            os._exit(1)
    bpy.app.timers.register(tick, first_interval=0.3)


if __name__ == '__main__':
    if '--worker' in sys.argv:
        import bpy
        bpy.context.preferences.view.show_splash = False
        def start():
            try:
                worker()
            except Exception:
                import traceback
                traceback.print_exc()
                os._exit(1)
        bpy.app.timers.register(start, first_interval=2)
    else:
        with tempfile.TemporaryDirectory(prefix='pawprint-selection-events-') as temp:
            env = os.environ.copy()
            env['PAWPRINT_PROBE_MANAGED'] = '1'
            for suffix in ('RESOURCES', 'CONFIG', 'SCRIPTS', 'DATAFILES', 'EXTENSIONS'):
                env[f'BLENDER_USER_{suffix}'] = str(Path(temp) / suffix.lower())
            subprocess.run(['blender', '--factory-startup', '--enable-event-simulate',
                            '--python-exit-code', '1', '--python', str(Path(__file__).resolve()),
                            '--', '--worker'], env=env, check=True, timeout=120)
