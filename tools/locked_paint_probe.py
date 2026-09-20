# SPDX-License-Identifier: GPL-3.0-or-later
"""Isolated UI feasibility probe for native painting on a saved-view canvas proxy."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def setup():
    import bpy
    from mathutils import Quaternion, Vector
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from projection_test import load_extension, setup_fixture

    extension = load_extension()
    scene, target, original = setup_fixture(bpy)
    for vertex in target.data.vertices:
        vertex.co.z += vertex.co.x * 0.5
    bpy.context.view_layer.update()
    area = next(a for a in bpy.context.screen.areas if a.type == 'VIEW_3D')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    space = area.spaces.active
    with bpy.context.temp_override(area=area, region=region):
        view = space.region_3d
        view.view_rotation = Quaternion((1, 0, 0, 0))
        view.view_location = (0, 0, 0)
        view.view_distance = 5
        view.view_perspective = 'PERSP'
        space.lens = 50
        bpy.ops.pawprint.create_stack()
        stack = target.active_material.pawprint
        stack.width, stack.height = 192, 128
        bpy.ops.pawprint.add_projection()
        bpy.ops.pawprint.add_projection()
        # Production layers start empty; populate only this paint-test fixture.
        for item in stack.layers:
            empty = item.image
            assert all(value == 0 for value in empty.pixels[:]), 'New layer is not transparent'
            item.image = extension.projection.new_image('Paint fixture', 192, 128, pattern=True)
            bpy.data.images.remove(empty)
        layer = stack.layers[1]
        if os.environ.get('PAWPRINT_PROBE_MANAGED'):
            assert bpy.ops.pawprint.paint_layer() == {'RUNNING_MODAL'}
            # Navigation is free during sessions: this deliberate change must
            # persist (no auto-restore) until a stroke begins.
            view.view_rotation = Quaternion((0.9238795, 0, 0.3826834, 0))
            view.view_distance = 20
            return extension, area, region, target.name, bpy.context.object.name, layer.image.name
        # A fronto-parallel proxy gives native UV painting an exact affine
        # screen-to-image map, independent of target topology and depth.
        inverse = extension.projection.matrix(layer.projection).inverted()
        corners = ((-1, -1), (1, -1), (1, 1), (-1, 1))
        vertices = []
        for x, y in corners:
            p = inverse @ Vector((x, y, 0, 1))
            vertices.append(p.xyz / p.w)
        mesh = bpy.data.meshes.new('Temporary saved-view canvas')
        mesh.from_pydata(vertices, [], [(0, 1, 2, 3)])
        uv = mesh.uv_layers.new()
        for loop, (x, y) in zip(uv.data, corners):
            loop.uv = ((x + 1) / 2, (y + 1) / 2)
        proxy = bpy.data.objects.new('Temporary paint proxy', mesh)
        scene.collection.objects.link(proxy)
        proxy.hide_render = True
        material = bpy.data.materials.new('Invisible paint proxy')
        material.use_nodes = True
        material.node_tree.nodes.clear()
        transparent = material.node_tree.nodes.new('ShaderNodeBsdfTransparent')
        output = material.node_tree.nodes.new('ShaderNodeOutputMaterial')
        material.node_tree.links.new(transparent.outputs[0], output.inputs['Surface'])
        mesh.materials.append(material)
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        proxy.select_set(True)
        bpy.context.view_layer.objects.active = proxy
        paint = scene.tool_settings.image_paint
        paint.mode = 'IMAGE'
        paint.canvas = layer.image
        paint.use_normal_falloff = False
        paint.seam_bleed = 0
        space.shading.type = 'MATERIAL'
        bpy.ops.object.mode_set(mode='TEXTURE_PAINT')
    return extension, area, region, target.name, proxy.name, layer.image.name


def strokes(state, u=0.3, v=0.35):
    import bpy
    from array import array

    extension, area, region, target_name, proxy_name, image_name = state
    image = bpy.data.images[image_name]
    paint = bpy.context.scene.tool_settings.image_paint
    assert paint.brush, 'Native texture paint brush was not loaded'
    brush = paint.brush
    unified = paint.unified_paint_settings
    unified.use_unified_size = True
    unified.size = 12
    unified.use_unified_strength = True
    unified.strength = 1
    image_width, image_height = image.size[:]
    width, height = extension.projection.frame_size(region, image_width / image_height)
    x = (region.width - width) / 2 + width * u
    y = (region.height - height) / 2 + height * v
    stroke = [dict(name='probe', mouse=(x, y), mouse_event=(x, y), pressure=1,
                   size=12, time=0, is_start=True, x_tilt=0, y_tilt=0, location=(0, 0, 0))]

    def pixels():
        return array('f', bpy.data.images[image_name].pixels[:])

    with bpy.context.temp_override(area=area, region=region):
        if extension.painting.active():
            # The navigation gate left the session aligned with the saved view.
            session = extension.painting._session
            assert extension.painting.aligned(session, area.spaces.active)
            assert not bpy.ops.pawprint.remove_layer.poll()
            assert not bpy.ops.pawprint.add_projection.poll()
        if os.environ.get('PAWPRINT_PROBE_SCREENSHOT'):
            bpy.ops.screen.screenshot(filepath=os.environ['PAWPRINT_PROBE_SCREENSHOT'])
        target = bpy.data.objects[target_name]
        original_uvs = [tuple(loop.uv) for loop in target.data.uv_layers.active.data]
        lower_image_name = target.active_material.pawprint.layers[0].image.name
        lower_pixels = array('f', bpy.data.images[lower_image_name].pixels[:])
        before = pixels()
        bpy.ops.ed.undo_push(message='Before native proxy stroke')
        assert bpy.ops.paint.image_paint(stroke=stroke, mode='NORMAL') == {'FINISHED'}
        colored = pixels()
        changed = [i // 4 for i in range(0, len(before), 4) if before[i:i + 4] != colored[i:i + 4]]
        assert changed, 'Stroke changed no image pixels'
        centroid = (sum(p % image_width + 0.5 for p in changed) / len(changed),
                    sum(p // image_width + 0.5 for p in changed) / len(changed))
        assert abs(centroid[0] - image_width * u) < 2, centroid
        assert abs(centroid[1] - image_height * v) < 2, centroid
        bpy.ops.ed.undo()
        assert pixels() == before, 'Native color stroke undo failed'
        bpy.ops.ed.redo()
        assert pixels() == colored, 'Native color stroke redo failed'
        brush = bpy.context.scene.tool_settings.image_paint.brush
        if extension.painting.active():
            bpy.ops.pawprint.paint_blend(erase=True)
        else:
            brush.blend = 'ERASE_ALPHA'
        assert bpy.ops.paint.image_paint(stroke=stroke, mode='NORMAL') == {'FINISHED'}
        erased = pixels()
        assert any(a < b for a, b in zip(erased[3::4], colored[3::4])), 'Eraser did not reduce alpha'
        bpy.ops.ed.undo()
        assert pixels() == colored, 'Native eraser undo failed'
        bpy.ops.ed.undo()
        assert pixels() == before, 'Interleaved stroke undo failed'
        bpy.ops.ed.redo()
        assert pixels() == colored
        bpy.ops.ed.redo()
        assert pixels() == erased
        outside = dict(stroke[0], mouse=(1, 1), mouse_event=(1, 1))
        bpy.ops.paint.image_paint(stroke=[outside], mode='NORMAL')
        assert pixels() == erased, 'Stroke outside canvas changed the image'
        # Proxy removal must leave the target's original UVs and image intact.
        if extension.painting.active():
            session = extension.painting._session
            assert bpy.ops.pawprint.finish_paint() == {'FINISHED'}
            paint = bpy.context.scene.tool_settings.image_paint
            assert paint.mode == session['paint']['mode']
            assert paint.use_normal_falloff == session['paint']['use_normal_falloff']
            assert paint.seam_bleed == session['paint']['seam_bleed']
            assert (paint.canvas.name if paint.canvas else None) == session['paint']['canvas']
            assert area.spaces.active.shading.type == session['shading']
            undo_count = 0
            while pixels() != before and undo_count < 8:
                assert bpy.ops.ed.undo() == {'FINISHED'}
                undo_count += 1
            assert pixels() == before, 'Cannot undo painting across session exit'
            for _ in range(undo_count):
                assert bpy.ops.ed.redo() == {'FINISHED'}
            assert pixels() == erased, 'Cannot redo painting across session exit'
            assert not extension.painting.active()
            assert bpy.context.object.name == target_name
            assert bpy.context.object.mode == 'OBJECT'
            assert proxy_name not in bpy.data.objects
            # Return through the session boundary and verify that native pixel
            # undo remains usable after leaving painting.
            assert bpy.ops.pawprint.paint_layer() == {'RUNNING_MODAL'}
            assert pixels() == erased
            assert bpy.ops.pawprint.finish_paint() == {'FINISHED'}
        else:
            bpy.ops.object.mode_set(mode='OBJECT')
            proxy = bpy.data.objects[proxy_name]
            mesh = proxy.data
            bpy.data.objects.remove(proxy, do_unlink=True)
            bpy.data.meshes.remove(mesh)
        target = bpy.data.objects[target_name]
        assert target.active_material.pawprint.layers[1].image == bpy.data.images[image_name]
        assert [tuple(loop.uv) for loop in target.data.uv_layers.active.data] == original_uvs
        assert array('f', bpy.data.images[lower_image_name].pixels[:]) == lower_pixels
        assert pixels() == erased
        brush_name = brush.name
        if os.environ.get('PAWPRINT_PROBE_MANAGED'):
            target.active_material.pawprint.active_index = 0
            assert bpy.ops.pawprint.paint_layer() == {'RUNNING_MODAL'}
            bpy.ops.pawprint.paint_blend(erase=False)
            assert bpy.context.scene.tool_settings.image_paint.canvas.name == lower_image_name
            assert bpy.ops.paint.image_paint(stroke=stroke, mode='NORMAL') == {'FINISHED'}
            lower_colored = array('f', bpy.data.images[lower_image_name].pixels[:])
            assert lower_colored != lower_pixels
            bpy.ops.pawprint.finish_paint()
            bpy.ops.ed.undo()
            assert array('f', bpy.data.images[lower_image_name].pixels[:]) == lower_pixels
            assert pixels() == erased
            bpy.ops.ed.redo()
            assert array('f', bpy.data.images[lower_image_name].pixels[:]) == lower_colored
            assert pixels() == erased
            bpy.ops.ed.undo()
            bpy.data.objects[target_name].active_material.pawprint.active_index = 1
            # Disabling during a session must return to the target.
            assert bpy.ops.pawprint.paint_layer() == {'RUNNING_MODAL'}
            extension.unregister()
            assert bpy.context.object.name == target_name
            assert not any(obj.get('_pawprint_paint_proxy') for obj in bpy.data.objects)
            extension.register()
            # Save must persist the image and target, never the temporary canvas.
            with tempfile.TemporaryDirectory(prefix='pawprint-session-save-') as temp:
                path = str(Path(temp) / 'painted.blend')
                assert bpy.ops.pawprint.paint_layer() == {'RUNNING_MODAL'}
                bpy.data.images[image_name].pack()
                bpy.ops.wm.save_as_mainfile(filepath=path)
                assert not extension.painting.active()
                assert not any(obj.get('_pawprint_paint_proxy') for obj in bpy.data.objects)
                assert bpy.ops.pawprint.paint_layer() == {'RUNNING_MODAL'}
                bpy.ops.wm.open_mainfile(filepath=path, load_ui=False)
                assert not extension.painting.active()
                assert pixels() == erased
                assert not any(obj.get('_pawprint_paint_proxy') for obj in bpy.data.objects)
    print({'native_proxy_paint': True, 'stroke_centroid': centroid,
           'alpha_erase': True, 'interleaved_undo_redo': True,
           'brush': brush_name}, flush=True)
    if os.environ.get('PAWPRINT_PROBE_MANAGED'):
        return extension, target_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blender', default='blender')
    parser.add_argument('--screenshot', help='Optional path for a viewport review PNG')
    parser.add_argument('--managed', action='store_true', help='Exercise the implemented Paint Layer session')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='pawprint-locked-paint-') as temp:
        env = os.environ.copy()
        if args.managed:
            env['PAWPRINT_PROBE_MANAGED'] = '1'
        if args.screenshot:
            env['PAWPRINT_PROBE_SCREENSHOT'] = str(Path(args.screenshot).resolve())
        for suffix in ('RESOURCES', 'CONFIG', 'SCRIPTS', 'DATAFILES', 'EXTENSIONS'):
            env[f'BLENDER_USER_{suffix}'] = str(Path(temp) / suffix.lower())
        subprocess.run([args.blender, '--factory-startup', '--enable-event-simulate',
                        '--python-exit-code', '1',
                        '--python', str(Path(__file__).resolve()), '--', '--worker'],
                       env=env, check=True, timeout=180)


if __name__ == '__main__':
    if '--worker' in sys.argv:
        import bpy
        bpy.context.preferences.view.show_splash = False
        state = None

        def frame_position(region, extension, image_name):
            import bpy
            image_width, image_height = bpy.data.images[image_name].size[:]
            width, height = extension.projection.frame_size(region, image_width / image_height)
            x = (region.width - width) / 2 + width * 0.3
            y = (region.height - height) / 2 + height * 0.35
            return round(region.x + x), round(region.y + y)

        def navigation_check(data):
            """Free navigation: a misaligned view persists until a stroke."""
            import bpy
            from array import array
            from mathutils import Quaternion
            extension, area, region, target_name, _, image_name = data
            session = extension.painting._session
            space = area.spaces.active
            layer = extension.painting.session_stack().layers[1]
            assert extension.painting.active()
            assert not extension.painting.aligned(session, space), 'Watchdog still auto-restores'
            view = space.region_3d
            assert view.view_rotation.rotation_difference(Quaternion(layer.view_rotation)).angle > 0.001
            center = (round(region.x + region.width / 2), round(region.y + region.height / 2))
            before = array('f', bpy.data.images[image_name].pixels[:])
            # A misaligned press must be swallowed: no pixels change, the saved
            # view returns, and the session survives for the next stroke.
            window = bpy.context.window
            window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=center[0], y=center[1])
            window.event_simulate(type='LEFTMOUSE', value='PRESS', x=center[0], y=center[1])
            window.event_simulate(type='LEFTMOUSE', value='RELEASE', x=center[0], y=center[1])
            return before

        def header_check(data):
            """The native header (with the brush color chooser) must stay visible during a session."""
            import bpy
            from array import array
            from pathlib import Path
            import tempfile
            extension, area, region, _, _, _ = data
            with tempfile.TemporaryDirectory(prefix='pawprint-header-') as temp:
                first, second = Path(temp) / 'header-a.png', Path(temp) / 'header-b.png'
                with bpy.context.temp_override(area=area, region=region):
                    bpy.ops.screen.screenshot(filepath=str(first))
                    # Clearing an unset header text is a no-op; if the session
                    # had installed a banner these captures would differ.
                    extension.painting.paint_header(area, None)
                    bpy.ops.screen.screenshot(filepath=str(second))
                loaded = []
                try:
                    for path in (first, second):
                        loaded.append(bpy.data.images.load(str(path), check_existing=False))
                    assert loaded[0].size[:] == loaded[1].size[:], 'Header screenshots differ in size'
                    rows = [array('f', image.pixels[:]) for image in loaded]
                    width, height = loaded[0].size[:]
                    for y in range(min(height, 70)):
                        base = y * width * 4
                        for index in range(base, base + width * 4):
                            assert abs(rows[0][index] - rows[1][index]) <= 1 / 255, \
                                'Session hid the native header (color chooser disappeared)'
                finally:
                    for image in loaded:
                        bpy.data.images.remove(image)

        def gate_check(data, before):
            """Verify the gate, then drive one real aligned stroke."""
            import bpy
            from array import array
            extension, area, region, _, _, image_name = data
            session = extension.painting._session
            space = area.spaces.active
            assert extension.painting.active(), 'Gate ended the session'
            assert extension.painting.aligned(session, space), 'Gate did not restore the saved view'
            assert array('f', bpy.data.images[image_name].pixels[:]) == before, 'Misaligned press painted'
            window = bpy.context.window
            start = frame_position(region, extension, image_name)
            window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=start[0], y=start[1])
            window.event_simulate(type='LEFTMOUSE', value='PRESS', x=start[0], y=start[1])
            window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=start[0] + 2, y=start[1] + 2)
            window.event_simulate(type='LEFTMOUSE', value='RELEASE', x=start[0] + 2, y=start[1] + 2)

        def stroke_check(data, before):
            import bpy
            from array import array
            extension, area, _, _, _, image_name = data
            session = extension.painting._session
            space = area.spaces.active
            image = bpy.data.images[image_name]
            colored = array('f', image.pixels[:])
            changed = [i // 4 for i in range(0, len(before), 4) if before[i:i + 4] != colored[i:i + 4]]
            assert changed, 'Aligned event stroke changed no pixels'
            centroid = (sum(p % image.size[0] + 0.5 for p in changed) / len(changed),
                        sum(p // image.size[0] + 0.5 for p in changed) / len(changed))
            assert abs(centroid[0] - image.size[0] * 0.3) < 2, centroid
            assert abs(centroid[1] - image.size[1] * 0.35) < 2, centroid
            assert extension.painting.aligned(session, space)
            print({'free_navigation': True, 'stroke_gate': True,
                   'event_stroke_centroid': centroid}, flush=True)

        def run():
            global state
            try:
                if state is None:
                    data = setup()
                    managed = os.environ.get('PAWPRINT_PROBE_MANAGED')
                    state = dict(phase='header' if managed else 'strokes', data=data,
                                 managed=bool(managed))
                    return 30.0 if os.environ.get('PAWPRINT_PROBE_SCREENSHOT') else (0.5 if managed else 2.0)
                if state['phase'] == 'header':
                    header_check(state['data'])
                    state['phase'] = 'navigation'
                    return 0.4
                if state['phase'] == 'navigation':
                    state['before'] = navigation_check(state['data'])
                    state['phase'] = 'gate'
                    return 0.4
                if state['phase'] == 'gate':
                    gate_check(state['data'], state['before'])
                    state['phase'] = 'event_stroke'
                    return 0.4
                if state['phase'] == 'event_stroke':
                    stroke_check(state['data'], state['before'])
                    state['phase'] = 'strokes'
                    return 0.2
                if state['phase'] == 'strokes':
                    # The real event stroke already painted at (.3, .35); run the
                    # scripted EXEC phase at a different spot so it has fresh pixels.
                    result = strokes(state['data'], u=0.7, v=0.65) if state['managed'] else strokes(state['data'])
                    if result:
                        state = dict(phase='scene_enter', data=result)
                        return 1.0
                elif state['phase'] == 'scene_enter':
                    extension, target_name = state['data']
                    window = bpy.context.window_manager.windows[0]
                    area = next(a for a in window.screen.areas if a.type == 'VIEW_3D')
                    region = next(r for r in area.regions if r.type == 'WINDOW')
                    with bpy.context.temp_override(window=window, area=area, region=region):
                        assert bpy.ops.pawprint.paint_layer() == {'RUNNING_MODAL'}
                        original_scene = window.scene.name
                        other = bpy.data.scenes.new('Paint session scene-switch check')
                        empty = bpy.data.objects.new('Preserve new-scene selection', None)
                        other.collection.objects.link(empty)
                        window.scene = other
                        empty.select_set(True)
                        window.view_layer.objects.active = empty
                    state = dict(phase='scene_verify',
                                 data=(extension, original_scene, target_name, empty.name))
                    return 1.0
                elif state['phase'] == 'scene_verify':
                    extension, original_scene, target_name, empty_name = state['data']
                    assert not extension.painting.active(), 'Scene switch did not stop painting'
                    assert bpy.context.object.name == empty_name
                    assert bpy.context.object.select_get()
                    assert bpy.data.objects[target_name].mode == 'OBJECT'
                    assert not any(obj.get('_pawprint_paint_proxy') for obj in bpy.data.objects)
                    bpy.context.window.scene = bpy.data.scenes[original_scene]
                    assert bpy.context.object.name == target_name
                    print({'managed_lifecycle': True, 'scene_switch_cleanup': True}, flush=True)
                bpy.ops.wm.quit_blender()
            except Exception:
                import traceback
                traceback.print_exc()
                os._exit(1)
        bpy.app.timers.register(run, first_interval=1, persistent=True)
    else:
        main()
