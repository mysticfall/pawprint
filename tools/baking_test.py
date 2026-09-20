# SPDX-License-Identifier: GPL-3.0-or-later
"""Isolated UV commit rendering, one-step undo/redo and persistence checks."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def worker(directory):
    import bpy
    import numpy as np
    from mathutils import Vector
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from projection_test import load_extension, setup_fixture, create_projection, render_pixels
    ext = load_extension()
    scene, obj, original = setup_fixture(bpy)
    # Disjoint target UV islands. The unrelated slot overlaps deliberately: it
    # must never be baked into this slot's base.
    uv = obj.data.uv_layers.active
    for face in obj.data.polygons:
        left, right = (.1, .45) if face.index == 0 else (.55, .9)
        if face.index == 1:
            left, right = .1, .45
        for loop, co in zip(face.loop_indices, [(left,.1),(right,.1),(right,.9),(left,.9)]):
            uv.data[loop].uv = co
    first = create_projection(bpy, ext, obj, scene.camera)
    stack = obj.active_material.pawprint
    # Commit bakes read the scene's configured base resolution; pin it so the
    # same-size assertions below are independent of the 2048 default.
    scene.pawprint_base_width = scene.pawprint_base_height = 512
    material_name, object_name = obj.active_material.name, obj.name
    base_name = stack.base.name
    # Smooth source makes bake accuracy measurable independently of aliasing.
    values = np.zeros((128,128,4), dtype=np.float32)
    values[:,:,0] = np.linspace(.1,.8,128)[None,:]
    values[:,:,1] = np.linspace(.1,.7,128)[:,None]
    values[:,:,2], values[:,:,3] = .2, .65
    first.image.pixels.foreach_set(values.ravel())
    first.image.update()
    first.image.pack()
    first.name = 'Front partial alpha'
    fields = ('projection','view_matrix','view_rotation','view_location','view_distance',
              'lens','clip_start','clip_end','width','height')
    def add(name, color, angled=False, visible=True):
        layer = stack.layers.add()
        layer.name = name
        for key in fields:
            setattr(layer, key, getattr(first, key))
        if angled:
            scene.camera.location = (1,0,5)
            scene.camera.rotation_euler = (Vector((-1,0,0)) - scene.camera.location).to_track_quat('-Z','Y').to_euler()
            bpy.context.view_layer.update()
            view = scene.camera.matrix_world.inverted()
            layer.view_matrix = ext.projection.flattened(view)
            layer.projection = ext.projection.flattened(scene.camera.calc_matrix_camera(
                bpy.context.evaluated_depsgraph_get(), x=128,y=128) @ view)
        layer.depth = ext.projection.depth_image(bpy.context, layer)
        layer.image = ext.projection.new_image(name,128,128,transparent=True)
        layer.image.pixels.foreach_set(np.tile(color,128*128).astype(np.float32))
        layer.image.update()
        layer.image.pack()
        layer.visible = visible
        return layer
    add('Angled green', (.1,.8,.2,.4), angled=True)
    add('Hidden magenta', (1,0,1,1), visible=False)
    upper = add('Remaining upper', (.2,.3,.9,.2))
    upper.generation_positive = 'retain settings'
    upper.selection_paths = '[["REPLACE",[[0,0],[1,0],[1,1]]]]'
    upper_selection = upper.selection_paths
    upper_pixels = ext.capture.pixels(upper.image).copy()
    stack['active_index'] = 2
    scene.camera.location = (0,0,5)
    scene.camera.rotation_euler = (0,0,0)
    bpy.context.view_layer.update()
    ext.projection.build_material(obj.active_material)
    def render(name):
        return np.array(render_pixels(bpy, directory / name)).reshape((128,128,4))
    before = render('before.exr')
    old_base = ext.capture.pixels(stack.base).copy()
    state = (scene.render.engine, scene.camera.name, obj.active_material_index,
             tuple(o.name for o in bpy.context.selected_objects))
    counts = {key: len(getattr(bpy.data,key)) for key in ('scenes','objects','meshes','materials')}
    bpy.context.preferences.edit.use_global_undo = True
    bpy.ops.ed.undo_push(message='Before UV commit')
    assert bpy.ops.pawprint.commit_base('EXEC_DEFAULT', True) == {'FINISHED'}
    assert len(stack.layers) == 1 and stack.layers[0].name == 'Remaining upper'
    assert stack.layers[0].generation_positive == 'retain settings'
    assert stack.layers[0].selection_paths == upper_selection
    assert np.array_equal(ext.capture.pixels(stack.layers[0].image), upper_pixels)
    assert np.array_equal(ext.capture.pixels(bpy.data.images[base_name]),old_base)
    assert stack.base.name != base_name and stack.base.packed_file
    baked_name = stack.base.name
    baked = ext.capture.pixels(stack.base).copy()
    assert np.array_equal(baked[0,0], old_base[0,0]), 'Unused base texel changed'
    # Island padding extends 8px outside the UV boundary.
    assert not np.array_equal(baked[256,48],old_base[256,48]), 'No bake edge padding'
    assert all(len(getattr(bpy.data,key)) == value for key,value in counts.items()), 'Temporary datablock leak'
    assert (scene.render.engine,scene.camera.name,obj.active_material_index,
            tuple(o.name for o in bpy.context.selected_objects)) == state
    assert obj.material_slots[1].material == original
    after = render('after.exr')
    error = np.abs(before[20:108,10:60,:3]-after[20:108,10:60,:3])
    assert np.quantile(error,.99) < .025, ('Bake mismatch',np.max(error),np.quantile(error,.99))
    assert np.max(np.abs(before[:,70:110,:3]-after[:,70:110,:3])) < .002, 'Other slot changed'
    assert bpy.ops.ed.undo() == {'FINISHED'}
    stack = bpy.data.materials[material_name].pawprint
    assert len(stack.layers) == 4 and stack.active_index == 2 and stack.base.name == base_name
    assert np.array_equal(ext.capture.pixels(stack.base),old_base)
    assert bpy.ops.ed.redo() == {'FINISHED'}
    stack = bpy.data.materials[material_name].pawprint
    assert len(stack.layers) == 1 and stack.base.name == baked_name
    assert np.array_equal(ext.capture.pixels(stack.base),baked), 'Redo lost baked pixels'
    if '--interactive' in sys.argv:
        # Interleave a real native image edit on the remaining projection with
        # the commit's memfile undo. Native mode changes may add their own steps.
        assert bpy.ops.pawprint.paint_layer() == {'RUNNING_MODAL'}
        paint = bpy.context.scene.tool_settings.image_paint
        paint.unified_paint_settings.use_unified_size = True
        paint.unified_paint_settings.size = 18
        paint.unified_paint_settings.use_unified_strength = True
        paint.unified_paint_settings.strength = 1
        bpy.ops.pawprint.paint_blend(erase=True)
        region = bpy.context.region
        width,height = ext.projection.frame_size(region,1)
        x,y = (region.width-width)/2+width*.7, (region.height-height)/2+height*.3
        stroke = [dict(name='bake interleave',mouse=(x,y),mouse_event=(x,y),pressure=1,
                       size=18,time=0,is_start=True,x_tilt=0,y_tilt=0,location=(0,0,0))]
        assert bpy.ops.paint.image_paint('EXEC_DEFAULT',True,stroke=stroke) == {'FINISHED'}
        image_name = ext.painting.session_stack().layers[0].image.name
        painted = ext.capture.pixels(bpy.data.images[image_name]).copy()
        assert not np.array_equal(painted,upper_pixels), 'Native interleaved stroke did nothing'
        ext.painting.finish()
        undo_count = 0
        for _ in range(6):
            bpy.ops.ed.undo()
            undo_count += 1
            current = bpy.data.materials[material_name].pawprint
            if len(current.layers) == 4:
                break
        assert len(current.layers) == 4 and current.base.name == base_name
        assert np.array_equal(ext.capture.pixels(bpy.data.images[image_name]),upper_pixels)
        for _ in range(undo_count):
            bpy.ops.ed.redo()
        stack = bpy.data.materials[material_name].pawprint
        assert len(stack.layers) == 1 and stack.base.name == baked_name
        assert np.array_equal(ext.capture.pixels(stack.base),baked)
        assert np.array_equal(ext.capture.pixels(bpy.data.images[image_name]),painted)
        assert not ext.painting.active()
        print({'native_paint_commit_interleaving':True},flush=True)
    # Configurable resolution: a commit resamples the base to the configured
    # size, and preservation keeps the committed sources hidden instead of
    # removing them, so resolutions can be retried without losing sources.
    bpy.context.scene.pawprint_base_width, bpy.context.scene.pawprint_base_height = 768, 1024
    stack = bpy.data.materials[material_name].pawprint
    stack.commit_preserve = True
    before_resampled = render('before-resampled.exr')
    bpy.ops.ed.undo_push(message='Before resampled commit')
    assert bpy.ops.pawprint.commit_base('EXEC_DEFAULT', True) == {'FINISHED'}
    stack = bpy.data.materials[material_name].pawprint
    assert len(stack.layers) == 1 and not stack.layers[0].visible, 'Preserved layer not hidden'
    assert tuple(stack.base.size) == (768, 1024) and stack.base.name != baked_name
    resampled_name = stack.base.name
    preserved_pixels = ext.capture.pixels(stack.base).copy()
    assert np.array_equal(ext.capture.pixels(bpy.data.images[baked_name]), baked), 'Prior base mutated'
    after_resampled = render('after-resampled.exr')
    resampled_error = np.abs(before_resampled[20:108,10:60,:3]-after_resampled[20:108,10:60,:3])
    assert np.quantile(resampled_error,.99) < .025, ('Resample mismatch', np.max(resampled_error))
    assert bpy.ops.ed.undo() == {'FINISHED'}
    stack = bpy.data.materials[material_name].pawprint
    assert len(stack.layers) == 1 and stack.layers[0].visible and stack.base.name == baked_name
    assert tuple(stack.base.size) == (512, 512)
    assert bpy.ops.ed.redo() == {'FINISHED'}
    stack = bpy.data.materials[material_name].pawprint
    assert tuple(stack.base.size) == (768, 1024)
    assert np.array_equal(ext.capture.pixels(stack.base), preserved_pixels), 'Redo lost resampled pixels'
    stack.commit_preserve = False
    # Missing UV fails without changing base/layers or leaking bake resources.
    old_uv = stack.uv_name
    stack.uv_name = 'missing'
    try:
        ext.baking.bake_base(bpy.context,stack,0)
        raise AssertionError('Missing UV accepted')
    except ValueError:
        pass
    stack.uv_name = old_uv
    assert len(stack.layers) == 1 and stack.base.name == resampled_name
    # Commit final remaining layer: base-only stack remains usable.
    assert bpy.ops.pawprint.commit_base('EXEC_DEFAULT', True) == {'FINISHED'}
    assert not stack.layers and stack.preview
    # Emptying the stack snapshots the committed layer's settings for new layers.
    assert 'retain settings' in stack.last_settings
    final_pixels = ext.capture.pixels(stack.base).copy()
    path = directory / 'committed.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False)
    stack = bpy.data.objects[object_name].active_material.pawprint
    assert not stack.layers and stack.base.packed_file
    assert 'retain settings' in stack.last_settings
    assert np.array_equal(ext.capture.pixels(stack.base),final_pixels)
    print({'uv_commit':True,'different_views_alpha_visibility':True,'single_step_undo_redo':True,
           'upper_layers_and_other_slot_preserved':True,'padding_and_persistence':True},flush=True)


def main():
    with tempfile.TemporaryDirectory(prefix='pawprint-bake-test-') as temp:
        directory = Path(temp)
        env = os.environ.copy()
        for suffix in ('RESOURCES','CONFIG','SCRIPTS','DATAFILES','EXTENSIONS'):
            env['BLENDER_USER_' + suffix] = str(directory / suffix.lower())
        interactive = '--interactive' in sys.argv
        subprocess.run(['blender',*([] if interactive else ['--background']),
                        '--factory-startup','--python-exit-code','1',
                        '--python',str(Path(__file__).resolve()),'--','--worker',
                        *(['--interactive'] if interactive else []),str(directory)],
                       env=env,check=True,timeout=180)


if __name__ == '__main__':
    if '--worker' in sys.argv:
        if '--interactive' in sys.argv:
            import bpy
            def run():
                try:
                    area = next(area for area in bpy.context.screen.areas if area.type == 'VIEW_3D')
                    region = next(region for region in area.regions if region.type == 'WINDOW')
                    with bpy.context.temp_override(area=area,region=region):
                        worker(Path(sys.argv[-1]))
                    bpy.ops.wm.quit_blender()
                except Exception:
                    import traceback
                    traceback.print_exc()
                    os._exit(1)
            bpy.app.timers.register(run,first_interval=1)
        else:
            worker(Path(sys.argv[-1]))
    else:
        main()
