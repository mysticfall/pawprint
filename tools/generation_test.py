# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise live Generate, or --capture-only without ComfyUI, in isolated Blender."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def worker():
    import bpy
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from result_apply_probe import setup
    state = {}
    deadline = time.monotonic() + 600
    log_path = Path(os.environ.get('PAWPRINT_TEST_LOG', '/tmp/opencode/gentest-run.log'))
    start = time.monotonic()

    def log(message):
        # GUI Blender buffers stdout; phase progress must survive a hang.
        with open(log_path, 'a') as handle:
            handle.write(f'{time.monotonic() - start:7.1f}s {message}\n')

    def tick():
        try:
            assert time.monotonic() < deadline, 'Generation test timeout'
            if state.get('phase') != state.get('logged_phase'):
                state['logged_phase'] = state.get('phase')
                log(f"phase -> {state.get('phase')}")
            if not state:
                extension, area, name = setup()
                extension.painting.finish()
                state.update(extension=extension, area=area, name=name, phase='connect')
                return 1
            ext, area, name = state['extension'], state['area'], state['name']
            region = next(r for r in area.regions if r.type == 'WINDOW')
            with bpy.context.temp_override(area=area, region=region):
                layer = ext.model.active_layer(ext.model.active_stack(bpy.context))
                def pixels():
                    return np.array(bpy.data.images[name].pixels[:])
                if state['phase'] == 'connect':
                    if '--capture-only' in sys.argv:
                        scene = bpy.context.scene
                        layer.generation_resolution = 256
                        ext.selection.add_path(layer, [(.15,.2),(.4,.2),(.4,.65),(.15,.65)], 'REPLACE')
                        layer.generation_padding = 4
                        for engine in ('BLENDER_EEVEE', 'BLENDER_WORKBENCH', 'CYCLES'):
                            scene.render.engine = engine
                            original = (scene.render.filepath, scene.render.resolution_x,
                                        scene.render.resolution_y, scene.camera)
                            scenes = set(bpy.data.scenes.keys())
                            with tempfile.TemporaryDirectory(prefix='pawprint-capture-test-') as temp:
                                directory = Path(temp)
                                metadata = ext.capture.prepare(bpy.context, layer, directory, 256)
                                assert all((directory / file).is_file() for file in
                                           ('composite.png', 'input.png', 'mask.png'))
                                composite = bpy.data.images.load(str(directory / 'composite.png'))
                                values = ext.capture.pixels(composite)
                                assert np.any((values[:, :, 0] > .95) &
                                              (values[:, :, 1] < .03)), 'Material color missing'
                                bpy.data.images.remove(composite)
                                assert max(metadata['request_size']) == 256
                                expected = bpy.data.images.load(str(directory / 'input.png'))
                                expected_pixels = ext.capture.pixels(expected).copy()
                                bpy.data.images.remove(expected)
                                assert bpy.ops.pawprint.preview_context() == {'FINISHED'}
                                assert area.type == 'IMAGE_EDITOR'
                                preview = area.spaces.active.image
                                assert preview.packed_file and tuple(preview.size) == tuple(metadata['request_size'])
                                assert np.array_equal(ext.capture.pixels(preview), expected_pixels), 'Preview differs from server input'
                                area.type = 'VIEW_3D'
                                bpy.data.images.remove(preview)
                            assert scene.render.engine == engine, 'Original engine changed'
                            assert (scene.render.filepath, scene.render.resolution_x,
                                    scene.render.resolution_y, scene.camera) == original
                            assert set(bpy.data.scenes.keys()) == scenes, 'Capture scene leaked'
                            print({'capture_engine': engine, 'passed': True}, flush=True)
                        bpy.ops.wm.quit_blender()
                        return None
                    assert bpy.ops.pawprint.connect() == {'FINISHED'}
                    state['phase'] = 'wait_connection'
                elif state['phase'] == 'wait_connection' and not ext.generation.active():
                    assert ext.generation.connected(bpy.context), ext.generation.status()
                    state['zit'] = '--zit' in sys.argv
                    state['batch'] = '--batch' in sys.argv
                    if '--ipadapter' in sys.argv:
                        assert ext.generation._capabilities['ipadapter_models'], 'IPAdapter model not discovered'
                    if state['zit']:
                        assert ext.generation._capabilities['zit_unets'], 'Z model not discovered'
                        assert ext.generation._capabilities['zit_controlnets'], 'ZIT ControlNet patch not discovered'
                    state['masked'] = any(flag in sys.argv for flag in ('--inpaint', '--depth', '--ipadapter', '--zit', '--batch'))
                    if '--depth' in sys.argv:
                        assert ext.generation._capabilities['controlnet_models'], 'Union ControlNet not discovered'
                    layer.generation_checkpoint = 'epicrealismXL_pureFix.safetensors'
                    layer.generation_positive = 'hand painted blue ceramic, fine glazed texture'
                    layer.generation_negative = 'text, watermark, blurry'
                    layer.generation_steps, layer.generation_cfg = 4, 3.5
                    layer.generation_clip_skip = 2
                    layer.generation_seed = '18446744073709551615'
                    layer.generation_denoise = .4
                    layer.generation_resolution = 256
                    layer.generation_depth_enabled = '--depth' in sys.argv
                    if state['zit']:
                        layer.generation_adapter = 'ZIT'
                        caps = ext.generation._capabilities
                        layer.generation_zit_unet = 'z_image_turbo_bf16.safetensors'
                        loras = caps['zit_loras']
                        entry = layer.generation_loras.add()
                        entry.model = next(
                            (name for name in loras if 'skin texture' in name.lower()), loras[0] if loras else '')
                        entry.strength = 1.0
                        layer.generation_steps, layer.generation_cfg = 8, 1.0
                        layer.generation_sampler, layer.generation_scheduler = 'res_multistep', 'simple'
                        layer.generation_denoise = 1.0
                        # Depth guidance flows through the DiffSynth ControlNet
                        # patch with the union weights the user confirmed.
                        layer.generation_depth_enabled = True
                    if '--ipadapter' in sys.argv:
                        if '--ipadapter' in sys.argv:
                            layer.generation_ipadapter_enabled = True
                            layer.generation_ipadapter_weight = 0.8
                        reference = bpy.data.images.new('IPAdapter reference', width=96, height=64)
                        reference_rgba = np.zeros((64,96,4), dtype=np.float32)
                        reference_rgba[:, :, 0] = np.linspace(.1, .9, 96)[None, :]
                        reference_rgba[:, :, 1] = np.linspace(.9, .2, 64)[:, None]
                        reference_rgba[:, :, 2] = .3
                        reference_rgba[:, :, 3] = 1
                        reference.pixels.foreach_set(reference_rgba.ravel())
                        reference.update()
                        layer.generation_ipadapter_image = reference
                    if '--inpaint' in sys.argv:
                        # Complete replacement: a selection routes the request
                        # through the inpaint workflow at full denoise.
                        layer.generation_denoise = 1.0
                    if state['batch']:
                        # Three random-seed candidates reviewed before apply.
                        layer.generation_batch = 3
                        layer.generation_denoise = 1.0
                    # Guidance runs keep a selection to exercise the inpainting
                    # path; the plain default run clears it so the whole frame
                    # updates as ordinary img2img.
                    if state['masked']:
                        ext.selection.add_path(layer, [(.15,.2),(.4,.2),(.4,.65),(.15,.65)], 'REPLACE')
                        ext.selection.add_path(layer, [(.8,.8),(.85,.8),(.85,.85),(.8,.85)], 'ADD')
                        layer.generation_feather, layer.generation_padding = 3, 4
                    else:
                        layer.selection_paths = '[]'
                    state['before'] = pixels()
                    state['positive'] = layer.generation_positive
                    state['mask'] = ext.selection.footprint(layer, generation=True).copy()
                    state['others'] = {l.image.name: np.array(l.image.pixels[:]) for l in
                                       ext.model.active_stack(bpy.context).layers if l.image.name != name}
                    scene = bpy.context.scene
                    scene.render.resolution_x, scene.render.resolution_y = 321, 234
                    scene.render.resolution_percentage = 73
                    scene.render.filepath = '/do/not/write/here'
                    scene.view_settings.view_transform = 'AgX'
                    scene.view_settings.exposure = 1.3
                    state['render'] = (scene.render.filepath, scene.render.resolution_x,
                                       scene.render.resolution_y, scene.render.resolution_percentage,
                                       scene.view_settings.view_transform, scene.view_settings.exposure, scene.camera)
                    state['visibility'] = {o.name: o.hide_render for o in scene.objects}
                    paint = scene.tool_settings.image_paint
                    paint.clone_offset, paint.clone_alpha = (.13,.27), .37
                    paint.use_symmetry_x, paint.dither = True, .3
                    state['paint'] = (tuple(paint.clone_offset), paint.clone_alpha, paint.use_symmetry_x, paint.dither)
                    state['brush'] = (paint.brush.image_brush_type, paint.brush.blend,
                                      paint.brush.curve_distance_falloff_preset, paint.brush.use_alpha)
                    # Context includes visible upper layers and viewport-visible
                    # objects even if they are disabled in final renders.
                    stack = ext.model.active_stack(bpy.context)
                    upper = stack.layers.add()
                    upper.name = 'Upper context'
                    for attr in ('projection', 'view_matrix', 'view_rotation', 'view_location',
                                 'view_distance', 'lens', 'clip_start', 'clip_end', 'width', 'height'):
                        setattr(upper, attr, getattr(layer, attr))
                    upper.depth = layer.depth
                    upper.image = bpy.data.images.new('Upper context pixels', width=192, height=128)
                    upper_rgba = np.zeros((128,192,4), dtype=np.float32)
                    upper_rgba[90:105,50:70] = (0,1,1,1)
                    upper.image.pixels.foreach_set(upper_rgba.ravel())
                    upper.image.update()
                    ext.projection.build_material(stack.id_data)
                    from mathutils import Vector
                    view = ext.projection.matrix(layer.view_matrix)
                    projection = ext.projection.matrix(layer.projection) @ view.inverted()
                    points = [view.inverted() @ Vector(((u*2-1)*2/projection[0][0],
                              (v*2-1)*2/projection[1][1], -2)) for u,v in
                              ((.73,.35),(.88,.35),(.88,.55),(.73,.55))]
                    mesh = bpy.data.meshes.new('Context object mesh')
                    mesh.from_pydata(points, [], [(0,1,2,3)])
                    surrounding = bpy.data.objects.new('Context object', mesh)
                    scene.collection.objects.link(surrounding)
                    surrounding.hide_render = True
                    material = bpy.data.materials.new('Context blue')
                    material.use_nodes = True
                    nodes = material.node_tree.nodes
                    nodes.clear()
                    emission, output = nodes.new('ShaderNodeEmission'), nodes.new('ShaderNodeOutputMaterial')
                    emission.inputs[0].default_value = (0,0,1,1)
                    material.node_tree.links.new(emission.outputs[0], output.inputs['Surface'])
                    mesh.materials.append(material)
                    bpy.context.view_layer.update()
                    state['others'][upper.image.name] = np.array(upper.image.pixels[:])
                    state['visibility'] = {o.name:o.hide_render for o in scene.objects}
                    bpy.ops.ed.undo_push(message='Before live generation')
                    assert bpy.ops.pawprint.generate() == {'FINISHED'}
                    if '--ipadapter' in sys.argv:
                        assert (ext.generation._job['directory'] / 'reference.png').is_file(), 'Reference not exported'
                    composite = bpy.data.images.load(str(ext.generation._job['directory'] / 'composite.png'))
                    values = ext.capture.pixels(composite)
                    assert np.max(np.abs(values[97,60,:3] - (0,1,1))) < .03, 'Upper layer absent'
                    assert np.max(np.abs(values[int(128*.45),int(192*.8),:3] - (0,0,1))) < .03, 'Viewport-only object absent'
                    assert np.any((values[:,:,0] > .95) & (values[:,:,1] < .03)), 'Other material slot absent'
                    bpy.data.images.remove(composite)
                    if layer.generation_depth_enabled:
                        # Depth guidance is adapter-independent: the freshly
                        # captured depth.png feeds whichever graph consumes it.
                        depth = bpy.data.images.load(str(ext.generation._job['directory'] / 'depth.png'))
                        depth.colorspace_settings.name = 'Non-Color'
                        metadata = ext.generation._job['metadata']
                        assert tuple(depth.size) == metadata['request_size']
                        values = ext.capture.pixels(depth)
                        assert values[:, :, 0].max() > .99
                        assert values[:, :, 0].min() < .01
                        assert np.any((values[:, :, 0] > .05) & (values[:, :, 0] < .95)), 'Lost surface depth gradient'
                        # The viewport-visible, render-hidden context quad is nearer
                        # than the target and must be bright in the aligned crop.
                        x0, y0, x1, y1 = metadata['bounds']
                        x = int((192*.8-x0)/(x1-x0)*depth.size[0])
                        y = int((128*.45-y0)/(y1-y0)*depth.size[1])
                        assert values[y,x,0] > .99, 'Depth lost viewport-visible context'
                        bpy.data.images.remove(depth)
                    assert (scene.render.filepath, scene.render.resolution_x, scene.render.resolution_y,
                            scene.render.resolution_percentage, scene.view_settings.view_transform,
                            scene.view_settings.exposure, scene.camera) == state['render'], 'Render settings changed'
                    assert {o.name: o.hide_render for o in scene.objects} == state['visibility']
                    assert not any(s.name.startswith('Pawprint temporary capture') for s in bpy.data.scenes)
                    state['phase'] = 'batch_wait1' if state.get('batch') else 'graph'
                elif state['phase'] == 'batch_wait1' and not ext.generation.active():
                    # The batch completes into review mode instead of an applied
                    # result: candidates are held in memory behind a preview
                    # datablock swapped into the layer's texture node.
                    assert ext.generation.status().startswith('Reviewing candidates'), ext.generation.status()
                    review = ext.generation._review
                    info = ext.generation.review_info()
                    assert review and info['count'] == 3 and info['index'] == 0
                    # Random-seed candidates must genuinely differ somewhere.
                    assert any(not np.array_equal(c, review['candidates'][0])
                               for c in review['candidates'][1:]), 'Batch candidates are identical'
                    preview = review['preview']
                    preview_name = preview.name
                    stale_pointer = preview.as_pointer()
                    def display_node():
                        active = ext.generation._review
                        return next(n for n in ext.model.active_stack(bpy.context).id_data.node_tree.nodes
                                    if n.type == 'TEX_IMAGE' and n.name.startswith('Pawprint Layer')
                                    and (n.image == layer.image or (active and n.image == active['preview'])))
                    assert not ext.painting.active(), 'Review must keep the viewport interactive'
                    assert not any(o.get('_pawprint_paint_proxy') for o in bpy.data.objects)
                    assert tuple(preview.size) == tuple(layer.image.size)
                    assert np.allclose(ext.capture.pixels(preview), review['candidates'][0] / 255)
                    assert display_node().image.as_pointer() == preview.as_pointer()
                    assert bpy.ops.pawprint.candidate_select(direction=1) == {'FINISHED'}
                    assert ext.generation.review_info()['index'] == 1
                    # Switching swaps in a fresh datablock: the stale preview is
                    # gone, the display node binds the new one and it carries
                    # candidate 2's pixels (a shared, mutated-in-place datablock
                    # left the viewport texture stale in interactive use).
                    assert stale_pointer not in {image.as_pointer() for image in bpy.data.images}
                    preview = review['preview']
                    assert np.allclose(ext.capture.pixels(preview), review['candidates'][1] / 255)
                    assert display_node().image.as_pointer() == preview.as_pointer()
                    assert bpy.ops.pawprint.candidate_select(direction=-1) == {'FINISHED'}
                    assert ext.generation.review_info()['index'] == 0
                    # Commit the second candidate: review closes first, the
                    # result applies through the native clone (one undo step)
                    # and the layer adopts that candidate's seed.
                    chosen = review['seeds'][1]
                    assert bpy.ops.pawprint.candidate_select(direction=1) == {'FINISHED'}
                    assert bpy.ops.pawprint.commit_candidate() == {'FINISHED'}
                    assert ext.generation._review is None
                    assert preview_name not in bpy.data.images
                    assert layer.generation_seed == chosen
                    assert layer.generation_last_seed == chosen
                    after = pixels()
                    mask = state['mask'].reshape(-1)
                    assert not np.array_equal(after, state['before']), 'Candidate left the image unchanged'
                    assert np.array_equal(after.reshape(-1,4)[mask == 0], state['before'].reshape(-1,4)[mask == 0])
                    bpy.ops.ed.undo()
                    assert np.array_equal(pixels(), state['before']), 'Candidate commit undo'
                    bpy.ops.ed.redo()
                    assert np.array_equal(pixels(), after), 'Candidate commit redo'
                    # A second batch ends in discard: review closes, the preview
                    # disappears, the node shows the layer image again and the
                    # seed keeps the committed value.
                    # Start the second batch on a later timer tick: invoking
                    # operators in the same callback as undo/redo is fragile.
                    state['phase'] = 'batch_second'
                elif state['phase'] == 'batch_second':
                    layer.generation_batch = 2
                    bpy.ops.ed.undo_push(message='Before second batch')
                    assert bpy.ops.pawprint.generate() == {'FINISHED'}
                    state['phase'] = 'batch_wait2'
                elif state['phase'] == 'batch_wait2' and not ext.generation.active():
                    assert ext.generation.status().startswith('Reviewing candidates'), ext.generation.status()
                    assert ext.generation.review_info()['count'] == 2
                    review = ext.generation._review
                    preview_name = review['preview'].name
                    committed = layer.generation_seed
                    other = review['candidates'][1]
                    # Discard drops only the shown candidate: one stays under
                    # review, showing the survivor's pixels.
                    assert bpy.ops.pawprint.discard_candidates() == {'FINISHED'}
                    assert ext.generation._review is not None
                    info = ext.generation.review_info()
                    assert info['count'] == 1 and info['index'] == 0
                    assert np.allclose(ext.capture.pixels(ext.generation._review['preview']), other / 255)
                    # Discarding the last candidate closes the review.
                    assert bpy.ops.pawprint.discard_candidates() == {'FINISHED'}
                    assert ext.generation._review is None
                    assert preview_name not in bpy.data.images
                    assert layer.generation_seed == committed
                    stack = ext.model.active_stack(bpy.context)
                    assert not any(node.type == 'TEX_IMAGE' and node.image
                                   and node.image.name == 'Pawprint Candidate Preview'
                                   for node in stack.id_data.node_tree.nodes)
                    assert any(node.type == 'TEX_IMAGE' and node.image == layer.image
                               for node in stack.id_data.node_tree.nodes)
                    state['phase'] = 'batch_third'
                elif state['phase'] == 'batch_third':
                    # Third session exercises the remaining review option: keep
                    # the shown candidate as a new editable layer.
                    layer.generation_batch = 2
                    bpy.ops.ed.undo_push(message='Before third batch')
                    assert bpy.ops.pawprint.generate() == {'FINISHED'}
                    state['phase'] = 'batch_wait3'
                elif state['phase'] == 'batch_wait3' and not ext.generation.active():
                    log('batch_wait3 entered')
                    assert ext.generation.status().startswith('Reviewing candidates'), ext.generation.status()
                    assert ext.generation.review_info()['count'] == 2
                    review = ext.generation._review
                    assert bpy.ops.pawprint.candidate_select(direction=1) == {'FINISHED'}
                    chosen_seed, candidate = review['seeds'][1], review['candidates'][1]
                    stack = ext.model.active_stack(bpy.context)
                    state['source_pixels'] = pixels()
                    assert bpy.ops.pawprint.candidate_layer() == {'FINISHED'}
                    log('candidate_layer applied')
                    assert ext.generation._review is None
                    assert 'Pawprint Candidate Preview' not in bpy.data.images
                    # The operator's collection add/move invalidates Python
                    # property wrappers, so everything is re-fetched.
                    stack = ext.model.active_stack(bpy.context)
                    layer = stack.layers[1]
                    log(f'post-op layers={len(stack.layers)} active={stack.active_index} '
                        f'names={[item.name for item in stack.layers]}')
                    # A new layer sits directly above the source with the
                    # candidate pixels, the shared saved view/depth snapshot,
                    # inherited settings and the candidate's seed. The batch
                    # fixture stack also carries an upper-context layer.
                    assert len(stack.layers) == 4 and stack.active_index == 2, \
                        f'{len(stack.layers)} layers, active {stack.active_index}'
                    new_layer = stack.layers[2]
                    assert new_layer.name.startswith('Candidate')
                    assert tuple(new_layer.image.size) == tuple(layer.image.size)
                    assert new_layer.image.packed_file is not None
                    assert np.allclose(ext.capture.pixels(new_layer.image), candidate / 255)
                    assert new_layer.generation_seed == chosen_seed
                    assert new_layer.generation_last_seed == chosen_seed
                    assert new_layer.depth == layer.depth
                    assert tuple(new_layer.view_matrix) == tuple(layer.view_matrix)
                    assert tuple(new_layer.projection) == tuple(layer.projection)
                    assert new_layer.generation_positive == layer.generation_positive
                    assert new_layer.selection_paths == '[]'
                    assert new_layer.visible
                    assert np.array_equal(pixels(), state['source_pixels']), 'Add-as-layer touched the source image'
                    nodes = stack.id_data.node_tree.nodes
                    assert sum(1 for node in nodes if node.type == 'TEX_IMAGE'
                               and node.name.startswith('Pawprint Layer')) == 4
                    assert any(node.type == 'TEX_IMAGE' and node.image == new_layer.image
                               for node in nodes)
                    # One native undo step removes the layer again, on a later
                    # tick: undo in the same callback as layer/collection
                    # surgery has deadlocked the GUI before.
                    state['phase'] = 'batch_undo'
                elif state['phase'] == 'batch_undo':
                    bpy.ops.ed.undo()
                    stack = ext.model.active_stack(bpy.context)
                    assert len(stack.layers) == 3 and stack.active_index == 1, \
                        f'{len(stack.layers)} layers, active {stack.active_index}'
                    assert not any(node.type == 'TEX_IMAGE' and node.image
                                   and node.image.name.startswith('Pawprint Candidate')
                                   for node in stack.id_data.node_tree.nodes)
                    assert np.array_equal(pixels(), state['source_pixels']), 'Add-as-layer undo'
                    state['phase'] = 'cleanup'
                elif state['phase'] == 'graph':
                    # Capture the prompt id from the live job so the workflow the
                    # server actually received can be inspected after completion.
                    if ext.generation.active():
                        queued = ext.generation._job['directory'] / 'queued.json'
                        if queued.is_file():
                            import json
                            state['prompt_id'] = json.loads(queued.read_text())['prompt_id']
                            state['phase'] = 'waiting'
                    else:
                        raise AssertionError('Job finished before its submitted workflow could be inspected')
                elif state['phase'] == 'waiting' and not ext.generation.active():
                    assert ext.generation.status().startswith('Applied generation'), ext.generation.status()
                    if state.get('prompt_id'):
                        # Verify the submitted graph straight from server history:
                        # inpaint mode must carry the specialised encode and the
                        # preprocessor-routed repaint ControlNet, nothing quieter.
                        import json
                        import urllib.request
                        server = bpy.context.scene.pawprint_server
                        base = server if server.startswith(('http://', 'https://')) else 'http://' + server
                        url = base.rstrip('/') + '/history/' + state['prompt_id']
                        with urllib.request.urlopen(url, timeout=30) as response:
                            entry = json.loads(response.read()).get(state['prompt_id'])
                        assert entry, 'Server history lost the completed job'
                        graph = entry['prompt'][2]
                        classes = {key: value['class_type'] for key, value in graph.items()}
                        # The submitted graph follows the selection and the
                        # adapter: SDXL selections run InpaintModelConditioning,
                        # ZIT selections run the reference inpaint pipeline and
                        # unmasked requests plain whole-frame img2img.
                        if state.get('zit'):
                            # Z Image Turbo reference inpainting: MAT pre-fill,
                            # Fun ControlNet in inpaint mode, DifferentialDiffusion
                            # and the advanced sampling stack with colour match.
                            assert classes['30'] == 'UNETLoader', classes.get('30')
                            assert classes['34'] == 'CLIPLoader'
                            assert graph['34']['inputs']['type'] == 'lumina2'
                            assert classes['37'] == 'CLIPSetLastLayer'
                            assert graph['37']['inputs']['clip'] == ['34', 0]
                            assert graph['37']['inputs']['stop_at_clip_layer'] == -layer.generation_clip_skip
                            assert classes['38'] == 'CLIPTextEncode'
                            assert graph['38']['inputs']['clip'] == ['37', 0]
                            assert graph['38']['inputs']['text'] == layer.generation_positive
                            feather = layer.generation_feather
                            assert classes['26'] == 'INPAINT_ExpandMask'
                            assert graph['26']['inputs'] == {'mask': ['4', 0], 'grow': feather,
                                                             'blur': int(feather * 1.7), 'blur_type': 'linear'}
                            assert classes['27'] == 'INPAINT_StabilizeMask'
                            assert graph['27']['inputs'] == {'mask': ['26', 0], 'epsilon': 0.01}
                            assert classes['28'] == 'ThresholdMask'
                            assert graph['28']['inputs'] == {'mask': ['27', 0], 'value': 0.0}
                            assert classes['33'] == 'INPAINT_ExpandMask'
                            assert graph['33']['inputs'] == {'mask': ['4', 0], 'grow': 4, 'blur': 0,
                                                             'blur_type': 'gaussian'}
                            assert classes['42'] == 'INPAINT_LoadInpaintModel'
                            assert classes['43'] == 'INPAINT_InpaintWithModel'
                            assert graph['43']['inputs']['image'] == ['36', 0]
                            assert graph['43']['inputs']['mask'] == ['33', 0]
                            assert classes['44'] == 'ModelPatchLoader'
                            assert classes['45'] == 'ZImageFunControlnet'
                            assert graph['45']['inputs']['model_patch'] == ['44', 0]
                            assert graph['45']['inputs']['vae'] == ['35', 0]
                            assert graph['45']['inputs']['inpaint_image'] == ['36', 0]
                            assert graph['45']['inputs']['mask'] == ['28', 0]
                            assert graph['45']['inputs']['strength'] == layer.generation_zit_strength
                            assert classes['25'] == 'DifferentialDiffusion'
                            assert classes['46'] == 'VAEEncode'
                            assert graph['46']['inputs']['pixels'] == ['43', 0]
                            assert classes['47'] == 'SetLatentNoiseMask'
                            assert graph['47']['inputs']['mask'] == ['27', 0]
                            assert classes['48'] == 'RandomNoise'
                            assert classes['49'] == 'KSamplerSelect'
                            assert classes['52'] == 'BasicScheduler'
                            assert graph['52']['inputs']['denoise'] == 1.0
                            assert classes['54'] == 'BasicGuider'
                            assert graph['54']['inputs']['conditioning'] == ['38', 0]
                            assert classes['55'] == 'SamplerCustomAdvanced'
                            assert graph['55']['inputs']['latent_image'] == ['47', 0]
                            assert graph['55']['inputs']['sigmas'] == ['52', 0]
                            assert classes['41'] == 'VAEDecode'
                            assert graph['41']['inputs']['samples'] == ['55', 1]
                            assert classes['56'] == 'INPAINT_ColorMatch'
                            assert graph['56']['inputs']['reference'] == ['43', 0]
                            assert graph['56']['inputs']['exclude_mask'] == ['27', 0]
                            assert classes['10'] == 'PreviewImage'
                            assert graph['10']['inputs']['images'] == ['56', 0]
                            if len(layer.generation_loras):
                                assert classes['50'] == 'LoraLoaderModelOnly'
                                assert graph['50']['inputs']['lora_name'] == layer.generation_loras[0].model
                                assert graph['45']['inputs']['model'] == ['50', 0]
                            if layer.generation_depth_enabled:
                                assert graph['45']['inputs']['image'] == ['12', 0]
                            assert 'ControlNetLoader' not in classes.values()
                            assert 'ControlNetApplyAdvanced' not in classes.values()
                            assert 'VAEEncodeForInpaint' not in classes.values()
                            assert 'InpaintPreprocessor' not in classes.values()
                            assert 'ConditioningZeroOut' not in classes.values()
                            assert 'TextEncodeQwenImageEditPlus' not in classes.values()
                        elif state.get('masked'):
                            # Model conditioning builds the inpaint latent and
                            # the masked composite preserves the unselected
                            # pixels server-side; no inpaint VAE or
                            # preprocessor nodes exist anywhere.
                            assert classes['25'] == 'InpaintModelConditioning', classes.get('25')
                            assert classes['28'] == 'ImageCompositeMasked', classes.get('28')
                            assert graph['28']['inputs']['destination'] == ['2', 0]
                            assert graph['28']['inputs']['source'] == ['11', 0]
                            assert graph['10']['inputs']['images'] == ['28', 0]
                            assert graph['9']['inputs']['latent_image'] == ['25', 2]
                            assert graph['9']['inputs']['positive'] == ['25', 0]
                            assert 'VAEEncodeForInpaint' not in classes.values()
                            assert 'InpaintPreprocessor' not in classes.values()
                            if '--depth' in sys.argv:
                                assert classes['14'] == 'ControlNetApplyAdvanced'
                                assert graph['25']['inputs']['positive'] == ['14', 0]
                        else:
                            assert classes['5'] == 'VAEEncode' and classes['6'] == 'SetLatentNoiseMask'
                            assert 'VAEEncodeForInpaint' not in classes.values()
                            assert 'InpaintPreprocessor' not in classes.values()
                            assert 'InpaintModelConditioning' not in classes.values()
                    after = pixels()
                    assert not np.array_equal(after, state['before'])
                    mask = state['mask'].reshape(-1)
                    if '--inpaint' in sys.argv or state.get('zit'):
                        full = mask > .99
                        delta = np.abs(after.reshape(-1, 4)[full] - state['before'].reshape(-1, 4)[full])
                        assert delta.mean() > .05, 'Selection content was not replaced'
                    assert np.array_equal(after.reshape(-1,4)[mask == 0], state['before'].reshape(-1,4)[mask == 0])
                    paint = bpy.context.scene.tool_settings.image_paint
                    assert (tuple(paint.clone_offset), paint.clone_alpha, paint.use_symmetry_x, paint.dither) == state['paint']
                    assert (paint.brush.image_brush_type, paint.brush.blend, paint.brush.curve_distance_falloff_preset,
                            paint.brush.use_alpha) == state['brush']
                    assert not ext.painting.active()
                    assert not any(o.get('_pawprint_paint_proxy') for o in bpy.data.objects)
                    # Generate recorded both prompts; the history dedupes and
                    # menu entries apply remembered text back to the layer.
                    scene = bpy.context.scene
                    history, negative = scene.pawprint_positive_history, scene.pawprint_negative_history
                    assert history[-1].text == state['positive'], 'Positive prompt not recorded'
                    assert negative[-1].text == layer.generation_negative, 'Negative prompt not recorded'
                    ext.generation.record_prompt(history, state['positive'])
                    assert len(history) == 1, 'Duplicate history entries'
                    ext.generation.record_prompt(history, 'older prompt')
                    assert [item.text for item in history] == [state['positive'], 'older prompt']
                    assert bpy.ops.pawprint.apply_prompt(prompt='older prompt') == {'FINISHED'}
                    assert layer.generation_positive == 'older prompt'
                    assert bpy.ops.pawprint.apply_prompt(prompt=state['positive']) == {'FINISHED'}
                    assert layer.generation_positive == state['positive']
                    bpy.ops.pawprint.clear_prompt_history(negative=True)
                    assert not len(negative)
                    bpy.ops.ed.undo()
                    assert np.array_equal(pixels(), state['before']), 'Single-step generation undo'
                    bpy.ops.ed.redo()
                    assert np.array_equal(pixels(), after), 'Single-step generation redo'
                    state['after'] = after
                    for other, before in state['others'].items():
                        assert np.array_equal(np.array(bpy.data.images[other].pixels[:]), before)
                    # Cancellation and target switching are exercised through the real lifecycle.
                    assert bpy.ops.pawprint.connect() == {'FINISHED'}
                    assert bpy.ops.pawprint.cancel_generation() == {'FINISHED'}
                    assert not ext.generation.active()
                    state['phase'] = 'cleanup'
                elif state['phase'] == 'cleanup' and not ext.generation._retired:
                    # Use a completed worker result to exercise stale-pixel rejection
                    # deterministically without queuing additional GPU requests.
                    directory = Path(tempfile.mkdtemp(prefix='pawprint-stale-test-'))
                    ext.backend.publish(directory, 'done.json', {})
                    process = subprocess.Popen([sys.executable, '--version'], stdout=subprocess.DEVNULL)
                    process.wait()
                    stack = ext.model.active_stack(bpy.context)
                    original_digest = ext.generation.digest(layer.image)
                    ext.generation._job = dict(operation='generate', directory=directory, process=process,
                        window=bpy.context.window, area=area, scene=bpy.context.scene.as_pointer(),
                        view_layer=bpy.context.view_layer.name, owner=bpy.context.object.as_pointer(),
                        material=stack.id_data.as_pointer(), slot=bpy.context.object.active_material_index,
                        fingerprint=ext.generation.fingerprint(layer), digest=original_digest)
                    value = layer.image.pixels[0]
                    layer.image.pixels[0] = 1 - value
                    edited = pixels()
                    ext.generation.tick()
                    assert 'edited while generating' in ext.generation.status(), ext.generation.status()
                    assert not ext.generation.active() and not directory.exists()
                    assert np.array_equal(pixels(), edited), 'Stale result overwrote newer pixels'
                    # Layer switch cancels before examining any result.
                    directory = Path(tempfile.mkdtemp(prefix='pawprint-target-test-'))
                    ext.generation._job = dict(operation='generate', directory=directory, process=process,
                        window=bpy.context.window, area=area, scene=bpy.context.scene.as_pointer(),
                        view_layer=bpy.context.view_layer.name, owner=bpy.context.object.as_pointer(),
                        material=stack.id_data.as_pointer(), slot=bpy.context.object.active_material_index,
                        fingerprint=ext.generation.fingerprint(layer), digest=ext.generation.digest(layer.image))
                    stack.active_index = 0
                    ext.generation.tick()
                    assert not ext.generation.active() and 'target layer' in ext.generation.status()
                    ext.generation.tick()
                    assert not directory.exists()
                    if '--depth' in sys.argv:
                        assert bpy.ops.pawprint.preview_depth() == {'FINISHED'}
                        assert area.type == 'IMAGE_EDITOR'
                        preview = area.spaces.active.image
                        assert preview.packed_file and preview.colorspace_settings.name == 'Non-Color'
                        area.type = 'VIEW_3D'
                    print({'live_generate_operator': True, 'native_single_step_undo_redo': True,
                           'render_brush_clone_settings_restored': True, 'cancel_cleanup': True,
                           'stale_pixels_and_target_change_rejected': True}, flush=True)
                    bpy.ops.wm.quit_blender()
                    return None
            return .25
        except Exception:
            import traceback
            with open(log_path, 'a') as handle:
                handle.write(f'{time.monotonic() - start:7.1f}s FAILED\n')
                traceback.print_exc(file=handle)
            traceback.print_exc()
            os._exit(1)
    bpy.context.preferences.view.show_splash = False
    bpy.app.timers.register(tick, first_interval=2)


if __name__ == '__main__':
    if '--worker' in sys.argv:
        worker()
    else:
        with tempfile.TemporaryDirectory(prefix='pawprint-generation-test-') as temp:
            env = os.environ.copy()
            env['PAWPRINT_PROBE_MANAGED'] = '1'
            for suffix in ('RESOURCES', 'CONFIG', 'SCRIPTS', 'DATAFILES', 'EXTENSIONS'):
                env[f'BLENDER_USER_{suffix}'] = str(Path(temp) / suffix.lower())
            subprocess.run(['blender', '--factory-startup', '--python-exit-code', '1', '--python',
                            str(Path(__file__).resolve()), '--', '--worker',
                            *[flag for flag in ('--capture-only', '--depth', '--ipadapter', '--inpaint', '--zit', '--batch') if flag in sys.argv]],
                           env=env, check=True, timeout=600)
