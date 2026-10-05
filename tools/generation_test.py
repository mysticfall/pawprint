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
                def fixture_review(count=1, alpha=1, matching=False, normal_matching=False):
                    # Exercise completion via tick, with deterministic PNGs and
                    # an already-exited worker: no server or GPU job required.
                    stack = ext.model.active_stack(bpy.context)
                    width, height = layer.image.size
                    directory = Path(tempfile.mkdtemp(prefix='pawprint-review-test-'))
                    seeds = [str(100 + index) for index in range(count)]
                    settings = {'seed': seeds[0]}
                    if count > 1:
                        settings['seeds'] = seeds
                    metadata = dict(bounds=(0, 0, width, height), request_size=(width, height),
                                    weights=np.ones((height, width), dtype=np.float32))
                    if matching:
                        metadata['weights'][:] = 0
                        metadata['weights'][40:88, 40:88] = 1
                        reference = np.ones((height, width, 4), dtype=np.float32)
                        reference[:, :, :3] = (.25, .45, .65)
                        metadata['albedo_reference'] = reference
                    if normal_matching:
                        reference = np.ones((height, width, 4), dtype=np.float32)
                        reference[..., :3] = (0, .6, .8)
                        metadata['mesh_normals'] = reference
                    for index in range(count):
                        rgba = np.ones((height, width, 4), dtype=np.float32)
                        rgba[:, :, :3] = (.3 + index * .1, .5, .7)
                        ext.capture.save_input(directory / (f'result-{index}.png' if count > 1 else 'result.png'),
                                               rgba, (width, height))
                        normal = np.ones_like(rgba)
                        normal[:, :, :3] = (.5 + index * .1, .8, .9)
                        ext.capture.save_input(directory / (f'normal-{index}.png' if count > 1 else 'normal.png'),
                                               normal, (width, height), non_color=True)
                        rgba[:, :, :3] *= .5
                        rgba[:, :, 3] = alpha
                        original = bpy.data.images.new('Fixture generated RGBA', width=width, height=height, alpha=True)
                        original.pixels.foreach_set(rgba.ravel())
                        original.filepath_raw = str(directory / (f'original-{index}.png' if count > 1 else 'original.png'))
                        original.file_format = 'PNG'
                        original.save()
                        bpy.data.images.remove(original)
                    process = subprocess.Popen([sys.executable, '--version'], stdout=subprocess.DEVNULL)
                    process.wait()
                    ext.backend.publish(directory, 'done.json', {})
                    ext.generation._job = dict(operation='generate', directory=directory, process=process,
                        window=bpy.context.window, area=area, scene=bpy.context.scene.as_pointer(),
                        view_layer=bpy.context.view_layer.name, owner=bpy.context.object.as_pointer(),
                        material=stack.id_data.as_pointer(), slot=bpy.context.object.active_material_index,
                        fingerprint=ext.generation.fingerprint(layer), digest=ext.generation.digest(layer.image),
                        settings=settings, metadata=metadata)
                    ext.generation.tick()
                    assert not directory.exists() and ext.generation.review_active(), ext.generation.status()
                    return ext.generation._review
                if state['phase'] == 'connect':
                    if '--review-only' in sys.argv:
                        before_repair = ext.capture.pixels(layer.image).copy()
                        orphan = bpy.data.images.new('Pawprint Orphan Preview Test', width=128, height=128)
                        node = ext.generation._display_node(layer)
                        node.image = orphan
                        # Simulate an older graph, whose node cannot be located
                        # by image after a preview was left bound without review.
                        if 'pawprint_albedo_layer' in node:
                            del node['pawprint_albedo_layer']
                        alpha_review = fixture_review(alpha=.4)
                        assert np.array_equal(alpha_review['originals'][0][:, :, 3],
                                              alpha_review['candidates'][0][:, :, 3])
                        assert np.all(alpha_review['candidates'][0][:, :, 3] == 102)
                        ext.generation.end_review('Alpha preservation verified')
                        assert ext.generation._display_node(layer).image == layer.image
                        assert np.array_equal(ext.capture.pixels(layer.image), before_repair)
                        bpy.data.images.remove(orphan)
                        node = ext.generation._display_node(layer)
                        layer.id_data.node_tree.nodes.remove(node)
                        repaired = ext.generation._ensure_layer_display(layer)
                        assert repaired.image == layer.image and repaired.get('pawprint_albedo_layer') == layer.path_from_id()
                        assert np.array_equal(ext.capture.pixels(layer.image), before_repair)
                        matched_review = fixture_review(count=2, matching=True)
                        for index in range(2):
                            np.testing.assert_allclose(matched_review['candidates'][index][64, 64, :3] / 255,
                                                       (.25, .45, .65), atol=2 / 255)
                            np.testing.assert_allclose(matched_review['normals'][index][64, 64, :3],
                                                       (.5 + index * .1, .8, .9), atol=1 / 255)
                            np.testing.assert_allclose(matched_review['originals'][index][64, 64, :3] / 255,
                                                       ((.3 + index * .1) * .5, .25, .35), atol=1 / 255)
                        assert np.array_equal(ext.capture.pixels(layer.image), before_repair)
                        ext.generation.end_review('Boundary matching verified')
                        normal_review = fixture_review(normal_matching=True)
                        np.testing.assert_allclose(normal_review['normals'][0][64, 64, :3],
                                                   (.5, .5, 1), atol=1 / 255)
                        assert normal_review['normals'][0][64, 64, 3] == 1
                        ext.generation.end_review('Mesh normal matching verified')
                        layer.selection_paths = '[]'
                        state['before'] = pixels()
                        bpy.ops.ed.undo_push(message='Before single review')
                        review = fixture_review()
                        assert np.array_equal(pixels(), state['before']), 'Single result applied before review'
                        assert review['version'] == 'ALBEDO' and len(review['originals']) == 1
                        state['expected'] = review['candidates'][0].copy()
                        assert not np.array_equal(review['originals'][0], state['expected'])
                        assert bpy.ops.pawprint.candidate_version(version='ORIGINAL') == {'FINISHED'}
                        assert bpy.ops.pawprint.candidate_editor(editor='IMAGE_EDITOR') == {'FINISHED'}
                        state['phase'] = 'review_image'
                        return .5
                    if '--capture-only' in sys.argv:
                        scene = bpy.context.scene
                        subdivision = bpy.context.object.modifiers.new('Capture viewport subdivision', 'SUBSURF')
                        subdivision.levels, subdivision.render_levels = 0, 2
                        subdivision.show_render = False
                        render_states = []
                        def check_subdivision(render_scene):
                            render_states.append((subdivision.show_render, subdivision.render_levels))
                        bpy.app.handlers.render_pre.append(check_subdivision)
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
                                assert metadata['albedo_reference'].shape == (*tuple(layer.image.size)[::-1], 4)
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
                            assert render_states[-1] == (True, 0), 'Capture used render-only subdivision'
                            assert subdivision.render_levels == 2 and not subdivision.show_render, 'Subdivision settings leaked'
                            assert (scene.render.filepath, scene.render.resolution_x,
                                    scene.render.resolution_y, scene.camera) == original
                            assert set(bpy.data.scenes.keys()) == scenes, 'Capture scene leaked'
                            print({'capture_engine': engine, 'passed': True}, flush=True)
                        bpy.app.handlers.render_pre.remove(check_subdivision)
                        matrix = ext.projection.matrix
                        def fail_capture(_):
                            raise RuntimeError('Injected capture setup failure')
                        ext.projection.matrix = fail_capture
                        try:
                            with tempfile.TemporaryDirectory(prefix='pawprint-capture-failure-') as temp:
                                try:
                                    ext.capture.render_frame(bpy.context, layer, Path(temp) / 'failed.png')
                                    raise AssertionError('Capture should have failed')
                                except RuntimeError as exc:
                                    assert 'Injected' in str(exc)
                        finally:
                            ext.projection.matrix = matrix
                        assert subdivision.render_levels == 2 and not subdivision.show_render
                        assert set(bpy.data.scenes.keys()) == scenes, 'Failed capture leaked scene'
                        bpy.ops.wm.quit_blender()
                        return None
                    assert bpy.ops.pawprint.connect() == {'FINISHED'}
                    state['phase'] = 'wait_connection'
                elif state['phase'] == 'review_image':
                    review = ext.generation._review
                    assert review and area.type == 'IMAGE_EDITOR' and review['version'] == 'ORIGINAL'
                    assert area.spaces.active.image == review['preview']
                    assert not review['preview'].use_view_as_render
                    bpy.ops.image.view_zoom_ratio(ratio=2)
                    bpy.ops.image.view_pan(offset=(37, -19))
                    state['phase'] = 'review_toggle'
                elif state['phase'] == 'review_toggle':
                    # Read mapping after redraw so it reflects both zoom and pan.
                    zoom = tuple(area.spaces.active.zoom)
                    state['review_zoom'] = zoom
                    state['review_mapping'] = tuple(region.view2d.region_to_view(0, 0))
                    assert bpy.ops.pawprint.candidate_version(version='ALBEDO') == {'FINISHED'}
                    assert tuple(area.spaces.active.zoom) == zoom, 'Version toggle reset zoom'
                    state['phase'] = 'review_toggle_original'
                elif state['phase'] == 'review_toggle_original':
                    assert tuple(area.spaces.active.zoom) == state['review_zoom'], 'Version toggle reset zoom after redraw'
                    assert np.allclose(region.view2d.region_to_view(0, 0), state['review_mapping']), 'Version toggle reset pan'
                    assert np.allclose(ext.capture.pixels(ext.generation._review['preview']), state['expected'] / 255)
                    assert bpy.ops.pawprint.candidate_version(version='ORIGINAL') == {'FINISHED'}
                    state['phase'] = 'review_return_view'
                elif state['phase'] == 'review_return_view':
                    assert tuple(area.spaces.active.zoom) == state['review_zoom'], 'Original toggle reset zoom'
                    assert np.allclose(region.view2d.region_to_view(0, 0), state['review_mapping']), 'Original toggle reset pan'
                    review = ext.generation._review
                    assert np.allclose(ext.capture.pixels(review['preview']), review['originals'][0] / 255)
                    assert bpy.ops.pawprint.candidate_editor(editor='VIEW_3D') == {'FINISHED'}
                    assert ext.generation._review['version'] == 'ORIGINAL'
                    assert bpy.ops.pawprint.candidate_editor(editor='IMAGE_EDITOR') == {'FINISHED'}
                    state['phase'] = 'review_apply'
                elif state['phase'] == 'review_apply':
                    review = ext.generation._review
                    assert bpy.ops.pawprint.commit_candidate('EXEC_DEFAULT', True) == {'FINISHED'}
                    assert not ext.generation.review_active() and not review['originals']
                    assert area.type == 'IMAGE_EDITOR' and area.spaces.active.image == bpy.data.images[name]
                    after = pixels()
                    assert not np.array_equal(after, state['before'])
                    assert np.max(np.abs(after.reshape(state['expected'].shape) - state['expected'] / 255)) <= 2/255
                    state['after'] = after
                    assert layer.normal and layer.normal.packed_file
                    state['normal_after'] = ext.capture.pixels(layer.normal).copy()
                    assert not ext.painting.active()
                    assert not any(o.get('_pawprint_paint_proxy') for o in bpy.data.objects)
                    state['phase'] = 'review_undo'
                elif state['phase'] == 'review_undo':
                    bpy.ops.ed.undo()
                    assert np.array_equal(pixels(), state['before']), 'Image Editor Apply undo'
                    assert ext.model.active_layer(ext.model.active_stack(bpy.context)).normal is None, 'Normal Apply undo'
                    bpy.ops.ed.redo()
                    assert np.array_equal(pixels(), state['after']), 'Image Editor Apply redo'
                    assert np.array_equal(ext.capture.pixels(ext.model.active_layer(ext.model.active_stack(bpy.context)).normal),
                                          state['normal_after']), 'Normal Apply redo'
                    state['phase'] = 'review_batch'
                elif state['phase'] == 'review_batch':
                    area.type = 'VIEW_3D'
                    stack = ext.model.active_stack(bpy.context)
                    state['source_name'] = layer.image.name
                    state['layer_source_pixels'] = pixels()
                    state['layer_count'] = len(stack.layers)
                    bpy.ops.ed.undo_push(message='Before Image Editor candidate layer')
                    review = fixture_review(2)
                    weights = review['weights']
                    weights[:] = 0
                    weights[32:96, 32:96] = 1
                    weights[64:96, 64:96] = 0  # L selection inside a larger context rectangle.
                    weights[31, 32:96] = .5
                    state['masked_weights'] = weights.copy()
                    ext.generation._show_candidate(review)
                    outside = weights == 0
                    assert np.allclose(ext.capture.pixels(review['preview'])[outside],
                                       ext.capture.pixels(layer.image)[outside], atol=1/255), 'Review changed padded context'
                    assert np.array_equal(ext.capture.pixels(review['normal_preview'])[outside],
                                          ext.capture.pixels(layer.normal)[outside]), 'Normal review changed padded context'
                    assert bpy.ops.pawprint.candidate_version(version='ORIGINAL') == {'FINISHED'}
                    assert bpy.ops.pawprint.candidate_select(direction=1) == {'FINISHED'}
                    assert review['version'] == 'ORIGINAL'
                    expected = review['candidates'][1].copy()
                    expected[:, :, 3] = np.round(expected[:, :, 3] * weights).astype(np.uint8)
                    assert bpy.ops.pawprint.candidate_editor(editor='IMAGE_EDITOR') == {'FINISHED'}
                    # Python operator calls default to no automatic global-undo
                    # push; request it as a real UI button invocation would.
                    assert bpy.ops.pawprint.candidate_layer('EXEC_DEFAULT', True) == {'FINISHED'}
                    layer = ext.model.active_layer(ext.model.active_stack(bpy.context))
                    assert np.allclose(ext.capture.pixels(layer.image), expected / 255)
                    assert np.all(ext.capture.pixels(layer.normal)[:, :, 3][outside] == 0), 'Normal Layer kept context padding'
                    assert area.spaces.active.image == layer.image
                    assert np.array_equal(pixels(), state['layer_source_pixels'])
                    assert not review['originals'] and not review['candidates']
                    state['layer_expected'] = expected
                    state['phase'] = 'review_layer_undo'
                elif state['phase'] == 'review_layer_undo':
                    bpy.ops.ed.undo()
                    stack = ext.model.active_stack(bpy.context)
                    assert len(stack.layers) == state['layer_count'], 'Image Editor Layer undo'
                    assert ext.model.active_layer(stack).image.name == state['source_name']
                    assert np.array_equal(pixels(), state['layer_source_pixels'])
                    state['phase'] = 'review_layer_redo'
                elif state['phase'] == 'review_layer_redo':
                    assert bpy.ops.ed.redo() == {'FINISHED'}
                    stack = ext.model.active_stack(bpy.context)
                    assert len(stack.layers) == state['layer_count'] + 1, 'Image Editor Layer redo'
                    layer = ext.model.active_layer(stack)
                    assert np.allclose(ext.capture.pixels(layer.image), state['layer_expected'] / 255)
                    assert np.array_equal(pixels(), state['layer_source_pixels'])
                    assert layer.normal and layer.normal.colorspace_settings.name == 'Non-Color'
                    state['phase'] = 'review_pair_apply'
                elif state['phase'] == 'review_pair_apply':
                    area.type = 'VIEW_3D'
                    state['pair_before'] = ext.capture.pixels(layer.image).copy()
                    state['pair_normal_before'] = ext.capture.pixels(layer.normal).copy()
                    bpy.ops.ed.undo_push(message='Before replacing existing normal')
                    review = fixture_review()
                    review['weights'][:, :64] = 0
                    assert bpy.ops.pawprint.commit_candidate('EXEC_DEFAULT', True) == {'FINISHED'}
                    state['pair_after'] = ext.capture.pixels(layer.image).copy()
                    state['pair_normal_after'] = ext.capture.pixels(layer.normal).copy()
                    assert np.array_equal(state['pair_after'][:, :64], state['pair_before'][:, :64]), 'Apply changed unselected albedo'
                    assert np.array_equal(state['pair_normal_after'][:, :64], state['pair_normal_before'][:, :64])
                    assert not np.array_equal(state['pair_normal_after'], state['pair_normal_before'])
                    state['phase'] = 'review_pair_undo'
                elif state['phase'] == 'review_pair_undo':
                    bpy.ops.ed.undo()
                    layer = ext.model.active_layer(ext.model.active_stack(bpy.context))
                    assert np.array_equal(ext.capture.pixels(layer.image), state['pair_before'])
                    assert np.array_equal(ext.capture.pixels(layer.normal), state['pair_normal_before'])
                    bpy.ops.ed.redo()
                    layer = ext.model.active_layer(ext.model.active_stack(bpy.context))
                    assert np.array_equal(ext.capture.pixels(layer.image), state['pair_after'])
                    assert np.array_equal(ext.capture.pixels(layer.normal), state['pair_normal_after'])
                    state['phase'] = 'review_guards'
                elif state['phase'] == 'review_guards':
                    area.type = 'VIEW_3D'
                    review = fixture_review(2)
                    assert bpy.ops.pawprint.discard_candidates() == {'FINISHED'}
                    assert len(review['originals']) == len(review['candidates']) == 1
                    assert bpy.ops.pawprint.discard_candidates() == {'FINISHED'}
                    assert not review['originals'] and not review['candidates']
                    review = fixture_review()
                    layer.image.pixels[0] = 1 - layer.image.pixels[0]
                    ext.generation._maintain_review()
                    assert not ext.generation.review_active() and not review['originals']
                    review = fixture_review()
                    assert bpy.ops.pawprint.candidate_editor(editor='IMAGE_EDITOR') == {'FINISHED'}
                    area.spaces.active.mode = 'PAINT'
                    ext.generation._maintain_review()
                    assert not ext.generation.review_active() and not review['originals']
                    area.type = 'VIEW_3D'
                    review = fixture_review()
                    ext.generation.before_data_change()
                    assert not ext.generation.review_active() and not review['originals']
                    review = fixture_review()
                    review['preview'].pixels[0] = 1 - review['preview'].pixels[0]
                    ext.generation._maintain_review()
                    assert not ext.generation.review_active() and not review['originals']
                    review = fixture_review()
                    assert bpy.ops.pawprint.candidate_editor(editor='IMAGE_EDITOR') == {'FINISHED'}
                    apply = ext.result.apply
                    def fail_apply(*args):
                        raise ValueError('Injected candidate apply failure')
                    ext.result.apply = fail_apply
                    try:
                        try:
                            assert bpy.ops.pawprint.commit_candidate() == {'CANCELLED'}
                        except RuntimeError as exc:
                            assert 'Injected candidate apply failure' in str(exc)
                    finally:
                        ext.result.apply = apply
                    assert area.type == 'IMAGE_EDITOR' and area.spaces.active.image == layer.image
                    assert not ext.generation.review_active() and not review['originals']
                    assert not ext.painting.active()
                    area.type = 'VIEW_3D'
                    review = fixture_review()
                    ext.model.active_stack(bpy.context).active_index = 0
                    ext.generation._maintain_review()
                    assert not ext.generation.review_active() and not review['originals']
                    assert not any(i.name.startswith('Pawprint Candidate Preview') for i in bpy.data.images)
                    assert not any(i.name.startswith('Pawprint Normal Preview') for i in bpy.data.images)
                    print({'single_batch_review': True, 'shared_version_editor_switch': True,
                             'albedo_only_apply_layer': True, 'image_editor_apply_native_undo': True,
                            'image_editor_layer_native_undo_redo': True,
                           'review_cleanup_guards': True}, flush=True)
                    bpy.ops.wm.quit_blender()
                    return None
                elif state['phase'] == 'wait_connection' and not ext.generation.active():
                    assert ext.generation.connected(bpy.context), ext.generation.status()
                    # Missing Chord must fail before capture or any submission.
                    chord = ext.generation._capabilities['chord']
                    before = pixels()
                    ext.generation._capabilities['chord'] = None
                    try:
                        try:
                            assert bpy.ops.pawprint.generate() == {'CANCELLED'}
                        except RuntimeError as exc:
                            assert 'Chord' in str(exc)
                        assert not ext.generation.active() and np.array_equal(pixels(), before)
                    finally:
                        ext.generation._capabilities['chord'] = chord
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
                    layer.generation_resolution = 1024 if '--high-resolution' in sys.argv else 256
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
                    cyan = values[97,60,:3]
                    assert cyan[0] < .03 and min(cyan[1:]) > .8 and abs(cyan[1]-cyan[2]) < .03, 'Upper layer absent'
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
                    assert len(ext.generation._review['candidates']) == 3
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
                    assert ext.generation.status().startswith('Reviewing candidates'), ext.generation.status()
                    assert ext.generation._review['version'] == 'ALBEDO'
                    assert np.array_equal(pixels(), state['before']), 'Single generation bypassed review'
                    assert bpy.ops.pawprint.commit_candidate() == {'FINISHED'}
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
                        # adapter: SDXL selections run the reference inpaint
                        # pipeline, ZIT selections its own reference pipeline
                        # and unmasked requests plain whole-frame img2img.
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
                            assert '26' not in classes
                            assert classes['27'] == 'INPAINT_StabilizeMask'
                            assert graph['27']['inputs'] == {'mask': ['4', 0], 'epsilon': 0.01}
                            assert classes['28'] == 'ThresholdMask'
                            assert graph['28']['inputs'] == {'mask': ['4', 0], 'value': 0.5}
                            assert '33' not in classes
                            assert classes['42'] == 'INPAINT_LoadInpaintModel'
                            assert classes['43'] == 'INPAINT_InpaintWithModel'
                            assert graph['43']['inputs']['image'] == ['36', 0]
                            assert graph['43']['inputs']['mask'] == ['28', 0]
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
                            # The reference inpaint pipeline: the Fooocus patch
                            # joins the model chain, inpaint conditioning
                            # builds the latent and the advanced sampling stack
                            # ends in a colour match. Full denoise pre-fills
                            # through a MAT inpaint model; refinement encodes
                            # the original pixels with a split sigma schedule.
                            full = layer.generation_denoise >= 1.0
                            assert '26' not in classes
                            assert classes['25'] == 'SelfAttentionGuidance'
                            assert classes['29'] == 'DifferentialDiffusion'
                            assert classes['21'] == 'INPAINT_VAEEncodeInpaintConditioning'
                            assert classes['22'] == 'INPAINT_LoadFooocusInpaint'
                            assert classes['31'] == 'INPAINT_ApplyFooocusInpaint'
                            assert graph['31']['inputs']['latent'] == ['21', 2]
                            assert classes['44'] == 'RandomNoise'
                            assert classes['45'] == 'KSamplerSelect'
                            assert classes['46'] == 'CFGGuider'
                            assert graph['46']['inputs']['positive'] == ['21', 0]
                            assert classes['47'] == 'BasicScheduler'
                            assert graph['47']['inputs']['denoise'] == 1.0
                            assert classes['49'] == 'SamplerCustomAdvanced'
                            assert graph['49']['inputs']['latent_image'] == ['21', 3]
                            assert classes['41'] == 'VAEDecode'
                            assert graph['41']['inputs']['samples'] == ['49', 1]
                            assert classes['56'] == 'INPAINT_ColorMatch'
                            assert classes['10'] == 'PreviewImage'
                            assert graph['10']['inputs']['images'] == ['56', 0]
                            if full:
                                assert classes['27'] == 'INPAINT_StabilizeMask'
                                assert classes['42'] == 'INPAINT_LoadInpaintModel'
                                assert classes['43'] == 'INPAINT_InpaintWithModel'
                                assert graph['21']['inputs']['pixels'] == ['43', 0]
                                assert graph['21']['inputs']['mask'] == ['27', 0]
                                assert graph['49']['inputs']['sigmas'] == ['47', 0]
                                assert graph['56']['inputs']['reference'] == ['43', 0]
                                assert graph['56']['inputs']['exclude_mask'] == ['27', 0]
                            else:
                                assert '27' not in classes and '43' not in classes
                                assert graph['21']['inputs']['pixels'] == ['2', 0]
                                assert graph['21']['inputs']['mask'] == ['4', 0]
                                assert classes['48'] == 'SplitSigmas'
                                assert graph['48']['inputs']['step'] == round(
                                    layer.generation_steps * (1 - layer.generation_denoise))
                                assert graph['49']['inputs']['sigmas'] == ['48', 1]
                                assert graph['56']['inputs']['reference'] == ['2', 0]
                                assert graph['56']['inputs']['exclude_mask'] == ['4', 0]
                            assert 'KSampler' not in classes.values()
                            assert 'ImageCompositeMasked' not in classes.values()
                            assert 'InpaintModelConditioning' not in classes.values()
                            assert 'VAEEncode' not in classes.values()
                            assert 'VAEEncodeForInpaint' not in classes.values()
                            assert 'InpaintPreprocessor' not in classes.values()
                            if '--depth' in sys.argv:
                                assert classes['14'] == 'ControlNetApplyAdvanced'
                                assert graph['21']['inputs']['positive'] == ['14', 0]
                            if '--ipadapter' in sys.argv:
                                assert classes['18'] == 'IPAdapterAdvanced'
                                assert graph['25']['inputs']['model'] == ['18', 0]
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
                        actual = np.array(bpy.data.images[other].pixels[:])
                        assert np.array_equal(actual, before), (other, float(np.max(np.abs(actual - before))))
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
                            *[flag for flag in ('--capture-only', '--review-only', '--depth', '--ipadapter', '--inpaint', '--zit', '--batch', '--high-resolution') if flag in sys.argv]],
                           env=env, check=True, timeout=600)
