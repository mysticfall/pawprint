# SPDX-License-Identifier: GPL-3.0-or-later
"""Main-thread generation lifecycle. Networking lives in an isolated process."""
import hashlib
import json
from pathlib import Path
import secrets
import shutil
import subprocess
import tempfile

import bpy
import numpy as np

from . import backend, capture, model, painting, result

_job = None
_retired = []
_capabilities = None
_server = ''
_review = None
_status = 'Connect to ComfyUI to discover models'


def active():
    return _job is not None


def review_active():
    return _review is not None


def review_info():
    """Display data for the sidebar while candidates are under review."""
    if _review is None:
        return None
    seed = _review['seeds'][_review['index']]
    return dict(index=_review['index'], count=len(_review['candidates']), seed=seed)


def status():
    return _status


def connected(context):
    return _capabilities is not None and _server == context.scene.pawprint_server.rstrip('/')


def redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()


HISTORY_LIMIT = 32


def record_prompt(history, text, limit=HISTORY_LIMIT):
    """Append text as the most recent entry, dropping duplicates and overflow."""
    text = (text or '').strip()
    if not text:
        return
    for index, item in enumerate(history):
        if item.text == text:
            history.remove(index)
            break
    history.add().text = text
    while len(history) > limit:
        history.remove(0)


class PAWPRINT_PG_prompt(bpy.types.PropertyGroup):
    text: bpy.props.StringProperty()


def fingerprint(layer):
    return (layer.as_pointer(), layer.image.as_pointer(), tuple(layer.image.size),
            tuple(layer.projection), tuple(layer.view_matrix), layer.selection_paths,
            layer.generation_feather, layer.generation_padding, layer.generation_context)


def digest(image):
    return hashlib.sha256(capture.pixels(image).tobytes()).digest()


def _display_node(layer):
    """The layer's texture node inside its own material, matched by image."""
    tree = layer.id_data.node_tree
    if not tree:
        return None
    for node in tree.nodes:
        if node.type == 'TEX_IMAGE' and node.name.startswith('Pawprint Layer') and node.image == layer.image:
            return node
    return None


def _show_candidate(review):
    """Display one candidate.

    Every switch builds a fresh datablock: rewriting the pixels of an image the
    viewport shader already bound does not reliably refresh its GPU texture,
    so the datablock swap that displayed the first candidate is repeated per
    candidate instead of mutating one shared preview.
    """
    material, layer_image = review['material'], review['layer_image']
    previous = review.get('preview')
    candidate = review['candidates'][review['index']]
    values = candidate.astype(np.float32) / 255
    preview = bpy.data.images.new('Pawprint Candidate Preview',
                                  width=candidate.shape[1], height=candidate.shape[0], alpha=True)
    preview.pixels.foreach_set(values.ravel())
    preview.update()
    try:
        tree = material.node_tree
        targets = [node for node in tree.nodes if node.type == 'TEX_IMAGE'
                   and (node.image == layer_image or (previous is not None and node.image == previous))]
    except ReferenceError:
        targets = []
    for node in targets:
        node.image = preview
    review['preview'] = preview
    if previous is not None and previous.name in bpy.data.images:
        bpy.data.images.remove(previous)
        preview.name = 'Pawprint Candidate Preview'


def end_review(message):
    """Restore the layer display and drop the preview datablock."""
    global _review, _status
    review, _review = _review, None
    if not review:
        return
    preview, image = review.get('preview'), review['layer_image']
    try:
        tree = review['material'].node_tree
        if tree and preview is not None:
            for node in tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image == preview:
                    node.image = image
    except ReferenceError:
        pass
    if preview is not None and preview.name in bpy.data.images:
        bpy.data.images.remove(preview)
    _status = message
    redraw()


def _begin_review(context, job, seeds):
    """Swap the layer display to a preview datablock driven by the candidates."""
    global _review, _status
    stack = model.active_stack(context)
    layer = model.active_layer(stack)
    frame = layer.image.size
    candidates = []
    for index in range(len(seeds)):
        rgba = capture.returned_pixels(job['directory'] / f'result-{index}.png', frame, job['metadata'])
        candidates.append(np.asarray(rgba * 255).round().clip(0, 255).astype(np.uint8))
    node = _display_node(layer)
    if node is None:
        raise ValueError('Layer texture node not found; rebuild the material before reviewing')
    _review = dict(window=job['window'], area=job['area'], scene=job['scene'], view_layer=job['view_layer'],
                   owner=job['owner'], material=stack.id_data, material_ptr=job['material'], slot=job['slot'],
                   index_in_stack=stack.active_index, layer_image=layer.image, image_ptr=layer.image.as_pointer(),
                   fingerprint=job['fingerprint'], digest=job['digest'], weights=job['metadata']['weights'],
                   candidates=candidates, seeds=list(seeds), index=0)
    # _show_candidate creates the preview datablock and binds the layer node.
    _show_candidate(_review)
    _status = f"Reviewing candidates: 1/{len(seeds)} · seed {seeds[0]}"


def _maintain_review():
    """Keep review honest: verify the target, self-heal the display node."""
    review = _review
    try:
        window, area = review['window'], review['area']
        if (window not in bpy.context.window_manager.windows[:] or area not in window.screen.areas[:]
                or area.type != 'VIEW_3D'):
            end_review('Candidate review closed: viewport changed')
            return
        region = next(r for r in area.regions if r.type == 'WINDOW')
        with bpy.context.temp_override(window=window, area=area, region=region):
            context = bpy.context
            if (context.scene.as_pointer() != review['scene'] or context.view_layer.name != review['view_layer']
                    or painting.active() or active()
                    or context.active_object is None or context.active_object.as_pointer() != review['owner']
                    or context.object.mode != 'OBJECT' or context.object.active_material_index != review['slot']):
                end_review('Candidate review closed: target changed')
                return
            stack = model.active_stack(context)
            layer = model.active_layer(stack)
            if not (stack and stack.id_data.as_pointer() == review['material_ptr'] and layer and layer.image
                    and layer.image.as_pointer() == review['image_ptr']
                    and stack.active_index == review['index_in_stack']
                    and fingerprint(layer) == review['fingerprint']):
                end_review('Candidate review closed: target changed')
                return
            if digest(layer.image) != review['digest']:
                end_review('Candidate review closed: target image edited')
                return
            node = _display_node(layer)
            if node is not None and node.image != review['preview']:
                # A material rebuild re-referenced the layer image; preview again.
                node.image = review['preview']
    except Exception as exc:
        end_review('Candidate review closed: ' + str(exc))


def cancel(message='Cancelled; queued job removed, running output discarded'):
    global _job, _status
    job, _job = _job, None
    if job:
        (job['directory'] / 'cancel').touch()
        _retired.append(job)
        _status = message
        redraw()


def launch(operation, context, directory=None, **data):
    global _job, _status
    directory = directory or Path(tempfile.mkdtemp(prefix='pawprint-job-'))
    config = dict(operation=operation, server=context.scene.pawprint_server.rstrip('/'))
    if 'settings' in data:
        config['settings'] = data['settings']
    backend.publish(directory, 'request.json', config)
    try:
        process = subprocess.Popen([bpy.app.binary_path, '--background', '--factory-startup',
                                    '--disable-autoexec', '--python-exit-code', '1',
                                    '--python', str(Path(backend.__file__).resolve()), '--', str(directory)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    _job = dict(operation=operation, directory=directory, process=process,
                server=config['server'], window=context.window, area=context.area,
                scene=context.scene.as_pointer(), view_layer=context.view_layer.name, **data)
    _status = 'Connecting…' if operation == 'discover' else 'Uploading and queuing…'
    redraw()


def target_matches(context, job):
    if (context.scene.as_pointer() != job['scene'] or context.view_layer.name != job['view_layer']
            or painting.active() or context.active_object is None
            or context.active_object.as_pointer() != job['owner'] or context.object.mode != 'OBJECT'
            or context.object.active_material_index != job['slot']):
        return False
    stack = model.active_stack(context)
    layer = model.active_layer(stack)
    return bool(stack and stack.id_data.as_pointer() == job['material'] and layer and layer.image
                and fingerprint(layer) == job['fingerprint'])


def tick():
    global _job, _status, _capabilities, _server
    for old in _retired[:]:
        if old['process'].poll() is not None:
            shutil.rmtree(old['directory'], ignore_errors=True)
            _retired.remove(old)
    if _review is not None:
        _maintain_review()
    job = _job
    if not job:
        return 0.25
    try:
        window, area = job['window'], job['area']
        if (window not in bpy.context.window_manager.windows[:] or area not in window.screen.areas[:]
                or area.type != 'VIEW_3D' or window.scene.as_pointer() != job['scene']):
            cancel('Cancelled: originating scene or viewport changed')
            return 0.25
        region = next(r for r in area.regions if r.type == 'WINDOW')
        with bpy.context.temp_override(window=window, area=area, region=region):
            if job['operation'] == 'generate' and not target_matches(bpy.context, job):
                cancel('Cancelled: target layer, image or selection changed')
                return 0.25
            queued = job['directory'] / 'queued.json'
            if queued.exists():
                prompt_id = json.loads(queued.read_text())['prompt_id']
                _status = 'ComfyUI queued/running: ' + prompt_id[:8]
            progress = job['directory'] / 'progress.json'
            if progress.exists():
                data = json.loads(progress.read_text())
                _status = f"Generated {data['done']}/{data['total']} candidates…"
            if job['process'].poll() is None:
                redraw()
                return 0.25
            done = job['directory'] / 'done.json'
            response = json.loads(done.read_text()) if done.exists() else {'error': 'ComfyUI worker exited unexpectedly'}
            if 'error' in response:
                raise RuntimeError(response['error'])
            if job['operation'] == 'discover':
                if bpy.context.scene.pawprint_server.rstrip('/') != job['server']:
                    raise ValueError('Server address changed; connect again')
                _capabilities, _server = response['capabilities'], job['server']
                for name in ('checkpoints', 'samplers', 'schedulers', 'controlnet_models', 'ipadapter_models',
                             'zit_unets', 'zit_loras', 'zit_controlnets', 'loras'):
                    entries = getattr(bpy.context.window_manager, 'pawprint_' + name)
                    entries.clear()
                    for value in _capabilities[name]:
                        entries.add().name = value
                _status = (f"Connected: {len(_capabilities['checkpoints'])} SDXL checkpoints, "
                           f"{len(_capabilities['zit_unets'])} Z models")
            else:
                layer = model.active_layer(model.active_stack(bpy.context))
                if digest(layer.image) != job['digest']:
                    raise ValueError('Target image was edited while generating; result discarded')
                seeds = job['settings'].get('seeds') or [job['settings']['seed']]
                if len(seeds) > 1:
                    # Batch results wait for review; the display swaps to the
                    # preview datablock and the viewport stays interactive.
                    _begin_review(bpy.context, job, seeds)
                else:
                    rgba = capture.returned_pixels(job['directory'] / 'result.png', layer.image.size, job['metadata'])
                    result.apply(bpy.context, rgba, job['metadata']['weights'])
                    layer.generation_last_seed = str(seeds[0])
                    _status = 'Applied generation · Seed ' + str(seeds[0])
    except Exception as exc:
        _status = 'Generation: ' + str(exc)
        print('Pawprint:', _status)
        if _job is job and job['process'].poll() is None:
            cancel(_status)
    finally:
        if _job is job and job['process'].poll() is not None:
            _job = None
            shutil.rmtree(job['directory'], ignore_errors=True)
        redraw()
    return 0.25


class PAWPRINT_PG_choice(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty()


class PAWPRINT_OT_connect(bpy.types.Operator):
    bl_idname = 'pawprint.connect'
    bl_label = 'Connect / Refresh'
    bl_description = 'Discover installed ComfyUI nodes, checkpoints and sampling choices'

    @classmethod
    def poll(cls, context):
        return not active()

    def execute(self, context):
        global _capabilities
        _capabilities = None
        try:
            launch('discover', context)
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


class PAWPRINT_OT_generate(bpy.types.Operator):
    bl_idname = 'pawprint.generate'
    bl_label = 'Generate'
    bl_description = 'Capture the visible saved-view composite and generate one result or a random-seed batch to review'

    @classmethod
    def poll(cls, context):
        from .operators import viewport
        layer = model.active_layer(model.active_stack(context))
        return bool(not active() and _review is None and not painting.active() and connected(context)
                    and layer and layer.image
                    and context.object and context.object.mode == 'OBJECT' and viewport(context)[0])

    def execute(self, context):
        global _status
        directory = None
        try:
            stack = model.active_stack(context)
            layer = model.active_layer(stack)
            result.validate_image(layer.image)
            settings = {key: getattr(layer, 'generation_' + key) for key in backend.ALL_PARAMETERS}
            settings['adapter'] = layer.generation_adapter
            if settings['batch'] > 1:
                # Batch candidates each get their own random seed; the plain
                # seed field only steers single-image requests.
                seeds = [str(secrets.randbits(64)) for _ in range(settings['batch'])]
                settings['seeds'] = seeds
                settings['seed'] = seeds[0]
            elif settings['seed'].strip() == '-1':
                settings['seed'] = str(secrets.randbits(64))
            # A selection routes the request through the inpainting workflow;
            # without one the whole frame updates as ordinary img2img.
            settings['masked'] = layer.selection_paths != '[]'
            # The LoRA stack travels as name/strength pairs; validation and the
            # workflow chain consume it per adapter.
            settings['loras'] = [{'name': item.model, 'strength': item.strength}
                                 for item in layer.generation_loras]
            backend.validate(settings, _capabilities)
            record_prompt(context.scene.pawprint_positive_history, settings['positive'])
            record_prompt(context.scene.pawprint_negative_history, settings['negative'])
            directory = Path(tempfile.mkdtemp(prefix='pawprint-job-'))
            metadata = capture.prepare(context, layer, directory, settings['resolution'])
            # Depth guidance is adapter-independent; the worker uploads the
            # freshly written depth.png for whichever graph consumes it.
            if settings['depth_enabled']:
                capture.geometry_depth(context, layer, directory, metadata)
            if settings['adapter'] == 'SDXL' and settings['ipadapter_enabled']:
                if layer.generation_ipadapter_image is None:
                    raise ValueError('IPAdapter needs a reference image in the Guidance panel')
                capture.save_reference(directory / 'reference.png', layer.generation_ipadapter_image)
            launch('generate', context, directory, settings=settings, metadata=metadata,
                   owner=context.object.as_pointer(), material=stack.id_data.as_pointer(),
                   slot=context.object.active_material_index, fingerprint=fingerprint(layer), digest=digest(layer.image))
            return {'FINISHED'}
        except Exception as exc:
            if directory:
                shutil.rmtree(directory, ignore_errors=True)
            _status = str(exc)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


class PAWPRINT_OT_apply_prompt(bpy.types.Operator):
    bl_idname = 'pawprint.apply_prompt'
    bl_label = 'Apply prompt from history'
    bl_description = 'Replace the active layer prompt with this remembered text'
    bl_options = {'INTERNAL'}
    negative: bpy.props.BoolProperty()
    prompt: bpy.props.StringProperty()

    @classmethod
    def poll(cls, context):
        return bool(not painting.active() and not active()
                    and model.active_layer(model.active_stack(context)))

    def execute(self, context):
        layer = model.active_layer(model.active_stack(context))
        if self.negative:
            layer.generation_negative = self.prompt
        else:
            layer.generation_positive = self.prompt
        if context.area:
            context.area.tag_redraw()
        return {'FINISHED'}


class PAWPRINT_OT_clear_prompt_history(bpy.types.Operator):
    bl_idname = 'pawprint.clear_prompt_history'
    bl_label = 'Clear prompt history'
    bl_description = 'Remove every remembered prompt of this kind'
    bl_options = {'INTERNAL'}
    negative: bpy.props.BoolProperty()

    @classmethod
    def poll(cls, context):
        return hasattr(context.scene, 'pawprint_positive_history')

    def execute(self, context):
        history = (context.scene.pawprint_negative_history if self.negative
                   else context.scene.pawprint_positive_history)
        while len(history):
            history.remove(0)
        return {'FINISHED'}


class PAWPRINT_OT_cancel_generation(bpy.types.Operator):
    bl_idname = 'pawprint.cancel_generation'
    bl_label = 'Cancel Generation'
    bl_description = 'Remove this queued request or discard its running output; other server jobs are untouched'

    @classmethod
    def poll(cls, context):
        return active()

    def execute(self, context):
        cancel()
        return {'FINISHED'}


class PAWPRINT_OT_candidate_select(bpy.types.Operator):
    bl_idname = 'pawprint.candidate_select'
    bl_label = 'Browse Candidates'
    bl_description = 'Show the previous or next generated candidate on the layer'

    direction: bpy.props.IntProperty(default=1)

    @classmethod
    def poll(cls, context):
        return _review is not None and not painting.active()

    def execute(self, context):
        global _status
        review = _review
        if not review:
            return {'CANCELLED'}
        count = len(review['candidates'])
        review['index'] = (review['index'] + self.direction) % count
        _show_candidate(review)
        _status = f"Reviewing candidates: {review['index'] + 1}/{count} · seed {review['seeds'][review['index']]}"
        redraw()
        return {'FINISHED'}


class PAWPRINT_OT_commit_candidate(bpy.types.Operator):
    bl_idname = 'pawprint.commit_candidate'
    bl_label = 'Apply Candidate'
    bl_description = ('Apply the shown candidate to the layer with one native undo step '
                      'and adopt its seed for the next request')

    @classmethod
    def poll(cls, context):
        from .operators import viewport
        return bool(_review is not None and not active() and not painting.active()
                    and context.object and context.object.mode == 'OBJECT' and viewport(context)[0])

    def execute(self, context):
        global _status
        review = _review
        if not review:
            return {'CANCELLED'}
        index, seeds = review['index'], review['seeds']
        rgba = review['candidates'][index].astype(np.float32) / 255.0
        weights = review['weights']
        # Restore the layer display before applying so the managed paint session
        # and the material both show the real layer image again.
        end_review(f'Applying candidate {index + 1}/{len(seeds)}…')
        try:
            layer = model.active_layer(model.active_stack(context))
            result.apply(context, rgba, weights)
            layer.generation_seed = layer.generation_last_seed = seeds[index]
            _status = f'Applied candidate {index + 1}/{len(seeds)} · seed {seeds[index]}'
            redraw()
            return {'FINISHED'}
        except Exception as exc:
            _status = 'Candidate apply: ' + str(exc)
            print('Pawprint:', _status)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


class PAWPRINT_OT_candidate_layer(bpy.types.Operator):
    bl_idname = 'pawprint.candidate_layer'
    bl_label = 'Add Candidate as Layer'
    bl_description = ('Keep the shown candidate as a new editable layer above the source, '
                      'adopting its seed there; the source layer stays unchanged')
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _review is not None and not painting.active()

    def execute(self, context):
        global _status
        review = _review
        if not review:
            return {'CANCELLED'}
        index, seeds = review['index'], review['seeds']
        candidate, seed = review['candidates'][index], str(seeds[index])
        material, source_index = review['material'], review['index_in_stack']
        end_review(f'Adding candidate {index + 1}/{len(seeds)} as a layer…')
        image = None
        try:
            source = material.pawprint.layers[source_index]
            image = bpy.data.images.new('Pawprint Candidate', width=candidate.shape[1],
                                        height=candidate.shape[0], alpha=True)
            image.pixels.foreach_set((candidate.astype(np.float32) / 255.0).ravel())
            image.update()
            image.pack()
            stack = material.pawprint
            layer = stack.layers.add()
            # Mirror the proven add_projection order: no field written here
            # may rebuild the material. depth_tolerance owns a sync callback,
            # so it travels through raw dict access; the material rebuilds
            # once, at the end, from a fully initialized layer.
            for key in model.INHERIT_FIELDS:
                setattr(layer, key, getattr(source, key))
            layer.generation_ipadapter_image = source.generation_ipadapter_image
            for entry in source.generation_loras:
                item = layer.generation_loras.add()
                item.model, item.strength = entry.model, entry.strength
            layer.generation_seed = layer.generation_last_seed = seed
            layer.name = 'Candidate'
            # The candidate lives in the source layer's saved view; its depth
            # snapshot still describes that view, so both are shared as-is.
            layer.depth = source.depth
            for field in ('width', 'height', 'view_matrix', 'projection', 'view_rotation',
                          'view_location', 'view_distance', 'lens', 'clip_start', 'clip_end'):
                setattr(layer, field, getattr(source, field))
            layer['depth_tolerance'] = source.depth_tolerance
            layer['mirror'] = source.mirror
            target = source_index + 1
            stack.layers.move(len(stack.layers) - 1, target)
            stack['active_index'] = target
            # Rebind: collection moves invalidate the wrapper layers.add()
            # returned, and later writes through it would be silently lost.
            layer = stack.layers[target]
            # Assign last so the material rebuild sees the final stack order.
            layer.image = image
            _status = f'Added candidate {index + 1}/{len(seeds)} as a new layer · seed {seed}'
            redraw()
            return {'FINISHED'}
        except Exception as exc:
            if image is not None and image.users == 0:
                bpy.data.images.remove(image)
            _status = 'Candidate layer: ' + str(exc)
            print('Pawprint:', _status)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


class PAWPRINT_OT_discard_candidates(bpy.types.Operator):
    bl_idname = 'pawprint.discard_candidates'
    bl_label = 'Discard Candidate'
    bl_description = 'Drop the shown candidate from the batch; reviewing ends when the last one is gone'

    @classmethod
    def poll(cls, context):
        return _review is not None

    def execute(self, context):
        global _status
        review = _review
        index = review['index']
        del review['candidates'][index]
        del review['seeds'][index]
        if not review['seeds']:
            end_review('All candidates discarded')
            return {'FINISHED'}
        # The next candidate slides into the vacated slot.
        review['index'] = min(index, len(review['seeds']) - 1)
        _show_candidate(review)
        _status = f"Reviewing candidates: {review['index'] + 1}/{len(review['seeds'])} · seed {review['seeds'][review['index']]}"
        redraw()
        return {'FINISHED'}


class PAWPRINT_OT_preview_context(bpy.types.Operator):
    bl_idname = 'pawprint.preview_context'
    bl_label = 'Preview Generation Context'
    bl_description = 'Inspect the exact cropped input image prepared for ComfyUI; Shift-F5 returns to 3D'

    @classmethod
    def poll(cls, context):
        from .operators import viewport
        layer = model.active_layer(model.active_stack(context))
        return bool(not active() and not painting.active() and layer and layer.image
                    and context.object and context.object.mode == 'OBJECT' and viewport(context)[0])

    def execute(self, context):
        image = None
        try:
            layer = model.active_layer(model.active_stack(context))
            with tempfile.TemporaryDirectory(prefix='pawprint-context-preview-') as temp:
                directory = Path(temp)
                capture.prepare(context, layer, directory, layer.generation_resolution)
                image = bpy.data.images.load(str(directory / 'input.png'), check_existing=False)
                image.name = 'Pawprint Generation Context Preview'
                image.pack()
            context.area.type = 'IMAGE_EDITOR'
            context.area.spaces.active.image = image
            context.area.spaces.active.mode = 'VIEW'
            return {'FINISHED'}
        except Exception as exc:
            if image is not None:
                bpy.data.images.remove(image)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


class PAWPRINT_OT_preview_depth(bpy.types.Operator):
    bl_idname = 'pawprint.preview_depth'
    bl_label = 'Preview Geometry Depth'
    bl_description = 'Inspect a fresh saved-view depth crop in the Image Editor; Shift-F5 returns to 3D'

    @classmethod
    def poll(cls, context):
        from .operators import viewport
        layer = model.active_layer(model.active_stack(context))
        return bool(not active() and not painting.active() and layer and layer.image
                    and context.object and context.object.mode == 'OBJECT' and viewport(context)[0])

    def execute(self, context):
        from . import selection
        try:
            layer = model.active_layer(model.active_stack(context))
            bounds = selection.context_bounds(layer)
            if bounds is None:
                raise ValueError('Selection has no editable pixels')
            width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
            scale = layer.generation_resolution / max(width, height)
            size = tuple(max(64, round(value * scale / 8) * 8) for value in (width, height))
            with tempfile.TemporaryDirectory(prefix='pawprint-depth-preview-') as temp:
                directory = Path(temp)
                capture.geometry_depth(context, layer, directory, dict(bounds=bounds, request_size=size))
                image = bpy.data.images.load(str(directory / 'depth.png'), check_existing=False)
                image.name = 'Pawprint Geometry Depth Preview'
                image.colorspace_settings.name = 'Non-Color'
                image.pack()
            context.area.type = 'IMAGE_EDITOR'
            context.area.spaces.active.image = image
            context.area.spaces.active.mode = 'VIEW'
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


@bpy.app.handlers.persistent
def before_data_change(*args):
    if _review is not None:
        end_review('Candidate review closed by save, load or undo/redo')
    cancel('Cancelled by save, load or undo/redo')


@bpy.app.handlers.persistent
def after_load(*args):
    global _capabilities, _server, _status, _review
    _capabilities, _server, _review = None, '', None
    _status = 'Connect to ComfyUI to refresh installed models'


class PAWPRINT_OT_lora_add(bpy.types.Operator):
    bl_idname = 'pawprint.lora_add'
    bl_label = 'Add LoRA'
    bl_description = 'Append an empty LoRA stack entry'
    bl_options = {'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return bool(not painting.active() and not active()
                    and model.active_layer(model.active_stack(context)))

    def execute(self, context):
        layer = model.active_layer(model.active_stack(context))
        layer.generation_loras.add()
        return {'FINISHED'}


class PAWPRINT_OT_lora_remove(bpy.types.Operator):
    bl_idname = 'pawprint.lora_remove'
    bl_label = 'Remove LoRA'
    bl_description = 'Remove this LoRA stack entry'
    bl_options = {'INTERNAL'}
    index: bpy.props.IntProperty()

    @classmethod
    def poll(cls, context):
        return bool(not painting.active() and not active()
                    and model.active_layer(model.active_stack(context)))

    def execute(self, context):
        layer = model.active_layer(model.active_stack(context))
        layer.generation_loras.remove(min(self.index, len(layer.generation_loras) - 1))
        return {'FINISHED'}


CLASSES = (PAWPRINT_PG_choice, PAWPRINT_PG_prompt, PAWPRINT_OT_connect, PAWPRINT_OT_generate,
           PAWPRINT_OT_cancel_generation, PAWPRINT_OT_apply_prompt,
           PAWPRINT_OT_clear_prompt_history, PAWPRINT_OT_preview_context, PAWPRINT_OT_preview_depth,
           PAWPRINT_OT_candidate_select, PAWPRINT_OT_commit_candidate, PAWPRINT_OT_candidate_layer,
           PAWPRINT_OT_discard_candidates, PAWPRINT_OT_lora_add, PAWPRINT_OT_lora_remove)
HANDLERS = ('save_pre', 'load_pre', 'undo_pre', 'redo_pre')


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.pawprint_server = bpy.props.StringProperty(name='ComfyUI server', default='http://127.0.0.1:8188')
    bpy.types.Scene.pawprint_positive_history = bpy.props.CollectionProperty(type=PAWPRINT_PG_prompt)
    bpy.types.Scene.pawprint_negative_history = bpy.props.CollectionProperty(type=PAWPRINT_PG_prompt)
    bpy.types.Scene.pawprint_base_width = bpy.props.IntProperty(
        name='Base width', default=2048, min=64, subtype='PIXEL',
        description='Resolution for newly created bases and commit bakes; committed bases are resampled to this size')
    bpy.types.Scene.pawprint_base_height = bpy.props.IntProperty(
        name='Base height', default=2048, min=64, subtype='PIXEL',
        description='Resolution for newly created bases and commit bakes; committed bases are resampled to this size')
    for name in ('checkpoints', 'samplers', 'schedulers', 'controlnet_models', 'ipadapter_models',
                 'zit_unets', 'zit_loras', 'zit_controlnets', 'loras'):
        setattr(bpy.types.WindowManager, 'pawprint_' + name, bpy.props.CollectionProperty(type=PAWPRINT_PG_choice))
    for name in HANDLERS:
        getattr(bpy.app.handlers, name).append(before_data_change)
    bpy.app.handlers.load_post.append(after_load)
    bpy.app.timers.register(tick, first_interval=0.25, persistent=True)


def unregister():
    cancel('Cancelled by extension reload/disable')
    if bpy.app.timers.is_registered(tick):
        bpy.app.timers.unregister(tick)
    for name in HANDLERS:
        getattr(bpy.app.handlers, name).remove(before_data_change)
    bpy.app.handlers.load_post.remove(after_load)
    for name in ('checkpoints', 'samplers', 'schedulers', 'controlnet_models', 'ipadapter_models',
                 'zit_unets', 'zit_loras', 'zit_controlnets', 'loras'):
        delattr(bpy.types.WindowManager, 'pawprint_' + name)
    del bpy.types.Scene.pawprint_base_width
    del bpy.types.Scene.pawprint_base_height
    del bpy.types.Scene.pawprint_positive_history
    del bpy.types.Scene.pawprint_negative_history
    del bpy.types.Scene.pawprint_server
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
