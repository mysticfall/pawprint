# SPDX-License-Identifier: GPL-3.0-or-later
"""Live ComfyUI round-trip feasibility in an isolated Blender UI process.

This is a development probe, not the asynchronous extension implementation.
It uploads two uniquely named inputs and queues one masked img2img job.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib import error, parse, request
import uuid


class Client:
    def __init__(self, url):
        self.url = url.rstrip('/')

    def get(self, path):
        with request.urlopen(self.url + path, timeout=30) as response:
            return response.read()

    def post(self, path, data, content_type='application/json'):
        req = request.Request(self.url + path, data=data,
                              headers={'Content-Type': content_type})
        try:
            with request.urlopen(req, timeout=30) as response:
                return json.load(response)
        except error.HTTPError as exc:
            raise RuntimeError(exc.read().decode()) from exc

    def upload(self, path):
        boundary = uuid.uuid4().hex
        name = f'pawprint-probe-{uuid.uuid4().hex}-{path.name}'
        data = (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
                f'filename="{name}"\r\nContent-Type: image/png\r\n\r\n').encode()
        data += path.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
        result = self.post('/upload/image', data, f'multipart/form-data; boundary={boundary}')
        return '/'.join(p for p in (result.get('subfolder', ''), result['name']) if p)

    def generate(self, workflow, directory):
        queued = self.post('/prompt', json.dumps({'prompt': workflow}).encode())
        (directory / 'queued.json').write_text(json.dumps(queued, indent=2))
        prompt_id = queued['prompt_id']
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            history = json.loads(self.get('/history/' + prompt_id)).get(prompt_id)
            if history:
                (directory / 'history.json').write_text(json.dumps(history, indent=2))
                if history.get('status', {}).get('status_str') == 'error':
                    raise RuntimeError(history['status'])
                images = history.get('outputs', {}).get('10', {}).get('images', [])
                if images:
                    output = directory / 'result.png'
                    output.write_bytes(self.get('/view?' + parse.urlencode(images[0])))
                    return output
            time.sleep(0.5)
        raise TimeoutError(f'Prompt {prompt_id} timed out; see queued.json. Server job was not interrupted.')


def workflow(checkpoint, image, mask):
    def node(kind, **inputs):
        return {'class_type': kind, 'inputs': inputs}
    return {
        '1': node('CheckpointLoaderSimple', ckpt_name=checkpoint),
        '2': node('LoadImage', image=image),
        '3': node('LoadImage', image=mask),
        '4': node('ImageToMask', image=['3', 0], channel='red'),
        '5': node('VAEEncode', pixels=['2', 0], vae=['1', 2]),
        '6': node('SetLatentNoiseMask', samples=['5', 0], mask=['4', 0]),
        '7': node('CLIPTextEncode', clip=['1', 1], text='hand painted ceramic, colorful glazed surface, fine texture'),
        '8': node('CLIPTextEncode', clip=['1', 1], text='text, watermark, blurry'),
        '9': node('KSampler', model=['1', 0], positive=['7', 0], negative=['8', 0],
                  latent_image=['6', 0], seed=731, steps=12, cfg=5.0,
                  sampler_name='dpmpp_2m', scheduler='karras', denoise=0.35),
        '11': node('VAEDecode', samples=['9', 0], vae=['1', 2]),
        '10': node('PreviewImage', images=['11', 0]),
    }


def prepare(state, directory):
    """Render the saved perspective composite, then crop image and mask identically."""
    import bpy
    import numpy as np
    extension, area, image_name = state
    scene = bpy.context.scene
    stack = extension.painting.session_stack()
    layer = extension.model.active_layer(stack)
    width, height = layer.image.size
    # Disconnected remote island is retained even outside editable target coverage.
    extension.selection.add_path(layer, [(0.15, 0.2), (0.4, 0.2), (0.4, 0.65), (0.15, 0.65)], 'REPLACE')
    extension.selection.add_path(layer, [(0.8, 0.8), (0.85, 0.8), (0.85, 0.85), (0.8, 0.85)], 'ADD')
    layer.generation_feather, layer.generation_padding = 3, 4
    layer.generation_context = 'SELECTION'
    weights = extension.selection.footprint(layer, generation=True).copy()
    bounds = extension.selection.context_bounds(layer)
    x0, y0, x1, y1 = bounds
    assert x1 > width * 0.85 and y1 > height * 0.85, bounds

    view = extension.projection.matrix(layer.view_matrix)
    window = extension.projection.matrix(layer.projection) @ view.inverted()
    camera_data = bpy.data.cameras.new('Probe saved frame')
    camera = bpy.data.objects.new('Probe saved frame', camera_data)
    scene.collection.objects.link(camera)
    camera.matrix_world = view.inverted()
    camera_data.sensor_fit = 'HORIZONTAL'
    camera_data.sensor_width = 36
    camera_data.lens = window[0][0] * 18
    camera_data.clip_start, camera_data.clip_end = layer.clip_start, layer.clip_end
    # A visible upper layer must stay in context even though the request edits below it.
    upper = stack.layers.add()
    upper.name = 'Visible upper context'
    for attr in ('projection', 'view_matrix', 'view_rotation', 'view_location',
                 'view_distance', 'lens', 'clip_start', 'clip_end', 'width', 'height'):
        setattr(upper, attr, getattr(layer, attr))
    upper.depth, upper.has_view = layer.depth, True
    upper.image = bpy.data.images.new('Upper context pixels', width=width, height=height)
    upper_pixels = np.zeros((height, width, 4), dtype=np.float32)
    upper_pixels[90:105, 50:70] = (0, 1, 1, 1)
    upper.image.pixels.foreach_set(upper_pixels.ravel())
    upper.image.update()
    extension.projection.build_material(stack.id_data)
    # A separate object is visible context, independent of the target's material slots.
    from mathutils import Vector
    points = [view.inverted() @ Vector(((u*2-1)*2/window[0][0], (v*2-1)*2/window[1][1], -2))
              for u, v in ((.73, .35), (.88, .35), (.88, .55), (.73, .55))]
    mesh = bpy.data.meshes.new('Surrounding object mesh')
    mesh.from_pydata(points, [], [(0, 1, 2, 3)])
    surrounding = bpy.data.objects.new('Surrounding object', mesh)
    scene.collection.objects.link(surrounding)
    material = bpy.data.materials.new('Blue surrounding object')
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    emission, output = nodes.new('ShaderNodeEmission'), nodes.new('ShaderNodeOutputMaterial')
    emission.inputs[0].default_value = (0, 0, 1, 1)
    material.node_tree.links.new(emission.outputs[0], output.inputs['Surface'])
    mesh.materials.append(material)
    bpy.context.view_layer.update()
    actual = camera.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=width, y=height)
    assert max(abs(actual[r][c] - window[r][c]) for r in range(4) for c in range(4)) < 1e-4
    scene.camera = camera
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 8
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'
    scene.view_settings.view_transform, scene.view_settings.look = 'Standard', 'None'
    scene.view_settings.exposure, scene.view_settings.gamma = 0, 1
    scene.render.film_transparent = False
    scene.render.filepath = str(directory / 'composite.png')
    bpy.ops.render.render(write_still=True)
    composite = bpy.data.images.load(scene.render.filepath, check_existing=False)
    pixels = np.array(composite.pixels[:]).reshape(height, width, 4)
    assert np.any((pixels[:, :, 0] > 0.95) & (pixels[:, :, 1] < 0.03)), 'Other slot absent'
    assert np.any((pixels[:, :, 1] > 0.2) & (pixels[:, :, 2] > 0.2)), 'Layer composite absent'
    assert np.max(np.abs(pixels[97, 60, :3] - (0, 1, 1))) < 0.02, 'Visible upper layer absent'
    assert np.max(np.abs(pixels[int(height*.45), int(width*.8), :3] - (0, 0, 1))) < 0.02, 'Surrounding object absent'

    crop_width, crop_height = x1 - x0, y1 - y0
    request_width = 512
    request_height = max(64, round(crop_height / crop_width * request_width / 8) * 8)
    def save_input(name, values, non_color=False):
        image = bpy.data.images.new(name, width=crop_width, height=crop_height)
        if non_color:
            image.colorspace_settings.name = 'Non-Color'
        image.pixels.foreach_set(values.astype(np.float32).ravel())
        image.scale(request_width, request_height)
        image.filepath_raw = str(directory / name)
        image.file_format = 'PNG'
        image.save()
        bpy.data.images.remove(image)
    save_input('input.png', pixels[y0:y1, x0:x1])
    mask = np.ones((crop_height, crop_width, 4), dtype=np.float32)
    mask[:, :, :3] = weights[y0:y1, x0:x1, None]
    save_input('mask.png', mask, non_color=True)
    metadata = dict(image=image_name, frame=[width, height], bounds=list(bounds),
                    request_size=[request_width, request_height], seed=731, denoise=0.35,
                    projection=list(layer.projection), selection_paths=layer.selection_paths)
    (directory / 'request.json').write_text(json.dumps(metadata, indent=2))
    return weights, bounds, metadata


def check(state, args):
    import bpy
    import numpy as np
    from result_apply_probe import check as apply_check
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    client = Client(args.server)
    info = json.loads(client.get('/object_info'))
    draft = workflow(args.checkpoint, '', '')
    required = {node['class_type'] for node in draft.values()}
    assert required <= info.keys(), required - info.keys()
    checkpoints = info['CheckpointLoaderSimple']['input']['required']['ckpt_name'][0]
    assert args.checkpoint in checkpoints, (args.checkpoint, checkpoints)
    (directory / 'capabilities.json').write_text(json.dumps({
        'system': json.loads(client.get('/system_stats')),
        'nodes': sorted(required), 'checkpoints': checkpoints,
        'diffusion_models': info.get('UNETLoader', {}).get('input', {}).get('required', {}).get('unet_name', [[]])[0],
        'gguf_diffusion_models': info.get('UnetLoaderGGUF', {}).get('input', {}).get('required', {}).get('unet_name', [[]])[0],
    }, indent=2))
    weights, bounds, metadata = prepare(state, directory)
    graph = workflow(args.checkpoint, client.upload(directory / 'input.png'), client.upload(directory / 'mask.png'))
    (directory / 'workflow.json').write_text(json.dumps(graph, indent=2))
    result = client.generate(graph, directory)
    image = bpy.data.images.load(str(result), check_existing=False)
    assert list(image.size) == metadata['request_size'], list(image.size)
    x0, y0, x1, y1 = bounds
    image.scale(x1 - x0, y1 - y0)
    width, height = bpy.data.images[state[2]].size
    returned = np.zeros((height, width, 4), dtype=np.float32)
    returned[y0:y1, x0:x1] = np.array(image.pixels[:]).reshape(y1-y0, x1-x0, 4)
    # Placement and masking use the original footprint, never the resized server mask.
    apply_check(state, generated=returned, weights=weights)
    applied = bpy.data.images[state[2]]
    applied.filepath_raw, applied.file_format = str(directory / 'applied-layer.png'), 'PNG'
    applied.save()
    report = dict(live_masked_img2img=True, crop_roundtrip=True,
                  unselected_pixels_preserved=True, native_result_undo_redo=True,
                  checkpoint=args.checkpoint, prompt_id=json.loads((directory / 'queued.json').read_text())['prompt_id'])
    (directory / 'report.json').write_text(json.dumps(report, indent=2))
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', default='http://127.0.0.1:8188')
    parser.add_argument('--checkpoint', default='epicrealismXL_pureFix.safetensors')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    argv = sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    if args.worker:
        import bpy
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from result_apply_probe import setup
        bpy.context.preferences.view.show_splash = False
        state = []
        def run():
            try:
                if not state:
                    state.append(setup())
                    return 1
                check(state[0], args)
                bpy.ops.wm.quit_blender()
            except Exception:
                import traceback
                traceback.print_exc()
                os._exit(1)
        bpy.app.timers.register(run, first_interval=2)
    else:
        with tempfile.TemporaryDirectory(prefix='pawprint-generation-') as temp:
            env = os.environ.copy()
            env['PAWPRINT_PROBE_MANAGED'] = '1'
            for suffix in ('RESOURCES', 'CONFIG', 'SCRIPTS', 'DATAFILES', 'EXTENSIONS'):
                env[f'BLENDER_USER_{suffix}'] = str(Path(temp) / suffix.lower())
            subprocess.run(['blender', '--factory-startup', '--python-exit-code', '1', '--python',
                            str(Path(__file__).resolve()), '--', '--worker', '--server', args.server,
                            '--checkpoint', args.checkpoint, '--output', str(args.output.resolve())],
                           env=env, check=True, timeout=600)


if __name__ == '__main__':
    main()
