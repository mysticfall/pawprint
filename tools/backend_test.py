# SPDX-License-Identifier: GPL-3.0-or-later
"""Network/lifecycle contract checks without Blender or a GPU server."""
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest

spec = importlib.util.spec_from_file_location('pawprint_backend', Path(__file__).resolve().parents[1] / 'pawprint/backend.py')
assert spec and spec.loader
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


class BackendTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.settings = {key: value['default'] for key, value in backend.ALL_PARAMETERS.items()}
        self.settings.update(checkpoint='test.safetensors', positive='positive prompt', negative='negative prompt',
                             seed=str(2**64-1), steps=13, cfg=6.25, denoise=.42)
        self.calls = []
        self.mode = 'cancel'
        self.prompts = 0
        self.cancel_at = 2
        self.info = {name: {} for name in backend.REQUIRED}
        self.info.update({name: {} for name in backend.CHORD_REQUIRED})
        self.info['ChordLoadModel'] = {'input': {'required': {'ckpt_name': [['chord_v1.safetensors']]}}}
        self.info['CheckpointLoaderSimple'] = {'input': {'required': {'ckpt_name': [['test.safetensors']]}}}
        self.info['INPAINT_LoadInpaintModel'] = {'input': {'required': {'model_name': [[
            'MAT_Places512_G_fp16.safetensors', 'lama_large_512px.safetensors']]}}}
        self.info['KSampler'] = {'input': {'required': {
            'sampler_name': [['dpmpp_2m', 'euler', 'res_multistep']], 'scheduler': [['karras', 'simple']],
            'steps': ['INT', {'min':1, 'max':10000}], 'cfg': ['FLOAT', {'min':0, 'max':100}],
            'denoise': ['FLOAT', {'min':0, 'max':1}], 'seed': ['INT', {'min':0, 'max':2**64-1}],
        }}}
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def respond(self, payload, status=200):
                self.send_response(status)
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())

            def do_GET(self):
                owner.calls.append(('GET', self.path, None))
                if self.path == '/object_info':
                    self.respond(owner.info)
                elif owner.mode in ('chord_error', 'chord_missing') and owner.prompts == 2 and self.path.startswith('/history/'):
                    self.respond({self.path[len('/history/'):]: {
                        'status': {'status_str': 'error' if owner.mode == 'chord_error' else 'success',
                                    'completed': True, 'messages': ['out of memory']},
                        'outputs': {'10': {'images': [{'filename': 'albedo.png', 'subfolder': '', 'type': 'temp'}]}}}})
                elif owner.mode in ('batch', 'batch_cancel', 'chord_error', 'chord_missing') and self.path.startswith('/history/'):
                    self.respond({self.path[len('/history/'):]: {
                        'status': {'status_str': 'success', 'completed': True},
                         'outputs': {'10': {'images': [
                             {'filename': 'result.png', 'subfolder': '', 'type': 'temp'}]},
                             '11': {'images': [{'filename': 'normal.png', 'subfolder': '', 'type': 'temp'}]}}}})
                elif owner.mode in ('batch', 'batch_cancel', 'estimate', 'chord_error', 'chord_missing') and self.path.startswith('/view'):
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'png bytes')
                elif owner.mode == 'estimate' and self.path.startswith('/history/'):
                    self.respond({self.path[len('/history/'):]: {
                        'status': {'status_str': 'success', 'completed': True},
                        'outputs': {
                            '10': {'images': [{'filename': 'basecolor.png', 'subfolder': '', 'type': 'temp'}]},
                            '11': {'images': [{'filename': 'normal.png', 'subfolder': '', 'type': 'temp'}]},
                        }}})
                elif owner.mode in ('batch', 'batch_cancel') and self.path.startswith('/view'):
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'png bytes')
                elif self.path == '/history/own-prompt':
                    self.respond({'own-prompt': {'status': {'status_str':'error', 'messages':['GPU execution failed']}}})
                else:
                    self.respond({}, 404)

            def do_POST(self):
                body = self.rfile.read(int(self.headers['Content-Length']))
                parsed = json.loads(body) if self.headers['Content-Type'] == 'application/json' else None
                owner.calls.append(('POST', self.path, parsed))
                if self.path == '/upload/image':
                    self.respond({'name':'uploaded.png', 'subfolder':''})
                elif self.path == '/prompt':
                    if owner.mode == 'reject':
                        self.respond({'error':'Invalid checkpoint'}, 400)
                    else:
                        if owner.mode == 'cancel':
                            # The crucial race: cancellation occurs during POST,
                            # before the worker knows its server-assigned prompt ID.
                            (owner.directory / 'cancel').touch()
                        if owner.mode == 'batch_cancel':
                            owner.prompts += 1
                            if owner.prompts == owner.cancel_at:
                                # Cancel once the first image is finished and the
                                # second request is already submitted.
                                (owner.directory / 'cancel').touch()
                            self.respond({'prompt_id': f'own-prompt-{owner.prompts}'})
                        elif owner.mode in ('batch', 'chord_error', 'chord_missing'):
                            owner.prompts += 1
                            self.respond({'prompt_id': f'own-prompt-{owner.prompts}'})
                        else:
                            self.respond({'prompt_id': 'own-prompt'})
                elif self.path == '/queue':
                    self.respond({})
                else:
                    self.respond({}, 404)
        self.server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        backend.publish(self.directory, 'request.json', dict(operation='generate', settings=self.settings,
                        server=f'http://127.0.0.1:{self.server.server_port}'))
        for name in ('input.png','mask.png'):
            (self.directory / name).write_bytes(b'fixture PNG bytes')

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def test_cancel_during_submission_owns_only_one_job(self):
        backend.run(self.directory)
        self.assertFalse(self.directory.exists())
        deletes = [body for _, path, body in self.calls if path == '/queue']
        self.assertEqual(deletes, [{'delete':['own-prompt']}])
        self.assertFalse(any(path == '/interrupt' for _, path, _ in self.calls))
        graph = next(body['prompt'] for _,path,body in self.calls if path == '/prompt')
        self.assertEqual(graph['7']['inputs']['text'], self.settings['positive'])
        self.assertEqual(graph['8']['inputs']['text'], self.settings['negative'])
        self.assertEqual(graph['24']['class_type'], 'CLIPSetLastLayer')
        self.assertEqual(graph['24']['inputs'], {'clip': ['1', 1], 'stop_at_clip_layer': -1})
        self.assertEqual(graph['7']['inputs']['clip'], ['24', 0])
        for key in ('steps','cfg','denoise'):
            self.assertEqual(graph['9']['inputs'][key], self.settings[key])
        self.assertEqual(graph['9']['inputs']['seed'], 2**64-1)

    def test_server_validation_error_is_reported(self):
        self.mode = 'reject'
        backend.run(self.directory)
        self.assertIn('Invalid checkpoint', json.loads((self.directory / 'done.json').read_text())['error'])
        self.assertFalse(any(path == '/queue' for _,path,_ in self.calls))

    def test_clip_skip_controls_prompt_encoder_layer(self):
        caps = backend.capabilities(self.info)
        settings = dict(self.settings, clip_skip=4)
        backend.validate(settings, caps)
        graph = backend.workflow(settings, 'in.png', 'mask.png')
        self.assertEqual(graph['24']['inputs']['stop_at_clip_layer'], -4)
        self.assertEqual(graph['7']['inputs']['clip'], ['24', 0])
        self.assertEqual(graph['8']['inputs']['clip'], ['24', 0])
        with self.assertRaisesRegex(ValueError, 'CLIP skip'):
            backend.validate(dict(settings, clip_skip=25), caps)

    def test_execution_error_is_reported(self):
        self.mode = 'execution_error'
        backend.run(self.directory)
        self.assertIn('GPU execution failed', json.loads((self.directory / 'done.json').read_text())['error'])

    def test_missing_dependencies_reject_before_upload(self):
        del self.info['SetLatentNoiseMask']
        backend.run(self.directory)
        self.assertIn('SetLatentNoiseMask', json.loads((self.directory / 'done.json').read_text())['error'])
        self.assertFalse(any(method == 'POST' for method,_,_ in self.calls))

    def test_depth_dependencies_are_optional_but_required_when_enabled(self):
        caps = backend.capabilities(self.info)
        backend.validate(self.settings, caps)
        self.settings['depth_enabled'] = True
        with self.assertRaisesRegex(ValueError, 'ControlNet model'):
            backend.validate(self.settings, caps)
        self.info['ControlNetLoader'] = {'input': {'required': {'control_net_name': [[
            'controlnet_depth_sdxl.safetensors', 'sdxl_promax.safetensors', 'unrelated.safetensors']]}}}
        self.info['ControlNetApplyAdvanced'] = {}
        # Without the union type selector the shared weight cannot be tuned.
        caps = backend.capabilities(self.info)
        self.assertEqual(caps['controlnet_models'], [])
        with self.assertRaisesRegex(ValueError, 'ControlNet model'):
            backend.validate(self.settings, caps)
        self.info['SetUnionControlNetType'] = {}
        caps = backend.capabilities(self.info)
        self.assertEqual(caps['controlnet_models'], ['sdxl_promax.safetensors'])
        backend.validate(self.settings, caps)
        backend.publish(self.directory, 'request.json', dict(operation='generate', settings=self.settings,
                        server=f'http://127.0.0.1:{self.server.server_port}'))
        (self.directory / 'depth.png').write_bytes(b'fixture depth')
        backend.run(self.directory)
        self.assertEqual(sum(path == '/upload/image' for _,path,_ in self.calls), 3)
        graph = next(body['prompt'] for _,path,body in self.calls if path == '/prompt')
        self.assertEqual(graph['23']['class_type'], 'SetUnionControlNetType')
        self.assertEqual(graph['23']['inputs']['type'], 'depth')
        self.assertEqual(graph['14']['class_type'], 'ControlNetApplyAdvanced')
        self.assertEqual(graph['14']['inputs']['image'], ['12', 0])
        self.assertEqual(graph['14']['inputs']['control_net'], ['23', 0])
        self.assertEqual(graph['14']['inputs']['vae'], ['1', 2])
        self.assertEqual(graph['9']['inputs']['positive'], ['14', 0])
        self.assertEqual(graph['9']['inputs']['negative'], ['14', 1])
        self.assertEqual(graph['6']['inputs']['mask'], ['4', 0])
        self.assertFalse(any(path == '/interrupt' for _,path,_ in self.calls))

    def test_ipadapter_requires_paired_encoder_and_patches_model(self):
        caps = backend.capabilities(self.info)
        self.assertEqual(caps['ipadapter_models'], [])
        self.settings['ipadapter_enabled'] = True
        with self.assertRaisesRegex(ValueError, 'IPAdapter model'):
            backend.validate(self.settings, caps)
        self.info['IPAdapterModelLoader'] = {'input': {'required': {'ipadapter_file': [[
            'ip-adapter-plus_sdxl_vit-h.safetensors', 'ip-adapter_sdxl_vit-g.safetensors',
            'ip-adapter_sd15.safetensors']]}}}
        self.info['IPAdapterAdvanced'] = {}
        self.info['CLIPVisionLoader'] = {'input': {'required': {'clip_name': [[
            'CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors']]}}}
        # The ViT-G adapter stays unadvertised: its bigG encoder is absent, and
        # SD15 weights have no pairing in this SDXL contract.
        caps = backend.capabilities(self.info)
        self.assertEqual(caps['ipadapter_models'], ['ip-adapter-plus_sdxl_vit-h.safetensors'])
        self.settings['ipadapter_model'] = 'ip-adapter-plus_sdxl_vit-h.safetensors'
        self.settings['ipadapter_weight'] = 0.85
        backend.validate(self.settings, caps)
        self.settings['depth_enabled'] = True
        with self.assertRaisesRegex(ValueError, 'ControlNet model'):
            backend.validate(self.settings, caps)
        self.info['ControlNetLoader'] = {'input': {'required': {'control_net_name': [[
            'sdxl_promax.safetensors']]}}}
        self.info['ControlNetApplyAdvanced'] = {}
        self.info['SetUnionControlNetType'] = {}
        caps = backend.capabilities(self.info)
        backend.validate(self.settings, caps)
        graph = backend.workflow(self.settings, 'in.png', 'mask.png', 'depth.png', 'reference.png')
        self.assertEqual(graph['15']['class_type'], 'IPAdapterModelLoader')
        self.assertEqual(graph['16']['inputs']['clip_name'], 'CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors')
        self.assertEqual(graph['18']['class_type'], 'IPAdapterAdvanced')
        self.assertEqual(graph['18']['inputs']['model'], ['1', 0])
        self.assertEqual(graph['18']['inputs']['image'], ['17', 0])
        self.assertEqual(graph['18']['inputs']['clip_vision'], ['16', 0])
        self.assertEqual(graph['18']['inputs']['weight'], 0.85)
        self.assertEqual(graph['9']['inputs']['model'], ['18', 0])
        self.assertEqual(graph['9']['inputs']['positive'], ['14', 0])
        self.assertEqual(graph['9']['inputs']['negative'], ['14', 1])
        self.settings['ipadapter_enabled'] = False
        graph = backend.workflow(self.settings, 'in.png', 'mask.png', 'depth.png')
        self.assertNotIn('18', graph)
        self.assertEqual(graph['9']['inputs']['model'], ['1', 0])
        with self.assertRaisesRegex(ValueError, 'reference image is missing'):
            backend.workflow(dict(self.settings, ipadapter_enabled=True, depth_enabled=False), 'in.png', 'mask.png')

    def test_inpaint_selection_drives_reference_pipeline(self):
        caps = backend.capabilities(self.info)
        self.assertEqual(caps['sdxl_inpaint'], 'MAT_Places512_G_fp16.safetensors')
        # Without a selection none of the inpaint machinery is needed; a plain
        # server without any ControlNet machinery still accepts whole-frame
        # img2img.
        backend.validate(dict(self.settings, masked=False), caps)
        # A selection runs the reference pipeline; its nodes are part of the
        # SDXL contract, so any server offering SDXL accepts masked requests
        # below full denoise without extra weights.
        self.settings['masked'] = True
        backend.validate(self.settings, caps)
        # Depth guidance keeps requiring the union ControlNet weight.
        self.settings['depth_enabled'] = True
        with self.assertRaisesRegex(ValueError, 'ControlNet model'):
            backend.validate(self.settings, caps)
        self.info['ControlNetLoader'] = {'input': {'required': {'control_net_name': [[
            'controlnet_depth_sdxl.safetensors', 'sdxl_promax.safetensors', 'unrelated.safetensors']]}}}
        self.info['ControlNetApplyAdvanced'] = {}
        caps = backend.capabilities(self.info)
        self.assertEqual(caps['controlnet_models'], [])
        with self.assertRaisesRegex(ValueError, 'ControlNet model'):
            backend.validate(self.settings, caps)
        self.info['SetUnionControlNetType'] = {}
        caps = backend.capabilities(self.info)
        self.assertEqual(caps['controlnet_models'], ['sdxl_promax.safetensors'])
        backend.validate(self.settings, caps)
        # Below full denoise the refine pipeline runs: depth patches the
        # conditioning first, the once-feathered upload gates the inpaint
        # conditioning of the original pixels, the Fooocus patch joins the
        # model chain, and the split sigma schedule enters at the step the
        # denoise strength skips. Colors re-match against the original input.
        settings = dict(self.settings, feather=12)
        graph = backend.workflow(settings, 'in.png', 'mask.png', 'depth.png', caps=caps)
        self.assertNotIn('26', graph)  # Capture already feathered the mask.
        self.assertEqual(graph['21']['class_type'], 'INPAINT_VAEEncodeInpaintConditioning')
        self.assertEqual(graph['21']['inputs'], {'vae': ['1', 2], 'pixels': ['2', 0],
                                                 'mask': ['4', 0], 'positive': ['14', 0],
                                                 'negative': ['14', 1]})
        self.assertEqual(graph['25']['class_type'], 'SelfAttentionGuidance')
        self.assertEqual(graph['25']['inputs'], {'model': ['1', 0], 'scale': 0.5, 'blur_sigma': 2.0})
        self.assertEqual(graph['29']['class_type'], 'DifferentialDiffusion')
        self.assertEqual(graph['29']['inputs'], {'model': ['25', 0]})
        self.assertEqual(graph['22']['class_type'], 'INPAINT_LoadFooocusInpaint')
        self.assertEqual(graph['22']['inputs'],
                         {'head': 'fooocus_inpaint_head.pth', 'patch': 'inpaint_v26.fooocus.patch'})
        self.assertEqual(graph['31']['class_type'], 'INPAINT_ApplyFooocusInpaint')
        self.assertEqual(graph['31']['inputs'],
                         {'model': ['29', 0], 'patch': ['22', 0], 'latent': ['21', 2]})
        self.assertEqual(graph['44']['inputs'], {'noise_seed': 2**64-1})
        self.assertEqual(graph['45']['inputs'], {'sampler_name': settings['sampler']})
        self.assertEqual(graph['46']['class_type'], 'CFGGuider')
        self.assertEqual(graph['46']['inputs'], {'model': ['31', 0], 'positive': ['21', 0],
                                                 'negative': ['21', 1], 'cfg': settings['cfg']})
        self.assertEqual(graph['47']['class_type'], 'BasicScheduler')
        self.assertEqual(graph['47']['inputs']['denoise'], 1.0)
        self.assertEqual(graph['48']['class_type'], 'SplitSigmas')
        self.assertEqual(graph['48']['inputs'], {'sigmas': ['47', 0], 'step': 8})
        self.assertEqual(graph['49']['class_type'], 'SamplerCustomAdvanced')
        self.assertEqual(graph['49']['inputs'],
                         {'noise': ['44', 0], 'guider': ['46', 0], 'sampler': ['45', 0],
                          'sigmas': ['48', 1], 'latent_image': ['21', 3]})
        self.assertEqual(graph['41']['class_type'], 'VAEDecode')
        self.assertEqual(graph['41']['inputs'], {'samples': ['49', 1], 'vae': ['1', 2]})
        self.assertEqual(graph['56']['class_type'], 'INPAINT_ColorMatch')
        self.assertEqual(graph['56']['inputs'], {'target': ['41', 0], 'reference': ['2', 0],
                                                 'exclude_mask': ['4', 0], 'strength': 1.0})
        self.assertEqual(graph['10']['inputs']['images'], ['56', 0])
        for key in ('5', '6', '9', '11', '27', '33', '42', '43'):
            self.assertNotIn(key, graph)
        classes = {node['class_type'] for node in graph.values()}
        self.assertNotIn('InpaintModelConditioning', classes)
        self.assertNotIn('ImageCompositeMasked', classes)
        self.assertNotIn('VAEEncode', classes)
        # Full denoise: the replace pipeline stabilizes the uploaded mask,
        # pre-fills the selection through a MAT inpaint model behind a tight
        # mask, encodes the pre-filled pixels as the latent and ColorMatch
        # reference, and keeps a whole sigma schedule.
        full = dict(self.settings, denoise=1.0)
        backend.validate(full, caps)
        graph = backend.workflow(full, 'in.png', 'mask.png', 'depth.png', caps=caps)
        self.assertEqual(graph['27']['class_type'], 'INPAINT_StabilizeMask')
        self.assertEqual(graph['27']['inputs'], {'mask': ['4', 0], 'epsilon': 0.01})
        self.assertEqual(graph['33']['class_type'], 'ThresholdMask')
        self.assertEqual(graph['33']['inputs'],
                         {'mask': ['4', 0], 'value': 0.5})
        self.assertEqual(graph['42']['class_type'], 'INPAINT_LoadInpaintModel')
        self.assertEqual(graph['42']['inputs'], {'model_name': 'MAT_Places512_G_fp16.safetensors'})
        self.assertEqual(graph['43']['class_type'], 'INPAINT_InpaintWithModel')
        self.assertEqual(graph['43']['inputs'], {'inpaint_model': ['42', 0], 'image': ['2', 0],
                                                 'mask': ['33', 0], 'seed': 2**64-1})
        self.assertEqual(graph['21']['inputs']['pixels'], ['43', 0])
        self.assertEqual(graph['21']['inputs']['mask'], ['27', 0])
        self.assertNotIn('48', graph)
        self.assertEqual(graph['49']['inputs']['sigmas'], ['47', 0])
        self.assertEqual(graph['56']['inputs']['reference'], ['43', 0])
        self.assertEqual(graph['56']['inputs']['exclude_mask'], ['27', 0])
        # Without depth the conditioning flows straight from the text encodes.
        graph = backend.workflow(dict(self.settings, depth_enabled=False), 'in.png', 'mask.png')
        self.assertEqual(graph['21']['inputs']['positive'], ['7', 0])
        self.assertNotIn('13', graph)
        # A server without any inpaint pre-fill model rejects full-denoise
        # selections; refinement keeps working.
        self.info['INPAINT_LoadInpaintModel'] = {'input': {'required': {'model_name': [[]]}}}
        caps = backend.capabilities(self.info)
        self.assertIsNone(caps['sdxl_inpaint'])
        with self.assertRaisesRegex(ValueError, 'Inpaint model'):
            backend.validate(full, caps)
        backend.validate(self.settings, caps)
        self.info['INPAINT_LoadInpaintModel'] = {'input': {'required': {'model_name': [[
            'MAT_Places512_G_fp16.safetensors', 'lama_large_512px.safetensors']]}}}
        # Without a selection none of the inpaint machinery appears: plain VAE
        # encode, whole-frame noise mask, conditioning straight to the depth
        # apply, decoded image as output.
        graph = backend.workflow(dict(self.settings, masked=False), 'in.png', 'mask.png', 'depth.png')
        self.assertEqual(graph['5']['class_type'], 'VAEEncode')
        self.assertEqual(graph['6']['inputs']['mask'], ['4', 0])
        self.assertEqual(graph['9']['inputs']['latent_image'], ['6', 0])
        self.assertEqual(graph['9']['inputs']['positive'], ['14', 0])
        self.assertEqual(graph['10']['inputs']['images'], ['11', 0])
        for key in ('21', '25', '26'):
            self.assertNotIn(key, graph)
        # The worker uploads exactly the composite, mask and depth maps; the
        # masked graph consumes the single input upload.
        backend.publish(self.directory, 'request.json', dict(operation='generate', settings=self.settings,
                        server=f'http://127.0.0.1:{self.server.server_port}'))
        (self.directory / 'depth.png').write_bytes(b'fixture depth')
        backend.run(self.directory)
        self.assertEqual(sum(path == '/upload/image' for _, path, _ in self.calls), 3)
        graph = next(body['prompt'] for _, path, body in self.calls if path == '/prompt')
        self.assertEqual(graph['21']['class_type'], 'INPAINT_VAEEncodeInpaintConditioning')
        self.assertNotIn('6', graph)
        self.assertEqual(graph['49']['inputs']['latent_image'], ['21', 3])
        self.assertFalse(any(path == '/interrupt' for _, path, _ in self.calls))


    def test_zit_adapter(self):
        caps = backend.capabilities(self.info)
        self.assertFalse(caps['adapters']['ZIT'])
        self.assertTrue(caps['adapters']['SDXL'])
        zit = {key: spec['default'] for key, spec in backend.ZIT_PARAMETERS.items()}
        zit.update(adapter='ZIT', zit_unet='z_image_turbo_bf16.safetensors',
                   loras=[{'name': 'zit/skin texture Photorealistic style v4.5.safetensors', 'strength': 0.85}],
                   positive='weathered oak', negative='seams', masked=True, depth_enabled=True, feather=12)
        with self.assertRaisesRegex(ValueError, 'Z Image Turbo nodes'):
            backend.validate(dict(zit, loras=[]), caps)
        for name, payload in {
            'UNETLoader': {'input': {'required': {'unet_name': [[
                'qwen_image_edit_2509_int8_convrot.safetensors', 'z_image_turbo_bf16.safetensors']]}}},
            'CLIPLoader': {'input': {'required': {'clip_name': [[
                'qwen_2_5_vl_7b_fp8_scaled.safetensors', 'qwen_3_4b.safetensors']]}}},
            'VAELoader': {'input': {'required': {'vae_name': [[
                'ae.safetensors', 'qwen_image_vae.safetensors', 'pixel_space']]}}},
            'CLIPTextEncode': {},
            'ConditioningZeroOut': {},
            'LoraLoaderModelOnly': {'input': {'required': {'lora_name': [[
                'zit/Z Seasian.safetensors', 'zit/skin texture Photorealistic style v4.5.safetensors',
                'sdxl_lightning_8step.safetensors']]}}},
            'ModelSamplingAuraFlow': {},
            'DifferentialDiffusion': {},
            'ZImageFunControlnet': {},
            'INPAINT_StabilizeMask': {},
            'INPAINT_ColorMatch': {},
            'INPAINT_InpaintWithModel': {},
            'INPAINT_LoadInpaintModel': {'input': {'required': {'model_name': [[
                'MAT_Places512_G_fp16.safetensors']]}}},
            'ThresholdMask': {},
            'SplitSigmas': {},
            'RandomNoise': {},
            'KSamplerSelect': {},
            'BasicScheduler': {},
            'BasicGuider': {},
            'SamplerCustomAdvanced': {},
        }.items():
            self.info[name] = payload
        caps = backend.capabilities(self.info)
        self.assertTrue(caps['adapters']['ZIT'] and caps['adapters']['SDXL'])
        self.assertEqual(caps['zit_unets'],
                         ['qwen_image_edit_2509_int8_convrot.safetensors', 'z_image_turbo_bf16.safetensors'])
        # The SDXL lightning LoRA stays out of the zit style list.
        self.assertEqual(caps['zit_loras'],
                         ['zit/Z Seasian.safetensors', 'zit/skin texture Photorealistic style v4.5.safetensors'])
        self.assertEqual(caps['zit_clip'], 'qwen_3_4b.safetensors')
        self.assertEqual(caps['zit_vae'], 'ae.safetensors')
        self.assertEqual(caps['zit_inpaint'], 'MAT_Places512_G_fp16.safetensors')
        # Depth guidance needs the DiffSynth patch loader pair.
        self.assertEqual(caps['zit_controlnets'], [])
        with self.assertRaisesRegex(ValueError, 'ControlNet patch'):
            backend.validate(zit, caps)
        self.info['ModelPatchLoader'] = {'input': {'required': {'name': [[
            'Z-Image-Turbo-Fun-Controlnet-Tile-2.1-lite-2601-8steps.safetensors',
            'Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2601-8steps.safetensors']]}}}
        self.info['QwenImageDiffsynthControlnet'] = {}
        caps = backend.capabilities(self.info)
        self.assertEqual(len(caps['zit_controlnets']), 2)
        with self.assertRaisesRegex(ValueError, 'Z model'):
            backend.validate(dict(zit, zit_unet='missing.safetensors'), caps)
        # The LoRA stack validates against the zit-filtered weight list.
        with self.assertRaisesRegex(ValueError, 'LoRA is not available'):
            backend.validate(dict(zit, loras=[{'name': 'sdxl_lightning_8step.safetensors', 'strength': 1.0}]), caps)
        backend.validate(dict(zit, loras=[]), caps)
        backend.validate(dict(zit, batch=3), caps)
        graph = backend.workflow(dict(zit, seed='11'), 'in.png', 'mask.png', 'depth.png', caps=caps)
        self.assertEqual(graph['30']['class_type'], 'UNETLoader')
        self.assertEqual(graph['50']['class_type'], 'LoraLoaderModelOnly')
        self.assertEqual(graph['50']['inputs']['lora_name'], 'zit/skin texture Photorealistic style v4.5.safetensors')
        self.assertEqual(graph['50']['inputs']['strength_model'], 0.85)
        self.assertEqual(graph['34']['class_type'], 'CLIPLoader')
        self.assertEqual(graph['34']['inputs']['clip_name'], 'qwen_3_4b.safetensors')
        self.assertEqual(graph['34']['inputs']['type'], 'lumina2')
        self.assertEqual(graph['37']['class_type'], 'CLIPSetLastLayer')
        self.assertEqual(graph['37']['inputs'], {'clip': ['34', 0], 'stop_at_clip_layer': -1})
        self.assertEqual(graph['35']['inputs']['vae_name'], 'ae.safetensors')
        self.assertEqual(graph['38']['class_type'], 'CLIPTextEncode')
        self.assertEqual(graph['38']['inputs']['clip'], ['37', 0])
        self.assertEqual(graph['38']['inputs']['text'], 'weathered oak')
        # Masked at full denoise: the reference pipeline. A MAT inpaint model
        # pre-fills the selection, the Fun ControlNet runs in inpaint mode as
        # the context provider, DifferentialDiffusion wraps the chain and the
        # advanced sampling stack ends in a colour match against the pre-fill.
        self.assertNotIn('26', graph)
        self.assertEqual(graph['27']['class_type'], 'INPAINT_StabilizeMask')
        self.assertEqual(graph['27']['inputs'], {'mask': ['4', 0], 'epsilon': 0.01})
        self.assertEqual(graph['28']['class_type'], 'ThresholdMask')
        self.assertEqual(graph['28']['inputs'], {'mask': ['4', 0], 'value': 0.5})
        self.assertNotIn('33', graph)
        self.assertEqual(graph['42']['class_type'], 'INPAINT_LoadInpaintModel')
        self.assertEqual(graph['42']['inputs']['model_name'], 'MAT_Places512_G_fp16.safetensors')
        self.assertEqual(graph['43']['class_type'], 'INPAINT_InpaintWithModel')
        self.assertEqual(graph['43']['inputs'], {'inpaint_model': ['42', 0], 'image': ['36', 0],
                                                 'mask': ['28', 0], 'seed': 11})
        self.assertEqual(graph['44']['class_type'], 'ModelPatchLoader')
        self.assertEqual(graph['44']['inputs']['name'],
                         'Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2601-8steps.safetensors')
        self.assertEqual(graph['45']['class_type'], 'ZImageFunControlnet')
        self.assertEqual(graph['45']['inputs']['model'], ['50', 0])
        self.assertEqual(graph['45']['inputs']['model_patch'], ['44', 0])
        self.assertEqual(graph['45']['inputs']['vae'], ['35', 0])
        self.assertEqual(graph['45']['inputs']['image'], ['12', 0])
        self.assertEqual(graph['45']['inputs']['inpaint_image'], ['36', 0])
        self.assertEqual(graph['45']['inputs']['mask'], ['28', 0])
        self.assertEqual(graph['45']['inputs']['strength'], zit['zit_strength'])
        self.assertEqual(graph['25']['class_type'], 'DifferentialDiffusion')
        self.assertEqual(graph['25']['inputs']['model'], ['45', 0])
        self.assertEqual(graph['46']['class_type'], 'VAEEncode')
        self.assertEqual(graph['46']['inputs'], {'pixels': ['43', 0], 'vae': ['35', 0]})
        self.assertEqual(graph['47']['class_type'], 'SetLatentNoiseMask')
        self.assertEqual(graph['47']['inputs'], {'samples': ['46', 0], 'mask': ['27', 0]})
        self.assertEqual(graph['48']['inputs'], {'noise_seed': 11})
        self.assertEqual(graph['49']['inputs'], {'sampler_name': 'res_multistep'})
        self.assertEqual(graph['52']['class_type'], 'BasicScheduler')
        self.assertEqual(graph['52']['inputs']['model'], ['25', 0])
        self.assertEqual(graph['52']['inputs']['scheduler'], 'simple')
        self.assertEqual(graph['52']['inputs']['steps'], 8)
        self.assertEqual(graph['52']['inputs']['denoise'], 1.0)
        self.assertEqual(graph['54']['class_type'], 'BasicGuider')
        self.assertEqual(graph['54']['inputs'], {'model': ['25', 0], 'conditioning': ['38', 0]})
        self.assertEqual(graph['55']['class_type'], 'SamplerCustomAdvanced')
        self.assertEqual(graph['55']['inputs'], {'noise': ['48', 0], 'guider': ['54', 0], 'sampler': ['49', 0],
                                                 'sigmas': ['52', 0], 'latent_image': ['47', 0]})
        self.assertEqual(graph['41']['class_type'], 'VAEDecode')
        self.assertEqual(graph['41']['inputs'], {'samples': ['55', 1], 'vae': ['35', 0]})
        self.assertEqual(graph['56']['class_type'], 'INPAINT_ColorMatch')
        self.assertEqual(graph['56']['inputs'], {'target': ['41', 0], 'reference': ['43', 0],
                                                 'exclude_mask': ['27', 0], 'strength': 1.0})
        self.assertEqual(graph['10']['inputs']['images'], ['56', 0])
        for key in ('5', '6', '32', '39', '40'):
            self.assertNotIn(key, graph)
        classes = {node['class_type'] for node in graph.values()}
        self.assertNotIn('ControlNetLoader', classes)
        self.assertNotIn('ControlNetApplyAdvanced', classes)
        self.assertNotIn('VAEEncodeForInpaint', classes)
        self.assertNotIn('InpaintPreprocessor', classes)
        self.assertNotIn('InpaintModelConditioning', classes)
        self.assertNotIn('ImageCompositeMasked', classes)
        # An empty stack leaves the sampler chain on the raw UNet.
        graph = backend.workflow(dict(zit, loras=[]), 'in.png', 'mask.png', 'depth.png', caps=caps)
        self.assertNotIn('50', graph)
        self.assertEqual(graph['45']['inputs']['model'], ['30', 0])
        self.assertEqual(graph['25']['inputs']['model'], ['45', 0])
        # No depth guidance: the Fun ControlNet keeps running in inpaint mode
        # as the context provider, just without a control image.
        graph = backend.workflow(dict(zit, loras=[], depth_enabled=False),
                                  'in.png', 'mask.png', None, caps=caps)
        self.assertNotIn('12', graph)
        self.assertIsNone(graph['45']['inputs']['image'])
        self.assertEqual(graph['45']['inputs']['model'], ['30', 0])
        # Below full denoise the pipeline refines the existing pixels: no
        # pre-fill, but the Fun ControlNet keeps steering the refinement, so
        # depth guidance and Control strength stay effective at any denoise;
        # the sigma schedule still drops its strongest step.
        refine = dict(zit, loras=[], denoise=0.6)
        backend.validate(refine, caps)
        graph = backend.workflow(refine, 'in.png', 'mask.png', 'depth.png', caps=caps)
        for key in ('27', '33', '42', '43'):
            self.assertNotIn(key, graph)
        self.assertEqual(graph['45']['inputs']['model'], ['30', 0])
        self.assertEqual(graph['45']['inputs']['image'], ['12', 0])
        self.assertEqual(graph['45']['inputs']['mask'], ['28', 0])
        self.assertEqual(graph['28']['inputs'], {'mask': ['4', 0], 'value': 0.5})
        self.assertEqual(graph['25']['inputs']['model'], ['45', 0])
        self.assertEqual(graph['46']['inputs']['pixels'], ['36', 0])
        self.assertEqual(graph['47']['inputs']['mask'], ['4', 0])
        self.assertEqual(graph['52']['inputs']['denoise'], 0.6)
        self.assertEqual(graph['53']['class_type'], 'SplitSigmas')
        self.assertEqual(graph['53']['inputs'], {'sigmas': ['52', 0], 'step': 1})
        self.assertEqual(graph['55']['inputs']['sigmas'], ['53', 1])
        self.assertEqual(graph['56']['inputs']['reference'], ['36', 0])
        self.assertEqual(graph['56']['inputs']['exclude_mask'], ['4', 0])
        # Refine selections validate the ControlNet patch like full-denoise ones.
        with self.assertRaisesRegex(ValueError, 'ControlNet patch'):
            backend.validate(dict(refine, zit_controlnet='missing.safetensors'), caps)
        # Without a selection the whole frame updates through the plain
        # img2img latent: VAE encode plus a whole-frame noise mask.
        graph = backend.workflow(dict(zit, loras=[], masked=False),
                                 'in.png', 'mask.png', 'depth.png', caps=caps)
        self.assertNotIn('25', graph)
        self.assertNotIn('28', graph)
        self.assertEqual(graph['5']['class_type'], 'VAEEncode')
        self.assertEqual(graph['5']['inputs']['pixels'], ['36', 0])
        self.assertEqual(graph['6']['class_type'], 'SetLatentNoiseMask')
        self.assertEqual(graph['6']['inputs']['mask'], ['4', 0])
        self.assertEqual(graph['40']['inputs']['latent_image'], ['6', 0])
        self.assertEqual(graph['40']['inputs']['positive'], ['38', 0])
        self.assertEqual(graph['10']['inputs']['images'], ['41', 0])
        # The worker uploads input, mask and depth: every ZIT graph consumes
        # the mask and depth rides on the prepared file.
        backend.publish(self.directory, 'request.json', dict(operation='generate', settings=zit,
                        server=f'http://127.0.0.1:{self.server.server_port}'))
        (self.directory / 'depth.png').write_bytes(b'fixture depth')
        backend.run(self.directory)
        self.assertEqual(sum(path == '/upload/image' for _, path, _ in self.calls), 3)
        graph = next(body['prompt'] for _, path, body in self.calls if path == '/prompt')
        self.assertEqual(graph['45']['class_type'], 'ZImageFunControlnet')
        self.assertFalse(any(path == '/interrupt' for _, path, _ in self.calls))

    def test_sdxl_lora_stack(self):
        self.info['LoraLoaderModelOnly'] = {'input': {'required': {'lora_name': [[
            'first.safetensors', 'second.safetensors']]}}}
        caps = backend.capabilities(self.info)
        settings = dict(self.settings, loras=[
            {'name': 'first.safetensors', 'strength': 0.25},
            {'name': 'second.safetensors', 'strength': -0.5},
        ])
        backend.validate(settings, caps)
        graph = backend.workflow(settings, 'in.png', 'mask.png')
        self.assertEqual(graph['51']['inputs'], {
            'model': ['50', 0], 'lora_name': 'second.safetensors', 'strength_model': -0.5})
        self.assertEqual(graph['50']['inputs']['model'], ['1', 0])
        self.assertEqual(graph['9']['inputs']['model'], ['51', 0])
        with self.assertRaisesRegex(ValueError, 'LoRA is not available'):
            backend.validate(dict(settings, loras=[{'name': 'missing.safetensors', 'strength': 1.0}]), caps)

    def test_batch_submits_one_prompt_per_seed_and_cancels_all_together(self):
        # A complete batch: every candidate gets its own prompt and seed, the
        # uploads are shared, results arrive as result-<i>.png with progress,
        # and nothing needs deleting afterwards.
        # Batch counts carry no fixed upper limit.
        backend.validate(dict(self.settings, batch=32, seeds=['1', '2']),
                         backend.capabilities(self.info))
        self.mode = 'batch'
        settings = dict(self.settings, seeds=['11', '22', '33'])
        backend.publish(self.directory, 'request.json', dict(operation='generate', settings=settings,
                        server=f'http://127.0.0.1:{self.server.server_port}'))
        backend.run(self.directory)
        prompts = [body for _, path, body in self.calls if path == '/prompt']
        self.assertEqual(len(prompts), 6)
        self.assertEqual([body['prompt']['9']['inputs']['seed'] for body in prompts[:3]], [11, 22, 33])
        for body in prompts[3:]:
            self.assertEqual(body['prompt']['20']['class_type'], 'ChordMaterialEstimation')
            self.assertIn('11', body['prompt'])
        self.assertEqual(sum(path == '/upload/image' for _, path, _ in self.calls), 5)
        self.assertFalse(any(path == '/queue' for _, path, _ in self.calls))
        for index in range(3):
            self.assertEqual((self.directory / f'result-{index}.png').read_bytes(), b'png bytes')
            self.assertEqual((self.directory / f'original-{index}.png').read_bytes(), b'png bytes')
        self.assertEqual(json.loads((self.directory / 'progress.json').read_text()),
                          {'stage': 'Estimating albedo + normal', 'done': 3, 'total': 3})
        self.assertEqual(json.loads((self.directory / 'done.json').read_text())['seeds'], ['11', '22', '33'])
        # Cancelling mid-batch: the first image finished, the second was queued.
        # One /queue call must retire every prompt this worker owns together.
        self.temp.cleanup()
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.mode = 'batch_cancel'
        self.prompts = 0
        for name in ('input.png', 'mask.png'):
            (self.directory / name).write_bytes(b'fixture PNG bytes')
        backend.publish(self.directory, 'request.json', dict(operation='generate', settings=settings,
                        server=f'http://127.0.0.1:{self.server.server_port}'))
        backend.run(self.directory)
        self.assertFalse(self.directory.exists())
        deletes = [body for _, path, body in self.calls if path == '/queue']
        self.assertEqual(deletes[-1:], [{'delete': ['own-prompt-1', 'own-prompt-2']}])
        self.assertFalse(any(path == '/interrupt' for _, path, _ in self.calls))


    def test_estimate_downloads_both_maps_without_mask(self):
        # Estimation is a separate operation: one upload (the composite), one
        # prompt, both map outputs collected, no queue cleanup afterwards.
        for name, payload in {name: {} for name in backend.CHORD_REQUIRED}.items():
            self.info[name] = payload
        # New-style combo serialization: ['COMBO', {'options': [...]}].
        self.info['ChordLoadModel'] = {'input': {'required': {'ckpt_name': [
            'COMBO', {'options': ['sdxl_base.safetensors', 'chord_v1.safetensors']}]}}}
        caps = backend.capabilities(self.info)
        self.assertEqual(caps['chord'], 'chord_v1.safetensors')
        self.mode = 'estimate'
        settings = dict(chord='chord_v1.safetensors', tile=1024, overlap=128)
        backend.publish(self.directory, 'request.json', dict(operation='estimate', settings=settings,
                        server=f'http://127.0.0.1:{self.server.server_port}'))
        backend.run(self.directory)
        self.assertEqual(sum(path == '/upload/image' for _, path, _ in self.calls), 1)
        self.assertEqual(sum(path == '/prompt' for _, path, _ in self.calls), 1)
        self.assertFalse(any(path == '/queue' for _, path, _ in self.calls))
        self.assertEqual((self.directory / 'result-basecolor.png').read_bytes(), b'png bytes')
        self.assertEqual((self.directory / 'result-normal.png').read_bytes(), b'png bytes')
        self.assertEqual(json.loads((self.directory / 'done.json').read_text())['maps'],
                         ['result-basecolor.png', 'result-normal.png'])
        graph = next(body['prompt'] for _, path, body in self.calls if path == '/prompt')
        self.assertEqual(len(graph), 9)
        self.assertEqual(graph['1'], {'class_type': 'ChordLoadModel',
                                      'inputs': {'ckpt_name': 'chord_v1.safetensors'}})
        self.assertEqual(graph['13'], {'class_type': 'GetImageSize', 'inputs': {'image': ['2', 0]}})
        self.assertEqual(graph['7'], {'class_type': 'SplitImageToTileList',
                                      'inputs': {'image': ['2', 0], 'tile_width': 1024,
                                                 'tile_height': 1024, 'overlap': 128}})
        self.assertEqual(graph['20'], {'class_type': 'ChordMaterialEstimation',
                                       'inputs': {'chord_model': ['1', 0], 'image': ['7', 0]}})
        for key, slot in (('8', 0), ('9', 1)):
            self.assertEqual(graph[key], {'class_type': 'ImageMergeTileList',
                                          'inputs': {'image_list': ['20', slot], 'overlap': 128,
                                                     'final_width': ['13', 0], 'final_height': ['13', 1]}})
        self.assertEqual(graph['10'], {'class_type': 'PreviewImage', 'inputs': {'images': ['8', 0]}})
        self.assertEqual(graph['11'], {'class_type': 'PreviewImage', 'inputs': {'images': ['9', 0]}})

    def test_estimate_without_chord_model_fails_before_upload(self):
        del self.info['ChordLoadModel']
        caps = backend.capabilities(self.info)
        self.assertIsNone(caps['chord'])
        backend.publish(self.directory, 'request.json',
                        dict(operation='estimate', settings=dict(chord='chord_v1.safetensors', tile=1024,
                                                                 overlap=128),
                             server=f'http://127.0.0.1:{self.server.server_port}'))
        backend.run(self.directory)
        self.assertIn('Chord', json.loads((self.directory / 'done.json').read_text())['error'])
        self.assertFalse(any(method == 'POST' for method, _, _ in self.calls))

    def test_generation_without_chord_fails_before_upload(self):
        del self.info['ChordLoadModel']
        backend.run(self.directory)
        self.assertIn('Chord', json.loads((self.directory / 'done.json').read_text())['error'])
        self.assertFalse(any(method == 'POST' for method, _, _ in self.calls))

    def test_cancel_during_chord_retires_both_phases(self):
        self.mode = 'batch_cancel'
        self.cancel_at = 2  # single generation finished; Chord POST is in flight
        backend.run(self.directory)
        self.assertFalse(self.directory.exists())
        deletes = [body for _, path, body in self.calls if path == '/queue']
        self.assertEqual(deletes, [{'delete': ['own-prompt-1', 'own-prompt-2']}])
        self.assertFalse(any(path in ('/interrupt', '/free') for _, path, _ in self.calls))

    def test_chord_failure_never_publishes_a_reviewable_result(self):
        self.mode = 'chord_error'
        backend.run(self.directory)
        self.assertIn('out of memory', json.loads((self.directory / 'done.json').read_text())['error'])
        self.assertTrue((self.directory / 'original.png').exists())
        self.assertFalse((self.directory / 'result.png').exists())
        self.assertEqual([body for _, path, body in self.calls if path == '/queue'],
                         [{'delete': ['own-prompt-1', 'own-prompt-2']}])

    def test_chord_missing_albedo_is_an_error(self):
        self.mode = 'chord_missing'
        backend.run(self.directory)
        self.assertIn('without paired albedo and normal', json.loads((self.directory / 'done.json').read_text())['error'])
        self.assertFalse((self.directory / 'result.png').exists())

    def test_estimate_cancellation_deletes_only_own_prompt(self):
        for name in backend.CHORD_REQUIRED:
            self.info[name] = {'input': {'required': {'ckpt_name': [['chord_v1.safetensors']]}}} \
                if name == 'ChordLoadModel' else {}
        self.mode = 'cancel'
        backend.publish(self.directory, 'request.json',
                        dict(operation='estimate', settings=dict(chord='chord_v1.safetensors', tile=1024,
                                                                 overlap=128),
                             server=f'http://127.0.0.1:{self.server.server_port}'))
        backend.run(self.directory)
        self.assertFalse(self.directory.exists())
        deletes = [body for _, path, body in self.calls if path == '/queue']
        self.assertEqual(deletes, [{'delete': ['own-prompt']}])
        self.assertFalse(any(path == '/interrupt' for _, path, _ in self.calls))


    def test_guidance_tables_declare_concepts_per_adapter(self):
        # The Guidance panel renders one generic section from these tables,
        # so an adapter that does not declare a concept never shows it.
        sdxl = backend.guidance_for('SDXL')
        zit = backend.guidance_for('ZIT')
        sdxl_features = {section['concept'] for section in sdxl if section['kind'] == 'feature'}
        zit_features = {section['concept'] for section in zit if section['kind'] == 'feature'}
        self.assertEqual(sdxl_features, {'depth', 'reference'})
        self.assertEqual(zit_features, {'depth'})
        sdxl_depth = next(s for s in sdxl if s.get('concept') == 'depth')
        zit_depth = next(s for s in zit if s.get('concept') == 'depth')
        self.assertEqual(sdxl_depth['picker'], 'controlnet_model')
        self.assertEqual(sdxl_depth['choices'], 'controlnet_models')
        self.assertEqual(sdxl_depth['strength'], 'depth_strength')
        self.assertEqual(sdxl_depth['preview'], 'pawprint.preview_depth')
        self.assertEqual(zit_depth['picker'], 'zit_controlnet')
        self.assertEqual(zit_depth['choices'], 'zit_controlnets')
        self.assertEqual(zit_depth['strength'], 'zit_strength')
        self.assertEqual(zit_depth['preview'], 'pawprint.preview_depth')
        self.assertEqual(zit_depth['source'], 'Source: Geometry')
        sdxl_reference = next(s for s in sdxl if s.get('concept') == 'reference')
        self.assertEqual(sdxl_reference['image'], 'ipadapter_image')
        self.assertEqual(sdxl_reference['weight'], 'ipadapter_weight')
        for table, adapter in ((sdxl, 'SDXL'), (zit, 'ZIT')):
            self.assertEqual(sum(s['kind'] == 'status' for s in table), 1, adapter)
            self.assertEqual(sum(s['kind'] == 'note' for s in table), 1, adapter)
            self.assertIn('masked', next(s for s in table if s['kind'] == 'status'))
            self.assertIn('text', next(s for s in table if s['kind'] == 'note'))


if __name__ == '__main__':
    unittest.main()
