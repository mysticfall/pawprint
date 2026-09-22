# SPDX-License-Identifier: GPL-3.0-or-later
"""Generation adapter contract and isolated, bpy-free ComfyUI worker.

Each adapter owns its parameter descriptors, capability discovery, validation
and workflow graph; the UI and request snapshot consume the same descriptions.
The worker runs in a separate factory Blender process, never a background
Python thread in the interactive Blender. Only JSON files and PNG bytes cross
processes.
"""
import json
from pathlib import Path
import time
from urllib import error, parse, request
import uuid


SDXL_PARAMETERS = {
    'checkpoint': dict(kind='STRING', name='SDXL checkpoint', default='', choices='checkpoints'),
    'positive': dict(kind='STRING', name='Positive prompt', default=''),
    'negative': dict(kind='STRING', name='Negative prompt', default=''),
    'denoise': dict(kind='FLOAT', name='Denoise strength', default=0.35, min=0.0, max=1.0),
    'cfg': dict(kind='FLOAT', name='CFG', default=5.0, min=0.0, max=100.0),
    'clip_skip': dict(kind='INT', name='CLIP skip', default=1, min=1, max=24,
                      description='Final CLIP layers to skip before encoding prompts'),
    'seed': dict(kind='STRING', name='Seed', default='731', description='Unsigned 64-bit seed; -1 chooses a random seed for each request'),
    'steps': dict(kind='INT', name='Sampling steps', default=20, min=1, max=10000),
    'sampler': dict(kind='STRING', name='Sampler', default='dpmpp_2m', choices='samplers'),
    'scheduler': dict(kind='STRING', name='Scheduler', default='karras', choices='schedulers'),
    'resolution': dict(kind='INT', name='Context long edge', default=1024, min=64, max=2048,
                       description='Request pixels on the longer crop edge; dimensions round to multiples of eight'),
    'batch': dict(kind='INT', name='Batch', default=1, min=1, soft_max=8,
                  description='Random-seed candidates to generate; above one, review them and pick a result to apply'),
    'depth_enabled': dict(kind='BOOL', name='Depth guidance', default=False),
    'depth_strength': dict(kind='FLOAT', name='Depth strength', default=0.5, min=0.0, max=2.0),
    'ipadapter_enabled': dict(kind='BOOL', name='IPAdapter reference', default=False),
    'ipadapter_model': dict(kind='STRING', name='IPAdapter model',
                            default='ip-adapter-plus_sdxl_vit-h.safetensors', choices='ipadapter_models'),
    'ipadapter_weight': dict(kind='FLOAT', name='IPAdapter weight', default=1.0, min=0.0, max=2.0),
    'controlnet_model': dict(kind='STRING', name='ControlNet model', default='sdxl_promax.safetensors',
                             choices='controlnet_models',
                             description='Single union ControlNet weight shared by depth and inpaint guidance'),
}
# Z Image Turbo is a distilled flow model sampled at CFG 1 with a zeroed
# negative conditioning; guidance flows through a DiffSynth model patch.
ZIT_PARAMETERS = {
    'zit_unet': dict(kind='STRING', name='Z model', default='', choices='zit_unets'),
    'positive': dict(kind='STRING', name='Positive prompt', default=''),
    'negative': dict(kind='STRING', name='Negative prompt', default=''),
    'denoise': dict(kind='FLOAT', name='Denoise strength', default=1.0, min=0.0, max=1.0),
    'cfg': dict(kind='FLOAT', name='CFG', default=1.0, min=0.0, max=100.0),
    'clip_skip': dict(kind='INT', name='CLIP skip', default=1, min=1, max=24,
                      description='Final CLIP layers to skip before encoding prompts'),
    'seed': dict(kind='STRING', name='Seed', default='731', description='Unsigned 64-bit seed; -1 chooses a random seed for each request'),
    'steps': dict(kind='INT', name='Sampling steps', default=8, min=1, max=10000),
    'sampler': dict(kind='STRING', name='Sampler', default='res_multistep', choices='samplers'),
    'scheduler': dict(kind='STRING', name='Scheduler', default='simple', choices='schedulers'),
    'resolution': dict(kind='INT', name='Context long edge', default=1024, min=64, max=2048,
                       description='Request pixels on the longer crop edge; dimensions round to multiples of eight'),
    'batch': dict(kind='INT', name='Batch', default=1, min=1, soft_max=8,
                  description='Random-seed candidates to generate; above one, review them and pick a result to apply'),
    'depth_enabled': dict(kind='BOOL', name='Depth guidance', default=False,
                          description='Steer generation with the saved-view geometry depth map through the DiffSynth ControlNet patch'),
    'zit_controlnet': dict(kind='STRING', name='ControlNet patch', default='Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2601-8steps.safetensors',
                           choices='zit_controlnets'),
    'zit_strength': dict(kind='FLOAT', name='Control strength', default=1.0, min=0.0, max=10.0),
}
# The union of every adapter's parameters backs the shared RNA properties and
# the last-used snapshot; shared keys keep the SDXL defaults.
ALL_PARAMETERS = dict(SDXL_PARAMETERS)
for _key, _spec in ZIT_PARAMETERS.items():
    if _key not in ALL_PARAMETERS:
        ALL_PARAMETERS[_key] = _spec

ADAPTERS = ('SDXL', 'ZIT')


def parameters_for(adapter):
    return ZIT_PARAMETERS if adapter == 'ZIT' else SDXL_PARAMETERS


# Concept-level guidance sections each adapter offers. The Guidance panel
# renders this table generically, so conceptually identical features share one
# presentation while each adapter keeps its own wiring, weights and wording;
# an adapter that does not declare a concept never shows it (selecting Z Image
# Turbo hides every SDXL-only control by construction).
SDXL_GUIDANCE = (
    dict(kind='feature', concept='depth', toggle='depth_enabled',
         picker='controlnet_model', choices='controlnet_models',
         empty='No supported SDXL union ControlNet found',
         unconnected='Connect in Generation to list models',
         source='Source: Geometry', strength='depth_strength',
         preview='pawprint.preview_depth'),
    dict(kind='status', masked='Selection: inpaint regeneration',
         unmasked='No selection: whole-frame img2img'),
    dict(kind='feature', concept='reference', toggle='ipadapter_enabled',
         picker='ipadapter_model', choices='ipadapter_models',
         empty='No supported SDXL IPAdapter model found',
         unconnected='Connect in Generation to list models',
         image='ipadapter_image', missing_image='Assign a reference image to generate',
         weight='ipadapter_weight'),
    dict(kind='loras', concept='loras', choices='loras',
         empty='No LoRAs found on this server',
         unconnected='Connect in Generation to list LoRAs',
         description='Stacked in order on top of the checkpoint; each entry has its own strength'),
    dict(kind='note', text='Planned: edge controls'),
)
ZIT_GUIDANCE = (
    dict(kind='feature', concept='depth', toggle='depth_enabled',
         picker='zit_controlnet', choices='zit_controlnets',
         empty='No ZIT ControlNet patch found',
         unconnected='Connect in Generation to list patches',
         strength='zit_strength'),
    dict(kind='loras', concept='loras', choices='zit_loras',
         empty='No zit LoRAs found on this server',
         unconnected='Connect in Generation to list LoRAs',
         description='Stacked in order on top of the UNet; each entry has its own strength'),
    dict(kind='status', masked='Selection: noise-mask inpainting',
         unmasked='No selection: whole-frame img2img'),
    dict(kind='note', text='Planned: reference guidance for Z Image Turbo'),
)


def guidance_for(adapter):
    return ZIT_GUIDANCE if adapter == 'ZIT' else SDXL_GUIDANCE


def _basename(name):
    return name.replace('\\', '/').split('/')[-1]


def _prefer(names, token):
    matches = [name for name in names if token in _basename(name).lower()]
    return matches[0] if matches else (names[0] if names else None)


# One union ControlNet loader serves depth guidance.
CONTROLNET_NODES = {'ControlNetLoader', 'ControlNetApplyAdvanced', 'SetUnionControlNetType'}
IPADAPTER_NODES = {'IPAdapterModelLoader', 'IPAdapterAdvanced', 'CLIPVisionLoader'}
# Dedicated SDXL IPAdapter weights pair with a specific CLIP Vision encoder.
IPADAPTER_ENCODERS = {'vit-h': 'CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors',
                      'vit-g': 'CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors'}
# Xinsir ControlNet Pro Max SDXL supports the union "depth" type through
# SetUnionControlNetType.
UNION_MODELS = {'sdxl_promax.safetensors'}
REQUIRED = {'CheckpointLoaderSimple', 'LoadImage', 'ImageToMask', 'VAEEncode',
             'SetLatentNoiseMask', 'CLIPSetLastLayer', 'CLIPTextEncode', 'KSampler', 'VAEDecode',
            'PreviewImage', 'InpaintModelConditioning', 'ImageCompositeMasked'}
ZIT_REQUIRED = {'UNETLoader', 'CLIPLoader', 'VAELoader', 'CLIPSetLastLayer', 'CLIPTextEncode',
                'ConditioningZeroOut', 'ModelSamplingAuraFlow', 'VAEEncode',
                'SetLatentNoiseMask', 'KSampler', 'VAEDecode', 'LoadImage', 'PreviewImage',
                'DifferentialDiffusion', 'ZImageFunControlnet', 'INPAINT_ExpandMask',
                'INPAINT_StabilizeMask', 'INPAINT_ColorMatch', 'INPAINT_LoadInpaintModel',
                'INPAINT_InpaintWithModel', 'ThresholdMask', 'SplitSigmas', 'RandomNoise',
                'KSamplerSelect', 'BasicScheduler', 'BasicGuider', 'SamplerCustomAdvanced'}
# DiffSynth ControlNet patches (model_patch) loaded through ModelPatchLoader; the
# Fun ControlNet apply covers inpaint mode (ZImageFunControlnet) and depth
# guidance (QwenImageDiffsynthControlnet).
ZIT_CONTROLNET_NODES = {'ModelPatchLoader', 'QwenImageDiffsynthControlnet', 'ZImageFunControlnet'}


def encoder_for(model):
    name = _basename(model).lower()
    for token, encoder in IPADAPTER_ENCODERS.items():
        if token in name:
            return encoder
    return None


def capabilities(info):
    sdxl_missing = REQUIRED - info.keys()
    zit_missing = ZIT_REQUIRED - info.keys()
    if sdxl_missing and zit_missing:
        raise ValueError('Missing generation nodes: ' + ', '.join(sorted(sdxl_missing | zit_missing)))
    caps = {}
    caps['adapters'] = dict(SDXL=not sdxl_missing, ZIT=not zit_missing)
    caps['checkpoints'] = caps['controlnet_models'] = caps['ipadapter_models'] = []
    caps['zit_unets'] = caps['zit_loras'] = caps['zit_controlnets'] = []
    caps['zit_inpaint'] = None
    caps['loras'] = []
    caps['samplers'] = caps['schedulers'] = []
    caps['ranges'] = {}
    caps['zit_clip'] = caps['zit_vae'] = None
    if not sdxl_missing:
        caps['checkpoints'] = info['CheckpointLoaderSimple']['input']['required']['ckpt_name'][0]
        installed = info['ControlNetLoader']['input']['required']['control_net_name'][0] \
            if 'ControlNetLoader' in info else []
        if CONTROLNET_NODES <= info.keys():
            caps['controlnet_models'] = [name for name in installed if _basename(name) in UNION_MODELS]
        if IPADAPTER_NODES <= info.keys():
            encoders = {_basename(name)
                        for name in info['CLIPVisionLoader']['input']['required']['clip_name'][0]}
            for name in info['IPAdapterModelLoader']['input']['required']['ipadapter_file'][0]:
                encoder = encoder_for(name)
                if encoder and encoder in encoders:
                    caps['ipadapter_models'].append(name)
    if not zit_missing:
        caps['zit_unets'] = info['UNETLoader']['input']['required']['unet_name'][0]
        # Style LoRAs live in the zit folder; the folder prefix is part of the
        # server name, so match it before the basename.
        caps['zit_loras'] = [name for name in info['LoraLoaderModelOnly']['input']['required']['lora_name'][0]
                             if 'zit' in name.lower() or 'z-image' in _basename(name).lower()]
        caps['zit_clip'] = _prefer(info['CLIPLoader']['input']['required']['clip_name'][0], 'qwen_3')
        caps['zit_vae'] = _prefer(info['VAELoader']['input']['required']['vae_name'][0], 'ae')
        if ZIT_CONTROLNET_NODES <= info.keys():
            caps['zit_controlnets'] = info['ModelPatchLoader']['input']['required']['name'][0]
        # The pre-fill stage of full-denoise inpainting picks a dedicated
        # inpaint model the way the CLIP/VAE names resolve. Newer servers
        # expose combos as [type, config] with the options inside the config.
        model_spec = info['INPAINT_LoadInpaintModel']['input']['required']['model_name']
        model_names = model_spec[0] if isinstance(model_spec[0], list) \
            else model_spec[1].get('options', [])
        caps['zit_inpaint'] = _prefer(model_names, 'mat')
    if 'LoraLoaderModelOnly' in info:
        # Every server LoRA is offered to SDXL; the ZIT table filters to zit/
        # weights through caps['zit_loras'] above.
        caps['loras'] = info['LoraLoaderModelOnly']['input']['required']['lora_name'][0]
    if 'KSampler' in info:
        inputs = info['KSampler']['input']['required']
        caps['samplers'] = inputs['sampler_name'][0]
        caps['schedulers'] = inputs['scheduler'][0]
        caps['ranges'] = {key: inputs[key][1] for key in ('steps', 'cfg', 'denoise', 'seed')}
    return caps


def _validate_numeric(settings, caps):
    seed = int(settings['seed'])
    if not 0 <= seed <= 2**64 - 1:
        raise ValueError('Seed must be -1 (random) or an unsigned 64-bit integer')
    for key, limits in caps.get('ranges', {}).items():
        value = seed if key == 'seed' else settings[key]
        if not limits.get('min', value) <= value <= limits.get('max', value):
            raise ValueError(f'{key} is outside the installed server range')


def validate(settings, caps):
    adapter = settings.get('adapter', 'SDXL')
    # The LoRA stack is a shared concept: each adapter validates it against the
    # weight list its family actually loads.
    allowed_loras = caps['zit_loras'] if adapter == 'ZIT' else caps['loras']
    for entry in settings.get('loras', []):
        if entry['name'] not in allowed_loras:
            raise ValueError(f"LoRA is not available on this server: {entry['name']}")
    if adapter == 'ZIT':
        if not caps.get('adapters', {}).get('ZIT'):
            raise ValueError('Z Image Turbo nodes are not installed on this server')
        for key, spec in ZIT_PARAMETERS.items():
            value = settings[key]
            if spec.get('choices') and value not in caps[spec['choices']]:
                raise ValueError(f"{spec['name']} is not available on this server: {value}")
            if 'min' in spec and not spec['min'] <= value <= spec.get('max', value):
                raise ValueError(f"{spec['name']} must be at least {spec['min']}")
        # The Fun ControlNet runs for every masked request (and for unmasked
        # depth guidance); full-denoise inpainting additionally needs a
        # pre-fill inpaint model.
        if (settings.get('masked') or settings.get('depth_enabled')) \
                and settings['zit_controlnet'] not in caps['zit_controlnets']:
            raise ValueError(f"ControlNet patch is not available on this server: {settings['zit_controlnet']}")
        if settings.get('masked') and settings.get('denoise', 1.0) >= 1.0 and not caps.get('zit_inpaint'):
            raise ValueError('Inpaint model is not available on this server')
        _validate_numeric(settings, caps)
        return
    optional = ('depth_', 'ipadapter_')
    for key, spec in SDXL_PARAMETERS.items():
        if key == 'controlnet_model':
            continue
        if any(key.startswith(prefix) and not settings.get(key.rsplit('_', 1)[0] + '_enabled', False)
               for prefix in optional):
            continue
        value = settings[key]
        if spec.get('choices') and value not in caps[spec['choices']]:
            raise ValueError(f"{spec['name']} is not available on this server: {value}")
        if 'min' in spec and not spec['min'] <= value <= spec.get('max', value):
            raise ValueError(f"{spec['name']} must be at least {spec['min']}")
    if settings.get('depth_enabled'):
        if settings['controlnet_model'] not in caps['controlnet_models']:
            raise ValueError(f"ControlNet model is not available on this server: {settings['controlnet_model']}")
    _validate_numeric(settings, caps)


def _sdxl_workflow(settings, image, mask, depth=None, reference=None):
    def node(kind, **inputs):
        return dict(class_type=kind, inputs=inputs)
    graph = {
        '1': node('CheckpointLoaderSimple', ckpt_name=settings['checkpoint']),
        '2': node('LoadImage', image=image),
        '3': node('LoadImage', image=mask),
        '4': node('ImageToMask', image=['3', 0], channel='red'),
        '24': node('CLIPSetLastLayer', clip=['1', 1], stop_at_clip_layer=-settings['clip_skip']),
        '7': node('CLIPTextEncode', clip=['24', 0], text=settings['positive']),
        '8': node('CLIPTextEncode', clip=['24', 0], text=settings['negative']),
        '9': node('KSampler', model=None,
                  seed=int(settings['seed']), steps=settings['steps'],
                  cfg=settings['cfg'], sampler_name=settings['sampler'],
                  scheduler=settings['scheduler'], denoise=settings['denoise']),
        '11': node('VAEDecode', samples=['9', 0], vae=['1', 2]),
        '10': node('PreviewImage', images=['11', 0]),
    }
    # The per-layer LoRA stack chains on top of the checkpoint in listed
    # order; ModelOnly loaders leave the CLIP untouched.
    model = ['1', 0]
    for index, lora in enumerate(settings.get('loras', [])):
        key = str(50 + index)
        graph[key] = node('LoraLoaderModelOnly', model=model,
                          lora_name=lora['name'], strength_model=lora['strength'])
        model = [key, 0]
    graph['9']['inputs']['model'] = model
    latent = None
    positive, negative = ['7', 0], ['8', 0]

    def loader():
        # One shared union ControlNet weight serves the depth apply.
        if '13' not in graph:
            graph['13'] = node('ControlNetLoader', control_net_name=settings['controlnet_model'])
        return ['13', 0]

    if settings.get('depth_enabled'):
        if depth is None:
            raise ValueError('Depth guidance image is missing')
        graph['12'] = node('LoadImage', image=depth)
        graph['23'] = node('SetUnionControlNetType', control_net=loader(), type='depth')
        graph['14'] = node('ControlNetApplyAdvanced', positive=positive, negative=negative,
                           control_net=['23', 0], image=['12', 0], vae=['1', 2],
                           strength=settings['depth_strength'], start_percent=0.0, end_percent=1.0)
        positive, negative = ['14', 0], ['14', 1]
    if settings.get('masked'):
        # A selection means model-level inpainting: InpaintModelConditioning
        # takes the conditioned text (after any depth apply), VAE-encodes the
        # input itself and attaches the noise mask, producing both the final
        # conditioning and the latent. ImageCompositeMasked then rebuilds the
        # output so pixels outside the mask stay exactly the original input.
        graph['25'] = node('InpaintModelConditioning', positive=positive, negative=negative,
                           vae=['1', 2], pixels=['2', 0], mask=['4', 0], noise_mask=True)
        graph['28'] = node('ImageCompositeMasked', destination=['2', 0], source=['11', 0],
                           mask=['4', 0], x=0, y=0, resize_source=False)
        graph['10']['inputs']['images'] = ['28', 0]
        # InpaintModelConditioning emits conditioning (slots 0/1) and the
        # noise-masked latent (slot 2); the sampler takes all three directly.
        positive, negative, latent = ['25', 0], ['25', 1], ['25', 2]
    else:
        # Without a selection the whole frame updates as ordinary img2img; the
        # latent always comes from the rendered composite, even at full denoise.
        graph['5'] = node('VAEEncode', pixels=['2', 0], vae=['1', 2])
        graph['6'] = node('SetLatentNoiseMask', samples=['5', 0], mask=['4', 0])
        latent = ['6', 0]
    graph['9']['inputs'].update(positive=positive, negative=negative, latent_image=latent)
    if settings.get('ipadapter_enabled'):
        if reference is None:
            raise ValueError('IPAdapter reference image is missing')
        graph['15'] = node('IPAdapterModelLoader', ipadapter_file=settings['ipadapter_model'])
        graph['16'] = node('CLIPVisionLoader', clip_name=encoder_for(settings['ipadapter_model']))
        graph['17'] = node('LoadImage', image=reference)
        # Reference conditioning patches the model; spatial depth keeps patching
        # conditioning, so both compose without competing for the same inputs.
        graph['18'] = node('IPAdapterAdvanced', model=model, ipadapter=['15', 0], image=['17', 0],
                           weight=settings['ipadapter_weight'], weight_type='linear',
                           combine_embeds='concat', start_at=0.0, end_at=1.0,
                           embeds_scaling='K+V w/ C penalty', clip_vision=['16', 0])
        graph['9']['inputs'].update(model=['18', 0])
    return graph


def _zit_workflow(settings, image, mask, depth=None, caps=None):
    def node(kind, **inputs):
        return dict(class_type=kind, inputs=inputs)
    graph = {
        '30': node('UNETLoader', unet_name=settings['zit_unet'], weight_dtype='default'),
        '34': node('CLIPLoader', clip_name=(caps or {}).get('zit_clip'), type='lumina2'),
        '35': node('VAELoader', vae_name=(caps or {}).get('zit_vae')),
        '36': node('LoadImage', image=image),
        '3': node('LoadImage', image=mask),
        '4': node('ImageToMask', image=['3', 0], channel='red'),
        '37': node('CLIPSetLastLayer', clip=['34', 0], stop_at_clip_layer=-settings['clip_skip']),
        '38': node('CLIPTextEncode', clip=['37', 0], text=settings['positive']),
    }
    # The per-layer LoRA stack chains on top of the UNet in listed order.
    model = ['30', 0]
    for index, lora in enumerate(settings.get('loras', [])):
        key = str(50 + index)
        graph[key] = node('LoraLoaderModelOnly', model=model,
                          lora_name=lora['name'], strength_model=lora['strength'])
        model = [key, 0]
    if settings.get('masked'):
        # Selections follow the confirmed external reference pipelines. The
        # Fun ControlNet always runs in inpaint mode, feeding the surrounding
        # context (plus an optional depth image) to the sampler. At full
        # denoise a dedicated inpaint model additionally pre-fills the
        # selection; below full denoise the original pixels refine directly
        # with a softened sigma schedule. Both paths re-match colors against
        # their reference outside the selection.
        full = settings['denoise'] >= 1.0
        feather = settings.get('feather', 0)
        graph['26'] = node('INPAINT_ExpandMask', mask=['4', 0], grow=feather,
                           blur=int(feather * 1.7), blur_type='linear')
        noise_mask = exclude = ['26', 0]
        control_mask = ['26', 0]
        reference = ['36', 0]
        if full:
            graph['27'] = node('INPAINT_StabilizeMask', mask=['26', 0], epsilon=0.01)
            graph['28'] = node('ThresholdMask', mask=['27', 0], value=0.0)
            noise_mask = exclude = ['27', 0]
            control_mask = ['28', 0]
            graph['33'] = node('INPAINT_ExpandMask', mask=['4', 0], grow=4, blur=0,
                               blur_type='gaussian')
            graph['42'] = node('INPAINT_LoadInpaintModel',
                               model_name=(caps or {}).get('zit_inpaint'))
            graph['43'] = node('INPAINT_InpaintWithModel', inpaint_model=['42', 0],
                               image=['36', 0], mask=['33', 0], seed=int(settings['seed']))
            reference = ['43', 0]
        graph['44'] = node('ModelPatchLoader', name=settings['zit_controlnet'])
        control_image = None
        if settings.get('depth_enabled'):
            if depth is None:
                raise ValueError('Depth guidance image is missing')
            graph['12'] = node('LoadImage', image=depth)
            control_image = ['12', 0]
        graph['45'] = node('ZImageFunControlnet', model=model, model_patch=['44', 0],
                           vae=['35', 0], image=control_image, inpaint_image=['36', 0],
                           mask=control_mask, strength=settings['zit_strength'])
        model = ['45', 0]
        graph['25'] = node('DifferentialDiffusion', model=model)
        model = ['25', 0]
        graph['46'] = node('VAEEncode', pixels=reference, vae=['35', 0])
        graph['47'] = node('SetLatentNoiseMask', samples=['46', 0], mask=noise_mask)
        graph['48'] = node('RandomNoise', noise_seed=int(settings['seed']))
        graph['49'] = node('KSamplerSelect', sampler_name=settings['sampler'])
        graph['52'] = node('BasicScheduler', model=model, scheduler=settings['scheduler'],
                           steps=settings['steps'], denoise=1.0 if full else settings['denoise'])
        sigmas = ['52', 0]
        if not full:
            # Dropping the strongest sigma softens the refinement start, as in
            # the reference refine workflow.
            graph['53'] = node('SplitSigmas', sigmas=['52', 0], step=1)
            sigmas = ['53', 1]
        graph['54'] = node('BasicGuider', model=model, conditioning=['38', 0])
        graph['55'] = node('SamplerCustomAdvanced', noise=['48', 0], guider=['54', 0],
                           sampler=['49', 0], sigmas=sigmas, latent_image=['47', 0])
        graph['41'] = node('VAEDecode', samples=['55', 1], vae=['35', 0])
        graph['56'] = node('INPAINT_ColorMatch', target=['41', 0], reference=reference,
                           exclude_mask=exclude, strength=1.0)
        graph['10'] = node('PreviewImage', images=['56', 0])
        return graph
    # Without a selection the whole frame updates as ordinary img2img; the
    # latent always comes from the rendered composite, even at full denoise.
    if settings.get('depth_enabled'):
        if depth is None:
            raise ValueError('Depth guidance image is missing')
        graph['12'] = node('LoadImage', image=depth)
        graph['44'] = node('ModelPatchLoader', name=settings['zit_controlnet'])
        graph['45'] = node('QwenImageDiffsynthControlnet', model=model, model_patch=['44', 0],
                           vae=['35', 0], image=['12', 0], strength=settings['zit_strength'])
        model = ['45', 0]
    graph['32'] = node('ModelSamplingAuraFlow', model=model, shift=3.0, sampling='flow')
    graph['39'] = node('ConditioningZeroOut', conditioning=['38', 0])
    graph['5'] = node('VAEEncode', pixels=['36', 0], vae=['35', 0])
    graph['6'] = node('SetLatentNoiseMask', samples=['5', 0], mask=['4', 0])
    graph['40'] = node('KSampler', model=['32', 0], positive=['38', 0], negative=['39', 0],
                       latent_image=['6', 0], seed=int(settings['seed']), steps=settings['steps'],
                       cfg=settings['cfg'], sampler_name=settings['sampler'],
                       scheduler=settings['scheduler'], denoise=settings['denoise'])
    graph['41'] = node('VAEDecode', samples=['40', 0], vae=['35', 0])
    graph['10'] = node('PreviewImage', images=['41', 0])
    return graph


def workflow(settings, image, mask, depth=None, reference=None, caps=None):
    if settings.get('adapter') == 'ZIT':
        # Guidance is a DiffSynth model patch; conditioning stays plain text
        # with a zeroed negative.
        return _zit_workflow(settings, image, mask, depth, caps=caps)
    return _sdxl_workflow(settings, image, mask, depth, reference)


class Client:
    def __init__(self, url):
        if parse.urlsplit(url).scheme not in {'http', 'https'}:
            raise ValueError('Server URL must start with http:// or https://')
        self.url = url.rstrip('/')

    def get(self, path):
        with request.urlopen(self.url + path, timeout=30) as response:
            return response.read()

    def post(self, path, data, content_type='application/json'):
        req = request.Request(self.url + path, data=data, headers={'Content-Type': content_type})
        try:
            with request.urlopen(req, timeout=30) as response:
                payload = response.read()
                return json.loads(payload) if payload else {}
        except error.HTTPError as exc:
            with exc:
                detail = exc.read().decode(errors='replace')
            raise RuntimeError(detail) from exc

    def upload(self, path):
        boundary = uuid.uuid4().hex
        name = f'pawprint-{uuid.uuid4().hex}-{path.name}'
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
                f'filename="{name}"\r\nContent-Type: image/png\r\n\r\n').encode()
        body += path.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
        result = self.post('/upload/image', body, f'multipart/form-data; boundary={boundary}')
        return '/'.join(p for p in (result.get('subfolder', ''), result['name']) if p)


def publish(directory, name, value):
    temporary = directory / (name + '.tmp')
    temporary.write_text(json.dumps(value), encoding='utf8')
    temporary.replace(directory / name)


def run(directory):
    """Own just these prompts. Cancellation never interrupts a shared running job."""
    prompt_ids = []
    client = None
    try:
        config = json.loads((directory / 'request.json').read_text())
        client = Client(config['server'])
        if (directory / 'cancel').exists():
            return
        caps = capabilities(json.loads(client.get('/object_info')))
        if config['operation'] == 'discover':
            publish(directory, 'done.json', dict(capabilities=caps))
            return
        settings = config['settings']
        validate(settings, caps)
        if (directory / 'cancel').exists():
            return
        # Both SDXL and ZIT graphs consume the mask (conditioning/noise mask).
        image = client.upload(directory / 'input.png')
        mask = client.upload(directory / 'mask.png')
        # Upload only the artifacts the client prepared: a layer switched
        # between adapters can still carry depth_enabled without a fresh
        # depth.png for the current request.
        depth = client.upload(directory / 'depth.png') if (directory / 'depth.png').exists() else None
        reference = client.upload(directory / 'reference.png') if (directory / 'reference.png').exists() else None
        if (directory / 'cancel').exists():
            return
        # Batch requests resolve one random seed per candidate client-side;
        # single-image requests keep using the plain seed field.
        seeds = [str(seed) for seed in settings.get('seeds') or [settings['seed']]]
        for index, seed in enumerate(seeds):
            if (directory / 'cancel').exists():
                return
            graph = workflow(dict(settings, seed=seed), image, mask, depth, reference, caps)
            prompt_id = client.post('/prompt', json.dumps({'prompt': graph}).encode())['prompt_id']
            prompt_ids.append(prompt_id)
            if len(seeds) == 1:
                publish(directory, 'queued.json', dict(prompt_id=prompt_id))
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline and not (directory / 'cancel').exists():
                history = json.loads(client.get('/history/' + parse.quote(prompt_id))).get(prompt_id)
                if history:
                    status = history.get('status', {})
                    if status.get('status_str') == 'error':
                        raise RuntimeError(str(status.get('messages', status)))
                    images = history.get('outputs', {}).get('10', {}).get('images', [])
                    if images:
                        name = 'result.png' if len(seeds) == 1 else f'result-{index}.png'
                        (directory / name).write_bytes(client.get('/view?' + parse.urlencode(images[0])))
                        publish(directory, 'progress.json', dict(done=index + 1, total=len(seeds)))
                        break
                    if status.get('completed'):
                        raise RuntimeError('ComfyUI completed without an image')
                time.sleep(0.5)
            else:
                if not (directory / 'cancel').exists():
                    raise TimeoutError(f'Prompt {prompt_id} exceeded 20 minutes; running output will be discarded')
        publish(directory, 'done.json', dict(seeds=seeds))
    except Exception as exc:
        publish(directory, 'done.json', dict(error=str(exc)))
    finally:
        complete = all((directory / ('result.png' if len(prompt_ids) == 1
                                     else f'result-{index}.png')).exists()
                       for index in range(len(prompt_ids)))
        if client and prompt_ids and ((directory / 'cancel').exists() or not complete):
            try:
                client.post('/queue', json.dumps({'delete': prompt_ids}).encode())
            except Exception:
                pass
        if (directory / 'cancel').exists():
            import shutil
            shutil.rmtree(directory, ignore_errors=True)


if __name__ == '__main__':
    import sys
    run(Path(sys.argv[sys.argv.index('--') + 1]))
