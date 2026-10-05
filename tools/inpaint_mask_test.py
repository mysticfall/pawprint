# SPDX-License-Identifier: GPL-3.0-or-later
"""Factory Blender: actual capture/PNG resizing and protected inpaint holes.

Run: blender --background --factory-startup --python-exit-code 1 --python tools/inpaint_mask_test.py
Rendering is stubbed; native image scaling/storage and production preparation run.
"""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from projection_test import load_extension


def rectangle(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def main():
    ext = load_extension()
    image = bpy.data.images.new('Mask fixture', width=512, height=512)
    layer = SimpleNamespace(image=image, generation_padding=8,
                            generation_context='SELECTION', generation_feather=24,
                            selection_paths=json.dumps([
                                ('REPLACE', rectangle(.1, .1, .8, .8)),
                                ('SUBTRACT', rectangle(.35, .35, .55, .55)),
                                ('ADD', rectangle(.9, .9, .95, .95)),
                            ]))
    composite = np.ones((512, 512, 4), dtype=np.float32)
    composite[..., :3] = (.4, .3, .2)
    settings = {key: spec['default'] for key, spec in ext.backend.ALL_PARAMETERS.items()}
    settings.update(masked=True, seed='11', loras=[])
    caps = dict(zit_clip='qwen_3_4b.safetensors', zit_vae='ae.safetensors',
                zit_inpaint='MAT.pth', sdxl_inpaint='MAT.pth')
    target_before = ext.capture.pixels(image).copy()
    with tempfile.TemporaryDirectory(dir='/tmp/opencode', prefix='pawprint-mask-') as temporary:
        directory = Path(temporary)
        for feather in (0, 24):
            layer.generation_feather = feather
            hard = ext.selection.footprint(layer).copy()
            expected = ext.selection.rasterize(json.loads(layer.selection_paths), 512, 512, feather)
            for resolution in (256, 1024):
                with patch.object(ext.capture, 'render_frame', return_value=composite), \
                     patch.object(ext.capture, 'albedo_reference', return_value=composite):
                    metadata = ext.capture.prepare(None, layer, directory, resolution)
                np.testing.assert_array_equal(metadata['weights'], expected)
                np.testing.assert_array_equal(ext.selection.footprint(layer), hard)
                uploaded = bpy.data.images.load(str(directory / 'mask.png'), check_existing=False)
                try:
                    uploaded.colorspace_settings.name = 'Non-Color'
                    mask = ext.capture.pixels(uploaded)[..., 0].copy()
                    assert tuple(uploaded.size) == metadata['request_size']
                finally:
                    bpy.data.images.remove(uploaded)
                x0, y0, x1, y1 = metadata['bounds']
                def sample(u, v):
                    x = int((u * 512 - x0) * mask.shape[1] / (x1 - x0))
                    y = int((v * 512 - y0) * mask.shape[0] / (y1 - y0))
                    return mask[y, x]
                assert sample(.45, .45) == 0, 'Protected hole lost during preparation'
                assert sample(.2, .2) == 1
                assert sample(.925, .925) > 0, 'Disconnected island lost'
                if feather:
                    # A point just inside the hole has soft coverage, but must
                    # remain context, rather than being hidden by a >0 threshold.
                    assert 0 < sample(.36, .45) < .5
                for adapter in ('SDXL', 'ZIT'):
                    for denoise in (.6, 1.0):
                        graph = ext.backend.workflow(dict(settings, adapter=adapter,
                            denoise=denoise, feather=feather), 'input.png', 'mask.png', caps=caps)
                        assert not any(node['class_type'] == 'INPAINT_ExpandMask'
                                       for node in graph.values()), 'Second dilation/blur returned'
                        source = ['27', 0] if denoise == 1.0 else ['4', 0]
                        if adapter == 'ZIT':
                            assert graph['47']['inputs']['mask'] == source
                            control = graph[graph['45']['inputs']['mask'][0]]
                            assert control['inputs'] == {'mask': ['4', 0], 'value': .5}
                            # ThresholdMask uses strict >, then ComfyUI internally
                            # inverts it to keep context. The hole remains visible.
                            keep = 1 - (mask > control['inputs']['value'])
                            assert np.all(keep[mask == 0] == 1)
                            assert np.all(keep[mask < .5] == 1)
                            if denoise == 1:
                                assert graph['43']['inputs']['mask'] == ['28', 0]
                        else:
                            assert graph['21']['inputs']['mask'] == source
                            if denoise == 1:
                                assert graph['33']['inputs'] == {'mask': ['4', 0], 'value': .5}
                # Exercise production patch resizing back to saved-frame space;
                # its final weights remain independent of model binarization.
                returned = ext.capture.returned_pixels(directory / 'input.png', (512, 512), metadata)
                blended = (returned * expected[..., None]
                           + composite * (1 - expected[..., None]))
                np.testing.assert_array_equal(blended[expected == 0], composite[expected == 0])
        np.testing.assert_array_equal(ext.capture.pixels(image), target_before)
        # Finite-support feather is not a promise to preserve arbitrarily small
        # holes: a hole narrower than the kernel can have no zero-weight core.
        tiny = [('REPLACE', rectangle(0, 0, 1, 1)),
                ('SUBTRACT', rectangle(.49, .49, .51, .51))]
        assert ext.selection.rasterize(tiny, 512, 512, 24)[256, 256] > .5
    bpy.data.images.remove(image)
    ext.unregister()
    print('INPAINT MASK TEST OK: holes, feather 0/24, crop resizing, both adapters/modes')


if __name__ == '__main__':
    main()
