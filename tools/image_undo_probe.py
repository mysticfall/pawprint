# SPDX-License-Identifier: GPL-3.0-or-later
"""Run in a disposable Blender: --background --factory-startup --python <this file>.

This is a feasibility probe, not an undo implementation. A datablock sentinel
distinguishes working memfile undo from actual image-buffer restoration.
"""

from array import array
import json

import bpy


class PAWPRINT_OT_pixel_undo_probe(bpy.types.Operator):
    bl_idname = "pawprint.pixel_undo_probe"
    bl_label = "Pawprint pixel undo probe"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        image = bpy.data.images['Pawprint Undo Probe']
        image.pixels.foreach_set(array('f', (0.8, 0.2, 0.1, 1)) * 16)
        image.update()
        context.scene['pawprint_undo_sentinel'] = 1
        return {'FINISHED'}


def sample():
    return {'sentinel': bpy.context.scene['pawprint_undo_sentinel'],
            'red': round(bpy.data.images['Pawprint Undo Probe'].pixels[0], 3)}


bpy.context.preferences.edit.use_global_undo = True
bpy.utils.register_class(PAWPRINT_OT_pixel_undo_probe)
image = bpy.data.images.new('Pawprint Undo Probe', 4, 4, alpha=True)
image.use_fake_user = True
image.pixels.foreach_set(array('f', (0.1, 0.2, 0.3, 1)) * 16)
image.update()
bpy.context.scene['pawprint_undo_sentinel'] = 0
bpy.ops.ed.undo_push(message='Pawprint baseline')
before = sample()
bpy.ops.pawprint.pixel_undo_probe()
bpy.ops.ed.undo_push(message='Pawprint direct pixel edit')
after = sample()
result = {'before': before, 'after': after, 'undo_available': bpy.ops.ed.undo.poll()}
if result['undo_available']:
    bpy.ops.ed.undo()
    result['undo'] = sample()
    bpy.ops.ed.redo()
    result['redo'] = sample()
    assert result['undo']['sentinel'] == 0 and result['redo']['sentinel'] == 1
    result['pixels_restored'] = result['undo']['red'] == before['red']
print('PAWPRINT_IMAGE_UNDO_PROBE ' + json.dumps(result, sort_keys=True))
bpy.utils.unregister_class(PAWPRINT_OT_pixel_undo_probe)
