# SPDX-License-Identifier: GPL-3.0-or-later
"""Derived PBR material: Chord map application and generation/material views.

The pawprint material keeps its baked-lighting composite for precise
inpainting; the derived PBR material is a separate datablock for previewing
and retouching with other painting extensions. Swapping the slot material
between the two is the view mode — painted layers are never destroyed.
"""

import bpy

from . import baking, capture, generation, model


def _painting():
    from . import painting
    return painting.active()


def _map_image(path, name, non_color):
    """Load an estimated map into the stable named datablock, packed.

    The datablock is reused across estimates instead of being removed and
    re-renamed: memfile undo cannot reliably restore node image pointers
    across such swaps, so map identity stays fixed and only the pixels
    change.
    """
    image = bpy.data.images.get(name)
    if image is None:
        image = bpy.data.images.load(str(path), check_existing=False)
        image.name = name
    else:
        if image.packed_file:
            image.unpack(method='REMOVE')
        image.filepath = str(path)
        image.reload()
    image.colorspace_settings.name = 'Non-Color' if non_color else 'sRGB'
    image.pack()
    image.filepath_raw = ''
    return image


def build(stack, basecolor, normal):
    """Create or rebuild the stack's derived PBR material from packed maps."""
    material = stack.pbr_material
    if material is None:
        material = bpy.data.materials.new('Pawprint PBR')
        stack.pbr_material = material
        material.use_nodes = True
        # Drop the default Principled/Output pair; named nodes are built below.
        material.node_tree.nodes.clear()
    material.pawprint_source = stack.id_data
    material.use_nodes = True
    tree = material.node_tree

    def node(blender_type, name, label=None):
        found = tree.nodes.get(name)
        if found is None or found.bl_idname != blender_type:
            found = tree.nodes.new(blender_type)
            found.name = name
        if label is not None:
            found.label = label
        return found

    # Nodes are found by name instead of clearing the tree, keeping node and
    # image identity stable so undo restores wiring without pointer rescramble.
    output = node('ShaderNodeOutputMaterial', 'Pawprint Material Output')
    output.location = (300, 0)
    principled = node('ShaderNodeBsdfPrincipled', 'Pawprint Principled BSDF')
    principled.location = (0, 0)
    uv = node('ShaderNodeUVMap', 'Pawprint UV')
    uv.uv_map = stack.uv_name
    uv.location = (-600, -200)
    albedo = node('ShaderNodeTexImage', 'Pawprint Albedo', 'Albedo')
    albedo.location = (-300, 200)
    albedo.image = _map_image(basecolor, 'Pawprint Albedo', non_color=False)
    normal_tex = node('ShaderNodeTexImage', 'Pawprint Normal', 'Normal')
    normal_tex.location = (-300, -200)
    normal_tex.image = _map_image(normal, 'Pawprint Normal', non_color=True)
    normal_map = node('ShaderNodeNormalMap', 'Pawprint Normal Map')
    normal_map.uv_map = stack.uv_name
    normal_map.location = (0, -200)

    def link(source, target):
        if not any(existing.from_socket == source and existing.to_socket == target
                   for existing in tree.links):
            tree.links.new(source, target)

    link(uv.outputs['UV'], albedo.inputs['Vector'])
    link(uv.outputs['UV'], normal_tex.inputs['Vector'])
    link(albedo.outputs['Color'], principled.inputs['Base Color'])
    link(normal_tex.outputs['Color'], normal_map.inputs['Color'])
    link(normal_map.outputs['Normal'], principled.inputs['Normal'])
    link(principled.outputs['BSDF'], output.inputs['Surface'])
    return material


def _job_stack(job):
    for material in bpy.data.materials:
        if material.as_pointer() == job['material']:
            stack = material.pawprint
            return stack if stack.enabled else None
    return None


def _stack_digest(stack):
    images = [stack.base] + [layer.image for layer in stack.layers]
    return tuple(generation.digest(image) for image in images if image)


_pending = None


def begin_apply(job):
    """Run the undoable apply from the generation timer's context."""
    global _pending
    _pending = job
    try:
        # The explicit undo argument makes script-context calls append a
        # memfile step, matching interactive invocation.
        bpy.ops.pawprint.apply_pbr('EXEC_DEFAULT', True)
    finally:
        _pending = None


class PAWPRINT_OT_estimate_pbr(bpy.types.Operator):
    bl_idname = 'pawprint.estimate_pbr'
    bl_label = 'Estimate Albedo & Normal'
    bl_description = ('Bake the base and layers into UV space, estimate albedo and normal maps '
                      'with Chord on ComfyUI, then rebuild the derived PBR material')
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        from .operators import viewport
        stack = model.active_stack(context)
        return bool(not generation.active() and not generation.review_active()
                    and not _painting() and stack and stack.base
                    and context.object and context.object.mode == 'OBJECT'
                    and generation.connected(context) and generation.chord_ready()
                    and viewport(context)[0])

    def execute(self, context):
        from pathlib import Path
        import tempfile
        stack = model.active_stack(context)
        composite = None
        directory = None
        try:
            composite = baking.bake_composite(context, stack)
            digest = _stack_digest(stack)
            directory = Path(tempfile.mkdtemp(prefix='pawprint-job-'))
            capture.save_reference(directory / 'input.png', composite)
            bpy.data.images.remove(composite)
            composite = None
            generation.launch('estimate', context, directory,
                              settings=dict(chord=generation.chord_model(), tile=1024, overlap=128),
                              owner=context.object.as_pointer(), material=stack.id_data.as_pointer(),
                              slot=context.object.active_material_index,
                              fingerprint=model.fingerprint_stack(stack), digest=digest)
            return {'FINISHED'}
        except Exception as exc:
            if composite is not None:
                bpy.data.images.remove(composite)
            if directory:
                import shutil
                shutil.rmtree(directory, ignore_errors=True)
            generation.set_status(str(exc))
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


class PAWPRINT_OT_apply_pbr(bpy.types.Operator):
    bl_idname = 'pawprint.apply_pbr'
    bl_label = 'Apply PBR Maps'
    bl_description = 'Load the estimated maps and rebuild the derived PBR material'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _pending is not None

    def execute(self, context):
        job = _pending
        if job is None:
            return {'CANCELLED'}
        try:
            stack = _job_stack(job)
            if stack is None:
                raise ValueError('The Pawprint material for this estimate no longer exists')
            if model.fingerprint_stack(stack) != job['fingerprint'] or _stack_digest(stack) != job['digest']:
                raise ValueError('The layer stack changed while estimating; result discarded')
            build(stack, job['directory'] / 'result-basecolor.png', job['directory'] / 'result-normal.png')
            model.redraw(stack, context)
            return {'FINISHED'}
        except Exception as exc:
            generation.set_status('PBR apply: ' + str(exc))
            print('Pawprint: PBR apply:', exc)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


class PAWPRINT_OT_view_mode(bpy.types.Operator):
    bl_idname = 'pawprint.view_mode'
    bl_label = 'Switch Pawprint View'
    bl_description = ('Swap the active slot between the painted Pawprint material (generation view) '
                      'and its derived PBR material (material view)')
    bl_options = {'REGISTER', 'UNDO'}

    material_view: bpy.props.BoolProperty(default=True)

    @classmethod
    def poll(cls, context):
        stack = model.displayed_stack(context)
        if stack is None or _painting() or generation.active() or generation.review_active():
            return False
        if not (context.object and context.object.mode == 'OBJECT'):
            return False
        # Returning to generation view must stay possible even when the PBR
        # pointer was cleared while the slot still shows the old material.
        return model.in_material_view(context) is not None or stack.pbr_material is not None

    def execute(self, context):
        stack = model.displayed_stack(context)
        target = stack.pbr_material if self.material_view else stack.id_data
        if self.material_view and target is None:
            self.report({'ERROR'}, 'Estimate the PBR maps first')
            return {'CANCELLED'}
        obj = context.active_object
        for index, slot in enumerate(obj.material_slots):
            if slot.material is None:
                continue
            if slot.material == stack.id_data or slot.material.pawprint_source == stack.id_data:
                slot.material = target
                if obj.active_material_index != index:
                    obj.active_material_index = index
                model.redraw(stack, context)
                return {'FINISHED'}
        self.report({'ERROR'}, 'The Pawprint material slot was not found')
        return {'CANCELLED'}


CLASSES = (PAWPRINT_OT_estimate_pbr, PAWPRINT_OT_apply_pbr, PAWPRINT_OT_view_mode)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
