# SPDX-License-Identifier: GPL-3.0-or-later
"""UV-base commits through an isolated emission bake into a new native image."""

import bpy
import bmesh

from . import model, projection, painting, generation


def bake_base(context, stack, last):
    """Return a packed replacement base; leave all original pixels/data untouched."""
    owner = context.object
    base = stack.base
    if not base or base.source not in {'FILE', 'GENERATED'} or min(base.size) < 1:
        raise ValueError('The base must be a loaded, single UV image')
    if not stack.uv_name or stack.uv_name not in owner.data.uv_layers:
        raise ValueError('The saved base UV map is missing')
    for layer in list(stack.layers)[:last + 1]:
        if layer.visible and (not layer.image or not layer.depth):
            raise ValueError('A visible layer is missing its image or visibility snapshot')
    scene = mesh = obj = material = image = None
    window = context.window
    original_scene = window.scene
    original_view_layer = window.view_layer
    success = False
    try:
        # Evaluate in the original scene before creating the isolated bake scene.
        evaluated = owner.evaluated_get(context.evaluated_depsgraph_get())
        mesh = bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True,
                                              depsgraph=context.evaluated_depsgraph_get())
        bm = bmesh.new()
        try:
            bm.from_mesh(mesh)
            bmesh.ops.delete(bm, geom=[face for face in bm.faces
                                      if face.material_index != owner.active_material_index], context='FACES')
            bm.to_mesh(mesh)
        finally:
            bm.free()
        if not mesh.polygons or stack.uv_name not in mesh.uv_layers:
            raise ValueError('The target slot has no evaluated faces with the saved UV map')
        material = stack.id_data.copy()
        material.name = 'Pawprint Temporary Bake Material'
        temporary = material.pawprint
        for index in range(len(temporary.layers) - 1, last, -1):
            temporary.layers.remove(index)
        projection.build_material(material)
        # Copy rather than overwrite: reference replacement + layer removal is
        # ordinary datablock undo, independent of direct image-pixel undo.
        image = base.copy()
        image.name = 'Pawprint Committed Base'
        image.filepath_raw = ''
        image.use_fake_user = False
        # Resample to the configured base resolution; Image.scale keeps content.
        if tuple(image.size) != (context.scene.pawprint_base_width, context.scene.pawprint_base_height):
            image.scale(context.scene.pawprint_base_width, context.scene.pawprint_base_height)
        node = material.node_tree.nodes.new('ShaderNodeTexImage')
        node.image = image
        material.node_tree.nodes.active = node
        mesh.materials.clear()
        mesh.materials.append(material)
        for face in mesh.polygons:
            face.material_index = 0
        scene = bpy.data.scenes.new('Pawprint Temporary Bake')
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'CPU'
        scene.cycles.samples = 1
        obj = bpy.data.objects.new('Pawprint Temporary Bake Target', mesh)
        obj.matrix_world = owner.matrix_world.copy()
        scene.collection.objects.link(obj)
        window.scene = scene
        view_layer = window.view_layer
        view_layer.objects.active = obj
        obj.select_set(True)
        view_layer.update()
        with context.temp_override(scene=scene, view_layer=view_layer, object=obj,
                                   active_object=obj, selected_objects=[obj], selected_editable_objects=[obj]):
            outcome = bpy.ops.object.bake(type='EMIT', target='IMAGE_TEXTURES',
                                          use_selected_to_active=False, use_clear=False,
                                          margin=8, margin_type='EXTEND', uv_layer=stack.uv_name)
        if outcome != {'FINISHED'}:
            raise RuntimeError('UV emission bake did not finish')
        image.pack()
        success = True
        return image
    finally:
        window.scene = original_scene
        window.view_layer = original_view_layer
        if obj:
            bpy.data.objects.remove(obj, do_unlink=True)
        if mesh:
            bpy.data.meshes.remove(mesh)
        if material:
            bpy.data.materials.remove(material)
        if scene:
            bpy.data.scenes.remove(scene)
        if image and not success:
            bpy.data.images.remove(image)


class PAWPRINT_OT_commit_base(bpy.types.Operator):
    bl_idname = 'pawprint.commit_base'
    bl_label = 'Commit Through Selected to Base'
    bl_description = ('Bake the base and visible layers through the selected layer into UV space '
                      'at the configured base resolution, then remove that bottom group (including '
                      'hidden layers), or keep its sources hidden when the stack requests preservation; '
                      'higher layers remain editable')
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        stack = model.active_stack(context)
        return bool(not painting.active() and not generation.active() and stack and stack.base
                    and model.active_layer(stack) and context.object and context.object.mode == 'OBJECT')

    def execute(self, context):
        stack = model.active_stack(context)
        last = stack.active_index
        try:
            image = bake_base(context, stack, last)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        stack.base = image
        if stack.commit_preserve:
            # Hidden sources avoid double compositing; removal stays available
            # by leaving the option off.
            for layer in list(stack.layers)[:last + 1]:
                layer.visible = False
            message = f'Committed {last + 1} layers to {image.size[0]} × {image.size[1]} UV base; sources kept hidden'
        else:
            model.remember_last(stack, stack.layers[last])
            for index in range(last, -1, -1):
                stack.layers.remove(index)
            stack['active_index'] = 0
            stack.preview = not bool(stack.layers)
            message = f'Committed {last + 1} layers to {image.size[0]} × {image.size[1]} UV base'
        projection.build_material(stack.id_data)
        model.redraw(stack, context)
        self.report({'INFO'}, message)
        return {'FINISHED'}


CLASSES = (PAWPRINT_OT_commit_base,)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
