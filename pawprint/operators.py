# SPDX-License-Identifier: GPL-3.0-or-later

import bpy
from mathutils import Quaternion

from . import model, projection


def viewport(context):
    if bpy.app.background:
        return None, None
    if context.area and context.area.type == 'VIEW_3D':
        space = context.space_data
        if space.region_quadviews:
            return None, None
        region = next((r for r in context.area.regions if r.type == 'WINDOW'), None)
        return space, region
    return None, None


class PAWPRINT_OT_create_stack(bpy.types.Operator):
    bl_idname = "pawprint.create_stack"
    bl_label = "Create Texture Stack"
    bl_description = "Assign a new unlit Pawprint material to this slot, preserving the original material"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH' or context.mode != 'OBJECT':
            cls.poll_message_set("Select a mesh in Object Mode")
            return False
        if not obj.material_slots or not obj.data.uv_layers:
            cls.poll_message_set("An existing material slot and UV map are required")
            return False
        return model.active_stack(context) is None

    def execute(self, context):
        obj = context.active_object
        slot = obj.material_slots[obj.active_material_index]
        original = slot.material
        material = bpy.data.materials.new(f"Pawprint · {obj.name} · {obj.active_material_index + 1}")
        stack = material.pawprint
        stack.owner = obj
        stack.original_material = original
        stack.uv_name = next((uv.name for uv in obj.data.uv_layers if uv.active_render), obj.data.uv_layers.active.name)
        stack.base = projection.new_image("Pawprint Base", context.scene.pawprint_base_width,
                                          context.scene.pawprint_base_height)
        stack.enabled = True
        projection.build_material(material)
        # Object-linked assignment avoids changing other users of the mesh data.
        slot.link = 'OBJECT'
        slot.material = material
        self.report({'INFO'}, "Created unlit base; use Material Preview or Rendered shading")
        return {'FINISHED'}


class PAWPRINT_OT_add_projection(bpy.types.Operator):
    bl_idname = "pawprint.add_projection"
    bl_label = "Add Layer from View"
    bl_description = "Capture this perspective view and scene visibility; create an empty transparent layer"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        from . import painting
        if painting.active():
            return False
        stack = model.active_stack(context)
        space, region = viewport(context)
        valid = stack and space and region and space.region_3d.view_perspective == 'PERSP'
        if not valid:
            cls.poll_message_set("Create a stack and use a perspective user view")
        return bool(valid) and context.mode == 'OBJECT'

    def execute(self, context):
        stack = model.active_stack(context)
        space, region = viewport(context)
        fields = model.INHERIT_FIELDS + ('generation_ipadapter_image',)
        previous = model.active_layer(stack)
        if previous:
            settings = {key: getattr(previous, key) for key in fields}
            loras = [(entry.model, entry.strength) for entry in previous.generation_loras]
        else:
            settings = model.recall_last(stack)
            # The LoRA stack is a collection and must be copied, not setattr'd.
            loras = settings.pop('generation_loras', [])
            if stack.last_reference:
                settings['generation_ipadapter_image'] = stack.last_reference
        layer = stack.layers.add()
        for key, value in settings.items():
            setattr(layer, key, value)
        for name, strength in loras:
            item = layer.generation_loras.add()
            item.model, item.strength = name, strength
        layer.name = f"Projection {len(stack.layers)}"
        layer.width, layer.height = stack.width, stack.height
        projection.capture_view(layer, space, region)
        layer.depth = projection.depth_image(context, layer, space)
        layer.image = projection.new_image("Pawprint Projection", layer.width, layer.height, transparent=True)
        # Bookkeeping must not trigger the layer-click view restoration.
        stack['active_index'] = len(stack.layers) - 1
        stack.preview = False
        projection.build_material(stack.id_data)
        context.area.tag_redraw()
        return {'FINISHED'}


class PAWPRINT_OT_restore_view(bpy.types.Operator):
    bl_idname = "pawprint.restore_view"
    bl_label = "Restore Layer View"
    bl_description = "Return to the saved eye position and fit the layer frame to this viewport"

    @classmethod
    def poll(cls, context):
        stack = model.active_stack(context)
        return bool(model.active_layer(stack) and viewport(context)[0])

    def execute(self, context):
        owner = model.active_stack(context)
        owner.preview = False
        stack = model.active_layer(owner)
        space, region = viewport(context)
        view = space.region_3d
        view.view_perspective = 'PERSP'
        view.view_rotation = Quaternion(stack.view_rotation)
        view.view_location = stack.view_location
        view.view_distance = stack.view_distance
        space.lens = stack.lens
        space.clip_start, space.clip_end = stack.clip_start, stack.clip_end
        view.update()
        # Refit after a viewport resize without moving the eye or stretching pixels.
        window = projection.matrix(stack.projection) @ projection.matrix(stack.view_matrix).inverted()
        width, _ = projection.frame_size(region, stack.width / stack.height)
        space.lens *= (window[0][0] * width / region.width) / view.window_matrix[0][0]
        view.update()
        context.area.tag_redraw()
        return {'FINISHED'}


class PAWPRINT_OT_open_image(bpy.types.Operator):
    bl_idname = "pawprint.open_image"
    bl_label = "Edit Layer Image Here"
    bl_description = "Switch this area to the native Image Editor; Shift-F5 returns to the 3D Viewport"

    @classmethod
    def poll(cls, context):
        from . import painting
        if painting.active():
            return False
        stack = model.active_layer(model.active_stack(context))
        return bool(stack and stack.image and context.area)

    def execute(self, context):
        image = model.active_layer(model.active_stack(context)).image
        context.area.type = 'IMAGE_EDITOR'
        context.area.spaces.active.image = image
        context.area.spaces.active.ui_mode = 'PAINT'
        return {'FINISHED'}


class PAWPRINT_OT_remove_layer(bpy.types.Operator):
    bl_idname = "pawprint.remove_layer"
    bl_label = "Remove Layer"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        from . import painting
        if painting.active():
            return False
        return model.active_layer(model.active_stack(context)) is not None

    def execute(self, context):
        stack = model.active_stack(context)
        removed = stack.layers[stack.active_index]
        if len(stack.layers) == 1:
            model.remember_last(stack, removed)
        stack.layers.remove(stack.active_index)
        stack['active_index'] = max(0, min(stack.active_index, len(stack.layers) - 1))
        stack.preview = not bool(stack.layers)
        projection.build_material(stack.id_data)
        return {'FINISHED'}


class PAWPRINT_OT_move_layer(bpy.types.Operator):
    bl_idname = "pawprint.move_layer"
    bl_label = "Move Layer"
    bl_options = {'REGISTER', 'UNDO'}
    direction: bpy.props.IntProperty(default=1)

    @classmethod
    def poll(cls, context):
        from . import painting
        if painting.active():
            return False
        return model.active_layer(model.active_stack(context)) is not None

    def execute(self, context):
        stack = model.active_stack(context)
        destination = stack.active_index + self.direction
        if not 0 <= destination < len(stack.layers):
            return {'CANCELLED'}
        stack.layers.move(stack.active_index, destination)
        stack['active_index'] = destination
        projection.build_material(stack.id_data)
        return {'FINISHED'}


CLASSES = (PAWPRINT_OT_create_stack, PAWPRINT_OT_add_projection, PAWPRINT_OT_restore_view,
           PAWPRINT_OT_open_image, PAWPRINT_OT_remove_layer, PAWPRINT_OT_move_layer)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
