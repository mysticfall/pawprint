# SPDX-License-Identifier: GPL-3.0-or-later

import bpy

from .model import active_stack, active_layer
from . import painting, backend, generation


class PAWPRINT_UL_layers(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.prop(item, 'visible', text='', icon='HIDE_OFF' if item.visible else 'HIDE_ON', emboss=False)
        layout.prop(item, 'name', text='', emboss=False, icon='IMAGE_DATA' if item.image else 'ERROR')


class PAWPRINT_PT_context(bpy.types.Panel):
    bl_label = "Pawprint"
    bl_idname = "PAWPRINT_PT_context"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Pawprint"

    def draw(self, context):
        layout = self.layout
        if painting.active():
            stack = active_stack(context)
            layout.label(text=f"Painting: {stack.owner.name}" if stack else "Finishing painting")
            return

        obj = context.active_object
        if obj is None:
            layout.label(text="Select a mesh object to begin.", icon="MESH_DATA")
            return

        if obj.type != "MESH":
            layout.label(text="Texture layers will target mesh objects.")
            return

        if not obj.material_slots:
            layout.label(text="No material slots assigned.", icon="MATERIAL")
            return

        index = obj.active_material_index
        layout.label(text=f"{obj.name} · Slot {index + 1}", icon="MATERIAL")
        material = obj.active_material
        if material is None:
            layout.label(text="The active material slot is empty.")


class PAWPRINT_PT_layers(bpy.types.Panel):
    bl_label = "Projection Layers"
    bl_idname = "PAWPRINT_PT_layers"
    bl_parent_id = "PAWPRINT_PT_context"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Pawprint"

    def draw(self, context):
        layout = self.layout
        stack = active_stack(context)
        if painting.active():
            layout.label(text="Saved view painting")
            layout.label(text="Navigation is free; strokes restore the saved view.", icon='INFO')
            layer = active_layer(stack)
            if layer:
                layout.label(text=layer.name)
                if layer.selection_paths != '[]':
                    layout.label(text='Painting within image selection', icon='SELECT_SET')
                if not layer.visible:
                    layout.label(text="This layer is hidden", icon='INFO')
            row = layout.row(align=True)
            row.operator('pawprint.paint_blend', text='Paint').erase = False
            row.operator('pawprint.paint_blend', text='Erase Alpha').erase = True
            if layer and layer.selection_paths != '[]':
                layout.operator('pawprint.delete_selected_pixels', icon='TRASH')
            layout.label(text="Brush settings are in the native toolbar.")
            layout.operator('pawprint.finish_paint', icon='CHECKMARK')
            layout.label(text="Escape also finishes painting.")
            return
        if stack is None:
            layout.operator("pawprint.create_stack", icon='ADD')
            layout.label(text="Requires a UV map and material slot.")
            return
        row = layout.row()
        row.template_list('PAWPRINT_UL_layers', '', stack, 'layers', stack, 'active_index',
                             rows=4, sort_reverse=True)
        row = row.column(align=True)
        row.operator('pawprint.remove_layer', text='', icon='REMOVE')
        up = row.row(align=True)
        up.enabled = stack.active_index < len(stack.layers) - 1
        up.operator('pawprint.move_layer', text='', icon='TRIA_UP').direction = 1
        down = row.row(align=True)
        down.enabled = stack.active_index > 0
        down.operator('pawprint.move_layer', text='', icon='TRIA_DOWN').direction = -1
        layer = active_layer(stack)
        if layer:
            if not layer.image:
                layout.label(text="Assign an image in Layer Details.", icon='ERROR')
            row = layout.row(align=True)
            row.operator('pawprint.paint_layer', text='Paint', icon='BRUSH_DATA')
            row.operator('pawprint.restore_view', text='Restore', icon='VIEW_CAMERA')
            row.operator('pawprint.open_image', text='Image', icon='IMAGE_DATA')
            row = layout.row(align=True)
            row.operator('pawprint.commit_base', icon='RENDER_STILL')
            row.prop(stack, 'commit_preserve', text='Keep', toggle=False)
            layout.row(align=True).prop(layer, 'selection_operation', expand=True)
            row = layout.row(align=True)
            tools = row.row(align=True)
            tools.operator_context = 'INVOKE_DEFAULT'
            for shape, label in [('LASSO', 'Lasso'), ('BOX', 'Box')]:
                op = tools.operator('pawprint.select_lasso', text=label)
                op.shape = shape
                op.operation = layer.selection_operation
            row.operator('pawprint.selection_clear', text='Clear')


class PAWPRINT_PT_generation(bpy.types.Panel):
    bl_label = "Generation"
    bl_idname = "PAWPRINT_PT_generation"
    bl_parent_id = "PAWPRINT_PT_context"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Pawprint"

    def draw(self, context):
        layer = active_layer(active_stack(context))
        layout = self.layout
        info = generation.review_info()
        if info:
            column = layout.column(align=True)
            seed = info['seed']
            column.label(text=f"Candidate {info['index'] + 1}/{info['count']} · seed "
                              f"{seed if len(seed) <= 16 else seed[:15] + '…'}")
            row = column.row(align=True)
            row.operator('pawprint.candidate_select', text='Previous').direction = -1
            row.operator('pawprint.candidate_select', text='Next').direction = 1
            row = column.row(align=True)
            row.operator('pawprint.commit_candidate', text='Apply', icon='CHECKMARK')
            row.operator('pawprint.candidate_layer', text='Layer', icon='ADD')
            row.operator('pawprint.discard_candidates', text='Discard', icon='X')
            column.label(text='The viewport stays interactive while reviewing.', icon='INFO')
        column = layout.column(align=True)
        column.enabled = not generation.active()
        column.prop(context.scene, 'pawprint_server', text='Server')
        column.operator('pawprint.connect', icon='FILE_REFRESH')
        if layer:
            column = layout.column(align=True)
            column.enabled = not painting.active() and not generation.active()
            column.prop(layer, 'generation_adapter')
            for key, spec in backend.parameters_for(layer.generation_adapter).items():
                if key.startswith(('depth_', 'ipadapter_', 'inpaint_')) or key == 'controlnet_model':
                    continue
                prop = 'generation_' + key
                if spec.get('choices'):
                    column.prop_search(layer, prop, context.window_manager, 'pawprint_' + spec['choices'])
                elif key in {'positive', 'negative'}:
                    column.label(text=spec['name'])
                    row = column.row(align=True)
                    row.prop(layer, prop, text='')
                    row.menu('PAWPRINT_MT_' + key + '_history', text='', icon='DOWNARROW_HLT')
                else:
                    column.prop(layer, prop)
            if layer.generation_last_seed:
                column.label(text='Last seed: ' + layer.generation_last_seed)
            column.prop(layer, 'generation_feather')
            column.prop(layer, 'generation_context')
            row = column.row()
            row.enabled = layer.generation_context == 'SELECTION' and layer.selection_paths != '[]'
            row.prop(layer, 'generation_padding')
            column.prop(layer, 'generation_preview')
            column.operator('pawprint.preview_context', icon='IMAGE_DATA')
            column.operator('pawprint.generate', icon='RENDER_STILL')
        if generation.active():
            layout.operator('pawprint.cancel_generation', icon='CANCEL')
        # Wrap errors/status to remain legible in a narrow sidebar.
        import textwrap
        for line in textwrap.wrap(generation.status(), width=max(24, int(context.region.width / 7))):
            layout.label(text=line)


class PAWPRINT_PT_layer_details(bpy.types.Panel):
    bl_label = 'Layer Details'
    bl_idname = 'PAWPRINT_PT_layer_details'
    bl_parent_id = 'PAWPRINT_PT_context'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Pawprint'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        stack = active_stack(context)
        if not stack:
            return
        layout = self.layout
        layout.enabled = not painting.active()
        layer = active_layer(stack)
        if layer:
            layout.template_ID(layer, 'image')
            layout.label(text=f'Saved frame: {layer.width} × {layer.height}')
            layout.prop(layer, 'depth_tolerance')
            layout.prop(layer, 'mirror')
        layout.label(text='Committed UV base')
        layout.template_ID(stack, 'base')
        row = layout.row(align=True)
        row.prop(context.scene, 'pawprint_base_width')
        row.prop(context.scene, 'pawprint_base_height')
        layout.label(text='Applies to new bases and commits.', icon='INFO')


class PAWPRINT_PT_new_layer(bpy.types.Panel):
    bl_label = 'New Layer'
    bl_idname = 'PAWPRINT_PT_new_layer'
    bl_parent_id = 'PAWPRINT_PT_context'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Pawprint'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        stack = active_stack(context)
        if not stack:
            return
        layout = self.layout
        layout.enabled = not painting.active()
        layout.prop(stack, 'preview')
        row = layout.row(align=True)
        row.prop(stack, 'width')
        row.prop(stack, 'height')
        layout.operator('pawprint.add_projection', icon='ADD')


class PAWPRINT_PT_guidance(bpy.types.Panel):
    bl_label = "Guidance"
    bl_idname = "PAWPRINT_PT_guidance"
    bl_parent_id = "PAWPRINT_PT_context"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Pawprint"

    def draw(self, context):
        layout = self.layout
        layer = active_layer(active_stack(context))
        if not layer:
            layout.label(text="Select a layer to configure guidance.")
            return
        layout.enabled = not painting.active() and not generation.active()
        masked = layer.selection_paths != '[]'
        # One generic renderer draws whatever concepts the active adapter
        # declares; an adapter that does not declare a feature never shows it.
        for index, section in enumerate(backend.guidance_for(layer.generation_adapter)):
            if index:
                layout.separator()
            if section['kind'] == 'status':
                layout.label(text=section['masked'] if masked else section['unmasked'],
                             icon='SELECT_SET' if masked else 'INFO')
                continue
            if section['kind'] == 'note':
                layout.label(text=section['text'], icon='DOT')
                continue
            if section['kind'] == 'loras':
                layout.label(text=section['description'], icon='INFO')
                if generation.connected(context):
                    for entry_index, entry in enumerate(layer.generation_loras):
                        row = layout.row(align=True)
                        row.prop_search(entry, 'model', context.window_manager,
                                        'pawprint_' + section['choices'], text='')
                        row.prop(entry, 'strength', text='', slider=True)
                        remove = row.operator('pawprint.lora_remove', text='', icon='X')
                        remove.index = entry_index
                    layout.operator('pawprint.lora_add', icon='ADD')
                    if not len(getattr(context.window_manager, 'pawprint_' + section['choices'])):
                        layout.label(text=section['empty'], icon='ERROR')
                else:
                    layout.label(text=section['unconnected'], icon='INFO')
                continue
            toggle = 'generation_' + section['toggle']
            layout.prop(layer, toggle)
            if not getattr(layer, toggle):
                continue
            if section.get('source'):
                layout.label(text=section['source'])
            if section.get('picker'):
                if generation.connected(context):
                    layout.prop_search(layer, 'generation_' + section['picker'],
                                       context.window_manager, 'pawprint_' + section['choices'])
                    if not len(getattr(context.window_manager, 'pawprint_' + section['choices'])):
                        layout.label(text=section['empty'], icon='ERROR')
                else:
                    layout.label(text=section['unconnected'], icon='INFO')
            if section.get('image'):
                layout.template_ID(layer, 'generation_' + section['image'],
                                   new='image.new', open='image.open')
                if getattr(layer, 'generation_' + section['image']) is None:
                    layout.label(text=section['missing_image'], icon='ERROR')
            if section.get('weight'):
                layout.prop(layer, 'generation_' + section['weight'])
            if section.get('strength'):
                layout.prop(layer, 'generation_' + section['strength'])
            if section.get('preview'):
                layout.operator(section['preview'], icon='IMAGE_DATA')


def draw_prompt_history(layout, history, negative):
    if not len(history):
        layout.label(text='No prompts yet', icon='INFO')
        return
    for item in reversed(history):
        text = item.text
        label = text if len(text) <= 64 else text[:63] + '…'
        entry = layout.operator('pawprint.apply_prompt', text=label, icon='DOT')
        entry.prompt, entry.negative = text, negative
    layout.separator()
    clear = layout.operator('pawprint.clear_prompt_history', text='Clear history')
    clear.negative = negative


class PAWPRINT_MT_positive_history(bpy.types.Menu):
    bl_label = 'Positive prompt history'

    def draw(self, context):
        draw_prompt_history(self.layout, context.scene.pawprint_positive_history, False)


class PAWPRINT_MT_negative_history(bpy.types.Menu):
    bl_label = 'Negative prompt history'

    def draw(self, context):
        draw_prompt_history(self.layout, context.scene.pawprint_negative_history, True)


CLASSES = (
    PAWPRINT_UL_layers,
    PAWPRINT_MT_positive_history,
    PAWPRINT_MT_negative_history,
    PAWPRINT_PT_context,
    PAWPRINT_PT_layers,
    PAWPRINT_PT_generation,
    PAWPRINT_PT_layer_details,
    PAWPRINT_PT_new_layer,
    PAWPRINT_PT_guidance,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
