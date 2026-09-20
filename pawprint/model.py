# SPDX-License-Identifier: GPL-3.0-or-later
"""Material-owned projection layers and legacy single-projection migration."""

import json

import bpy
from bpy.props import BoolProperty, CollectionProperty, FloatProperty, FloatVectorProperty, IntProperty, PointerProperty, StringProperty


def sync_material(self, context):
    if self.id_data.pawprint.enabled:
        from .projection import build_material
        build_material(self.id_data)


def redraw(self, context):
    if context and context.screen:
        for area in context.screen.areas:
            area.tag_redraw()


def select_layer(self, context):
    redraw(self, context)
    if context and active_stack(context) == self and bpy.ops.pawprint.restore_view.poll():
        bpy.ops.pawprint.restore_view()


class PAWPRINT_PG_stack(bpy.types.PropertyGroup):
    # Legacy image/view fields remain registered so existing .blend files can
    # migrate on load and source-linked development reload.
    enabled: BoolProperty(default=False)
    has_view: BoolProperty(default=False)
    owner: PointerProperty(type=bpy.types.Object)
    original_material: PointerProperty(type=bpy.types.Material)
    base: PointerProperty(type=bpy.types.Image, update=sync_material)
    image: PointerProperty(type=bpy.types.Image, update=sync_material)
    depth: PointerProperty(type=bpy.types.Image)
    uv_name: StringProperty()
    visible: BoolProperty(name="Visible", default=True, update=sync_material)
    width: IntProperty(name="Width", default=512, min=64, max=2048, update=redraw)
    height: IntProperty(name="Height", default=512, min=64, max=2048, update=redraw)
    projection: FloatVectorProperty(size=16)
    view_matrix: FloatVectorProperty(size=16)
    view_rotation: FloatVectorProperty(size=4, default=(1, 0, 0, 0))
    view_location: FloatVectorProperty(size=3)
    view_distance: FloatProperty()
    lens: FloatProperty()
    clip_start: FloatProperty()
    clip_end: FloatProperty()
    depth_tolerance: FloatProperty(
        name="Depth tolerance", default=0.01, min=0.000001, soft_max=0.1,
        precision=5, update=sync_material,
        description="World-space surface-distance tolerance for sampled visibility; large values may leak onto nearby hidden surfaces",
    )
    active_index: IntProperty(default=0, update=select_layer)
    preview: BoolProperty(name="Preview New Layer Frame", default=True, update=redraw)
    last_settings: StringProperty(description="JSON snapshot of the last-used generation settings, used to prefill new layers when the stack is empty")
    last_reference: PointerProperty(type=bpy.types.Image, description="Last-used IPAdapter reference image")
    commit_preserve: BoolProperty(
        name="Keep source layers", default=False,
        description="Keep the committed layers hidden instead of removing them, so their editable sources survive the commit")


class PAWPRINT_PG_lora(bpy.types.PropertyGroup):
    model: StringProperty(name='LoRA')
    strength: FloatProperty(name='Strength', default=1.0, min=-3, max=3)


class PAWPRINT_PG_layer(bpy.types.PropertyGroup):
    generation_last_seed: StringProperty(name='Last used seed')
    generation_ipadapter_image: PointerProperty(name='Reference image', type=bpy.types.Image)
    generation_loras: CollectionProperty(type=PAWPRINT_PG_lora)
    selection_paths: StringProperty(default='[]', update=redraw)
    selection_operation: bpy.props.EnumProperty(name='Selection mode', items=[
        ('REPLACE', 'Replace', 'Replace the current selection'),
        ('ADD', 'Add', 'Add to the current selection'),
        ('SUBTRACT', 'Subtract', 'Subtract from the current selection')], default='REPLACE')
    generation_feather: IntProperty(name='Inpaint feather (px)', default=0, min=0, max=128, update=redraw)
    generation_padding: IntProperty(name='Context padding (px)', default=32, min=0, max=2048, update=redraw)
    generation_preview: BoolProperty(name='Show Context Bounds', default=False, update=redraw)
    generation_context: bpy.props.EnumProperty(name='Context', items=[
        ('FRAME', 'Full Frame', ''), ('SELECTION', 'Selection Bounds', '')],
        default='SELECTION', update=redraw)
    generation_adapter: bpy.props.EnumProperty(name='Model type', items=[
        ('SDXL', 'SDXL', 'Stable Diffusion XL workflow with ControlNet guidance'),
        ('ZIT', 'Z Image Turbo', 'Z Image Turbo workflow with DiffSynth ControlNet guidance')],
        default='SDXL', description='Generation backend family for this layer')
    name: StringProperty(name="Name", default="Projection")
    image: PointerProperty(type=bpy.types.Image, update=sync_material)
    depth: PointerProperty(type=bpy.types.Image)
    visible: BoolProperty(name="Visible", default=True, update=sync_material)
    has_view: BoolProperty(default=True)
    width: IntProperty(default=512)
    height: IntProperty(default=512)
    projection: FloatVectorProperty(size=16)
    view_matrix: FloatVectorProperty(size=16)
    view_rotation: FloatVectorProperty(size=4, default=(1, 0, 0, 0))
    view_location: FloatVectorProperty(size=3)
    view_distance: FloatProperty()
    lens: FloatProperty()
    clip_start: FloatProperty()
    clip_end: FloatProperty()
    depth_tolerance: FloatProperty(name="Depth tolerance", default=0.01, min=0.000001,
                                  precision=5, update=sync_material)
    mirror: bpy.props.EnumProperty(
        name="Mirror", items=[('NONE', 'None', 'Project the image as captured'),
                              ('X_PLUS', 'X+', 'Keep the +X half and mirror it onto −X; the −X half of the image is ignored'),
                              ('X_MINUS', 'X−', 'Keep the −X half and mirror it onto +X; the +X half of the image is ignored'),
                              ('Y_PLUS', 'Y+', 'Keep the +Y half and mirror it onto −Y; the −Y half of the image is ignored'),
                              ('Y_MINUS', 'Y−', 'Keep the −Y half and mirror it onto +Y; the +Y half of the image is ignored'),
                              ('Z_PLUS', 'Z+', 'Keep the +Z half and mirror it onto −Z; the −Z half of the image is ignored'),
                              ('Z_MINUS', 'Z−', 'Keep the −Z half and mirror it onto +Z; the +Z half of the image is ignored')],
        default='NONE', update=sync_material,
        description="Mirror the projection across the target object's local axis plane")


# The adapter owns parameter names, defaults, limits and the UI/request contract.
from .backend import ALL_PARAMETERS
for _name, _spec in ALL_PARAMETERS.items():
    _arguments = {key: value for key, value in _spec.items() if key not in {'kind', 'choices'}}
    PAWPRINT_PG_layer.__annotations__['generation_' + _name] = {
        'STRING': StringProperty, 'FLOAT': FloatProperty, 'INT': IntProperty, 'BOOL': BoolProperty,
    }[_spec['kind']](**_arguments)

# JSON-serializable settings inherited by new layers (the reference image is a
# pointer and travels separately).
INHERIT_FIELDS = tuple('generation_' + key for key in ALL_PARAMETERS) + (
    'generation_adapter', 'generation_feather', 'generation_padding', 'generation_context', 'generation_preview')


def remember_last(stack, layer):
    """Snapshot inheritable settings so a later first layer reuses them."""
    data = {key: getattr(layer, key) for key in INHERIT_FIELDS}
    # The LoRA stack is a collection, so it rides the snapshot as pair lists.
    data['generation_loras'] = [(item.model, item.strength) for item in layer.generation_loras]
    stack['last_settings'] = json.dumps(data)
    stack.last_reference = layer.generation_ipadapter_image


def recall_last(stack):
    """Return the stored last-used settings, ignoring corrupt snapshots."""
    try:
        data = json.loads(stack.last_settings or '{}')
    except ValueError:
        return {}
    result = {key: data[key] for key in INHERIT_FIELDS if key in data}
    if 'generation_loras' in data:
        result['generation_loras'] = data['generation_loras']
    return result


def active_layer(stack):
    if stack and 0 <= stack.active_index < len(stack.layers):
        return stack.layers[stack.active_index]
    return None


def migrate(material):
    stack = material.pawprint
    if not stack.enabled or not stack.has_view or stack.layers:
        return
    layer = stack.layers.add()
    # Copy data fields first; image/visibility updates rebuild the material.
    for name in ('depth', 'width', 'height', 'projection', 'view_matrix',
                 'view_rotation', 'view_location', 'view_distance', 'lens',
                 'clip_start', 'clip_end', 'depth_tolerance', 'visible', 'image'):
        setattr(layer, name, getattr(stack, name))
    stack.has_view = False
    stack.image = None
    stack.depth = None
    stack.preview = False


@bpy.app.handlers.persistent
def load_layers(_):
    for material in bpy.data.materials:
        migrate(material)


def migrate_existing():
    load_layers(None)


@bpy.app.handlers.persistent
def repair_missing_images(scene, depsgraph):
    # ID deletion bypasses PointerProperty update callbacks. Rebuild only when
    # a previously sampled image was removed, so its empty texture cannot cover
    # lower layers. Unlinking through the layer UI uses sync_material directly.
    from .projection import build_material
    for material in bpy.data.materials:
        if not material.pawprint.enabled or not material.node_tree:
            continue
        if any(node.type == 'TEX_IMAGE' and node.image is None
               and node.name.startswith(('Pawprint Layer', 'Captured Scene Depth'))
               for node in material.node_tree.nodes):
            build_material(material)


def active_stack(context):
    from . import painting
    if painting.active():
        return painting.session_stack()
    obj = context.active_object
    material = obj.active_material if obj and obj.type == "MESH" else None
    if material and material.pawprint.enabled and material.pawprint.owner == obj:
        return material.pawprint
    return None


def register():
    bpy.utils.register_class(PAWPRINT_PG_lora)
    bpy.utils.register_class(PAWPRINT_PG_layer)
    PAWPRINT_PG_stack.__annotations__['layers'] = CollectionProperty(type=PAWPRINT_PG_layer)
    bpy.utils.register_class(PAWPRINT_PG_stack)
    bpy.types.Material.pawprint = PointerProperty(type=PAWPRINT_PG_stack)
    bpy.app.handlers.load_post.append(load_layers)
    bpy.app.handlers.depsgraph_update_post.append(repair_missing_images)
    bpy.app.timers.register(migrate_existing)


def unregister():
    if bpy.app.timers.is_registered(migrate_existing):
        bpy.app.timers.unregister(migrate_existing)
    bpy.app.handlers.load_post.remove(load_layers)
    bpy.app.handlers.depsgraph_update_post.remove(repair_missing_images)
    del bpy.types.Material.pawprint
    bpy.utils.unregister_class(PAWPRINT_PG_stack)
    bpy.utils.unregister_class(PAWPRINT_PG_layer)
    bpy.utils.unregister_class(PAWPRINT_PG_lora)
