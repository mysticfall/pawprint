# SPDX-License-Identifier: GPL-3.0-or-later
"""Isolated PBR composite bake, material build, view-mode and persistence checks."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def worker(directory):
    import bpy
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from projection_test import load_extension, setup_fixture, create_projection
    ext = load_extension()
    scene, obj, original = setup_fixture(bpy)
    # Baking reads UVs (projection rendering never does): give the faces real,
    # disjoint islands like baking_test, or every sample stays at texel (0,0).
    uv = obj.data.uv_layers.active
    for face in obj.data.polygons:
        left, right = (.1, .45) if face.index == 0 else (.55, .9)
        if face.index == 1:
            left, right = .1, .45
        for loop, co in zip(face.loop_indices, [(left, .1), (right, .1), (right, .9), (left, .9)]):
            uv.data[loop].uv = co
    first = create_projection(bpy, ext, obj, scene.camera)
    stack = obj.active_material.pawprint
    bpy.context.preferences.edit.use_global_undo = True
    scene.pawprint_base_width = scene.pawprint_base_height = 256
    # A visible upper layer so the composite provably includes more than the base.
    values = np.zeros((128, 128, 4), dtype=np.float32)
    values[:, :, 0], values[:, :, 1], values[:, :, 2], values[:, :, 3] = .9, .2, .1, .8
    first.image.pixels.foreach_set(values.ravel())
    first.image.update()
    first.image.pack()

    # --- Composite bake: equals a full-range commit bake, survives removal ---
    composite = ext.baking.bake_composite(bpy.context, stack)
    assert composite.name == 'Pawprint Material Composite' and tuple(composite.size) == (256, 256)
    assert composite.packed_file and composite.filepath_raw == ''
    commit = ext.baking.bake_base(bpy.context, stack, len(stack.layers) - 1)
    assert np.allclose(ext.capture.pixels(composite), ext.capture.pixels(commit))
    bpy.data.images.remove(commit)
    composite_pixels = ext.capture.pixels(composite).copy()
    layer_pixels = ext.capture.pixels(first.image).copy()
    # Painted pixels appear in the composite: the uniform layer lands at
    # alpha .8 over the opaque gray base (46/255) inside face 0's UV island.
    # The bake blends in linear light and stores sRGB, so mirror that here.
    def to_linear(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    def to_srgb(l):
        return np.where(l <= 0.0031308, l * 12.92, 1.055 * np.maximum(l, 0) ** (1 / 2.4) - 0.055)

    base_rgb = np.full(3, 46 / 255)
    alpha = layer_pixels[0, 0, 3]
    expected = to_srgb(to_linear(layer_pixels[0, 0, :3]) * alpha + to_linear(base_rgb) * (1 - alpha))
    assert np.allclose(composite_pixels[128, 70, :3], expected, atol=.01), \
        'Layer content missing from the composite bake'
    bpy.data.images.remove(composite)
    # An emptied stack bakes the base alone without error.
    for index in range(len(stack.layers)):
        stack.layers.remove(0)
    ext.projection.build_material(stack.id_data)
    base_only = ext.baking.bake_composite(bpy.context, stack)
    assert base_only.name == 'Pawprint Material Composite' and base_only.packed_file
    bpy.data.images.remove(base_only)
    # Restore a working layer for the rest of the probe.
    layer = stack.layers.add()
    for key in ('width', 'height', 'projection', 'view_matrix', 'view_rotation',
                'view_location', 'view_distance', 'lens', 'clip_start', 'clip_end'):
        setattr(layer, key, getattr(first, key))
    layer.depth = first.depth
    layer.image = first.image
    layer.visible = True
    ext.projection.build_material(stack.id_data)

    # --- build: node wiring, colorspaces, name reuse, back-reference ---
    def write_map(name, rgb):
        data = np.zeros((32, 32, 4), dtype=np.float32)
        data[:, :, :3], data[:, :, 3] = rgb, 1
        image = bpy.data.images.new(name, width=32, height=32)
        image.pixels.foreach_set(data.ravel())
        image.filepath_raw, image.file_format = str(directory / name), 'PNG'
        image.save()
        bpy.data.images.remove(image)
        return directory / name

    albedo_path = write_map('basecolor.png', (.25, .5, .75))
    normal_path = write_map('normal.png', (.5, .5, 1.0))
    material = ext.pbr.build(stack, albedo_path, normal_path)
    assert material == stack.pbr_material and material.name == 'Pawprint PBR'
    assert material.pawprint_source == stack.id_data
    tree = material.node_tree
    principled = next(n for n in tree.nodes if n.type == 'BSDF_PRINCIPLED')
    output = next(n for n in tree.nodes if n.type == 'OUTPUT_MATERIAL')
    uv = next(n for n in tree.nodes if n.type == 'UVMAP')
    assert uv.name == 'Pawprint UV' and uv.uv_map == stack.uv_name
    tex_nodes = {n.label: n for n in tree.nodes if n.type == 'TEX_IMAGE'}
    assert tex_nodes['Albedo'].image.name == 'Pawprint Albedo'
    assert tex_nodes['Albedo'].image.colorspace_settings.name == 'sRGB'
    assert tex_nodes['Normal'].image.name == 'Pawprint Normal'
    assert tex_nodes['Normal'].image.colorspace_settings.name == 'Non-Color'
    normal_map = next(n for n in tree.nodes if n.type == 'NORMAL_MAP')
    assert normal_map.uv_map == stack.uv_name

    def linked(source, target):
        return any(l.from_socket == source and l.to_socket == target for l in tree.links)

    assert linked(tex_nodes['Albedo'].outputs['Color'], principled.inputs['Base Color'])
    assert linked(tex_nodes['Normal'].outputs['Color'], normal_map.inputs['Color'])
    assert linked(normal_map.outputs['Normal'], principled.inputs['Normal'])
    assert linked(principled.outputs['BSDF'], output.inputs['Surface'])
    assert linked(uv.outputs['UV'], tex_nodes['Albedo'].inputs['Vector'])
    assert linked(uv.outputs['UV'], tex_nodes['Normal'].inputs['Vector'])
    for node in tex_nodes.values():
        assert node.image.packed_file
    albedo_users = tex_nodes['Albedo'].image.users
    # Rebuilding keeps the material, replaces the maps and reuses the names.
    albedo_path2 = write_map('basecolor2.png', (.8, .6, .4))
    normal_path2 = write_map('normal2.png', (.5, .6, 1.0))
    material2 = ext.pbr.build(stack, albedo_path2, normal_path2)
    assert material2 == material
    tex_nodes = {n.label: n for n in material.node_tree.nodes if n.type == 'TEX_IMAGE'}
    assert tex_nodes['Albedo'].image.name == 'Pawprint Albedo'
    assert np.allclose(ext.capture.pixels(tex_nodes['Albedo'].image)[16, 16, :3], (.8, .6, .4), atol=.01)
    assert len([image for image in bpy.data.images if image.name == 'Pawprint Albedo']) == 1
    assert bpy.data.images.get('basecolor.png') is None, 'Loaded map leaked unpacked'

    # --- View mode: slot swap, editing gate, undo/redo ---
    # Undo/redo (memfile) invalidates every bpy wrapper, so re-resolve by name
    # after each step like baking_test does.
    object_name = obj.name
    pawprint_name = stack.id_data.name
    pbr_name = material.name
    other_name = original.name
    slot_index = obj.active_material_index
    assert ext.model.active_stack(bpy.context) is not None
    assert ext.model.displayed_stack(bpy.context) == stack
    assert ext.model.in_material_view(bpy.context) is None

    def resolve():
        o = bpy.data.objects[object_name]
        mat = bpy.data.materials[pawprint_name]
        return o, mat, mat.pawprint, bpy.data.materials[pbr_name]

    bpy.ops.ed.undo_push(message='Before material view')
    assert bpy.ops.pawprint.view_mode('EXEC_DEFAULT', True, material_view=True) == {'FINISHED'}
    assert obj.material_slots[slot_index].material == material
    assert ext.model.active_stack(bpy.context) is None, 'Editing stayed enabled in material view'
    assert ext.model.displayed_stack(bpy.context) == stack
    assert ext.model.in_material_view(bpy.context) == stack
    assert not bpy.ops.pawprint.commit_base.poll()
    assert not bpy.ops.pawprint.add_projection.poll()
    assert bpy.ops.pawprint.view_mode.poll()
    assert bpy.ops.ed.undo() == {'FINISHED'}
    obj, pawprint_material, stack, material = resolve()
    assert obj.material_slots[slot_index].material == pawprint_material
    assert ext.model.active_stack(bpy.context) == stack
    assert ext.model.in_material_view(bpy.context) is None
    assert bpy.ops.ed.redo() == {'FINISHED'}
    obj, pawprint_material, stack, material = resolve()
    assert obj.material_slots[slot_index].material == material
    assert ext.model.active_stack(bpy.context) is None
    # Returning works without another undo step.
    assert bpy.ops.pawprint.view_mode('EXEC_DEFAULT', True, material_view=False) == {'FINISHED'}
    assert obj.material_slots[slot_index].material == pawprint_material
    assert ext.model.active_stack(bpy.context) == stack
    other = (slot_index + 1) % len(obj.material_slots)
    assert obj.material_slots[other].material.name == other_name

    # --- Apply: begin_apply consumes a job, verifies targets, stays undoable ---
    albedo_path3 = write_map('basecolor3.png', (.1, .9, .3))
    normal_path3 = write_map('normal3.png', (.4, .5, 1.0))
    import shutil
    job_directory = Path(tempfile.mkdtemp(prefix='pawprint-pbr-probe-'))
    shutil.copy(albedo_path3, job_directory / 'result-basecolor.png')
    shutil.copy(normal_path3, job_directory / 'result-normal.png')
    job = dict(directory=job_directory, material=pawprint_material.as_pointer(),
               fingerprint=ext.model.fingerprint_stack(stack),
               digest=ext.pbr._stack_digest(stack))
    bpy.ops.ed.undo_push(message='Before PBR apply')
    ext.pbr.begin_apply(job)
    tex_nodes = {n.label: n for n in material.node_tree.nodes if n.type == 'TEX_IMAGE'}
    assert np.allclose(ext.capture.pixels(tex_nodes['Albedo'].image)[16, 16, :3], (.1, .9, .3), atol=.01)
    assert np.allclose(ext.capture.pixels(tex_nodes['Normal'].image)[16, 16, :3], (.4, .5, 1.0), atol=.01)
    # Map datablocks are reused, so undo cannot cross the node wiring; the
    # reloaded pixels themselves may survive memfile undo like any direct
    # image write.
    assert bpy.ops.ed.undo() == {'FINISHED'}
    obj, pawprint_material, stack, material = resolve()
    tex_nodes = {n.label: n for n in material.node_tree.nodes if n.type == 'TEX_IMAGE'}
    assert tex_nodes['Albedo'].image.name == 'Pawprint Albedo'
    assert tex_nodes['Normal'].image.name == 'Pawprint Normal'
    albedo_value = ext.capture.pixels(tex_nodes['Albedo'].image)[16, 16, :3]
    assert any(np.allclose(albedo_value, expected, atol=.01)
               for expected in ((.8, .6, .4), (.1, .9, .3))), albedo_value
    # Re-applying the same job after an undo refreshes the maps again.
    ext.pbr.begin_apply(job)
    tex_nodes = {n.label: n for n in material.node_tree.nodes if n.type == 'TEX_IMAGE'}
    assert np.allclose(ext.capture.pixels(tex_nodes['Albedo'].image)[16, 16, :3], (.1, .9, .3), atol=.01)
    # A stale structure is rejected without changes.
    stale = dict(directory=job_directory, material=pawprint_material.as_pointer(),
                 fingerprint=('broken',), digest=job['digest'])
    tex_before = {n.label: n.image.name for n in material.node_tree.nodes if n.type == 'TEX_IMAGE'}
    try:
        ext.pbr.begin_apply(stale)
        raise AssertionError('Stale job accepted')
    except RuntimeError as exc:
        assert 'changed while estimating' in str(exc), exc
    tex_after = {n.label: n.image.name for n in material.node_tree.nodes if n.type == 'TEX_IMAGE'}
    assert tex_before == tex_after
    # A first-ever apply is structural: one undo removes the derived material.
    bpy.data.materials.remove(material)
    stack.pbr_material = None
    bpy.ops.ed.undo_push(message='Before first PBR apply')
    ext.pbr.begin_apply(job)
    material = stack.pbr_material
    assert material is not None and bpy.data.materials.get('Pawprint PBR') is not None
    assert np.allclose(ext.capture.pixels(
        next(n for n in material.node_tree.nodes if n.label == 'Albedo').image)[16, 16, :3],
        (.1, .9, .3), atol=.01)
    assert bpy.ops.ed.undo() == {'FINISHED'}
    pawprint_material = bpy.data.materials[pawprint_name]
    stack = pawprint_material.pawprint
    assert stack.pbr_material is None
    assert bpy.data.materials.get('Pawprint PBR') is None
    # Bring the derived material back for the persistence check.
    ext.pbr.begin_apply(job)
    material = stack.pbr_material
    assert material is not None and material.node_tree is not None
    shutil.rmtree(job_directory, ignore_errors=True)

    # --- Persistence: the pointer cycle keeps both materials alive ---
    path = directory / 'pbr.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False)
    # Pre-reload wrappers are stale after the undos above; use stored names.
    obj = bpy.data.objects[object_name]
    stack = bpy.data.materials[pawprint_name].pawprint
    assert stack.pbr_material is not None and stack.pbr_material.name == 'Pawprint PBR'
    assert stack.pbr_material.pawprint_source == stack.id_data
    assert not stack.pbr_material.use_fake_user
    assert stack.pbr_material.node_tree is not None
    assert ext.model.active_stack(bpy.context) == stack
    print({'composite_bake': True, 'pbr_build_and_rebuild': True, 'view_mode_gate_and_undo': True,
           'apply_with_stale_rejection': True, 'pointer_cycle_persistence': True}, flush=True)


def main():
    with tempfile.TemporaryDirectory(prefix='pawprint-pbr-probe-') as temp:
        directory = Path(temp)
        env = os.environ.copy()
        for suffix in ('RESOURCES', 'CONFIG', 'SCRIPTS', 'DATAFILES', 'EXTENSIONS'):
            env['BLENDER_USER_' + suffix] = str(directory / suffix.lower())
        subprocess.run(['blender', '--background', '--factory-startup', '--python-exit-code', '1',
                        '--python', str(Path(__file__).resolve()), '--', '--worker', str(directory)],
                       env=env, check=True, timeout=180)


if __name__ == '__main__':
    if '--worker' in sys.argv:
        worker(Path(sys.argv[-1]))
    else:
        main()
