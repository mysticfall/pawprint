# SPDX-License-Identifier: GPL-3.0-or-later
"""Run with system Python; launch a disposable Blender registration/reload check."""

import argparse
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def check_in_blender():
    import addon_utils
    import bpy

    root = Path(__file__).resolve().parents[1]
    repository = Path(os.environ["PAWPRINT_TEST_REPOSITORY"])
    (repository / "pawprint").symlink_to(root / "pawprint", target_is_directory=True)
    repos = bpy.context.preferences.extensions.repos
    repo = repos.new(
        name="Pawprint smoke test",
        module="pawprint_test",
        custom_directory=str(repository),
    )
    module_name = f"bl_ext.{repo.module}.pawprint"

    def classes(module):
        return (module.model.PAWPRINT_PG_stack, module.model.PAWPRINT_PG_lora,
                module.model.PAWPRINT_PG_layer,
                *module.operators.CLASSES, *module.painting.CLASSES,
                *module.selection.CLASSES, *module.generation.CLASSES,
                *module.baking.CLASSES, *module.ui.CLASSES)

    def registered(module):
        assert all(cls.is_registered for cls in classes(module))
        assert hasattr(bpy.types.Material, 'pawprint')
        assert module.overlay._handle is not None
        assert bpy.app.handlers.load_post.count(module.model.load_layers) == 1
        assert bpy.app.handlers.save_pre.count(module.painting.stop_before_file_change) == 1
        assert bpy.app.handlers.undo_post.count(module.painting.reconcile_undo) == 1
        assert bpy.app.handlers.undo_pre.count(module.generation.before_data_change) == 1
        assert bpy.app.timers.is_registered(module.generation.tick)
        assert hasattr(bpy.types.Scene, 'pawprint_server')
        assert hasattr(bpy.types.Scene, 'pawprint_base_width') and hasattr(bpy.types.Scene, 'pawprint_base_height')
        assert hasattr(bpy.types.Scene, 'pawprint_positive_history')
        assert hasattr(bpy.types.Scene, 'pawprint_negative_history')

    def unregistered(classes):
        assert not any(cls.is_registered for cls in classes)

    try:
        module = addon_utils.enable(module_name, default_set=True)
        assert module is not None, "Extension enable failed"
        registered(module)

        obj = bpy.context.active_object
        original = obj.active_material
        twin = bpy.data.objects.new('Linked mesh isolation check', obj.data)
        bpy.context.scene.collection.objects.link(twin)
        obj.data.materials.append(original)
        assert bpy.ops.pawprint.create_stack() == {'FINISHED'}
        material = obj.active_material
        base = material.pawprint.base
        assert tuple(base.size) == (2048, 2048), 'Default base resolution'
        assert material.pawprint.owner == obj
        assert material.pawprint.original_material == original
        assert obj.material_slots[1].material == original
        assert twin.material_slots[0].material == original
        # Exercise actual capture operators against a viewport context, including
        # removal of the last layer and creating another one afterwards.
        area = next(a for a in bpy.context.screen.areas if a.type == 'VIEW_3D')
        region = next(r for r in area.regions if r.type == 'WINDOW')
        material.pawprint.width = material.pawprint.height = 64
        # RegionView3D.update requires an initialized window (it crashes in
        # background Blender); --interactive runs this block in an isolated UI.
        with bpy.context.temp_override(area=area, region=region):
            if bpy.app.background:
                layer = material.pawprint.layers.add()
                layer.image = module.projection.new_image('Reload image', 64, 64)
            else:
                check_capture(bpy, material, area)
        image = material.pawprint.layers[0].image
        module.selection.add_path(material.pawprint.layers[0],
                                  [(0, 0), (0.5, 0), (0.5, 1), (0, 1)], 'REPLACE')
        selection_paths = material.pawprint.layers[0].selection_paths
        material.pawprint.layers[0].generation_positive = 'Reloaded prompt'
        material.pawprint.layers[0].generation_negative = 'Reloaded negative'
        material.pawprint.layers[0].generation_seed = '18446744073709551615'
        material.pawprint.layers[0].generation_batch = 3
        material.pawprint.layers[0].generation_steps = 17
        lora = material.pawprint.layers[0].generation_loras.add()
        lora.model, lora.strength = 'first.safetensors', 0.25
        if not bpy.app.background:
            with bpy.context.temp_override(area=area, region=region):
                assert bpy.ops.pawprint.paint_layer() == {'RUNNING_MODAL'}

        old_classes = classes(module)
        bpy.utils.load_scripts(reload_scripts=True)
        module = importlib.import_module(module_name)
        registered(module)
        unregistered(old_classes)
        assert not module.painting.active()
        assert not any(item.get('_pawprint_paint_proxy') for item in bpy.data.objects)
        assert not any(item.get('_pawprint_paint_proxy') for item in bpy.data.images)
        assert bpy.context.view_layer.objects.active == obj
        assert classes(module)[0] is not old_classes[0]
        assert material.pawprint.enabled and material.pawprint.base == base
        assert material.pawprint.owner == obj
        assert material.pawprint.layers[0].image == image
        assert material.pawprint.layers[0].selection_paths == selection_paths
        assert material.pawprint.layers[0].generation_positive == 'Reloaded prompt'
        assert material.pawprint.layers[0].generation_negative == 'Reloaded negative'
        assert material.pawprint.layers[0].generation_seed == '18446744073709551615'
        assert material.pawprint.layers[0].generation_batch == 3
        assert material.pawprint.layers[0].generation_steps == 17
        expected_loras = [('first.safetensors', 0.25)]
        if not bpy.app.background:
            expected_loras = [
                ('first.safetensors', 0.25), ('second.safetensors', -0.5),
                ('first.safetensors', 0.25)]
        actual_loras = [(item.model, item.strength) for item in material.pawprint.layers[0].generation_loras]
        assert actual_loras == expected_loras, (actual_loras, expected_loras)

        old_classes = classes(module)
        addon_utils.disable(module_name, default_set=True)
        unregistered(old_classes)
        assert not hasattr(bpy.types.Material, 'pawprint')
        assert module.overlay._handle is None
        assert not hasattr(bpy.types.Scene, 'pawprint_server')
        assert not hasattr(bpy.types.Scene, 'pawprint_base_width')
        assert not hasattr(bpy.types.Scene, 'pawprint_base_height')
        assert not hasattr(bpy.types.Scene, 'pawprint_positive_history')
        assert not hasattr(bpy.types.Scene, 'pawprint_negative_history')
        assert not bpy.app.timers.is_registered(module.generation.tick)
        module = addon_utils.enable(module_name, default_set=True)
        assert module is not None, "Extension re-enable failed"
        registered(module)
        assert material.pawprint.enabled and material.pawprint.base == base
        old_classes = classes(module)
        addon_utils.disable(module_name, default_set=True)
        unregistered(old_classes)
        print(f"Pawprint smoke test passed on Blender {bpy.app.version_string}")
    finally:
        if addon_utils.check(module_name)[1]:
            addon_utils.disable(module_name, default_set=True)
        repos.remove(repo)


def check_capture(bpy, material, area):
    from mathutils import Quaternion
    extension = importlib.import_module(material.pawprint.__class__.__module__.rsplit('.', 1)[0])
    area.spaces.active.region_3d.view_perspective = 'PERSP'
    assert material.pawprint.preview
    assert bpy.ops.pawprint.add_projection() == {'FINISHED'}
    assert not material.pawprint.preview
    region = bpy.context.region
    points = extension.overlay.frame_corners(material.pawprint.layers[0], region, area.spaces.active.region_3d)
    width, height = extension.projection.frame_size(region, 1)
    assert abs(points[0][0] - (region.width - width) / 2) < 0.01
    assert abs(points[0][1] - (region.height - height) / 2) < 0.01
    area.spaces.active.region_3d.view_rotation = Quaternion((0.9238795, 0, 0.3826834, 0))
    previous = material.pawprint.layers[0]
    previous.generation_checkpoint = 'prefill-checkpoint'
    previous.generation_positive = 'painted ceramic'
    previous.generation_negative = 'text'
    previous.generation_cfg = 6.5
    previous.generation_padding = 45
    previous.generation_batch = 4
    previous.generation_last_seed = '123'
    previous.generation_adapter = 'ZIT'
    previous.generation_zit_unet = 'z_image_turbo_bf16.safetensors'
    previous.generation_zit_controlnet = 'Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2601-8steps.safetensors'
    previous.generation_ipadapter_image = bpy.data.images.new('Smoke reference', width=4, height=4)
    previous.generation_ipadapter_enabled = True
    previous.generation_ipadapter_weight = 1.5
    previous.generation_loras.clear()
    lora = previous.generation_loras.add()
    lora.model, lora.strength = 'first.safetensors', 0.25
    lora = previous.generation_loras.add()
    lora.model, lora.strength = 'second.safetensors', -0.5
    previous.selection_paths = '[["REPLACE", [[0,0],[1,0],[1,1]]]]'
    previous.mirror = 'Y_MINUS'
    assert bpy.ops.pawprint.add_projection() == {'FINISHED'}
    assert len(material.pawprint.layers) == 2
    created = material.pawprint.layers[1]
    for key in extension.backend.ALL_PARAMETERS:
        assert getattr(created, 'generation_' + key) == getattr(previous, 'generation_' + key)
    assert created.generation_adapter == 'ZIT'
    assert created.generation_zit_unet == 'z_image_turbo_bf16.safetensors'
    assert created.generation_ipadapter_image == previous.generation_ipadapter_image
    assert [(item.model, item.strength) for item in created.generation_loras] == [
        ('first.safetensors', 0.25), ('second.safetensors', -0.5)]
    bpy.data.images.remove(previous.generation_ipadapter_image)
    created.generation_ipadapter_enabled = False
    created.generation_ipadapter_weight = 0.25
    assert previous.generation_ipadapter_enabled and previous.generation_ipadapter_weight == 1.5
    assert created.generation_padding == 45
    assert created.selection_paths == '[]' and created.generation_last_seed == ''
    # Mirror is a per-layer projection setting and deliberately not inherited.
    assert created.mirror == 'NONE' and previous.mirror == 'Y_MINUS'
    assert all(value == 0 for value in created.image.pixels[:])
    created.generation_positive = 'independent prompt'
    assert previous.generation_positive == 'painted ceramic'
    material.pawprint.active_index = 0
    view = area.spaces.active.region_3d
    expected = Quaternion(material.pawprint.layers[0].view_rotation)
    assert view.view_rotation.rotation_difference(expected).angle < 0.0001
    material.pawprint.active_index = 1
    expected = Quaternion(material.pawprint.layers[1].view_rotation)
    assert view.view_rotation.rotation_difference(expected).angle < 0.0001
    assert all(layer.visible for layer in material.pawprint.layers)
    view.view_rotation = Quaternion((1, 0, 0, 0))
    assert bpy.ops.pawprint.restore_view() == {'FINISHED'}
    assert view.view_rotation.rotation_difference(expected).angle < 0.0001
    # Keep an IPAdapter-enabled layer active so the interactive sidebar renders
    # the reference picker with its datablock/file-browser controls.
    material.pawprint.active_index = 0
    assert bpy.ops.pawprint.remove_layer() == {'FINISHED'}
    assert bpy.ops.pawprint.remove_layer() == {'FINISHED'}
    assert material.pawprint.preview
    assert bpy.ops.pawprint.add_projection() == {'FINISHED'}
    # The new first layer inherits the last-used settings snapshotted when the
    # final layer was removed, instead of falling back to defaults.
    restored = material.pawprint.layers[0]
    assert restored.generation_checkpoint == 'prefill-checkpoint'
    assert restored.generation_positive == 'independent prompt'
    assert restored.generation_cfg == 6.5 and restored.generation_padding == 45
    assert restored.generation_ipadapter_enabled is False and restored.generation_ipadapter_weight == 0.25
    assert restored.generation_adapter == 'ZIT'
    assert restored.generation_zit_unet == 'z_image_turbo_bf16.safetensors'
    assert restored.generation_zit_controlnet == 'Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2601-8steps.safetensors'
    assert [(item.model, item.strength) for item in restored.generation_loras] == [
        ('first.safetensors', 0.25), ('second.safetensors', -0.5)]
    assert restored.selection_paths == '[]' and restored.generation_last_seed == ''
    assert bpy.ops.pawprint.restore_view() == {'FINISHED'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", default="blender", help="Blender executable")
    parser.add_argument('--interactive', action='store_true', help='Use an isolated UI window to test view capture')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="pawprint-smoke-") as temporary:
        root = Path(temporary)
        repository = root / "repository"
        repository.mkdir()
        env = os.environ.copy()
        env["BLENDER_USER_RESOURCES"] = str(root / "blender")
        for suffix in ("CONFIG", "SCRIPTS", "DATAFILES", "EXTENSIONS"):
            env[f"BLENDER_USER_{suffix}"] = str(root / suffix.lower())
        env["PAWPRINT_TEST_REPOSITORY"] = str(repository)
        subprocess.run(
            [
                args.blender,
                *([] if args.interactive else ['--background']),
                "--factory-startup",
                "--python-exit-code", "1",
                "--python", str(Path(__file__).resolve()),
                "--", "--blender-worker",
            ],
            env=env,
            check=True,
        )


if __name__ == "__main__":
    if "--blender-worker" in sys.argv:
        import bpy
        if bpy.app.background:
            check_in_blender()
        else:
            def run_ui_checks():
                try:
                    check_in_blender()
                except Exception:
                    import traceback
                    traceback.print_exc()
                    os._exit(1)
                bpy.ops.wm.quit_blender()
            bpy.app.timers.register(run_ui_checks, first_interval=1)
    else:
        main()
