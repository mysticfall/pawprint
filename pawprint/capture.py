# SPDX-License-Identifier: GPL-3.0-or-later
"""Clean saved-frame scene capture; all temporary render state is scoped here."""
import bpy
import numpy as np

from . import projection, selection


def pixels(image):
    values = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(values)
    return values.reshape(image.size[1], image.size[0], 4)


def prepare(context, layer, directory, long_edge):
    """Render the saved-frame composite and stage the request pixels.

    The mask PNG is always written alongside the input; every adapter graph
    consumes it and the client keeps using it as the result stencil.
    """
    width, height = layer.image.size
    weights = selection.footprint(layer, generation=True).copy()
    bounds = selection.context_bounds(layer)
    if bounds is None or not np.any(weights):
        raise ValueError('Selection has no editable pixels')
    x0, y0, x1, y1 = bounds
    scale = long_edge / max(x1 - x0, y1 - y0)
    request_size = tuple(max(64, round(value * scale / 8) * 8) for value in (x1-x0, y1-y0))
    scene = context.scene
    # Copy render/color/world settings, not geometry or materials. Use an isolated
    # scene so the artist's camera, output path, compositor and color settings stay intact.
    temporary = scene.copy()
    temporary.name = 'Pawprint temporary capture'
    camera_data = bpy.data.cameras.new('Pawprint saved frame')
    camera = bpy.data.objects.new('Pawprint saved frame', camera_data)
    temporary.collection.objects.link(camera)
    visibility = [(obj, obj.hide_render) for obj in scene.objects]
    modifiers = [(modifier, modifier.show_render) for obj in scene.objects for modifier in obj.modifiers]
    collections = [(collection, collection.hide_render) for collection in bpy.data.collections]
    composite = None
    try:
        space = context.space_data
        visible = {obj.as_pointer(): obj.visible_get(view_layer=context.view_layer, viewport=space)
                   for obj in scene.objects}
        for obj, _ in visibility:
            obj.hide_render = not visible[obj.as_pointer()]
        # Render-only restrictions must not override the captured viewport visibility.
        for collection, _ in collections:
            collection.hide_render = False
        for modifier, _ in modifiers:
            modifier.show_render = modifier.show_viewport
        for view_layer in temporary.view_layers:
            view_layer.use = view_layer.name == context.view_layer.name
            view_layer.material_override = None
        view = projection.matrix(layer.view_matrix)
        window = projection.matrix(layer.projection) @ view.inverted()
        camera.matrix_world = view.inverted()
        camera_data.sensor_fit = 'HORIZONTAL'
        camera_data.sensor_width = 36
        camera_data.lens = window[0][0] * 18
        camera_data.shift_x = window[0][2] / 2
        camera_data.shift_y = window[1][2] * height / width / 2
        camera_data.clip_start, camera_data.clip_end = layer.clip_start, layer.clip_end
        actual = camera.calc_matrix_camera(context.evaluated_depsgraph_get(), x=width, y=height)
        if max(abs(actual[r][c] - window[r][c]) for r in range(4) for c in range(4)) > 1e-4:
            raise ValueError('Saved frame cannot be reproduced by the render camera')
        temporary.camera = camera
        render = temporary.render
        if render.engine not in {'CYCLES', 'BLENDER_EEVEE'}:
            render.engine = 'BLENDER_EEVEE'
        render.resolution_x, render.resolution_y = width, height
        render.resolution_percentage = 100
        render.pixel_aspect_x = render.pixel_aspect_y = 1
        render.use_border = render.use_crop_to_border = False
        render.use_compositing = render.use_sequencer = False
        render.use_multiview = False
        render.film_transparent = False
        render.image_settings.file_format = 'PNG'
        render.image_settings.color_mode, render.image_settings.color_depth = 'RGBA', '8'
        # Images represent unlit sRGB color. Baking AgX/exposure into them here
        # would apply that display transform a second time when projected back.
        temporary.display_settings.display_device = 'sRGB'
        temporary.view_settings.view_transform = 'Standard'
        temporary.view_settings.look = 'None'
        temporary.view_settings.exposure, temporary.view_settings.gamma = 0, 1
        temporary.view_settings.use_curve_mapping = False
        render.filepath = str(directory / 'composite.png')
        bpy.ops.render.render(scene=temporary.name, write_still=True)
        composite = bpy.data.images.load(render.filepath, check_existing=False)
        rgba = pixels(composite)
        crop = rgba[y0:y1, x0:x1]
        save_input(directory / 'input.png', crop, request_size)
        mask = np.ones((y1-y0, x1-x0, 4), dtype=np.float32)
        mask[:, :, :3] = weights[y0:y1, x0:x1, None]
        save_input(directory / 'mask.png', mask, request_size, non_color=True)
        return dict(weights=weights, bounds=bounds, request_size=request_size)
    finally:
        for obj, hidden in visibility:
            obj.hide_render = hidden
        for collection, hidden in collections:
            collection.hide_render = hidden
        for modifier, enabled in modifiers:
            modifier.show_render = enabled
        if composite:
            bpy.data.images.remove(composite)
        bpy.data.objects.remove(camera, do_unlink=True)
        bpy.data.cameras.remove(camera_data)
        bpy.data.scenes.remove(temporary)
        context.view_layer.update()


def save_input(path, values, size, non_color=False):
    image = bpy.data.images.new('Pawprint request pixels', width=values.shape[1], height=values.shape[0])
    try:
        if non_color:
            image.colorspace_settings.name = 'Non-Color'
        image.pixels.foreach_set(values.astype(np.float32).ravel())
        image.scale(*size)
        image.filepath_raw, image.file_format = str(path), 'PNG'
        image.save()
    finally:
        bpy.data.images.remove(image)


def save_reference(path, reference):
    """Export the user's reference image at native resolution.

    Reference guidance (IPAdapter conditioning) is
    semantically separate from the target view, so the saved-view
    crop/scaling must not be applied here.
    """
    width, height = reference.size
    if width < 1 or height < 1:
        raise ValueError('Reference image has no pixels')
    image = bpy.data.images.new('Pawprint reference pixels', width=width, height=height)
    try:
        image.pixels.foreach_set(pixels(reference).astype(np.float32).ravel())
        image.filepath_raw, image.file_format = str(path), 'PNG'
        image.save()
    finally:
        bpy.data.images.remove(image)


def geometry_depth(context, layer, directory, metadata):
    """Fresh geometry guidance, independent of the layer's old visibility snapshot.

    Normalize positive camera-space depth across the full saved frame before
    cropping. Near is white, far/background black; a constant-depth surface is
    white. All viewport-visible surfaces participate, including other objects.
    """
    from types import SimpleNamespace
    frame = SimpleNamespace(width=layer.image.size[0], height=layer.image.size[1],
                            view_matrix=layer.view_matrix, projection=layer.projection,
                            clip_start=layer.clip_start, clip_end=layer.clip_end)
    image = projection.depth_image(context, frame, context.space_data)
    try:
        depths = pixels(image)[:, :, 0]
        valid = np.isfinite(depths) & (depths > 0)
        if not np.any(valid):
            raise ValueError('No visible geometry in the saved depth-guidance frame')
        near, far = float(depths[valid].min()), float(depths[valid].max())
        values = np.zeros_like(depths)
        if far - near <= max(1e-6, far * 1e-6):
            values[valid] = 1
        else:
            values[valid] = (far - depths[valid]) / (far - near)
        rgba = np.ones((*depths.shape, 4), dtype=np.float32)
        rgba[:, :, :3] = values[:, :, None]
        x0, y0, x1, y1 = metadata['bounds']
        save_input(directory / 'depth.png', rgba[y0:y1, x0:x1],
                   metadata['request_size'], non_color=True)
    finally:
        bpy.data.images.remove(image)


def returned_pixels(path, frame_size, metadata):
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        if tuple(image.size) != metadata['request_size']:
            raise ValueError('ComfyUI returned unexpected image dimensions')
        x0, y0, x1, y1 = metadata['bounds']
        image.scale(x1-x0, y1-y0)
        result = np.zeros((frame_size[1], frame_size[0], 4), dtype=np.float32)
        result[y0:y1, x0:x1] = pixels(image)
        return result
    finally:
        bpy.data.images.remove(image)
