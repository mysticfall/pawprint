# SPDX-License-Identifier: GPL-3.0-or-later

_previous_modules = globals().get("_MODULES")
if _previous_modules is not None:
    import importlib
    from pathlib import Path

    for _module in _previous_modules:
        # A development slice may remove a module while Blender is open.
        if Path(_module.__file__).is_file():
            importlib.reload(_module)
elif globals().get("ui") is not None:
    # Reloading from the original UI-only slice.
    import importlib
    importlib.reload(globals()["ui"])

from . import backend, model, projection, operators, overlay, painting, selection, capture, normals, colormatch, result, generation, baking, pbr, ui

_MODULES = (backend, model, projection, operators, overlay, painting, selection, capture, normals, colormatch, result, generation, baking, pbr, ui)


def register():
    model.register()
    operators.register()
    painting.register()
    selection.register()
    generation.register()
    baking.register()
    pbr.register()
    ui.register()
    overlay.register()


def unregister():
    ui.unregister()
    pbr.unregister()
    baking.unregister()
    generation.unregister()
    painting.unregister()
    selection.unregister()
    overlay.unregister()
    operators.unregister()
    model.unregister()
