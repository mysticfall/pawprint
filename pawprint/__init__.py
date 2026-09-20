# SPDX-License-Identifier: GPL-3.0-or-later

_previous_modules = globals().get("_MODULES")
if _previous_modules is not None:
    import importlib

    for _module in _previous_modules:
        importlib.reload(_module)
elif globals().get("ui") is not None:
    # Reloading from the original UI-only slice.
    import importlib
    importlib.reload(globals()["ui"])

from . import backend, model, projection, operators, overlay, painting, selection, capture, result, generation, baking, ui

_MODULES = (backend, model, projection, operators, overlay, painting, selection, capture, result, generation, baking, ui)


def register():
    model.register()
    operators.register()
    painting.register()
    selection.register()
    generation.register()
    baking.register()
    ui.register()
    overlay.register()


def unregister():
    baking.unregister()
    generation.unregister()
    painting.unregister()
    selection.unregister()
    overlay.unregister()
    ui.unregister()
    operators.unregister()
    model.unregister()
