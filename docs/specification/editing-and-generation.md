# Editing and generation

## Editing model

Prioritize painting in the locked saved viewpoint; also permit editing the native
layer image in Blender's 2D Image Editor. The layer image stores its own
contribution, not a flattened copy of lower layers.

- Lasso and box selections are in the layer's image space and can be disconnected.
- Feather and context padding are generation/inpainting parameters. They do not
  soften ordinary painting or deletion; brush falloff remains a native brush setting.
- Selection + Delete clears the selected image pixels to transparency.
- An eraser brush reduces pixel alpha, with adjustable strength/opacity and falloff.
- A color brush writes color/alpha, allowing rough sketches followed by low-denoise
  generation to refine them.
- No separate editable layer mask and no layer-level opacity in the initial scope.
- Generation always writes into the active layer, rather than creating a new layer
  above a sketch by default.

## Current image-space selection implementation

Each layer stores ordered polygons in normalized saved-image coordinates,
separately from generation feather radius, context mode, and padding. Metadata saves with
the material; it does not modify the image alpha or the projection visibility.
Lasso and Box restore the saved view and complete on mouse release. An explicit
Replace/Add/Subtract mode applies to either shape. Shift/Ctrl at gesture start
override it with Add/Subtract; Escape/right click cancels. Clear removes the
selection highlight and painting restriction. Subtracting everything also clears
the selection. No selection means unrestricted editing/full-frame generation;
there is no separate Select All command.

Rasterization samples pixel centers; blue shows the hard selection used by painting
and deletion. Generation feather is a separable box blur with a finite
radius measured in image pixels, extending on both sides of the original boundary;
frame-edge samples extend the edge values. The optional generation preview shows
green context bounds enclosing
all nonzero weights plus padding (clipped to the frame), or the full frame when
requested. Off-target areas and distant islands are preserved. Bounds are integer
half-open pixel rectangles; selection coordinates have their origin at image bottom
left. The editing UI exposes preview/data facilities; the isolated generation probe
now exercises these same bounds and weights through a live crop/request round-trip.

Paint Layer creates a temporary native UV stencil from the hard selection. Both color
and erase strokes respect it. Delete Selected Pixels (Delete in the painting
region) uses one constant-strength, constant-falloff native erase dab covering the
frame, restoring the touched brush settings afterward. Selected alpha is cleared;
generation feather and padding have no effect on this edit. Native image undo handles
the edit, including interleaving with ordinary strokes. Selection changes require
finishing painting; native 2D Image Editor painting does not consume this stencil.
The stencil is removed on session cleanup and is not a saved layer-mask asset.

`tools/selection_event_test.py` sends real queued Blender events through both shapes
and all three modes, modifier overrides and cancellation. `tools/selection_test.py`
validates generation feather/context bounds separately from hard painting masks,
selection undo, exact unselected-pixel preservation,
single-step deletion interleaved with strokes, other-layer isolation, restored
native stencil settings, and save/reopen. Source-linked reload checks retain the
selection metadata and remove an active temporary stencil. Freehand mouse/tablet
gestures and resizing remain interactive acceptance checks. The compact layout puts
Generation immediately after the layer list/edit controls; Layer Details and New
Layer are collapsed by default below it.

## Visible-composite context requirements

Use a clean rendered result from the saved viewpoint, with the configured frame.
Match the visible scene while excluding gizmos, grid/guidelines, selection outlines,
image-frame markers, and other editor overlays.

Include visible higher layers as well as lower layers and the active layer's
visible contents. Selecting an active layer must not automatically hide others.
The user controls visibility, including when higher layers obscure the edit result.
An active-layer sketch is not deliberately stripped from the context.

Other visible objects and material slots remain as scene context. Only the active
object–material-slot combination receives the generated projection. Do not
isolate the target simply because other surfaces are not editable.

Earlier proposals to capture only layers through the active one are superseded.
Earlier suggestions to strip lighting from generation context were rejected.
The subsequent unlit-stack decision means baked lighting remains in stack images
without being shaded twice; surrounding ordinary materials can still be shaded
in the scene render. PBR lighting removal belongs to a separate future workflow.

## Selection and context bounds

Support at least:

1. Full image-frame context.
2. Bounding rectangle of the entire edit selection, expanded by configurable
   padding and clipped to the available frame.

The selected/feathered footprint determines what changes, regardless of the size
of context sent to ComfyUI. Preserve unselected active-layer pixels. Write back
the generated patch at the correct original coordinates after any crop/resize.
Keep selection masks used in requests distinct from persistent layer-mask assets,
which are not part of this design.

Disconnected selections deliberately influence the bounding rectangle. A user
may select a tiny distant area so the crop includes useful *visible composite*
context from other layers, material slots, or objects. The extra generated patch
can subsequently be deleted. Do not force selections to target coverage before
computing their context bounds.

This technique does not depend on invisible pixels stored outside the active
layer's projected silhouette. Whether those pixels are retained is a separate
implementation choice.

## Applying results

Render/flatten context dynamically for the request; do not persist it as the
layer's background. Apply the selected generated pixels, including feathering,
to the existing active image. The image outside that footprint remains unchanged.

Spatial guidance, edit masks, and image context must agree on framing, cropping,
and resizing. Record enough request metadata to place the returned patch exactly.

**First implemented async policy:** one job at a time, bound to the originating
window/viewport, scene/view layer, object/slot/material, layer/image and saved frame.
Changing target, image identity/size, selection, feather/context bounds, or entering
painting cancels ownership. Save/load, undo/redo, disable/reload also cancel. Before
application the entire target pixel buffer is hashed against its submission snapshot;
newer image edits cause discard. Orbiting is allowed; application restores the saved
frame. Other scene edits do not alter the already-captured request. A completed result
never targets whichever layer happens to be active later. Cancellation deletes only
the owned queued prompt; running output is discarded without global interruption.

## Undo and redo

Integrate with Blender's native undo/redo where possible. Aim to cover painting,
erasing, deleting pixels, applying generated results, layer changes, and merges.
Applying one completed generation should be one undoable edit. Redo restores the
same pixels, never reruns ComfyUI.

**Implemented:** base commits are one native undo step. The bake targets a new
image datablock and removes layer list entries; ordinary datablock undo restores
both the previous base reference and the removed layers without touching image
buffers directly, verified interleaved with native paint undo.

Direct image buffer changes require a feasibility check: `UNDO` on an operator is
not by itself proof that Blender captures modified pixels. Test interleaving
native painting and extension edits, both undo and redo, before choosing the
implementation. Preserve this user-facing goal even if internal snapshots become
necessary.

**Prototype finding (Blender 5.2.2):** native Image Editor strokes on the layer
image correctly undo and redo. A separate `UNDO` operator writing `image.pixels`
does not restore those pixels through ordinary memfile undo, although a scene
property changed alongside it restores correctly. `tools/image_undo_probe.py`
reproduces the distinction. Selected deletion uses a native erase operation instead
of direct buffer writes. The result-application probe below establishes the native
clone-based path now used by the shipped Generate workflow.

### Generated-result application feasibility

`tools/result_apply_probe.py` tests a native Texture Paint **Clone** stroke on the
managed saved-frame proxy, using its matching clone UV map, a temporary generated
source image, and a temporary stencil containing generation feather weights.
Source and stencil texels are duplicated 2× per axis so projection-roundoff and
bilinear sampling do not bleed across original pixel centers. Only temporary
images receive direct buffer writes. A single full-frame native clone dab modifies
the existing target Image datablock through Blender's native image undo.

On Blender 5.2.2 LTS, byte RGBA target tests verify exact fully selected source
pixels and exact unselected pixels, source-over feather color/alpha within byte
rounding tolerance, unchanged other layers, one-step result undo/redo interleaved
with native erasing, and undo/redo after session exit and removal of source images.
Redo restores cached pixels and never calls ComfyUI. A 2D Image Editor clone
approach was rejected because it did not preserve the intended source-alpha mask;
the successful path is the 3D proxy with an explicit stencil.

`tools/generation_probe.py` additionally passes a real SDXL masked-img2img result
through this path. It reconstructs a render camera from the saved view/projection
and asserts projection agreement. Rendered context explicitly includes a visible
upper layer, another target slot, and a separate surrounding object; overlays
are excluded by native rendering. A disconnected off-target selection expands
the crop. Returned pixels are resized/placed using recorded original bounds, and
the original feather mask constrains application, preserving zero-weight pixels.

The shipped Generate operator now integrates this path. It restores modified
brush/unified-paint/clone/stencil/symmetry/dither settings and cleans up the temporary
session/images; `tools/generation_test.py` validates native single-step result undo
and redo after session exit. Supported targets are byte RGBA, straight-alpha sRGB
images up to 2048 pixels per axis. Native brush-radius limits are checked before
applying. Native Save/Pack conventions still apply to edited images.

Saved-frame capture uses a temporary scene with its own camera/render settings,
no compositor/sequencer/overlays, viewport-visible objects, and viewport modifier
enable states. All overridden visibility flags are restored. The existing stack
uses unlit color, while surroundings use scene lighting. Standard/sRGB, exposure 0
and gamma 1 avoid baking a display transform into color that will be displayed again;
the original scene color management is preserved. Material Preview studio lighting
and render-specific modifier detail levels may differ from this scene render.
The camera matrix is checked against the saved projection. Capture is synchronous;
network/generation/download are asynchronous in a separate process. Original crop
bounds and feather weights are retained throughout resize/return placement.

### Locked-view painting feasibility (Blender 5.2.2, Linux)

`tools/locked_paint_probe.py` validates a candidate native painting path in an
isolated UI process. A temporary fronto-parallel mesh fills the saved image frame,
with ordinary 0–1 UVs. Native Texture Paint uses the active layer's existing Image
as its single-image canvas. A transparent material leaves the actual scene and
layer composite visible in Material Preview; the proxy is excluded from renders.
This avoids making the target's UV map stand in for its per-fragment perspective
projection. The target geometry and its UVs are not modified by painting.

Verified with two layers over sloped target geometry and a 192×128 frame:

- A native stroke at frame coordinates (0.3, 0.35) changes pixels centered within
  two texels of the expected (57.6, 44.8) location.
- Native `ERASE_ALPHA` reduces alpha; interleaved color/erase undo and redo restore
  the entire image buffer exactly, using Blender's native image undo path.
- A stroke outside the proxy frame changes no image pixels.
- Lower-layer pixels and target UVs remain unchanged. Removing the proxy retains
  the edited image and its layer reference.
- A viewport screenshot after shader compilation shows the underlying projected
  composite and unrelated material slot through the invisible proxy.

The prototype now exposes this path as **Paint Layer**, with **Paint**, **Erase
Alpha**, and **Finish Painting** controls; Escape also exits. One session runs at
a time, from Object Mode in a normal perspective viewport. The active layer is
bound independently of the selected temporary proxy. Finish before switching
layers, managing the stack, or using the Image Editor. Other layers stay visible.
Navigation is free between strokes: orbit and zoom to inspect the model. A
LEFTMOUSE press in the painting region while the viewport pose differs from the
saved view is swallowed, the saved view is restored, and a header hint asks the
user to restart the stroke; an aligned press passes through and starts a stroke,
during which navigation input is held so the running stroke keeps its rays.
Alignment compares the eye pose only (RegionView3D view matrix, tolerance 1e-4):
from the same eye position and orientation every cursor ray hits the proxy point
and the mesh point on one ray, so painted texels match displayed texels exactly,
regardless of lens refits caused by viewport resizes; a perspective view is
required. Entry temporarily
uses Material Preview; exit restores selection, shading/gizmos and overridden
native canvas settings. Brush color/size/strength use Blender's native controls.
Color/erase button changes to a brush's blend mode are restored on exit. The
native header — including the Texture Paint brush color chooser — stays visible
during the whole session; only the gate's transient "start the stroke again"
hint occupies the header text (cleared by the next aligned press), because
`Area.header_text_set` replaces the entire header contents.

Saving/loading, disabling/reloading, or changing scenes ends painting and removes
the temporary canvas. Image edits still use Blender's normal Save/Pack convention.
The `--managed` probe validates the real operators with event-simulated input:
the native header remaining intact during a session (two screenshots compared
pixel-for-pixel around a no-op header clear — a session banner would make them
differ), free navigation (a deliberately misaligned view persists), the stroke gate (a
misaligned press restores the saved view and paints no pixels), a real aligned
event stroke landing within two texels of its expected position, whole-buffer
native undo/redo
across session exit and layer switching, restoration of overridden settings,
save/reopen without temporary geometry, disabling during painting, and scene
switch cleanup preserving the new scene's selection. Interactive source-linked
smoke testing validates Reload Scripts during painting. Undo-recreated temporary
objects are cleaned up and the original target reselected.

Physical mouse/tablet interaction, pressure/falloff, brush-size behavior, and the
feel of the stroke gate and snap-back remain interactive acceptance
checks. The full-frame canvas can retain off-target pixels;
the existing projection shader continues to restrict their visible contribution.
This result does not resolve undo for arbitrary generated image-buffer writes.

The SDXL adapter can generate a batch of random-seed candidates with an
interactive review mode; see the backend guidance page for that contract.
