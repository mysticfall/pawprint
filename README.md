# Pawprint

A Blender **5.x** extension for viewpoint-bound, editable texture layers backed
by ComfyUI. The intended workflow is manual seam repair with layered painting and
inpainting, inspired by Krita AI Diffusion and informed by StableGen.

**The development prototype supports multiple projection layers.** Create an
unlit UV base, preview and capture perspective layers, edit their native RGBA
images, and reorder or hide them to repair overlapping views.
**Editing works without a server; Generate now connects to ComfyUI.** The first
SDXL adapter provides per-layer prompts and sampling controls, selected img2img,
asynchronous requests, native result undo/redo, optional geometry-depth guidance,
and IPAdapter reference conditioning. Per-layer ordered LoRA stacks support a
configurable strength for each entry. A **Z Image Turbo** adapter offers the
same user-facing workflow through a DiffSynth ControlNet model patch and
reference inpainting with colour matching.
Lower layer groups can be baked into the UV base with one-step undo.
Composite-estimated depth remains a later slice.
The manifest remains at 0.2.0; development slices do not each get a version bump.

## Install the ZIP

1. In Blender 5.0 or newer, open **Edit → Preferences → Get Extensions**.
2. Open the top-right menu and choose **Install from Disk**.
3. Select `dist/pawprint-0.2.0.zip` and enable Pawprint if not enabled automatically.
4. In a 3D Viewport press **N**, then open the **Pawprint** sidebar tab.

## Try projection layers

1. Select a mesh in **Object Mode** with usable UVs and at least one material slot.
   Choose the slot whose faces should receive the projection.
2. In **Pawprint → Projection Layers**, choose **Create Texture Stack**. This assigns
   a new material to the active object slot, with an opaque gray UV base. The
   previous material is retained and can be reassigned through Blender's material UI.
3. Use **Material Preview** or **Rendered** shading. Frame the object in a normal
   user **Perspective** view; camera, orthographic, and quad views are unsupported
   for capture in this prototype.
4. Enable **Preview New Layer Frame** and set the image width/height. Navigate
    until the yellow frame covers the intended area, then choose **Add Layer
   from View**. The captured frame matches the preview; preview switches off.
    New layers start fully transparent, leaving the base and lower layers visible.
     Generation settings are prefilled from the selected layer: model, prompts,
     sampling, depth guidance and context options. They remain independently editable.
     Selections and last-result seed history are not copied. When the stack has no
     layers left (after committing everything or removing the last layer), new layers
     reuse the last-used settings instead of falling back to defaults.
5. Orbit to inspect. Only the active slot receives the image, on surfaces visible
   at capture time. **Restore Layer View** returns to the saved view and frame.
 6. Choose **Paint Layer** to paint directly over the composite in its saved view.
    **Paint** adds color; **Erase Alpha** reveals lower layers. Use native
    brush settings and Ctrl-Z / Shift-Ctrl-Z. **Finish Painting** or Escape returns
    to inspection. Finish before selecting another layer or capturing a new view.
    Alternatively, choose **Edit Layer Image Here** to switch that area to the native Image Editor
    in Paint mode. Use Blender's brushes and undo/redo. **Shift-F5** returns the area
    to the 3D Viewport; the material samples the same edited image. A separate Image
    Editor area can show the image alongside the viewport.
 7. Save the `.blend` and use Blender's normal image **Save/Pack** tools for edits.
     Initial images use native generated/packed storage; no custom
     storage format or automatic image-save handler is installed.

Paint Layer temporarily uses Material Preview and a transparent canvas mesh to
map native strokes to the saved image. Navigation is free between strokes:
orbit and zoom to inspect the model, and a stroke started from a different view
is swallowed while the saved view snaps back — start the stroke again and it
paints exactly where it displays. While a stroke is running, navigation is held
so its rays stay accurate. The native Texture Paint header — including the brush
color chooser — stays visible throughout; only the gate's "start the stroke
again" hint temporarily occupies the header text. Paint Layer preserves the
target geometry/UVs,
selection, and overridden paint settings. Saving, loading, disabling/reloading the
extension, or changing scenes ends the session and removes its temporary canvas.
The saved view stays available for inspection; normal image Save/Pack rules still
apply. The full image frame is paintable, including off-target pixels; only the
captured visible target surfaces display those pixels. Upper layers stay visible.

Repeat preview/capture from another viewpoint to add an overlapping layer. The
list shows the topmost projection first, above the separate UV base. Select and
rename layers, toggle the eye icons, and use the arrows to reorder. Selection does
not hide other layers. Selecting a layer automatically restores its saved view;
**Restore Layer View** remains available after orbiting to inspect.
Erase alpha in the upper image to reveal the lower layer.
The minus button removes the selected layer; Add remains available even after
removing the last layer. Unlinking/deleting an image leaves a layer with an explicit
missing-image indication: assign a replacement or remove the layer. Existing
single-projection scenes migrate to a one-entry list on load or script reload.

### Commit layers to the base

Once lower layers look right, bake them into the UV base:

1. Select the highest layer that should be committed. Everything from the
   bottom layer up to and including it is committed — name it first if the
   group matters; the group is positional, not picked one by one.
2. Choose **Commit Through Selected to Base** (Object Mode, not painting or
   generating). Pawprint bakes the base plus all committed layers' visible
   projections into a **new packed base image at the configured base
   resolution** (Layer Details → Base width/height, scene-wide, default
   2048 × 2048, no upper limit; the existing base content is resampled to
   match), with an 8-pixel margin extending edges past UV seams. Hidden layers
   inside the group contribute nothing and are removed with it; layers above
   the selection stay fully editable, keeping their images, settings, and
   selections.
3. Keep **Keep** checked to preserve the committed sources instead of removing
   them: they stay in the stack, hidden, so you can retry other resolutions or
   re-edit them after undoing the commit. Hidden sources are not re-baked by a
   later commit of the same group — use undo to return to the pre-commit base.
4. The viewport appearance does not change: committed projections now come from
   the base. One native undo restores the previous base and the removed layers
   (or visibility, when sources were kept); redo re-applies the commit,
   interleaved cleanly with painting undo steps. The old base image is kept in
   the file until Blender purges it.

Baking renders through Cycles on the CPU for the target slot only. An error
(missing UV map, missing image) leaves the stack and base untouched.

### Select in image space

1. Finish painting, select a layer, and choose **Lasso** or **Box**. Drag around an area in
   the restored saved frame, then release. Escape/right click cancels the gesture.
2. Set **Replace / Add / Subtract**, then use either shape. Add makes disconnected
   islands; Subtract cuts holes. Shift/Ctrl at gesture start also choose Add/Subtract.
   **Clear** removes the highlight and restriction, returning to unrestricted painting.
   Subtracting everything has the same effect. There is no separate Select All command.
3. The blue tint shows the hard selection. Enter **Paint Layer**. Color and alpha
   erasing respect this footprint, with the brush's own strength/falloff.
   **Delete Selected Pixels** (or Delete over the painting viewport) clears its
   selected pixels to transparency. Native
   undo/redo works across deletion and brush strokes. Finish to change the selection.
4. In **Generation**, configure **Inpaint feather**, **Context** and **Context
   padding**. These are generation-only settings, not painting/deletion controls.
    Enable **Show Context Bounds** to see the green rectangle, including
   feather support and padding. Disconnected islands and off-target areas count
    toward its bounds. With no selection the context is the full frame.
5. Click **Preview Generation Context** to inspect the actual cropped/resized
   input image in the Image Editor. This uses the same saved-view composite capture
   as Generate, including visible upper layers and surrounding objects, without
   contacting ComfyUI. **Shift-F5** returns to 3D. The packed preview is a snapshot;
   Generate captures again, so scene or setting changes can change the input.

Generation follows the compact layer list/edit controls directly. Image datablocks,
saved-frame details, depth tolerance and the UV base are in the collapsed **Layer
Details** panel; frame preview, resolution and capture are in **New Layer**.

Selections belong to individual layers and save with the `.blend`. They follow
the saved image frame, not object UVs or whichever view is currently on screen.
The preview disappears when inspecting from another viewpoint; Restore Layer View
brings it back. Blender's Show Overlays toggle hides the tint and bounds.
The native 2D Image Editor remains an independent painting route: it does **not**
apply Pawprint's selection restriction. Generation uses the previewed context bounds
and the original feathered image-space footprint when applying the returned patch.

Visibility is captured once from evaluated, viewport-visible geometry. Keep the
scene geometry/transforms stable. The depth snapshot treats surfaces as solid,
including transparent materials, and has finite image-resolution boundaries.
**Depth tolerance** is a world-space surface-distance tolerance: increasing it may
reduce boundary gaps but allow projection onto very close hidden surfaces. On top
of the plane-distance and facing tests, a winding-independent axial occlusion
z-test rejects fragments lying clearly behind the captured surface along the saved
view ray (`w − depth > w × margin`, with a two-texel grazing margin scaled by the
texel footprint), so far-side leaks no longer depend on normals having consistent
winding. Sub-texel-thin gaps can still pass, and surfaces genuinely visible
through holes keep receiving the projection by design.
**Mirror** (Layer Details) folds a layer's projection across the target object's
local axis plane: the chosen direction names the kept half (X+ keeps the +X side),
the discarded side samples the mirrored kept half, and the opposite half of the
image is never sampled. Symmetric geometry covers exactly; where the mirrored side
lacks matching geometry, the depth-coverage test leaves holes. The setting is
per-layer, travels to candidate layers, and is deliberately not inherited by new
layers.
This prototype fixes each layer's frame dimensions after capture; post-content
resizing remains later work. Existing
materials are not baked into the initial gray base.

### Generate with ComfyUI

1. Start ComfyUI. Finish painting, select a layer, and open **Generation**.
2. Enter the server URL (default `http://127.0.0.1:8188`) and click **Connect / Refresh**.
   Pick a **Model type** per layer: **SDXL** or **Z Image Turbo**. With SDXL, choose
   an installed **SDXL checkpoint**; `epicrealismXL_pureFix.safetensors` is the
   tested local model. The server lists all checkpoints, so choose an SDXL-compatible
   one rather than assuming every listed architecture works with this adapter. With
    Z Image Turbo, choose the **Z model** (`z_image_turbo_bf16.safetensors`, loaded
    through UNETLoader). In **Guidance**, add zero or more LoRAs and set each
    entry's strength (default 1.0, range -3 to 3). Entries apply in listed order.
    SDXL lists every LoRA reported by the server; Z Image Turbo lists compatible
    `zit/` weights. The tested ZIT setup uses
    `zit/skin texture Photorealistic style v4.5.safetensors` at steps 8, CFG 1,
    res_multistep/simple.
 3. Enter **Positive prompt** and **Negative prompt**. Set **Denoise strength**, **CFG**,
     **CLIP skip**,
     **Seed**, **Sampling steps**, **Sampler**, **Scheduler**, and **Context long edge**.
    These settings belong to the selected layer. Seed accepts an unsigned 64-bit
    integer or `-1` for random; **Last seed** records the actual applied seed.
    Each prompt field has a dropdown listing previously used prompts (recorded when
    Generate runs, kept per scene, newest first, up to 32 entries); choose one to
    replace the field, or use its **Clear history** entry to reset the list.
     4. Choose the full frame or selected context, feather and padding. A
         selection means inpainting. With SDXL the request routes through
         `InpaintModelConditioning` and an `ImageCompositeMasked` output stage, so the
         selected pixels are regenerated rather than refined (denoise 1.0 re-imagines
         them completely) while the surrounding context conditions each step for a
         seamless blend and the server returns the original pixels outside the
         selection exactly. With Z Image Turbo, selections run a reference inpainting
         pipeline: at denoise 1.0 the region is pre-filled by a dedicated MAT inpaint
         model and steered by the Fun ControlNet in inpaint mode, and at lower denoise
         the original latent is refined from a softened sigma schedule — both paths
         finish with an `INPAINT_ColorMatch` step so the repaired pixels blend with
         the surrounding context, while the client-side feathered stencil keeps every
         unselected pixel exact. **Clear**
        means unrestricted generation over the full image as ordinary img2img; the
        latent always comes from the rendered composite, even at denoise 1.0. There
        is no toggle: the selection alone picks the mode. Each
        request produces one result; Sampling steps is the diffusion
        iteration count, not a batch/candidate count.
  5. For structural guidance (especially on an initially flat, unlit object), open
      **Guidance** and enable **Depth guidance**. Choose the shared union ControlNet
      `sdxl_promax.safetensors` and set **Depth strength** (default 0.5). Geometry is
      the default source. **Preview Geometry Depth**
      opens the exact guidance crop in the Image Editor; Shift-F5 returns to 3D.
      Z Image Turbo layers get the same **Depth guidance** toggle, applied through
      a DiffSynth ControlNet patch: **Model Patch Loader** plus the **Apply Qwen
      Image DiffSynth ControlNet** node with the Fun ControlNet Union weights
      (default `Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2601-8steps.safetensors`)
      and a **Control strength** (default 1.0).
 6. For reference-image conditioning, open **Guidance**, enable **IPAdapter
     reference**, and assign a reference image. The picker's folder button opens a
     file browser, so external images can be loaded directly without creating a
     datablock elsewhere first. Set **IPAdapter weight** (default 1.0, range 0–2;
     the tested model is `ip-adapter-plus_sdxl_vit-h.safetensors`). The reference is
     uploaded whole at its native resolution — the saved-view crop is deliberately
     not applied to it. It composes with depth guidance when both are enabled. These
      settings are stored per layer; new
      layers inherit them like the other generation fields. Z Image Turbo has no
      reference guidance yet — it is planned future work.
    7. Plain selection inpainting needs no ControlNet and no extra server nodes —
        `InpaintModelConditioning` and `ImageCompositeMasked` are core ComfyUI nodes.
        A union ControlNet is only involved when **Depth guidance** is enabled
         (step 5). Z Image Turbo selections use the inpainting described in step 4.
   8. Z Image Turbo prompts are plain text — no edit-instruction wrapping — and the
      negative conditioning is zeroed (`ConditioningZeroOut`), so the negative prompt
      field has no effect for Z Image Turbo. The **Guidance** section renders only
      features the active Model type declares: SDXL layers get depth and reference
       guidance plus the shared LoRA stack, while Z Image Turbo layers see the depth
        ControlNet patch, compatible LoRA stack and the inpaint selection status.
  9. Click **Generate**. Clean saved-view capture is synchronous and may briefly pause
     Blender. Uploading, sampling and downloading then run in a separate background
     process while Blender stays interactive. Status and **Cancel Generation** appear below.
 10. The result replaces the selected footprint in the existing layer with one native
     image-undo step. Ctrl-Z / Shift-Ctrl-Z undo/redo the pixels without contacting ComfyUI.
     Save/Pack edited images normally.
    11. Set **Batch** above one (no fixed upper limit — the slider suggests up to
        eight but any count can be typed) to generate that many random-seed
      candidates instead. Each candidate gets its own server request; uploads are
      shared and progress is reported per image. When the batch finishes, review
      mode starts: **Previous/Next** switch candidates while their projection is
      shown live on the model — orbiting and zooming stay enabled. **Apply**
      commits the shown candidate through the same one-step native undo and sets
      the layer's seed to that candidate; **Layer** keeps the shown candidate as
      a new editable layer directly above the source (shared saved view and
      visibility snapshot, inherited generation settings, the candidate's seed;
      the source layer stays untouched; one undo step removes it again);
      **Discard** drops only the shown candidate; reviewing ends when
      the last one is gone. Review ends automatically on
      save/load/undo/redo or when the target layer changes.

Capture includes all visible layers, other slots and viewport-visible objects using
scene lighting. It excludes overlays and the scene compositor/sequencer. Input uses
Standard/sRGB with neutral exposure, avoiding a second display transform when the
result is projected; the artist's render/color/camera/visibility settings are restored.
Scene-lighting capture can differ from Material Preview's studio lighting. Modifiers
follow their viewport enable state; render-specific subdivision levels and other
engine differences still warrant scene-specific testing.

One request runs at a time. Stay on the originating object/slot/layer and keep its
viewport open; orbiting to inspect is fine. Switching scene, target or selection,
starting painting, saving/loading, undo/redo, or disabling/reloading cancels ownership.
Direct image edits made while waiting cause the result to be discarded at completion.
Other scene changes do not recapture context: a request uses its submission snapshot.
Cancel removes only this queued prompt; an already-running server job may finish,
but its output is discarded. Pawprint never uses ComfyUI's global interrupt.
Unique uploaded inputs and temporary server outputs remain on ComfyUI; local job
files are cleaned up. Requests time out after 20 minutes of queue/execution polling.

This slice accepts byte RGBA, straight-alpha sRGB layer images up to 2048 pixels per
axis, as created by Pawprint. An excessively large viewport may exceed Blender's
native brush radius; reduce the viewport if reported. Geometry depth is generated
fresh from visible surfaces, including other objects, independently of the layer's
stored occlusion snapshot. Near is white, far/background black; maps share the
image/mask crop and resolution. Transparent surfaces are treated as solid and
shader displacement/volumes are excluded. Geometry capture is synchronous too.

Depth guidance uses one union ControlNet: it requires the standard ControlNet
loader/apply nodes plus the `SetUnionControlNetType` selector, with Xinsir's Pro Max
SDXL weights (`sdxl_promax.safetensors`) tuning the union type for the depth apply,
which receives the checkpoint VAE. Selection inpainting itself uses only core nodes
(`InpaintModelConditioning`, `ImageCompositeMasked`) and needs neither a ControlNet
nor `comfyui_controlnet_aux`. The combined workflow was
live-tested at denoise 1.0 with full selection replacement. IPAdapter requires the ComfyUI_IPAdapter_plus loader and
apply nodes, an installed SDXL IPAdapter model, and its paired CLIP Vision encoder
(ViT-H adapters need the ViT-H encoder; ViT-G/bigG adapters need the bigG encoder).
Discovery only advertises models whose encoder is present, so connect again after
adding weights. The reference uses fixed first-slice wiring (linear weight type,
full sampling range, `K+V w/ C penalty` embeds scaling); this was live-tested with
the installed SDXL Plus ViT-H model. The Z Image Turbo adapter requires
`UNETLoader`, `CLIPLoader` (type `lumina2`), `VAELoader`, `CLIPTextEncode`,
`ConditioningZeroOut`, `LoraLoaderModelOnly` and `ModelSamplingAuraFlow`; depth
guidance additionally needs `ModelPatchLoader` and the **Apply Qwen Image
DiffSynth ControlNet** node (`QwenImageDiffsynthControlnet`). The tested local
weights are `z_image_turbo_bf16.safetensors`, `qwen_3_4b.safetensors`,
`ae.safetensors`, and a `zit/` LoRA such as
`zit/skin texture Photorealistic style v4.5.safetensors`, plus the Fun
ControlNet Union patch
(`Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2601-8steps.safetensors`).
Composite-estimated depth is
the approved second source but is not implemented yet; the local Marigold setup
needs additional depth weights for it. Other ControlNets, ZIT reference guidance
and baking are not exposed yet. Batch
review adds no server dependency: it reuses the single-image workflow.

## Fast development: link the source once

Link the source into Blender's existing local extension repository. Blender loads
the same files you edit, so ordinary Python/UI changes need no ZIP rebuild.

1. Locate the existing local repository directory. For Blender 5.2 on this Linux
   setup, it is `~/.config/blender/5.2/extensions/user_default`.
2. From this project root, link the inner `pawprint/` package into it:

   ```sh
   ln -s "$PWD/pawprint" "$HOME/.config/blender/5.2/extensions/user_default/pawprint"
   ```

   Adjust the repository path for your Blender version/platform. If Pawprint is
   already ZIP-installed at that destination, uninstall that copy first. On
   Windows, use a directory junction/symlink to the same inner package. The
   resulting `pawprint/blender_manifest.toml` must be directly under the repository
   directory. Do not link the whole project as the extension package.
3. Refresh local extensions (or restart Blender), find Pawprint under Add-ons,
   and enable the development copy. Disable any ZIP-installed copy first: the
   two copies register the same panel names.
4. Edit the source and save. In Blender use **F3 → Reload Scripts**. If the command
   is hidden, enable **Preferences → Interface → Developer Extras**. Alternatively
   run `bpy.utils.load_scripts(reload_scripts=True)` in Blender's Python Console.
5. Return to the viewport; the sidebar uses the reloaded code.

Reload Scripts reloads other enabled Python addons too. Pawprint uses relative
imports, symmetrical registration, and submodule reload support for this native
workflow. Disabling/re-enabling alone is not a reliable way to refresh imported
Python files. Restart Blender after manifest changes or if a broken edit interrupts
reload. Future data-model changes may need their own migration/restart procedure.

## Build a distribution package

From this project root:

```sh
mkdir -p dist
blender --background --factory-startup --command extension validate pawprint
blender --background --factory-startup --command extension build --source-dir pawprint --output-dir dist
```

The extension ZIP contains the manifest and Python modules at its root. Product
specifications and development tooling live outside the installable package.

## Verification

Run the focused smoke check with system Python:

```sh
python3 tools/smoke_test.py
python3 tools/smoke_test.py --interactive
python3 tools/locked_paint_probe.py
python3 tools/locked_paint_probe.py --managed
python3 tools/selection_test.py
python3 tools/selection_event_test.py
python3 tools/result_apply_probe.py
python3 tools/backend_test.py
python3 tools/generation_test.py --capture-only
# Requires the live local server and EpicRealismXL checkpoint:
python3 tools/generation_test.py
# Same live workflow with geometry depth, aligned-map assertions and preview:
python3 tools/generation_test.py --depth
# Same live workflow with IPAdapter reference conditioning:
python3 tools/generation_test.py --ipadapter
# Same live workflow with selection inpainting (InpaintModelConditioning + ImageCompositeMasked) at denoise 1.0:
python3 tools/generation_test.py --inpaint
# Same live workflow with the Z Image Turbo adapter (z_image_turbo + LoRA stack,
# DiffSynth ControlNet depth patch, reference selection inpainting):
python3 tools/generation_test.py --zit
# Same live workflow with a three-candidate batch: review, switching, commit
# (seed adoption, one-step undo) and a discarded second batch:
python3 tools/generation_test.py --batch
# Live runs also fetch the submitted workflow from server history and verify the
# encode/preprocessor/ControlNet nodes the server actually received.
python3 tools/projection_test.py
python3 tools/baking_test.py
# Native paint undo interleaved with the commit's undo step:
python3 tools/baking_test.py --interactive
blender --background --factory-startup --python-exit-code 1 --python tools/image_undo_probe.py
# Or select another Blender installation:
python3 tools/smoke_test.py --blender /path/to/blender
```

The smoke script creates a temporary source-linked repository and isolated Blender user
directories, launches with factory settings, and checks enable, native script
reload, disable, and re-enable. It does not save your preferences or a scene.
It checks real `bl_ext.<repository>.pawprint` imports, not just a stand-alone
Python package import. It also checks property/operator/overlay lifecycle, data
surviving reload, and slot/linked-mesh isolation. Windows may require permission
to create directory symlinks.
The optional `--interactive` run opens and closes a separate factory-settings
window, checking actual capture, preview/frame alignment, removal of all layers,
recreation, and view restoration. It requires a display: RegionView3D updates are
not safe in background Blender.

The projection check renders an isolated fixture in Cycles and Eevee, checking
perspective mapping on large quads, alpha, visibility, occlusion, sloped surfaces
at a different render resolution, backfaces, inspection from another angle, and
pixel edits. It also checks two-view alpha compositing, reorder/hide, missing and
deleted images, transparent upper-layer edits, removal undo/redo, and legacy data
migration. A fresh Blender process checks two-layer `.blend`/image persistence. The test
requires an environment capable of initializing Eevee graphics.

The baking check commits multi-view layers with partial alpha, a hidden layer, and
a remaining upper layer through the actual operator: appearance equality against
the pre-commit render, base-reference replacement at the same resolution, an 8-pixel
margin outside UV islands, unchanged unrelated slot/unselected pixels, one-step
undo/redo of base pixels and stack, retained upper-layer settings, missing-UV error
safety, temporary-datablock cleanup, and `.blend` persistence of the committed base.
Its `--interactive` variant additionally interleaves a real native paint stroke with
the commit's undo step. It requires a display.

The undo probe intentionally reports `pixels_restored: false` on Blender 5.2.2:
ordinary undo restores its scene-property sentinel but not a direct image-buffer
edit. This is not an undo implementation. The result-application probe now uses
a native clone stroke instead, through the saved-frame canvas and a temporary
feathered stencil. It verifies exact fully selected source pixels and unselected
pixel preservation, feathered alpha/color, other-layer isolation, interleaved brush
undo/redo, and redo after exiting painting and deleting the source image. Like the
painting probes, it requires a display and runs in a separate factory Blender window.

### Live generation checks

`tools/generation_test.py --capture-only` runs without ComfyUI in an isolated
Blender UI. It checks saved-view capture with Eevee and Cycles, the Workbench-to-Eevee
fallback, output images, original render-engine/settings preservation, and scene cleanup.

`tools/generation_test.py` exercises the shipped Connect/Generate operators in an
isolated factory Blender UI against the local server. It checks prompt/parameter
submission (including a full-width 64-bit seed), upper-layer/other-slot/viewport-only
object context, render/color/visibility and brush/clone settings restoration, exact
unselected/other-layer pixels, single-step native undo/redo, cancellation cleanup,
and rejection of stale pixels or changed target layers. `--zit` runs the same
pipeline through the Z Image Turbo adapter and additionally verifies the DiffSynth
ControlNet depth patch and inpaint wiring in the submitted graph.
`tools/backend_test.py`
uses a local fake HTTP server to check errors, missing nodes, parameter wiring and
cancellation during submission without interrupting unrelated jobs. Source-linked
smoke checks cover generation registration/timers and per-layer settings on reload.

The earlier diagnostic feasibility tool also remains available:

With ComfyUI running and the named SDXL checkpoint installed:

```sh
python3 tools/generation_probe.py --server http://127.0.0.1:8188 \
  --checkpoint epicrealismXL_pureFix.safetensors --output /tmp/pawprint-generation-probe
```

This opens an isolated Blender process, discovers the required server nodes and
checkpoint, renders the saved perspective frame, uploads uniquely named context
and mask PNGs, and queues one masked img2img job. It retains the inputs, crop/frame
metadata, workflow, server history, returned image, applied layer and test report
in the output directory. Uploads and the temporary output also remain on ComfyUI.
It does not clear the server queue or interrupt unrelated jobs.

The tested workflow uses ordinary VAE encoding (preserving the sketch), a latent
noise mask, seed 731, 12 steps and denoise 0.35. Visible upper layers, another
material slot and a surrounding object are explicitly checked in rendered context.
The disconnected selection includes an off-target island. Image/mask crop and
resize together; application uses the original image-space feather mask and bounds.
One returned patch is applied with native image undo, not direct target-buffer writes.

Verified with Blender 5.2.2 LTS and ComfyUI 0.36.0 on the local server. The retained
probe artifacts diagnose transport, placement and undo, not generation quality.
The shipped Generate workflow is tested separately as described above.
See [backend findings](docs/specification/backend-and-guidance.md#live-feasibility-findings).

### Painting and interactive checks

The locked-paint probe opens a separate factory-settings UI window and tests native
Texture Paint through a transparent, saved-frame canvas proxy. It verifies stroke
alignment on a rectangular frame over sloped geometry, alpha erasing, interleaved
native undo/redo, and preservation of the lower layer and target UVs. It requires
a display. Add `--managed` to exercise the actual Paint Layer operators: the
native header staying intact during a session (screenshot-compared, guarding
the brush color chooser), free
navigation (a misaligned view persists between strokes), the stroke gate (a
misaligned press restores the saved view and paints nothing), a real event-driven
stroke, restored settings, undo/redo across exit and layer switching, native
save/reopen, disabling during painting, and scene-switch cleanup. Source-linked
interactive smoke testing also reloads scripts during a painting session. Use
`--screenshot /absolute/path/review.png` to retain a visual-review screenshot
(this allows extra time for viewport shader compilation). See the
[feasibility findings](docs/specification/editing-and-generation.md#locked-view-painting-feasibility-blender-522-linux)
for the implementation and remaining interactive acceptance checks.

Interactive checks:

- Install/enable the ZIP and find the Pawprint tab.
- Inspect context with no active object, a non-mesh, a mesh without material
  slots, an empty slot, and an assigned material; switch active slots.
- Try the workflow above, including a rectangular image frame and a changed viewport
  size before restoring. Confirm the sidebar stays readable at normal widths.
- Paint/erase the image; check projection updates and native painting undo/redo.
- In Paint Layer, try mouse/tablet strokes, pressure/falloff, native brush sizing,
  orbiting between strokes and the snap-back stroke gate, and physical viewport
  resizing. Finish and switch layers.
- Try freehand lasso/Add/Subtract, cancellation and native selection undo. Adjust
  generation feather/padding, orbit/restore, then paint and Delete within disconnected areas.
- Save edited images using native storage controls, reopen, and restore the view.
- Try Generate with your own prompts, image sizes and scene lighting; inspect
  feathered seams and low-denoise sketch preservation. Test sidebar ergonomics
  and long prompts at your preferred UI scale.
- Disable/re-enable without duplicate panels or errors.
- With the source link, edit a label and Reload Scripts; confirm it updates.

Headless registration tests alone do not establish visual layout correctness or
image-pixel undo.

The isolated-UI selection check verifies rectangular saved-frame lasso mapping,
disconnected/subtracted raster footprints, feathering, clipped context bounds,
selection undo, native masked erasing and single-step deletion interleaved with
stroke undo/redo. It checks exact preservation outside hard selections with generation feather enabled,
other-layer isolation, native stencil-setting restoration, temporary-image cleanup,
and save/reopen of pixels and selection metadata. Reload smoke checks also cover
selection metadata and teardown while a selection stencil is active. The blue
footprint/green context preview was visually inspected in an isolated viewport;
physical freehand/tablet interaction and sidebar ergonomics remain manual checks.

### Verified for the current development prototype

On Blender **5.2.2 LTS** (Linux): source manifest validation, extension ZIP build
and validation, archive contents, the expanded source-linked registration/reload
smoke check, Cycles/Eevee projection tests, fresh-process save/reopen, and the
direct-pixel undo probe passed/reported as described above. Blender 5.0 is the
declared minimum but was not tested separately.
The user confirmed that linking the source directly into Blender 5.2's existing
`user_default` repository works. In connected Blender, a separate test scene
verified actual viewport capture, a 2:1 frame, orbit hiding the frame, restoring
after a lens change, and native Image Editor painting with undo/redo. The viewport
projection/frame were visually inspected. The multi-layer slice additionally
passed headless and isolated-UI smoke checks, expanded two-layer render/persistence
checks, and a connected visual inspection of the layer-list body using a temporary
review panel. Full-sidebar scrolling, physical viewport resizing, and the Install
from Disk UI remain manual checks.

## Design and development scope

- [Specification index](docs/specification/index.md)
- [Layers, projection, and base](docs/specification/layers-and-projection.md)
- [Editing and generation](docs/specification/editing-and-generation.md)
- [Backend adapters and guidance](docs/specification/backend-and-guidance.md)
- [Scope, open decisions, and validation](docs/specification/scope-and-validation.md)
- [Agent working conventions](AGENTS.md)

Implement in small slices; the full design is not the current feature list.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE). StableGen is a GPLv3 reference project;
the current implementation does not incorporate StableGen source code.
