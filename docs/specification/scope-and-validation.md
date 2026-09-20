# Scope and validation

## Current slice: ZIT batch review (version unchanged at 0.2.0)

The user found that the Batch property was only enabled with SDXL. Batch is now a
shared adapter parameter: Z Image Turbo submits one random-seed request per
candidate and uses the same review, apply, add-as-layer and discard flow as SDXL.

Verified on Blender 5.2.2 LTS / ComfyUI 0.36.0: `python3 tools/backend_test.py`
(12 tests), headless smoke, and `TMPDIR=/tmp/opencode python3
tools/generation_test.py --zit --batch`.

## Previous slice: LoRA stacks (version unchanged at 0.2.0)

The user asked: "Can you improve the LoRA UI, so I can use more than one of them
with configurable strength each?" and approved: "Good. Let's proceed."

Each layer now stores an ordered LoRA collection, with per-entry model and strength
(default 1.0, UI range -3 to 3), inherited by new layers and preserved in the
last-used snapshot. The generic adapter-declared Guidance renderer supplies an
add/remove editor. SDXL exposes every discovered LoRA; ZIT exposes only compatible
`zit/` or `z-image` weights. Both workflows chain `LoraLoaderModelOnly` entries in
listed order before their downstream model patches. This replaces ZIT's former
single style-LoRA field.

Verified on Blender 5.2.2 LTS: `python3 tools/backend_test.py` (11 tests, including
adapter-specific availability validation and SDXL/ZIT chain wiring) and headless
`TMPDIR=/tmp/opencode python3 tools/smoke_test.py` (reload, inheritance and
last-used stack persistence).

## Previous slice: unified guidance contract (version unchanged at 0.2.0)

The user directed: "Like I said before, we should generalise over different base
models, unifying the API/UI over conceptually same feature like controlnets, even
though they may differ at the workflow level. Also, when we select the ZiT
architecture, there's little reason to show SDXL-related features to the user
either." — and approved the adapter-declared feature registry ("Sounds good to
me. Let's proceed.").

Each adapter now publishes a concept-level guidance table (`SDXL_GUIDANCE` /
`ZIT_GUIDANCE` + `guidance_for()`, mirroring `parameters_for`), and one generic
Guidance renderer walks the active adapter's table. An adapter that does not
declare a concept never shows it, so Z Image Turbo structurally hides SDXL-only
features (reference/IPAdapter rows never draw); SDXL shows depth + reference.
RNA fields and backend workflows are unchanged — presentation only. One visible
behavior change: the SDXL union ControlNet picker moved inside the
depth-enabled block, since plain selection inpainting needs no ControlNet after
the InpaintModelConditioning redesign.

Verified on Blender 5.2.2 LTS / ComfyUI 0.36.0: `python3 tools/backend_test.py`
(10 tests, including the new guidance-table contract test: SDXL declares depth +
reference with the expected RNA keys, ZIT declares depth only), smoke headless
and interactive, and a live default-variant canary (all five report keys True).

## Previous slice: Z Image Turbo adapter (version unchanged at 0.2.0)

The user requested replacing the unfinished second adapter: "Let's add a model
type selection, so that the user can choose a Qwen instead of SDXL workflow"
(landed first as Qwen Image Edit), then "I want you to replace the incomplete
Qwen image support with Z Image Turbo. Use @z-image-turbo.json as a reference,
and we'll use Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2601-8steps.safetensors
as a control net." After a server inventory confusion the user confirmed the
controlnet wiring: "The models were incorrectly named. I confirmed it working
with the `Apply Qwen Image DiffSynth ControlNet` node, combined with a
`Model Patch Loader`." The Qwen adapter was removed without migration
(pre-release compatibility is not required); see the
[ZIT adapter contract](backend-and-guidance.md#implemented-z-image-turbo-adapter).

Verified on Blender 5.2.2 LTS / ComfyUI 0.36.0:

- Fake-server backend checks (`python3 tools/backend_test.py`, 9 tests): ZIT
  node-set capability gating, unet/lora/patch choice validation, the
  UNETLoader → optional LoRA → optional DiffSynth depth patch
  (ModelPatchLoader + QwenImageDiffsynthControlnet) → ModelSamplingAuraFlow
  → KSampler wiring, always-present `SetLatentNoiseMask`, `ConditioningZeroOut`
  negative, LoRA skipping when unset, and exactly three uploads
  (input/mask/depth) with no global interrupt.
- Live isolated UI (`TMPDIR=/tmp/opencode python3 tools/generation_test.py --zit`,
  with depth guidance through the Union patch): discovery, submitted-workflow
  assertions (UNETLoader, lumina2 CLIPLoader, ConditioningZeroOut,
  QwenImageDiffsynthControlnet + ModelPatchLoader, SetLatentNoiseMask, no
  ControlNet/Inpaint nodes), replacement at denoise 1.0, pixel isolation,
  native single-step undo/redo, settings restoration, cancellation and
  stale-target guards.
- Regression: the five other live variants (default, `--depth`, `--ipadapter`,
  `--inpaint`, `--batch`) still pass; headless and interactive smoke cover the
  enum/RNA registration, reload inheritance and last-used snapshot (smoke was
  re-run after fixing stale Qwen fields left in `check_capture`).

Known limits: no reference guidance for ZIT yet (planned), stored `QWEN` enum
values fall back to SDXL silently, geometry is the only depth source.

## Previous slice: occlusion z-test (version unchanged at 0.2.0)

User report: "sometimes a projection affects the opposite side of the mesh. Is
there a way to improve our code to project onto only visible faces from the
layer's viewpoint more reliably?" — approved with "Yes, let's try that."

Diagnosis: the plane-distance test is blind along the view ray (grazing far-side
fragments whose offset is parallel to their own normal pass), and the facing test
depends on winding (flipped/inconsistent normals let far-side faces pass as
front-facing). The fix adds a winding-independent axial occlusion z-test —
reject when `w − depth > w × margin`, with a two-texel grazing margin scaled by
the texel footprint at the fragment's axial distance.

Verified on Blender 5.2.2 LTS:

- `tools/projection_test.py` (Cycles, save/reopen, Eevee): a tolerance-relaxed
  far-side floor fixture — built to leak honestly through the plane test, after
  two rejected fixture designs that proved the original "leak" demonstrations
  vacuous — is rejected post-fix, while a no-coverage-cut canary and all prior
  slope/rear-gap/backfacing/mirror checks still pass.
- The margin is extracted from the pure saved-window matrix after a node-graph
  dump revealed the combined window×view diagonal is rotation-polluted (a tilted
  saved view collapsed the margin to a huge negative number and rejected the
  entire layer).
- Smoke tests headless and interactive.

Known limits: sub-texel-thin gaps still pass; surfaces genuinely visible through
holes still receive the projection (spec-correct); extreme-grazing far sides rely
on the plane and facing tests.

## Previous slice: per-layer mirror (version unchanged at 0.2.0)

User request: "Can you add a mirror option to the layer UI? It should provide
None/X+/X-/Y+/Y-/Z+/Z- as options. … When one of the axis is selected, the
projection should be mirrored in the target object's selected local axis. The part
of the source image lies on the other side … should be ignored." Implemented as a
per-layer enum in Layer Details; the direction names the kept half, the fold
happens before projection across the object's local plane, the discarded image
half is never sampled, and candidate layers copy the setting (new layers do not
inherit it). See the [mirror contract](layers-and-projection.md#mirror).

Verified on Blender 5.2.2 LTS:

- `tools/projection_test.py` (Cycles and Eevee workers plus the save/reopen
  stage): asymmetric left-red/right-blue image halves — `X+` shows the blue
  (mirrored) half on the left quad while `X−` keeps the red half — with the depth
  snapshot covering the mirrored sample, and the pre-mirror mapping restored
  afterwards. One test-side pitfall documented: `pixels.foreach_set` is row-major,
  so half-selection must key on `index % width`, not the naive row split.
- `tools/smoke_test.py` headless and interactive: the mirror field registers and
  reloads, and the non-inheritance assertion (previous layer `Y−`, new layer
  `None`) holds.

## Previous slice: brush color chooser regression fix (version unchanged at 0.2.0)

After the free-navigation slice the user reported: "the colour chooser has
disappeared after the update." Root cause: that slice's session-wide header
banner (`Area.header_text_set` at session entry) — `header_text_set` replaces
the *entire* header contents, and in Texture Paint mode that strip hosts the
brush color chooser (isolated-probe evidence: 365 unique colors in the top 70
screenshot rows with the banner versus 800 with the native header).

Fix: no session banner at all. The native header stays visible for the whole
session; the only header text is the stroke gate's transient "start the stroke
again" hint, cleared by the next aligned press and at finish.

Verified on Blender 5.2.2 LTS: `tools/locked_paint_probe.py --managed` gained a
permanent regression — two screenshots taken during a session around a no-op
header clear are compared pixel-for-pixel in their top rows (a session banner
would make them differ), followed by all existing navigation/gate/event-stroke
and managed lifecycle/scene-switch checks; headless and interactive smoke pass.

## Previous slice: free navigation during painting with a stroke gate

The user asked "Is it possible not to lock the viewport when painting on a
layer?" and approved free navigation with stroke gating. The earlier navigation
lock existed because strokes map through saved-view eye rays onto a transparent
billboard: from any other pose the painted texel differs from the displayed one
(parallax). The approved semantics:

- Between strokes the viewport is fully free — orbit, zoom, inspect.
- A LEFTMOUSE press inside the painting region while the eye pose differs from
  the saved view is swallowed, the saved view is restored, and the header asks
  the user to restart the stroke; nothing is painted from a misaligned pose.
- While a stroke runs, navigation events are held so its rays stay accurate.
- Alignment compares the eye pose only (`RegionView3D.view_matrix`, elementwise
  tolerance 1e-4, perspective required): from the same eye, cursor rays hit the
  proxy point and the mesh point on one ray, so lens refits caused by viewport
  resizes do not matter.

Verified on Blender 5.2.2 LTS: `tools/locked_paint_probe.py --managed` with real
event-simulated input — the deliberately misaligned view persists (no
auto-restore watchdog), a misaligned press paints no pixels and restores the
saved view, an aligned real stroke lands within two texels of its expected
position, and all prior managed checks (EXEC strokes, alpha erase, interleaved
undo/redo, settings restoration, re-entry, lower-layer session, disable/reload,
save/reopen, scene-switch cleanup) pass. `tools/result_apply_probe.py`
(realigned via restore_view before its EXEC dab), headless and interactive
smoke, and backend tests still pass. Physical tablet feel of the gate remains an
interactive acceptance check.

## Previous slice: unlimited batch and per-candidate discard (version unchanged at 0.2.0)

Two review refinements: the user asked to remove the eight-candidate Batch cap
("the batch is currently limited to 8. Let's remove the limitaion."), then
reported that "pressing discard on a candidate removes the entire batch. It
should drop just the current candidate."

- **Batch** keeps its soft maximum of eight for the slider but has no hard cap;
  any positive count can be typed. Validation now treats a missing maximum as
  unbounded for every descriptor (regression-tested with batch 32).
- **Discard** removes only the shown candidate: the survivor list shrinks, the
  next candidate slides into the vacated slot, and review continues until the
  last candidate is discarded, which closes it. The per-batch teardown guards
  (preview removal, node restore, seed untouched) apply as before.

Verified on Blender 5.2.2 LTS / ComfyUI 0.36.0: `python3 tools/backend_test.py`
(9 tests, including the unbounded-batch validation regression), headless and
interactive smoke, and the live `tools/generation_test.py --batch` whose second
review session now discards one of two candidates (asserting the survivor is
displayed with its exact pixels) before discarding the final one (asserting full
cleanup); apply, add-as-layer and single-image sessions pass unchanged.

## Current slice: add candidate as a new layer (version unchanged at 0.2.0)

The user, after batch review worked: "Can you add a button to add a candidate as
a new layer? So, we'll have 3 options for a candidate - apply, discard, or add
as a layer." Review now offers **Apply / Layer / Discard**.

Verified on Blender 5.2.2 LTS / ComfyUI 0.36.0:

- Live isolated UI (`python3 tools/generation_test.py --batch`, third review
  session): **Layer** creates a new layer directly above the source with the
  candidate's exact pixels (packed byte image), the source's saved view,
  projection and visibility-snapshot images, inherited generation settings and
  the candidate's seed; the source layer's pixels and seed stay untouched; the
  stack order, active selection and material nodes rebuild correctly; one
  native undo step removes the layer again; a later discard still cleans up.
- Debugging this slice pinned two Blender hazards now encoded in the operator:
  writing a sync-callback field (`depth_tolerance`) mid-initialization rebuilt
  the material and the following pointer write deadlocked the GUI (fixed by the
  `add_projection` write order with a raw-dict bypass), and writes through the
  layer wrapper after `layers.move()` are silently lost (the wrapper is
  re-fetched before the final image assignment).
- Backend tests (9), headless/interactive smoke and the default live variant
  pass unchanged.

## Current slice: batch generation and candidate review (version unchanged at 0.2.0)

The user requested: a batch mode generating a configured number of images with
random seeds, review with switching while projections stay visible, interactive
view manipulation during selection, and seed adoption on commit.

Verified on Blender 5.2.2 LTS / ComfyUI 0.36.0:

- Fake-server backend checks (`python3 tools/backend_test.py`, 9 tests): a
  three-seed batch submits one prompt per seed with distinct seeds and shared
  uploads, writes `result-<i>.png` plus `progress.json`/`done.json`, deletes
  nothing when complete, and a mid-batch cancellation retires all owned prompts
  with one `/queue` call (no `/interrupt`).
- Live isolated UI (`python3 tools/generation_test.py --batch`): review opens
  with three distinct candidates (pairwise difference asserted), every switch
  swaps in a fresh preview datablock that the layer's texture node rebinds (the
  original in-place pixel rewrite left the viewport GPU texture stale in
  interactive use), the
  viewport/painting stay unlocked, commit applies the second candidate with
  seed adoption and exact zero-mask preservation with one-step undo/redo, and a
  second batch discards cleanly (preview removed, node restored, seed kept).
- All other live variants (default, `--depth`, `--ipadapter`, `--inpaint`,
  `--zit`) and headless/interactive smoke still pass; the batch RNA field,
  candidate operators and reload persistence are smoke-checked.

Known limits: candidates are held in RAM as uint8 full-frame arrays; review
self-heals but does not survive undo (ends by design).

## Current slice: model-conditioning inpainting (version unchanged at 0.2.0)

The user supplied a new reference graph (`inpaint-new.json`) and redirected the SDXL
masked workflow: "Now we use `Inpaint Model Conditioning` instead of the `repaint`
controlnet, `Inpaint Preprocessor` node, and `VAEEncode (for inpainting)` node. We
use `Image Composite Masked` to preserve the area outside the mask." Selection
existence still alone selects the mode (no toggle, no denoise threshold); the
`Repaint strength` knob is gone with the repaint ControlNet. See the
[inpaint contract](backend-and-guidance.md#implemented-inpaint-guidance).

Verified on Blender 5.2.2 LTS / ComfyUI 0.36.0:

- Fake-server backend checks (`python3 tools/backend_test.py`, 8 tests): masked
  requests validate on a bare server (no ControlNet or preprocessor required);
  `controlnet_model` is validated only when depth is enabled; masked graph shape
  (25 `InpaintModelConditioning` after an optional depth apply, 28
  `ImageCompositeMasked` rebuilding from the original input, PreviewImage fed from
  the composite, sampler latent slot 2, no `VAEEncode`/`VAEEncodeForInpaint`/
  `InpaintPreprocessor` anywhere); unmasked graph unchanged; worker run with
  masked+depth performs exactly three uploads and never the global interrupt.
- Live isolated UI (`TMPDIR=/tmp/opencode python3 tools/generation_test.py`, all
  six variants — default, `--depth`, `--ipadapter`, `--inpaint`, `--batch`, `--zit`
  — with submitted-workflow assertions fetched from ComfyUI `/history/<prompt_id>`):
  masked runs submit `InpaintModelConditioning` + `ImageCompositeMasked`, unmasked
  runs plain `VAEEncode` + `SetLatentNoiseMask`, replacement at denoise 1.0, pixel
  isolation, native single-step undo/redo, settings restoration, cancellation and
  stale-target guards.
- Headless and interactive smoke: removed RNA field (`generation_inpaint_strength`)
  and capability key register, reload and disable cleanly.

Known limits: fixed full-range depth application, single known union model (depth
only), `noise_mask` fixed true, composite mask reuse of the feathered upload.
Low-denoise repair keeps the same masked path — the model-conditioning latent
carries the unselected composite.

## Previous slice: Qwen Image Edit adapter (historical, superseded by Z Image Turbo)

The user requested a per-layer model type selector so "the user can choose a Qwen
instead of SDXL workflow", preserving user-facing functionality while the underlying
workflow differs ("Qwen has its own way of doing inpainting"). Implementation
follows the StableGen-researched Qwen approach with an adapter registry in
`backend.py` and a **Model type** enum per layer. See the
[Qwen adapter contract](backend-and-guidance.md#implemented-qwen-image-edit-adapter).

Verified on Blender 5.2.2 LTS / ComfyUI 0.36.0:

- Fake-server backend checks (`python3 tools/backend_test.py`, 8 tests): adapter
  dispatch, Qwen node-set capability gating, unet/lora choice validation, wrapped
  prompt and magenta instruction, GGUF→LoRA→AuraFlow→CFGNorm→KSampler wiring,
  image1/image2 references, LoRA skipping when unset, no ControlNet/inpaint nodes,
  and exactly two uploads (input + reference; the mask is not uploaded). A carried
  SDXL `depth_enabled` toggle on a Qwen layer is ignored: the worker uploads only
  artifacts the client actually wrote, so no missing `depth.png` failure.
- Live isolated UI (`TMPDIR=/tmp/opencode python3 tools/generation_test.py --qwen`,
  run with the carried depth toggle deliberately enabled):
  discovery, magenta fill inside the selection footprint of the prepared input,
  submitted-workflow assertions (`TextEncodeQwenImageEditPlus`, `UnetLoaderGGUF`,
  Lightning LoRA, no ControlNet/`VAEEncodeForInpaint`/`InpaintPreprocessor`),
  selection-content replacement, pixel isolation, native single-step undo/redo,
  settings restoration, cancellation and stale-target guards.
- SDXL regression: default, `--depth`, `--ipadapter` and `--inpaint` live variants
  still pass; headless and interactive smoke cover the enum/property registration,
  reload inheritance and last-used snapshot across adapters.

Known limits: Qwen depth/edge guidance and the image 3 context slot are future
work; shared RNA defaults come from SDXL (a layer switched to Qwen keeps its
current values until adjusted); ZIT remains a separate future candidate.

## Current slice: selection-driven ControlNet inpainting (version unchanged at 0.2.0)

Following review of the union-ControlNet redesign, the user settled the design:
**the existence of a selection alone selects the workflow** — "whether to use a
`VAEEncodeForInpaint` or plain `VAEEncoder` should depend on the existence of a
selection, NOT on any denoise threshold... We only have two different modes." After
the first selection-driven build the user corrected the remaining toggle: "the new
controlnet-based inpainting workflow should be used exclusively when there's a
selected region. We don't need any other mode of inpainting, or an option to toggle
it." An empty-latent init for denoise 1.0 was explicitly dropped: the latent always
comes from the rendered composite so lighting context reaches the server.

- Selection present → the ControlNet inpaint workflow, exclusively:
  `VAEEncodeForInpaint` (noise-filled masked latent, `grow_mask_by` 6) at any
  denoise, plus `InpaintPreprocessor` → union `repaint` ControlNet on the
  conditioning path. Out-of-mask latent carries the composite. No toggle exists;
  **Repaint strength** (default 1.0, range 0–2) is the only inpaint control.
- No selection (`Clear`) → plain `VAEEncode` + `SetLatentNoiseMask` whole-frame
  img2img at any denoise. Denoise governs only content survival, never the graph.
- `Generate` derives `masked` from `layer.selection_paths`; it flows through the
  request JSON, not RNA storage. Validation rejects masked requests on servers
  lacking the union machinery or the preprocessor; unmasked requests never need it.

Verified on Blender 5.2.2 LTS / ComfyUI 0.36.0:

- Fake-server backend checks (`python3 tools/backend_test.py`, 7 tests): masked
  requests require the union weight and preprocessor before upload, the masked
  graph carries the full inpaint workflow, the unmasked graph carries none of it,
  single shared `ControlNetLoader`, exactly three uploads and no global interrupt.
- Live isolated UI (`tools/generation_test.py` all four variants: default unmasked,
  `--depth`, `--ipadapter`, `--inpaint` at denoise 1.0 with mean-delta
  replacement): discovery, submitted-workflow assertions against ComfyUI
  `/history/<prompt_id>` — every masked run submits `VAEEncodeForInpaint`,
  `InpaintPreprocessor` and the union `repaint` selector; the unmasked run submits
  none of them — pixel isolation, native single-step undo/redo, settings
  restoration, cancellation and stale-target guards.
- Headless and interactive smoke: the removed `inpaint_enabled` RNA field and its
  UI toggle register, reload and disable cleanly.

Known limits: fixed grow-mask of 6 latent pixels, fixed full-range controlnet
application, single known union model, InpaintPreprocessor is a server dependency.

## Previous slice: inpaint discoverability and submitted-workflow verification (historical)

A user-extracted output workflow showed a request that silently used the ordinary
`VAEEncode` + `SetLatentNoiseMask` path because the replacement toggle was off at
Generate time. That slice opened the Guidance panel by default, added a high-denoise
reminder label (both superseded by the selection-driven design above), and — still
current — made `tools/generation_test.py` live runs fetch the *submitted* workflow
from ComfyUI `/history/<prompt_id>` (captured from the live job's `queued.json`) and
assert the graph the server actually received. The test URL is built from the scene
server string without assuming a missing scheme (a doubled scheme produced a
misleading DNS failure).

## Current slice: shared union ControlNet and preprocessor-based inpainting (version unchanged at 0.2.0)

Following the user's root-cause debugging and reference workflow from the ControlNet
author's repository, inpainting was rebuilt as one coherent mode and depth/inpaint now
share a single union ControlNet: **Inpaint guidance** (Guidance panel) switches the
latent stage to `VAEEncodeForInpaint`, feeds an `InpaintPreprocessor`
(comfyui_controlnet_aux, `black_pixel_for_xinsir_cn=True`) control image into the union
`repaint` type, and a single `ControlNetLoader` on the shared `controlnet_model`
(default `sdxl_promax.safetensors`) serves both depth and inpaint applies, each
receiving the checkpoint VAE. The separate depth/inpaint model pickers and the
`replace_content` switch were removed; depth controls moved from the Generation panel
into Guidance. See the
[guidance contract](backend-and-guidance.md#implemented-inpaint-guidance).

Verified on Blender 5.2.2 LTS / ComfyUI 0.36.0:

- Fake-server backend checks (`python3 tools/backend_test.py`, 7 tests): inpaint
  requires the union type selector and the preprocessor node; the submitted graph uses
  `VAEEncodeForInpaint` (no `SetLatentNoiseMask`), one shared `ControlNetLoader`
  serving depth and repaint selectors, the preprocessor fed by the single input and
  mask uploads, and exactly three uploads (input/mask/depth) with depth enabled.
- Live isolated UI (`python3 tools/generation_test.py --inpaint` and `--depth`):
  discovery, complete replacement at denoise 1.0 (mean pixel delta inside the
  full-weight footprint), pixel isolation, native single-step undo/redo, settings
  restoration, cancellation and stale-target guards.
- Headless and interactive smoke: the renamed/removed RNA fields register, reload and
  disable cleanly.

Known limits: fixed grow-mask of 6 latent pixels, fixed full-range controlnet
application, single known union model, InpaintPreprocessor is a new server dependency.
Low-denoise repair keeps the ordinary VAE encode path.

## Current slice: last-used prefill and prompt history (version unchanged at 0.2.0)

Emptying a stack no longer discards the working generation configuration. New
layers normally inherit the selected layer's settings; when the stack has no
layers (after committing everything or removing the last layer), they reuse a
snapshot of the last-used settings taken at that moment, including the IPAdapter
reference datablock pointer but excluding selections and last-seed history.
Prompt fields gained a history dropdown recorded at Generate time.

Verified on Blender 5.2.2 LTS / ComfyUI 0.36.0:

- Interactive smoke: remove both layers, re-add — checkpoint, prompts, CFG,
  padding and IPAdapter toggle/weight come from the snapshot, selections stay
  empty; scene history properties register/unregister cleanly.
- Bake test: commit-all writes the snapshot (`last_settings`) and it survives a
  save/reopen round trip.
- Live isolated UI (`python3 tools/generation_test.py`): Generate records both
  prompts, the history dedupes exact repeats and keeps newest-last order, menu
  entries apply remembered text back to the layer, and clearing works.
- Existing fake-server backend checks and baking/smoke suites re-run unchanged.

Last-used prefill covers the empty-stack case only; layer-to-layer inheritance
is unchanged. History entries are per scene, not per layer or global.

## Current slice: IPAdapter reference guidance (version unchanged at 0.2.0)

Per-layer IPAdapter reference conditioning is implemented in the Guidance panel:
enable toggle, native image datablock, discovered SDXL adapter model, and weight.
The reference is uploaded whole — no saved-view crop — and composes with geometry
depth. See the [guidance contract](backend-and-guidance.md#implemented-ipadapter-reference-guidance).

Verified on Blender 5.2.2 LTS / ComfyUI 0.36.0:

- Live isolated UI (`python3 tools/generation_test.py --ipadapter`): discovery of the
  installed SDXL Plus ViT-H model, reference export, complete live generation with
  IPAdapter conditioning, unselected/other-layer pixel isolation, native single-step
  result undo/redo, render/brush settings restoration, cancellation cleanup and
  stale-target rejection.
- System Python fake server (`tools/backend_test.py`): capability filtering pairs
  adapters with their required CLIP Vision encoder (ViT-G without bigG and SD15
  filenames are not advertised), validation rejects unavailable models, the workflow
  patches the sampler model input while depth keeps patching conditioning, and a
  missing reference image is refused.
- Smoke checks cover property inheritance for new layers (scalar parameters and the
  reference datablock) across source reload.

Reference-image influence/quality, weight-type variants (style transfer etc.),
multi-image references, LoRA stacks and edge controls remain later slices.

## Current slice: configurable base resolution and source-preserving commits (version unchanged at 0.2.0)

Base resolution is now a scene-wide setting (**Base width/height**, default
2048 × 2048, minimum 64, no upper limit), shown in Layer Details. It is used
when a stack is created and as the target size of every commit bake: the
existing base content is resampled into the new packed base copy, so commits
can target 4K or higher regardless of the previous base size. A per-stack
**Keep source layers** option keeps the committed layers hidden instead of
removing them, so resolutions can be retried and sources re-edited after undo.
Details in [layers and projection](layers-and-projection.md#commit-to-base).

Verified on Blender 5.2.2 LTS:

- Isolated UI (`tools/baking_test.py`, headless and `--interactive`): commit
  resamples 512² → 768×1024 with appearance preserved, preserved sources stay
  hidden with prior base pixels untouched, undo restores visibility plus the
  old base, redo restores the resampled pixels, and the removal path still
  works afterwards.
- Smoke checks cover the 2048² default base at stack creation and the
  Scene-property lifecycle across reload and disable.

Merge down between layers, composite-estimated depth, and quality checks remain
later slices.

## Previous slice: commit through selected layer to the UV base (version unchanged at 0.2.0)

**Commit Through Selected to Base** bakes the base plus the bottom contiguous
layer group through the selected layer into a new packed base copy (Cycles CPU
emission bake of the isolated slot, 8-pixel
EXTEND margin), then removes the committed layers — hidden ones contribute
nothing and are removed with the group. Higher layers remain fully editable.
One native undo restores the previous base reference, its pixels, and the
removed layers; errors leave the stack untouched. Details in
[layers and projection](layers-and-projection.md#commit-to-base).

Verified on Blender 5.2.2 LTS:

- Isolated UI (`tools/baking_test.py`, headless and `--interactive`):
  multi-view partial-alpha compositing matches the pre-commit render, unrelated
  slot and unselected base texels unchanged, margin present outside UV islands,
  retained upper-layer settings/selection/pixels, one-step undo/redo including
  base pixels and interleaved native paint strokes, missing-UV error safety,
  temporary datablock cleanup, and `.blend` persistence of the committed base.
- Smoke checks include the commit operator registration and reload lifecycle.

Merge down between layers, composite-estimated depth, higher-resolution bases,
retention of committed layers, and quality checks remain later slices. IPAdapter
reference guidance is now implemented as the guidance slice above.

## Previous slice: user-facing SDXL generation (version unchanged at 0.2.0)

The Generation panel now provides positive/negative prompts, denoise, CFG, seed,
sampling steps, checkpoint, sampler/scheduler and context resolution per layer.
Adapter descriptors drive properties/UI/validation. Connect/Refresh discovers
installed choices. Generate captures the visible saved frame, submits one masked
img2img request in an isolated process, and applies it through native image undo.
The async ownership/cancel policy is recorded in [editing and generation](editing-and-generation.md).

Verified on Blender 5.2.2 LTS / ComfyUI 0.36.0:

- Live isolated UI: actual Connect/Generate operators, upper layers/other slots and
  viewport-only objects in context, crop/feather placement, render/color/visibility
  and brush/clone settings restoration, unselected/other-layer isolation, single-step
  native result undo/redo, cancellation cleanup and stale-pixel/target rejection.
- System Python fake server: cancellation during submission deletes only its owned
  prompt; missing nodes and server validation/execution errors; prompt and common
  parameter wiring including unsigned 64-bit seed.
- Headless and isolated-UI source-linked registration/reload checks include generation
  classes/timers/handlers and persistent per-layer prompt/settings properties.

Personal scene lighting, generation quality, sidebar ergonomics, long-prompt entry,
and render-specific modifier differences remain interactive acceptance checks.
Geometry-depth guidance is now implemented as the first guidance slice (below).
Composite-estimated depth, other model-family adapters, baking and candidate
browsing remain later slices.

### Geometry-depth guidance verification

The agreed depth sources are geometry (default) and composite estimation. This
slice implements geometry first, including a fresh saved-view map, aligned crop,
SDXL depth model/strength, and packed Image Editor preview. See the
[guidance contract](backend-and-guidance.md#implemented-geometry-depth-slice).

Headless `tools/backend_test.py` verifies optional-dependency handling, excludes
unsupported/Union models, checks both conditioning branches and the unchanged
noise mask, and exercises the depth upload with owned cancellation. Registration
and reload use the normal smoke checks.

Isolated UI `tools/generation_test.py --depth` passed on Blender 5.2.2 LTS with
live ComfyUI and the Xinsir SDXL depth model. It checks map dimensions, depth
gradients, black background, near-white viewport-visible context added after the
layer's visibility snapshot, native result undo/redo, exact unselected-pixel
preservation, settings restoration, cancellation/stale targets, and preview.
First-generation artistic quality on the user's cube and comfortable guidance
strength/denoise choices still need hands-on evaluation. No user scene was changed.

## Earlier generation feasibility follow-up

The first live ComfyUI round-trip now passes in an isolated factory Blender UI
process (`tools/generation_probe.py`), using the installed EpicRealism XL checkpoint.
Saved-view composite context includes upper layers and surrounding scene content;
disconnected selection crops/masks survive resizing and return placement. The
result is applied to the existing layer image with a native clone dab and feather
stencil. `tools/result_apply_probe.py` verifies unchanged unselected/other-layer
pixels, feathered alpha/color, and interleaved native undo/redo after temporary
source cleanup. Redo needs no server request.

That follow-up added development probes; the current slice integrates managed
application, scoped capture, adapter capabilities and an explicit async lifecycle.
No package version change is needed. Tests used Blender 5.2.2 LTS and ComfyUI 0.36.0;
ZIT weights were not advertised by the server, so ZIT remains unvalidated.

## Existing layer/editing foundation

Deliver an extension named **Pawprint**, maintained by **Pawprint contributors**,
licensed **GPL-3.0-or-later**, declaring Blender **5.0.0** as its minimum.

Backward compatibility is not required until the first official release; prototype
data migrations are not an acceptance requirement for development slices.

The current approved implementation includes:

- A new material assigned to the active object slot, retaining the original
  material, with prototype state on the new material and an owner reference.
- A plain gray, unlit UV base and ordered native editable RGBA projection layers.
- Active-layer list with naming, visibility, reorder and remove controls. Selection
  restores the selected viewpoint without affecting visibility. Restore Layer
  View remains available after navigation. The UV base appears separately below
  the list.
- Pre-capture frame preview using the same framing calculation as capture.
- Add remains available after removal; missing images can be replaced or their
  layer removed. Legacy single-projection data migrates on load/reload.
- Perspective view/frame capture, restoring the view, and a viewport-only frame.
- Per-fragment perspective mapping, image alpha, slot isolation, and saved
  visibility/backface filtering in both Cycles and Eevee.
- Opening the native Image Editor for painting, with native stroke undo/redo.
- Managed Paint Layer mode: native color/alpha-erasing strokes over the visible
  composite, saved-view locking, explicit finish/Escape, and temporary canvas
  cleanup on exit, save/load, scene changes, and extension disable/reload.
- Native image/view persistence and source-linked reloads.
- Per-layer image-space Lasso/Box with Replace/Add/Subtract and Clear; blue hard
  selection overlay and optional green generation-context preview. No Select All.
- Generation-only feather/context padding, independent of painting and deletion.
- Native selection-limited painting and Delete Selected Pixels, with interleaved
  native stroke/delete undo and session-only stencil cleanup.

New layers start fully transparent. Asymmetric test patterns are confined to test
fixtures. Preview Generation Context displays the prepared server input (saved-view
composite, crop and request resolution) without a server connection; Show Context
Bounds remains the separate viewport overlay. Geometry depth is covered above;
composite-estimated depth is deferred by user decision.

Provide a distributable ZIP, installation instructions, and a fast source-linked
development loop so Python/UI changes do not require repackaging/reinstallation.

The editing foundation has multiple layers per stack, with dimensions chosen before
capture, and needs no server. Generation requires live ComfyUI. Baking and
post-content resizing remain product requirements for later slices.

### Prototype representation and limitations

The shader derives image coordinates per shading point from its world position
and the saved projection matrix, avoiding vertex-interpolated perspective errors.
A packed float depth image is raycast against evaluated viewport-visible geometry
at capture time. Shading reconstructs the nearest captured point and compares it
to the current fragment's geometric plane, avoiding nearest-Z gaps on sloped faces.
Backface eligibility uses geometric winding relative to the saved eye, independent
of the current inspection view. The depth image is internal visibility data,
**not generation guidance**.

This is a finite-resolution, static visibility prototype. Silhouette boundaries,
very close surfaces, and extreme geometry need further validation. Transparent
materials are treated as solid occluders; volumes and shader displacement are not
captured. Local-view/viewport visibility is used rather than render-only hiding.
The capture loops over image pixels synchronously, so large captures can pause
Blender. Layer defaults are 512² with controls limited to 64–2048 per axis;
base resolution is a separate scene setting (default 2048², minimum 64, no
upper limit).
Geometry changes are not monitored or enforced. These choices do not settle all
production visibility/performance decisions.

### Verification results (Blender 5.2.2 LTS, Linux)

- Headless: extension registration/reload/disable/re-enable, property/overlay
  cleanup, surviving data, other-slot and linked-mesh isolation.
- Headless Cycles and Eevee renders: large-face mapping, opaque/transparent
  regions, visibility toggle, foreground occlusion, hidden occluders, sloping
  surfaces at a different resolution, rear-surface rejection while inspecting
  from the side, backface rejection, and edited image samples.
- Headless multi-layer checks: two-view alpha compositing, reorder, hiding,
  unlinking/deleting images, transparent upper images revealing lower images,
  layer-removal undo/redo, and migration of legacy single-projection state.
- Isolated UI smoke check: preview/capture frame agreement, multiple captures,
  removing all layers, adding again, and restoring the active viewpoint.
- Fresh Blender process: saved image pixels, packed visibility, owner and image
  references, saved projection state, and shader image references survive reopen.
- Connected Blender: actual view capture, 2:1 image frame, frame hidden during
  orbit, restoration after a lens change, native Image Editor paint/undo/redo.
  Original scene retained separately during testing.
- Direct image writes: ordinary memfile undo restores a scene-property sentinel
   but leaves changed image pixels intact, despite an `UNDO` operator. See
   `tools/image_undo_probe.py`. The follow-up clone-based probe above establishes
   a separate native undo path without writing the target buffer directly.
- Isolated UI selection checks: lasso-to-image mapping on a rectangular frame,
  disconnected selection bounds, subtraction, feathering, selection undo,
  masked erase and native Delete interleaved undo/redo, unchanged unselected pixels,
  restored stencil settings, cleanup and save/reopen. Overlay visually inspected.

Physical viewport resizing, detailed sidebar layout, Install from Disk, and
native strokes interleaved with future extension pixel operations remain manual
or future checks. Blender 5.0 has not been separately tested.

## Intended initial product

- Perspective projection layers and an unlit committed UV base per object/slot.
- Native image data, visibility, saved viewpoints, image frames, and scene-aware
  projection coverage.
- Locked-view painting/erasing, arbitrary/disconnected selections, cropped or
  full-frame generation context, and active-layer result application.
- Per-layer model, LoRA, ControlNet, and supported reference guidance settings.
- Capability-driven model adapters, with ZIT a candidate pending validation.
- Compatible adjacent merges and bottom-contiguous commits into the base.
- Native undo/redo integration where feasible.

## Deferred / not required initially

- PBR decomposition and non-destructive PBR preview, while preserving extensibility
  and original image detail for that future work.
- Baking an existing material as the initial base.
- Orthographic viewpoints, cross-view projection merges, layer-level opacity, and
  separate editable layer masks.
- Automatic UV mapping, enforcement of stable geometry, automatic seam solving,
  and automatic multi-model generation chains.

## Open decisions

Detailed open points live next to their affected requirements. Before relevant
implementation slices, settle:

- Projection/occlusion representation and grazing-angle behavior.
- Base initialization, texture resolution, bake padding, and hidden-layer commits.
- Merge resolution compatibility and post-content framing changes.
- Model switching UX, initial model support matrix, exact preprocessors, and
  reference-image sources.
- Candidate handling beyond the first single-result adapter; broader request
  concurrency beyond the current one-job ownership/cancel policy.
- PBR preview/decomposition architecture when that feature is taken up.

## Early technical checks

1. **Image undo:** native brush strokes interleaved with pixel deletion/generation,
   merge/commit undo, and redo without backend requests.
2. **Projection:** perspective accuracy on low-poly meshes, occlusion, material-slot
   isolation, frame alignment, transparent overlaps, and stability on save/reload.
3. **Editing:** locked-view brush-to-image mapping and synchronizing native 2D edits.
4. **Context fidelity:** higher visible layers and surrounding objects included;
   overlays excluded; unlit stack appearance preserved through color management.
5. **Crop fidelity:** disconnected selections, padding, feathering, resized controls,
   and returned patch placement leave unselected pixels unchanged.
6. **Backend:** discover actual nodes/models and verify each adapter's promised
   workflows, especially low-denoise ZIT sketch refinement if selected.
7. **Commit:** preserve appearance and ordering while baking only a bottom group,
   with retained layers disabled and other layers untouched.

## Skeleton acceptance

- Validate manifest and build a ZIP.
- Smoke-test registration, removal, and reload under isolated factory settings.
- Manually install, enable, locate the sidebar, inspect empty/object/material
  states, and disable/re-enable without duplicate panels.
- Verify source-linked reloads update the UI without a ZIP roundtrip.
- Document the actual tested Blender release; declaring 5.0 compatibility does
  not mean every 5.x release has been tested.
