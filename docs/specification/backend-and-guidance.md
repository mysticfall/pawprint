# Backend adapters and guidance

## ComfyUI reference environment

Assume a ComfyUI setup like StableGen's. Reuse its implementation as a reference
for communicating with the server, uploading input images, assembling workflows,
and retrieving results. New model support may require additional installed
models/nodes; do not assume that StableGen's existing setup includes everything.

Useful reference files:

- `../StableGen/stablegen/core/server_api.py`
- `../StableGen/stablegen/texturing/generator.py`
- `../StableGen/stablegen/texturing/workflows.py`
- `../StableGen/stablegen/texturing/rendering.py`
- `../StableGen/installer.py` (node/model metadata, source URLs and installed filenames)

Depth has two approved sources: **geometry (default)** for initial generation,
and **current-composite estimation** for preserving generated structure during
repair. This supersedes the earlier image-estimation-only decision: a flat unlit
base can lose the shape cues an image estimator needs. Other image-derived
spatial controls still use the current visible composite.

## Per-layer generation configuration

Each layer has one selected model family/base model, plus:

- Prompt and model-appropriate generation settings.
- An ordered LoRA stack with per-entry settings.
- A ControlNet stack with per-entry settings and supported preprocessing.
- Reference-image guidance such as IPAdapter where supported.

“Model stack” means that combination, not a sequential pipeline that generates
with one model and automatically refines with another.

**Proposed:** allow changing a layer's model family while preserving its pixels
and projection. Keep incompatible settings inactive and remember settings by
family. Exact switching/restoration UX remains to be settled; no cross-family
compatibility should be assumed.

## Capability-driven adapter API

The extension queries the current adapter through a common API and presents only
features it supports. Capability descriptions should cover:

- Operations: generation, img2img, masked editing/inpainting.
- Guidance types, preprocessors, and reference-image facilities.
- LoRA support and model/adapter compatibility.
- Parameter definitions, defaults, ranges, and choices.
- Image constraints and restrictions on combinations of features.

The same capability information must validate requests, not just drive the UI.
Saved settings can outlive model switches and backend changes.

Distinguish adapter implementation support from actual server availability of
nodes and weights. Unsupported controls should not be offered; supported
features missing a dependency may be explained as unavailable. Final schema and
discovery/caching details belong to a later implementation slice.

## Unified guidance contract
Guidance UI follows the same adapter-owned contract as the generation
parameters: each adapter publishes a concept-level table (`SDXL_GUIDANCE`,
`ZIT_GUIDANCE`) consumed through `guidance_for(adapter)`, mirroring
`parameters_for`. One generic renderer in `ui.py` walks the active adapter's
table, so a concept the adapter does not declare is structurally never drawn —
selecting Z Image Turbo hides every SDXL-only feature (IPAdapter reference,
union inpaint rows) by construction.

Each table is an ordered tuple of descriptor kinds:

- `feature` — concept entry (e.g. depth, reference): `toggle` RNA key, optional
  `picker` + `choices` collection + `empty` error + `unconnected` info,
  optional `source` label, `strength` RNA key, `image` template picker +
  `missing_image` error, `weight` RNA key, and `preview` operator id.
- `status` — `masked`/`unmasked` labels describing the selection-driven mode.
- `note` — a planned-work footer line.
- `loras` — an ordered stack editor: `choices` collection, `empty` error,
  `unconnected` info and `description`; each entry supplies its model and strength.

SDXL declares depth (union ControlNet picker, Source: Geometry label,
strength, depth preview) and reference (IPAdapter image/model/weight); ZIT
declares depth (DiffSynth patch picker and strength) only. The RNA properties
and the backend workflows are unchanged — this slice moves only presentation
into the tables. One small behavior change fell out of the rewrite: the SDXL
union ControlNet picker now sits inside the depth-enabled block, where it
belongs since plain selection inpainting stopped needing a ControlNet with the
InpaintModelConditioning redesign.

## Initial guidance priorities

1. **Depth:** saved-view geometry by default; alternatively image-based depth
   estimation from the current visible composite. Implement geometry first,
    with the composite-estimated option explicitly deferred now that geometry is available.
2. **Edges:** Canny and extensible edge/line variants from that composite.
3. **IPAdapter:** separate reference-image conditioning, presented alongside the
   spatial controls where the adapter supports it.

ControlNet and IPAdapter are not interchangeable mechanisms. Their capabilities,
weights, and UI parameters depend on model family.

Align spatial maps with context crops and model resolution requirements. Reference
images are semantically separate and should not automatically receive the spatial
crop applied to the target view.

**Open:** first concrete preprocessors/variants, whether preprocessing happens
before or after cropping, and how users select IPAdapter reference sources.

## Pluggable model families

Design for SDXL, Qwen-family workflows, Z-Image Turbo, and future implementations
without embedding family-specific assumptions throughout the layer system.
The final initial support matrix is not yet fixed.

**Candidate: Z-Image Turbo (ZIT).** Include in the initial product scope if the
required integration is reasonably small. ComfyUI's official tutorial documents
ZIT and Fun Union ControlNet support for Canny, HED, Depth, Pose, and MLSD:
<https://docs.comfy.org/tutorials/image/z-image/z-image-turbo>.

Before promising ZIT support, validate the actual installed workflow for masked
editing, low-denoise sketch refinement, LoRAs, and selected controls. Ordinary
masked img2img and a dedicated inpainting model are not equivalent guarantees of
quality. Also ensure sketch refinement preserves the sketch in the encoded input
rather than replacing the selected region with a blank fill.

Do not promise SDXL-compatible IPAdapter or universal feature parity across model
families. Capability discovery exists to express those differences.

## Live feasibility findings

The local ComfyUI 0.36.0 server was inspected through `/system_stats` and
`/object_info`. It advertises checkpoints `epicrealismXL_pureFix.safetensors`
and `chord_v1.safetensors`, plus Qwen Image Edit 2509 GGUF and associated text
encoder/VAE. The standard UNET loader has no installed weights; the GGUF loader
advertises Qwen only. ZIT nodes exist, but the required local ZIT model weights
were not advertised. Node presence alone is not model availability.

`tools/generation_probe.py` successfully tested the installed EpicRealism XL
checkpoint with ordinary `VAEEncode` → `SetLatentNoiseMask` → `KSampler` →
`VAEDecode`. This avoids blanking the sketch before encoding. One explicit test
configuration uses seed 731, 12 steps, CFG 5, DPM++ 2M/Karras and denoise 0.35;
these are probe settings, not a universal adapter parameter contract.

Inputs use `/upload/image`, requests `/prompt`, status `/history/{prompt_id}`,
and result download `/view`. Each upload has a unique name. The probe polls only
its own request, records its ID and errors, and does not interrupt unrelated jobs.
It uses `PreviewImage` for a temporary server output. Both cropped inputs use the
same model dimensions, rounded to multiples of eight; the returned patch is resized
to the original crop before application through the original feather footprint.

This establishes one masked img2img transport path, not dedicated inpainting
quality, LoRA/ControlNet/reference support, Qwen support, or the final model support
matrix. Those capabilities remain unvalidated.

## Implemented first SDXL adapter

The Generation panel now exposes per-layer positive/negative prompts, denoise,
CFG, seed, sampling steps, sampler, scheduler and request context resolution.
`pawprint/backend.py` owns their descriptors (names/defaults/ranges/choice sources);
model properties, UI drawing and request validation consume that same contract.
Discovery supplies installed checkpoint/sampler/scheduler choices and server ranges;
the worker validates again immediately before upload/submission. Checkpoint discovery
does not identify architecture, so the UI explicitly asks the user to choose SDXL.
The initial workflow is the tested ordinary-VAE latent-noise-mask path above.

Defaults are 20 steps, CFG 5, denoise 0.35, seed 731, DPM++ 2M/Karras, context long
edge 1024. Seed uses a string property to preserve all unsigned 64-bit values;
`-1` selects a random seed, and the last applied seed is shown. Context dimensions
round to multiples of eight, minimum 64 each. One image is generated per request;
sampling steps are iteration count, not candidate count. These are SDXL defaults,
not requirements for future adapters.

Prompt history: each prompt field has a dropdown listing previously used texts.
Entries are recorded when Generate validates a request, are stored per scene
(they persist in the `.blend`), deduplicate exact repeats, and are capped at 32
with the oldest dropped. Choosing an entry replaces the field; each list can be
cleared from its own menu. Applying a history entry is blocked while painting or
while a generation job is active.

Connect/Refresh and Generate launch a separate factory Blender process running a
bpy-free HTTP worker. The main thread captures/applies pixels and polls process
completion; imports, UI drawing and Blender-side background threads do no networking.
JSON/PNG temporary files carry immutable request snapshots. Only one job is owned
at a time. Cancellation during upload/submission is checked again after the server
assigns its prompt ID, so only that queued prompt is removed through `/queue`.
Running jobs are abandoned locally rather than globally interrupted. Server-side
uploads/temporary output remain; local files are removed after completion/cancellation.
HTTP calls time out after 30 seconds, queue/execution polling after 20 minutes.

## Implemented geometry-depth slice

Per-layer **Depth guidance** lives in the Guidance panel; it is opt-in and its
default source is geometry. It exposes strength (default 0.5, range 0–2) and
**Preview Geometry Depth**. The source selector will be added with the second
implemented source; composite estimation is not yet offered as an executable option.

The map is freshly raycast against evaluated viewport-visible geometry at the
saved view, including surrounding objects. It does not reuse or modify the layer's
static visibility snapshot. Positive camera-space depth is normalized across the
full frame: near white, far/background black; a constant-depth surface is white.
It then receives exactly the image/mask crop and request dimensions. The map is
Non-Color, independent of materials and display transforms. Preview works without
ComfyUI, opens a packed Image datablock in the current editor, and uses the same
crop/resizing as Generate. Shift-F5 returns to the 3D View.

Depth shares the union ControlNet contract with inpainting: one
`ControlNetLoader` on the shared `controlnet_model` (default Xinsir Pro Max
`sdxl_promax.safetensors`) feeds a `SetUnionControlNetType` selector pinned to the
`depth` type, whose output applies to both positive and negative conditioning,
active throughout sampling (start 0, end 1), with the checkpoint VAE attached.
Discovery and request validation require the loader/apply/selector nodes and a
supported union weight only when depth or inpaint is enabled. The earlier
dedicated depth-model allowlist (`controlnet_depth_sdxl.safetensors` /
`diffusion_pytorch_model.fp16.safetensors`, identified via StableGen metadata)
was superseded by the shared union design; those plain depth weights are no
longer wired.

Raycasting inherits projection-capture limits: surfaces are solid, with no shader
displacement or volumetric depth. Distant visible geometry influences normalization.
Guidance adds structural evidence; it does not guarantee a particular generated
texture or recover detail absent from geometry. The RGB input remains the current
visible composite with normal scene lighting.

The user-provided local installation is `~/workspace/ComfyUI`. StableGen's metadata
and the installed `ComfyUI-Marigold` node identify a possible composite estimator.
Its model loader downloads missing weights on first use. The current local
`models/diffusers` contains Marigold IID/material and normal weights but no Marigold
depth weights, so the second source needs additional weights and separate validation.
No weights were downloaded for the geometry slice.

## Implemented IPAdapter reference guidance
Per-layer **IPAdapter reference** lives in the Guidance panel: an enable toggle, a
native image datablock picker with Blender's
standard **new/open** controls — the folder button opens a file browser so external
images load without visiting another editor first — the adapter model, and a weight
(default 1.0, range 0–2). All of these are stored on the layer, matching the
saved-view each layer carries. The reference is exported whole at its native
resolution; the saved-view crop
and context scaling are deliberately not applied because reference images are
semantically separate from the target view. New layers inherit these settings with
the other generation fields.

The local ComfyUI_IPAdapter_plus installation advertises
`ip-adapter-plus_sdxl_vit-h.safetensors` plus both common CLIP Vision encoders.
Discovery requires the model loader, `IPAdapterAdvanced`, and `CLIPVisionLoader`
nodes, and only advertises adapters whose paired encoder is installed: ViT-H models
need `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors`, ViT-G/bigG models need
`CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors`. SD15-class filenames have no
pairing in this SDXL contract. Ordinary generation remains available when the
nodes/weights are absent.

The workflow adds `IPAdapterModelLoader` → `CLIPVisionLoader` → `IPAdapterAdvanced`
patching the base model, so it composes with depth guidance: IPAdapter rewires the
sampler's model input while ControlNet rewires its conditioning. First-slice wiring
is fixed at linear weight type, `concat` embeds, full sampling range (start 0, end 1)
and `K+V w/ C penalty` embeds scaling; those defaults are not exposed as per-layer
controls yet. The complete live path (discovery, reference export, generation,
native single-step undo/redo and stale-target guards) was verified with the installed
SDXL Plus ViT-H model and EpicRealismXL.

## Implemented inpaint guidance
After several rounds of user review, the design settled on two principles: **the
existence of a selection alone selects the workflow** (no toggle, no
denoise-threshold switching), and — per the user's latest reference graph — the
masked path uses **`InpaintModelConditioning`** rather than repaint ControlNets,
preprocessors or `VAEEncodeForInpaint`:

- **Selection present** (any size, any denoise): text encodes flow through the
  optional depth apply into `InpaintModelConditioning` (checkpoint VAE, input
  pixels, selection mask, `noise_mask=True`). The core node emits both the
  inpaint-aware conditioning (slots 0/1) and a noise-masked latent (slot 2); the
  sampler takes all three. The decoded result then passes through
  `ImageCompositeMasked` (destination = the original uploaded input, source = the
  decoded image, same mask, zero offset, no resize), so the server itself returns
  pixels that are exactly the original outside the mask. Core nodes only — the
  `comfyui_controlnet_aux` dependency is gone.
- **No selection** (`Clear`): plain `VAEEncode` + `SetLatentNoiseMask` over the whole
  frame — ordinary img2img. The latent is always created from the rendered composite
  of the framed area, even at denoise 1.0; an empty-latent init was considered and
  explicitly dropped during review.

Denoise governs only how much existing content survives sampling; it never changes
the graph structure. `Generate` derives the `masked` flag from
`layer.selection_paths` and passes it with the request; the worker's `validate` and
`workflow` read it, and it is not a persisted RNA property. The server-side
composite is a belt-and-braces preservation; the client still applies the result
through its own feathered selection stencil, so unselected layer pixels remain
untouched regardless of what the server returns. There is no inpaint strength knob:
selection alone routes inpainting.

Depth guidance is the only remaining ControlNet consumer: masked requests validate
on a bare server, and `controlnet_model` is checked only when `depth_enabled` is
set. Depth applies first on the conditioning path (nodes 12/23/14), the resulting
positive/negative feed `InpaintModelConditioning`, and IPAdapter keeps patching the
model separately, so all three compose. Every `ControlNetApplyAdvanced` still
receives the checkpoint VAE.

The masked path reuses the single composite upload, so no extra server input is
created. The complete path (discovery, generation in all five live-variant shapes —
unmasked plain img2img, masked model-conditioning runs with depth, IPAdapter, and
denoise 1.0 with mean-delta verification inside the full-weight footprint, plus the
former Qwen magenta variant, since removed — pixel isolation, single-step undo/redo, settings
restoration, cancellation and stale-target guards) was live-tested against ComfyUI
0.36.0 with EpicRealismXL and `sdxl_promax.safetensors`; unit tests cover capability
filtering, graph shapes for both modes, the model-conditioning/composite wiring and
exactly three uploads (input/mask/depth) without a server.

## Implemented LoRA stacks

Each layer owns an ordered LoRA stack whose entries are a server model name and a
strength (default 1.0, UI range -3 to 3). The Guidance panel renders the same
stack concept for both adapters from their guidance tables, while discovery keeps
their valid choices distinct: SDXL accepts every `LoraLoaderModelOnly` weight the
server reports and ZIT accepts only its `zit/` or `z-image` weights. Validation
rejects a stack entry outside the active adapter's list.

The workflow applies entries in listed order through `LoraLoaderModelOnly`: SDXL
chains from the checkpoint before the sampler and optional IPAdapter; ZIT chains
from its UNET before the optional DiffSynth depth patch. This supersedes ZIT's
former single Style LoRA field.

Both adapters expose **CLIP skip** (1--24, default 1). Its value feeds
`CLIPSetLastLayer` as the corresponding negative `stop_at_clip_layer` before
their prompt encoders, including ZIT's Lumina2 Qwen CLIP path.

## Implemented Z Image Turbo adapter

The second adapter realizes the pluggable-family design: `backend.py` now exposes a
small registry (`ADAPTERS`, `parameters_for(adapter)`, `ALL_PARAMETERS` merging the
per-family parameter tables) and `validate`/`workflow` dispatch on the request's
`adapter` field. Each layer picks its family with the **Model type** enum
(`generation_adapter`, SDXL by default); the field inherits to new layers and
last-used snapshots like every other generation setting. Shared RNA properties
(prompts, seed, denoise, CFG, steps, sampler, scheduler, resolution) take the SDXL
defaults; the ZIT table carries its own validation defaults (steps 8, CFG 1.0,
res_multistep/simple, denoise 1.0). The worker stays dispatch-based and per-family.
It replaces the earlier Qwen Image Edit adapter, which never shipped and was
removed without migration (pre-release compatibility is not required).

Z Image Turbo's workflow follows the user's reference template and confirmed node
choices (no code reused):

- **Guidance is a DiffSynth model patch, not a conditioning ControlNet.** The model
   chain is `UNETLoader` (`z_image_turbo_bf16.safetensors`) → optional ordered
   `LoraLoaderModelOnly` stack (skipped when empty) → optional depth
  guidance (`ModelPatchLoader` with the Fun ControlNet Union patch, default
  `Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2601-8steps.safetensors`, applied
  through `QwenImageDiffsynthControlnet` — "Apply Qwen Image DiffSynth ControlNet" —
  which receives the model patch, the checkpoint VAE, the depth image and a
  strength) → `ModelSamplingAuraFlow` (shift 3, sampling `flow`) → `KSampler`.
  The text encoder is `CLIPLoader` (`qwen_3_4b.safetensors`, type `lumina2`) with a
  plain `CLIPTextEncode` positive and a `ConditioningZeroOut` negative — turbo
  samples at CFG 1, so the negative prompt field has no effect for ZIT.
- **Selections use the reference inpainting pipeline.** A selection runs a
  dual-path workflow modelled on the user's known-working reference: the uploaded
   mask is expanded (`INPAINT_ExpandMask`, grow = the layer's feather value, linear
   blur) into the generation mask. The Fun ControlNet patch runs in inpaint mode
   (`ZImageFunControlnet` with `inpaint_image` + the mask — binarised at full
   denoise, the expanded feather mask below — and an optional depth image) at
   every denoise, so context, depth guidance and Control strength stay effective
   during refinement. At denoise 1.0 the region is additionally pre-filled by
   a dedicated MAT inpaint model (`INPAINT_LoadInpaintModel` +
   `INPAINT_InpaintWithModel`, auto-picked from the server list like the CLIP/VAE)
   and the pre-filled pixels are encoded as the latent.
   Below denoise 1.0 the original composite is encoded directly (no pre-fill)
   and `SplitSigmas` drops the strongest sigma for a softer refinement.
   Both paths wrap the model in `DifferentialDiffusion`, sample through the
  advanced stack (`RandomNoise`/`KSamplerSelect`/`BasicScheduler`/`BasicGuider`/
  `SamplerCustomAdvanced`) and finish with `INPAINT_ColorMatch` (against the
  pre-fill at full denoise, the original below) using the expanded mask as
  `exclude_mask`, so repaired pixels blend with the context while unselected
  pixels stay exact. Without a selection the latent is a plain `VAEEncode` of the
  composite plus `SetLatentNoiseMask` whole-frame img2img through
  `ModelSamplingAuraFlow` + `KSampler`. The mask is uploaded for both adapters.
- **Prompts are plain text** — no edit-instruction wrapping, no magenta marking.
  In the masked pipelines the positive-only `BasicGuider` drives sampling, so the
  negative prompt has no effect there either.

Discovery requires the ZIT node set (`UNETLoader`, `CLIPLoader`, `VAELoader`,
`CLIPSetLastLayer`, `CLIPTextEncode`, `ConditioningZeroOut`, `ModelSamplingAuraFlow`, `VAEEncode`,
`SetLatentNoiseMask`, `KSampler`, `VAEDecode`, `LoadImage`, `PreviewImage`,
`DifferentialDiffusion`, `ZImageFunControlnet`, `INPAINT_ExpandMask`,
`INPAINT_StabilizeMask`, `INPAINT_ColorMatch`, `INPAINT_LoadInpaintModel`,
`INPAINT_InpaintWithModel`, `ThresholdMask`, `SplitSigmas`, `RandomNoise`,
`KSamplerSelect`, `BasicScheduler`, `BasicGuider`, `SamplerCustomAdvanced`) and
advertises `zit_unets` (UNETLoader list), `zit_loras` (names containing `zit` or a
`z-image` basename), `zit_controlnets` (ModelPatchLoader names, when
`ModelPatchLoader` plus either ControlNet apply node are installed) and
`zit_inpaint` (the MAT-style inpaint model; combos in both the classic options-list
and the newer `[type, config]` object_info shapes are understood); the CLIP
(`qwen_3` token) and VAE (`ae` token) names are resolved automatically. Validation
   rejects ZIT requests on servers without those nodes, without the chosen unet,
   without an available ControlNet patch when depth guidance is enabled or any
   selection runs (masked requests use the patch at every denoise), and without
   an inpaint model for full-denoise
   selections. The
Guidance panel for ZIT layers offers the depth toggle, patch picker and Control
strength instead of the SDXL stack; reference guidance is future work.

The live path (discovery, connect, generation with a selection, submitted-graph
assertions including the DiffSynth patch chain, pixel isolation, single-step
undo/redo, cancellation and stale-target guards) was verified against ComfyUI
0.36.0 with `z_image_turbo_bf16.safetensors`, `qwen_3_4b.safetensors`,
`ae.safetensors`, the `zit/skin texture Photorealistic style v4.5` LoRA and the
Fun ControlNet Union patch; fake-server unit tests cover dispatch, capability
filtering, patch gating, node wiring, LoRA skipping and upload counts
(input/mask/depth).

## Implemented batch review
Both adapters expose a **Batch** count (any positive integer; the slider's soft
maximum is eight, larger counts can be typed). Above one, Generate resolves one
random 64-bit seed per candidate client-side (the seed field is bypassed) and the
worker submits one prompt per seed, reusing the single set of uploads. Results
arrive as `result-<i>.png` with `progress.json {done,total}`; single-image jobs
keep the previous `result.png`/`queued.json` contract. Cancellation and failure
retire every prompt this worker owns with one `/queue` delete call — never the
global interrupt.

On completion the UI enters review mode: candidates are held in memory and shown
through a temporary preview datablock swapped into the layer's texture node, so
the viewport stays fully interactive (no modal, no navigation lock). Every
Previous/Next switch swaps in a **fresh** preview datablock and removes the old
one: rewriting the pixels of an image the viewport shader already bound left the
GPU texture stale in interactive use, while a datablock swap reliably refreshes
it. The sidebar
offers Previous/Next, **Apply** (closes review, applies through the native clone
with one undo step, and adopts the chosen candidate's seed into the layer's seed
field and last-seed), **Layer** (keeps the shown candidate as a new editable layer
directly above the source: a packed byte image of the candidate pixels, the
source's saved view/projection metadata and visibility-snapshot images shared,
generation settings inherited, the candidate's seed adopted on the new layer, the
source layer untouched, one native undo step) and **Discard**, which drops only
the shown candidate from the batch — the next candidate slides into its place and
reviewing continues until the last candidate is discarded, which closes review.
Review validity is
enforced each timer tick: window/scene/owner/slot/layer/selection identity, image
digest and node self-healing after material rebuilds. Save/load/undo/redo end
review before the data change.

The add-as-layer operator mirrors `add_projection`'s proven write order: no field
written before the final image assignment may rebuild the material
(`depth_tolerance` owns a sync callback, so it travels through raw dict access),
and the layer wrapper is re-fetched after `layers.move()` because collection
moves invalidate the wrapper `layers.add()` returned — writes through it are
silently lost, and in one investigated ordering the mid-initialization rebuild
deadlocked the GUI.
