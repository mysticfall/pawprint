# Pawprint

Pawprint is a Blender 5.x extension for viewpoint-bound, editable texture layers,
using ComfyUI for generation. Its purpose is **manual seam repair through layered
painting and inpainting**, rather than automatic multi-view seamless texturing.

## Read first

- [Specification index](docs/specification/index.md)
- [Layers, projection, and base](docs/specification/layers-and-projection.md)
- [Editing and generation](docs/specification/editing-and-generation.md)
- [Backend adapters and guidance](docs/specification/backend-and-guidance.md)
- [Scope, open decisions, and validation](docs/specification/scope-and-validation.md)
- [Installation, development, and verification](README.md)

These pages are the durable record of the design conversation. Treat explicit
requirements as authoritative; do not silently promote open questions or proposed
approaches into requirements. Update relevant pages when decisions change.

## Current implementation

The development prototype supports an unlit UV base and ordered native RGBA
projection layers per target slot, each with a saved perspective view/frame and
visibility snapshot. A layer list supports selection, naming, ordering, visibility,
and removal; new frames can be previewed before capture. Native 2D Image Editor
painting and managed saved-view Paint Layer/alpha erasing work, with free
navigation between strokes and a stroke gate that restores the saved view before
an aligned dab (the native header and its brush color chooser stay visible; the
gate's hint is transient text only). Per-layer image-space
lasso/box selections, hard selection-limited painting/deletion and generation-only
feather/padded context preview work, and per-layer Mirror folding
(None/X±/Y±/Z± across the object's local axis plane; the chosen direction keeps
that half, the other image half is never sampled) is implemented. Clear means no selection/unrestricted editing;
there is no separate Select All command.
User-facing SDXL generation now supports per-layer positive/negative prompts,
denoise/CFG/seed/sampling steps, sampler/scheduler and context resolution. Clean
saved-view capture, isolated-process ComfyUI jobs, cancellation/stale-target guards,
 and native clone result undo are integrated. Optional SDXL geometry-depth guidance
with preview and strength is implemented; composite-estimated depth is explicitly deferred.
IPAdapter reference conditioning (native image datablock, discovered model with
paired CLIP Vision encoder, weight, uncropped upload) composes with depth guidance.
Depth guidance uses one union ControlNet (default Xinsir Pro Max
sdxl_promax.safetensors via SetUnionControlNetType; the apply receives the
checkpoint VAE). The latent mode is selected only by the existence of a
selection: selections use InpaintModelConditioning (core node emitting inpaint
conditioning plus a noise-masked latent) followed by ImageCompositeMasked to
preserve the unselected pixels server-side, with no ControlNet, no preprocessor
and no strength knob; no selection uses plain VAEEncode + SetLatentNoiseMask
whole-frame img2img; denoise never changes the graph.
A per-layer ordered LoRA stack with per-entry strength chains through
LoraLoaderModelOnly for both adapters; ZIT's former single style-LoRA field is
removed. A second adapter family is selectable per layer via Model type: Z Image Turbo
(UNETLoader z_image_turbo + optional LoRA stack, optional depth guidance through ModelPatchLoader +
QwenImageDiffsynthControlnet DiffSynth patch with the Fun ControlNet Union
weights, ModelSamplingAuraFlow shift 3 flow, CLIPLoader qwen_3_4b lumina2 +
CLIPSetLastLayer/CLIPTextEncode + ConditioningZeroOut negative at CFG 1, steps 8
res_multistep/simple denoise 1.0; selections run a reference inpainting pipeline —
denoise 1.0 pre-fills with a MAT inpaint model and steers ZImageFunControlnet in
inpaint mode, lower denoise refines the original latent through SplitSigmas, both
wrap DifferentialDiffusion, sample via the BasicGuider/BasicScheduler/
SamplerCustomAdvanced stack and finish with INPAINT_ColorMatch; no selection uses
VAEEncode + SetLatentNoiseMask whole-frame with AuraFlow/KSampler; plain prompts,
no instruction wrapping; shared batch review). backend.py
owns the adapter registry (ADAPTERS/parameters_for/ALL_PARAMETERS) with
dispatch in validate/workflow; shared RNA defaults come from SDXL. ZIT
reference guidance is future work. The Guidance panel renders adapter-declared
concept tables (backend.py SDXL_GUIDANCE/ZIT_GUIDANCE with guidance_for), so
adapters structurally show only the features they declare.
SDXL layers can generate a batch of any number of random-seed candidates; the worker
submits one prompt per seed with shared uploads, and completion opens an
interactive review mode (preview datablock swapped into the texture node,
Previous/Next switching, Apply commits via the native clone and adopts the
chosen seed, Layer keeps the shown candidate as a new editable layer above the
source sharing its saved view/depth snapshot with the candidate's seed, Discard
drops only the shown candidate (review ends when the last is gone);
save/load/undo/target change ends it).
Prompt fields keep a per-scene history dropdown recorded at Generate time (deduped,
capped at 32). New layers inherit the selected layer's generation settings, or the
last-used snapshot when the stack is empty (commit-all/removal). New layers start
transparent; Preview Generation Context shows the same prepared
input crop used by Generate without requiring ComfyUI. Commit Through Selected
to Base bakes the bottom contiguous group into a new packed base copy at the
scene-configured base resolution (default 2048², no upper limit; existing base
content resampled) with one-step undo; an optional per-stack Keep keeps the
committed sources hidden instead of removing them. Merge down and composite
depth remain future slices. Geometry is the agreed default depth source.
Existing single-projection data migrates on load/reload.

- `pawprint/blender_manifest.toml`: extension metadata, minimum Blender 5.0.0.
- `pawprint/__init__.py`: registration and module reload support.
- `pawprint/model.py`: material-owned prototype state and image references.
- `pawprint/projection.py`: per-fragment projection shader and saved visibility,
  including the winding-independent axial occlusion z-test with a two-texel
  grazing margin extracted from the pure saved-window matrix.
- `pawprint/operators.py`: stack/layer creation, view restoration, image editing.
- `pawprint/overlay.py`: viewport-only saved-image frame.
- `pawprint/painting.py`: managed native saved-frame painting and session cleanup.
- `pawprint/selection.py`: per-layer image-space lasso, raster footprints and context bounds.
- `pawprint/ui.py`: sidebar panels.
- `pawprint/backend.py`: SDXL parameter/capability contract and bpy-free HTTP worker.
- `pawprint/capture.py`: scoped saved-view render/crop and returned patch placement.
- `pawprint/result.py`: native clone result application and paint-setting restoration.
- `pawprint/generation.py`: discovery, per-request ownership, async process lifecycle/UI operators.
- `pawprint/baking.py`: slot-isolated Cycles emission bake and commit-through-selected operator.
- `tools/backend_test.py`: fake-server errors, capabilities and owned cancellation checks.
- `tools/generation_test.py`: live shipped-operator capture/apply/undo and stale-target checks.
- `tools/baking_test.py`: isolated UV commit bake, one-step undo/redo, interleave and persistence checks.
- `tools/smoke_test.py`: isolated Blender registration/reload smoke check.
- `tools/projection_test.py`: isolated Cycles/Eevee render and persistence checks.
- `tools/image_undo_probe.py`: direct-pixel/global-undo feasibility probe.
- `tools/result_apply_probe.py`: native clone result application with feathered
  stencil, exact unselected-pixel preservation and interleaved image undo/redo.
- `tools/generation_probe.py`: live ComfyUI discovery, saved-view composite capture,
  crop/mask round-trip and result application in a separate factory Blender process.
- `tools/selection_test.py`: isolated UI selection mapping, stencil editing,
  deletion/stroke undo, persistence and cleanup checks.
- `tools/selection_event_test.py`: real Blender event-loop lasso/box modes,
  modifiers, cancellation and clear semantics.
- `tools/locked_paint_probe.py`: isolated UI probe for native saved-frame proxy
  painting, alpha erasing, and interleaved native image undo/redo. Positive
  feasibility result; `--managed` exercises the actual Paint Layer session,
  cross-session/layer undo, settings restoration, save/reopen and scene cleanup.
- `dist/`: generated installable archives.

## Working conventions

- Backward compatibility is not required before the first official release.
  Do not add migration work to development slices solely to preserve older
  prototype data.

- Do not bump versions per slice. Keep the current manifest version unless a
  package distinction is necessary; the user requests `0.0.x` for development
  version distinctions, rather than increasing minor/major versions per slice.

- Implement in small, reviewed slices; an approved product specification is not
  authorization to implement every feature at once.
- Use native Blender Image datablocks. Keep pixels, projection/view metadata, and
  model-specific generation settings conceptually separate.
- Preserve Blender's normal editing/storage conventions; target native undo/redo.
  Validate image-pixel undo rather than assuming an operator's `UNDO` flag is enough.
  On Blender 5.2.2, native Image Editor strokes undo/redo correctly, but direct
  `image.pixels` writes survive ordinary memfile undo. A native clone dab through
  the saved-frame proxy now implements generated-result application with native
  image undo. Do not ship direct target-buffer writes as undoable generation.
- Selected deletion uses a full-frame native erase dab through a temporary image
  stencil, with one native image-undo step. This does not solve generated-result
  buffer application. Selection metadata is separate from image alpha; the stencil
  is session-only, not a persistent editable layer mask. Preserve off-target
  selections when computing padded context bounds.
- The layer's stored depth image is an internal visibility snapshot, not ControlNet
  guidance. Geometry guidance is freshly computed from the saved view and current
  visible geometry, independently of that snapshot. Geometry is the default depth
  source; composite-estimated depth is the approved second source for repair work.
  StableGen's `installer.py` records node/model dependencies. The local ComfyUI
  installation is `~/workspace/ComfyUI`; do not locate it through `/proc`.
- Backend adapters own capability descriptions; UI and request validation consume
  the same description. Do not hardcode a universal SDXL-style parameter set.
- Use package-relative imports suitable for `bl_ext.<repository>.pawprint`.
  Keep registration/unregistration symmetrical and development reloads working.
- Keep imports and UI drawing free of network requests and scene mutations.
- Generation HTTP runs in a separate factory Blender process, not a bpy thread.
  Cancellation removes only its own queued prompt and discards running output;
  never use the shared server's global interrupt. Keep stale-target checks intact.
- Run focused checks appropriate to each slice. Report separately what was tested
  headlessly and what needs interactive Blender verification.
- Keep source-linked development practical; ZIP reinstallation is for distribution,
  not the routine edit/test loop. Link the inner `pawprint/` package directly into
  Blender's existing `extensions/user_default` repository; no separate development
  repository is required. See README for the confirmed Blender 5.2 Linux path.

## Reference project

`../StableGen/AGENTS.md` maps a related addon. Useful reference areas include
`stablegen/texturing/{generator,workflows,rendering,projection}.py` and
`stablegen/core/server_api.py`. Assume the same starting ComfyUI installation;
new model adapters may need additional documented models/nodes.

StableGen is a reference, not Pawprint's specification. In particular, Pawprint
assumes existing UVs, uses explicit editable layers, preprocesses controls from
the visible composite, and does not attempt to automatically solve all seams.
If reusing its code, preserve applicable license and attribution notices.
