# Pawprint specification

## Purpose

Bring a Krita AI Diffusion-like editing workflow into Blender: frame a surface,
select an area, generate a patch, erase or paint it, and inpaint again. Users repair
seams between projections and across UV boundaries by creating views that span
the troublesome area. Automatic seamlessness is not assumed; tileable image
generation is not the motivating requirement.

The initial implementation is deliberately smaller than this product design.
See [scope and validation](scope-and-validation.md) for the boundary.

## Pages

- [Layers and projection](layers-and-projection.md): ownership, compositing,
  viewpoint frames, merging, and the committed base.
- [Editing and generation](editing-and-generation.md): pixel edits, selections,
  scene context, cropped requests, and undo.
- [Backend and guidance](backend-and-guidance.md): per-layer model settings,
  capability-driven adapters, and ControlNet/IPAdapter/LoRA support.
- [Scope and validation](scope-and-validation.md): implementation status, deferred
  work, unresolved choices, and technical prototypes.

## Terminology

| Term | Meaning |
| --- | --- |
| Projection layer stack | Ordered layers owned by one object–material-slot combination. |
| Base | Committed, UV-mapped baked appearance underneath projection layers; not tied to a viewpoint. |
| Projection layer | Editable RGBA image plus a saved perspective viewpoint and generation settings. |
| Active layer | The destination of painting and generation, regardless of which layers are visible. |
| Visible composite | The current scene appearance, including all visible layers and surrounding objects/materials. |
| Edit selection | Image-space region whose pixels may be overwritten, including feathering. |
| Context region | Full frame or padded selection bounds supplied to generation; may be larger than the edit selection. |
| Guidance | Spatial controls such as depth/edges, plus separate reference-image guidance such as IPAdapter. |
| Adapter | Model-family implementation that describes supported capabilities and constructs backend requests. |

An artist may call the first generated projection their “base layer.” It remains
an ordinary viewpoint-bound layer, distinct from the actual committed UV base.

## Design status

Requirements in these pages record the agreed workflow. Sections marked **Open**,
**Candidate**, or **Deferred** are not settled implementation promises. Blender
5.0.0 is the declared minimum; native APIs and backend workflows must be verified
against available releases before later slices rely on them.
