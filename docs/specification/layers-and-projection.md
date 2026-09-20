# Layers, projection, and base

## Ownership and assumptions

- Attach a stack to an object–material-slot combination. Assume the material is
  not shared with other targets.
- Objects already have usable UV maps for committing textures. On-demand UV
  generation, as offered by StableGen, is not required.
- Transforms, mesh shape, and topology are assumed stable during texturing.
  Do not enforce this assumption; consequences of changes belong to the user.
- Use native Blender Image datablocks and Blender's save/pack/unpack facilities.
  No custom image storage format or mandatory packing policy has been requested.

## Base and working display

The actual base is UV-mapped baked color, independent of a viewpoint. It displays
**unlit**, as do all projection layers: generated shadows and highlights must not
receive scene lighting a second time. The base is not directly editable through
viewpoint-layer tools.

Composite visible projections from bottom to top using image alpha over the base.
Lower-layer changes show dynamically through transparent regions. Opaque pixels
generated using an earlier context remain fixed until edited or regenerated.

The base need not already be a PBR material. A future non-destructive PBR preview
should allow testing the committed base without damaging its original appearance.
PBR decomposition is deferred; retain original images and leave room for derived
channel images. Decomposition may discard detail, so replacing the source image
with its output is not an acceptable assumed workflow. Whether decomposition is
an adjustment layer or another operation remains open.

Optional baking of an existing material is useful but not essential. The more
common bootstrap is a first ordinary projection generated with guidance.
**Open:** exact initial base fill and setup interaction.

The 0.2.0 prototype uses a fixed opaque gray generated image and a new material,
retaining the previous material by reference. This is a test bootstrap, not a
decision on final base-creation controls.

## Projection layers

Each layer has:

- An editable RGBA image; transparency belongs to that image, not to a separate
  user-editable layer mask.
- A saved perspective viewpoint with sufficient projection/framing metadata to
  reproduce its image mapping.
- Image resolution and aspect ratio.
- User-controlled visibility and [generation settings](backend-and-guidance.md).

The initial scope has no layer-level opacity control. Pixel alpha still supports
feathered patches and variable-strength erasing.

Only target surfaces visible from the saved viewpoint receive the projection.
Respect occlusion, backfaces, object identity, and material-slot membership.
Visibility eligibility is a geometry/projection concern, not a new editable mask.
The implementation must prevent painting through foreground surfaces.

Implemented visibility combines three per-fragment tests: the plane-distance
depth test against the snapshot, a winding-corrected facing test, and a
winding-independent axial occlusion z-test that rejects fragments clearly behind
the captured surface along the saved view ray (`w − depth > w × margin`). The
margin is two snapshot texels scaled by the texel footprint at the fragment's
axial distance (extracted from the pure saved-window matrix; the combined
window×view diagonal is rotation-polluted), so it absorbs pixel-center snap error
at grazing angles without admitting genuine occlusion gaps. Sub-texel-thin gaps
can still pass, surfaces genuinely visible through holes still receive the
projection by design, and extreme-grazing far sides rely on the plane and facing
tests.

**Open:** precise visibility representation, boundary tolerances, and optional
grazing-angle fading. Different viewpoints can still produce stretching or visual
inconsistency; manual layer editing is the intended remedy.

The current prototype uses a static raycast depth snapshot with per-fragment
projection. Its finite-resolution/surface assumptions are recorded under
[scope and validation](scope-and-validation.md#prototype-representation-and-limitations).

## Viewpoint and frame

- Create a layer from the current view; restore a selected layer's saved view.
- Selection, painting, and generation use its 2D image space while aligned to that
  view. Users may orbit for inspection and return to resume editing.
- Show a clear image-frame marker with the exact bounds/aspect when the selected
  layer's viewpoint is active. The marker is excluded from generation renders.
- Perspective-only initially. Orthographic support is deferred, not technically
  incompatible with compositing perspective layers.
- Resolution/aspect remain editable after image content exists. This is unusual
  in the intended workflow; do not require clearing or prevent changes. Users
  accept consequences. Exact resampling/reframing semantics remain open.
- Retaining image pixels outside target coverage is acceptable; trimming with
  safe padding is also acceptable. Such pixels are not required for the context
  selection technique described in [editing](editing-and-generation.md).

## Mirror

Each layer has an optional **Mirror** setting (`None`, `X±`, `Y±`, `Z±`) that folds
its projection across the target object's local axis plane. The chosen direction
names the *kept* half: with `X+`, fragments on the +X side of the object's local
X=0 plane project normally, fragments on the −X side are reflected onto the kept
half (`P' = P − fold·2·dot(P−O,n)·n` with `fold` gated by the fragment's side and
`n` the signed world-space axis, `O` the object origin), and image texels belonging
to the −X half are never sampled. `X−` differs only in sign — the other image half
is used. Surface normals fold the same way, gated by the fragment's side rather
than the normal's direction, so the saved-eye facing and depth-difference tests
keep working.

Plane constants are baked at material-build time, consistent with the
stable-transforms assumption. The depth snapshot is unchanged: at mirrored UVs it
stores the kept-side surface, so symmetric geometry covers exactly while asymmetric
geometry fails the depth tolerance and leaves holes. Painting, selection,
generation, and baking operate in image space and are unaffected.

The setting is per-layer, is copied to candidate layers created during batch
review, and is deliberately not inherited by new layers.

## Merge down

Initially merge only adjacent layers with the same compatible saved projection.
The former “same normal plane” rule was discarded: orientation alone does not
make perspective projections interchangeable.

Composite upper pixels over lower pixels into the lower image and remove the
upper layer. Other layers keep their content and order. Matching projection means
matching position, orientation, lens, and framing, not merely viewing direction.
Compatible visibility restrictions also matter for preserving appearance.

**Open:** whether matching resolutions are mandatory initially or whether merging
resamples into a selected resolution. Cross-view reprojection merges are deferred.

## Commit to base

Commit a contiguous group starting with the bottom projection layer into the
base's UV image. Arbitrary middle-layer commits can change ordering/appearance
and are not the initial operation.

**Implemented (0.2.0 prototype):** the group is the bottom layer through the
selected layer. Baking evaluates the target object, isolates the active slot's
faces with a copied material into a temporary Cycles CPU scene, bakes the
unlit emission into a **new packed copy of the base, resampled to the
scene-configured base resolution** (Base width/height, default 2048 × 2048,
minimum 64, no upper limit; also used when a stack is created), then swaps the
base reference. By default the committed layers are removed; a per-stack
**Keep source layers** option retains them hidden instead, so resolutions can
be retried or sources re-edited after undo without destroying them.
Layers above the selection keep their images, views, settings, and selections.
Hidden layers inside the group are removed (or kept hidden) without
contributing pixels.
One native undo restores the previous base image and the removed layers (or
their visibility, when sources were kept);
redo restores the commit. Errors (missing UV map or images) leave the stack
and base untouched. An 8-pixel EXTEND margin covers UV-island edges. The old
base image datablock is retained in the file until purged.

Offer removal of committed layers. Retained layers are disabled after committing
to avoid applying feathered/transparent contributions twice. Retaining them is
not equivalent to keeping the pre-commit base: undo or a base snapshot is needed
to restore that state.

**Open:** bake sampling/margin controls and the exact controls for choosing
the group.
