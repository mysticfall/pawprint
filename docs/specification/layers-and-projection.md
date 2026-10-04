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

The actual base is UV-mapped baked albedo, independent of a viewpoint. The base
and projection composite display through a diffuse-only Principled shader under
scene lighting (metallic 0, roughness 1, specular IOR level 0). This supersedes the
earlier unlit-stack decision: generation-time Chord albedo estimation
removes measured diffuse lighting before storing generated pixels. The base is
not directly editable through viewpoint-layer tools. Commit bakes still use an
emission-only material copy so they store albedo rather than scene illumination.

Composite visible projections from bottom to top using image alpha over the base.
Lower-layer changes show dynamically through transparent regions. Opaque pixels
generated using an earlier context remain fixed until edited or regenerated.

The base need not already be a PBR material. The derived PBR material
([below](#derived-pbr-material)) provides non-destructive preview without
damaging the original appearance. Original images are retained and derived
channel images are separate datablocks, so decomposition never replaces its
source. Whether decomposition beyond albedo/normal estimation (roughness,
metalness) is exposed remains open.

Optional baking of an existing material is useful but not essential. The more
common bootstrap is a first ordinary projection generated with guidance.
**Open:** exact initial base fill and setup interaction.

The 0.2.0 prototype uses a fixed opaque gray generated image and a new material,
retaining the previous material by reference. This is a test bootstrap, not a
decision on final base-creation controls.

## Surface-relative normal channel

Generated layers retain Chord's normal image alongside albedo at full strength
1.0. Its red/right, green/down, blue/toward-viewer vectors are interpreted as
mesh-relative estimates: saved-camera right is projected onto the mesh shading
normal's tangent plane, with a stable fallback at a degenerate projection.
Decoded vectors are transformed into world space and replace the lower normal
at opaque coverage; feathered coverage blends normalized vectors bottom to top.
The lower composed normal never defines the estimate's basis, avoiding repeated
addition of texture detail. Mirror folding reflects both basis and result.

Normal coverage uses the lesser of normal and albedo alpha, multiplied by the
same projection visibility and layer visibility as albedo. Erasing albedo thus
hides the corresponding normal contribution. Normal images are packed float
Non-Color datablocks with Channel Packed alpha; normal painting is not exposed.
Apply replaces a selection-limited normal snapshot together with the native
albedo dab in one undo step. Layer limits both maps' alpha to the selected/feathered
footprint; review preserves the existing layer outside that footprint.

Commit Through Selected additionally bakes the composed normal into a separate
tangent-space UV base image when normals are present. A native Cycles NORMAL bake
uses the saved UV tangent basis with +X/+Y/+Z (OpenGL) encoding; a neutral map is
approximately (0.5, 0.5, 1), even on curved geometry. Mesh geometry establishes
the conversion basis, not baked broad-shape detail. The working shader reads that
base through a tangent-space Normal Map node. An opaque neutral estimate replaces
lower detail with the mesh normal; transparent coverage preserves lower detail.
Normal textures extend their RGB at the saved-frame border to
avoid interpolation toward black; the existing coverage/alpha gates still bound
their visibility. Prior object-space prototype maps must be cleared/replaced or
rebaked from retained source layers without the old normal base; no migration is
added. Chord may infer broad forms already represented by geometry. New generations
experimentally remove mesh-correlated angular flow at multiple spatial scales
before review, using a fresh evaluated target-slot smooth-normal capture. Removal
is full-strength and ungated: unmatched variation is retained by the regression,
but meaningful relief coincident with mesh flow can also be erased. Existing maps
are not retroactively corrected; this is not a calibrated Chord normal model.

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

## Derived PBR material (superseded final-estimation workflow)

The final **Estimate Albedo & Normal** command described below has been removed.
Every generation now receives a Chord albedo pass before review/application;
the diffuse-lit working material and emission commit bakes already store albedo.
Existing derived-material viewing helpers remain. The following records the
earlier implementation, not a requirement to run a second albedo pass.

The stack's baked-lighting composite stays the inpainting work surface; a
**separate derived PBR material** previews the result and supports retouching
with other material-painting extensions. The two serve different purposes and
are never merged into one datablock.

**Estimate Albedo & Normal** bakes the current stack through its top layer
into a temporary UV composite (same slot-isolated emission bake as commits),
uploads it to ComfyUI, and runs the tiled Chord material-estimation workflow
(1024² tiles, 128px overlap, circular padding — the reference "Tiled Chord"
graph) producing **albedo and normal only**; roughness/metalness outputs stay
unconnected and Principled defaults are kept. Results are applied into packed
map datablocks (`Pawprint Albedo` sRGB, `Pawprint Normal` Non-Color) wired
through a UV-map node and a normal-map node on `stack.pbr_material`
("Pawprint PBR"). A `pawprint_source` back-reference pairs the two materials,
and the pointer cycle keeps both alive across saves without fake users.

**Generation/Material view** swaps the object's slot material between the
pawprint material and the derived PBR material. Material view shows a
read-only stack UI; all editing, generation, painting, selection, and commit
operators are gated off while it is active. Viewport shading is left to the
user.

Re-estimation reuses the same material, nodes, and map datablocks, refreshing
pixels and links in place: removing and re-renaming image datablocks scrambles
node→image pointers across memfile undo. Undo removes a first-ever derived
material in one step; later re-applies keep structure stable, and map pixel
reloads may survive memfile undo like other direct image writes. The estimate
job guards owner/slot/stack identity and pixel digests per tick and at
completion, cancelling itself when the target changes.

**Open:** roughness/metalness exposure, Chord normal-map orientation
verification (assumed OpenGL), tile-seam behavior on large composites, and
bleed through fully transparent regions.
