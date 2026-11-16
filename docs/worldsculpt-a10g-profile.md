# WorldSculpt profile for AWS g5.xlarge

## Hardware envelope

- GPU: NVIDIA A10G, 24 GiB VRAM, CUDA compute capability 8.6.
- Host: g5.xlarge, 4 vCPU and 16 GiB RAM.
- Storage: 150 GiB gp3.
- Build NATTEN with `NATTEN_CUDA_ARCH=8.6` and no more than four workers.
- Start with the normal GPU path. `--low_vram` lowers VRAM to roughly 10–12 GiB
  by offloading models, but the 16 GiB host is tight; use it only with swap or
  resize temporarily to g5.2xlarge.

## Released-model limitation

The AlayaLab model repository currently publishes multi-view checkpoints only
for sparse structure and shape:

- `ss_ft64_mv_lora_ibr_texverse`, step 15000
- `shape_ft1024_mv_lora_ibr_texverse_fixedmem05`, step 15000

It does not publish a matching multi-view texture checkpoint. The official
`inference.sh` therefore passes `--no_tex`. Treat WorldSculpt as the geometry
and rigid-placement stage. For the Unreal MVP, texture the accepted geometry by
multi-view projection/baking, or use TRELLIS.2/Pixal3D single-image texturing on
the best anchor crop. Do not invoke the experimental multi-view Stage 3 without
a matching checkpoint and config.

## Baseline geometry profile

| Parameter | Baseline | Reason |
|---|---:|---|
| crop resolution | 1024 | Fixed resolution used by the released shape model |
| canonical frame | anchor | Matches the canonical front-view training setup |
| alpha erosion | 0 / 0 | Preserves silhouettes for spatial validation |
| minimum mask ratio | 0.001 | Rejects almost invisible instances |
| maximum crop ratio | 3.0 | Prevents pathological off-axis allocations |
| mask-fit percentile | 95 | Recovers underestimated 3D boxes without chasing every outlier |
| selected views | 8 | Training uses 1–20 inputs; eight is the safe A10G starting point |
| view selection | fps | Maximizes angular coverage and removes near-duplicate video frames |
| seed | 42 | Deterministic baseline |
| sampler | official | Matches the command published in the repository README |
| sparse-structure step | 15000 | Only published release checkpoint |
| shape step | 15000 | Only published release checkpoint |
| max tokens | 32768 | Matches the shape training cap and limits A10G peaks |
| per-object composition budget | 250000 faces | Practical first Unreal import target |
| final scene GLB decimation | 100000 faces/object | Smoke-test target; raise only for accepted hero assets |
| render previews | disabled | Rendering dominates runtime and is not needed for first inference |

Resolution must remain 1024 for multi-view inference. Do not tune it as an
ordinary quality knob: the released models were fine-tuned at grid 64 / 1024.

## Controlled quality sweep

Run this only on three representative objects: one simple, one thin/concave and
one heavily occluded.

1. Baseline: 8 views, FPS, official sampler, seed 42, 32768 tokens.
2. Coverage: 12 views, FPS, otherwise baseline.
3. Visibility: 8 views, area selection, otherwise baseline.
4. Sampler: 8 views, FPS, `train` sampler (SS 50 steps / guidance 3; shape 12 / guidance 3).
5. Capacity: winning setup with 49152 tokens if no OOM occurred.

Select using reprojection silhouette IoU, world-space OBB error, visible-surface
coverage, runtime, peak VRAM and import size. Do not choose by a single beauty
render: position, rotation and scale are the acceptance criteria for the PCG
contract.

## Input requirements for WaveSpeed imagery

Independent text-to-image samples are not valid multi-view input. Each scene
must supply:

- one coherent camera trajectory around a persistent scene;
- camera intrinsics and camera-to-world transforms, or footage suitable for
  reliable pose recovery;
- persistent per-instance IDs across frames;
- instance masks;
- metric or scale-anchored depth from which 3D boxes can be produced.

Prefer an image-to-video model with explicit camera motion over multiple fresh
text-to-image calls. Reject frames with duplicated/disappearing objects, changed
topology, inconsistent focal length, or less than 0.1% visible mask area.

## Texture strategy for Unreal

1. Keep terrain, floors, roads, water and vegetation zones procedural. Export
   polygons/masks and PCG directives rather than dense reconstructed meshes.
2. For rigid objects, unwrap the accepted WorldSculpt mesh and bake base color
   from the calibrated views using visibility and view-angle weights.
3. Use a single-image texturing prior only to fill genuinely unobserved regions;
   mark those texels as generated in metadata.
4. Generate or estimate roughness/normal maps separately, then validate scale and
   channel conventions in Unreal.
5. Export transforms in meters and convert handedness in one dedicated adapter.

## Operational gates

- Keep the EC2 instance stopped outside installation and test windows.
- Cache Hugging Face assets on the 150 GiB volume; do not redownload per job.
- Accept any gated model licences in the Hugging Face account before bootstrap
  and provide a read-only `HF_TOKEN` only in the SSM session environment.
- Download one benchmark scene first, not the full dataset.
- Record the upstream Git commit, checkpoint names, seed and all parameters in
  every job manifest.
- Stop after the first successful object and inspect memory before batching a
  full scene.
