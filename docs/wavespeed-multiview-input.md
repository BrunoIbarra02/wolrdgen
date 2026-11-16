# WaveSpeed synthetic-input profile

## Recommended role

Use WaveSpeed to create the appearance reference and a temporally coherent
image-to-video sequence. Do not use independent text-to-image calls for every
view: repeated objects, layout and focal length will drift and WorldSculpt will
treat those inconsistencies as geometry.

Initial candidate endpoint:

`wavespeed-ai/wan-2.2/image-to-video`

Baseline request:

- resolution: 720p;
- duration: 5 seconds;
- fixed seed stored in the job manifest;
- static environment with no people, animals, vehicles, wind or moving props;
- slow orbital camera movement, constant focal length and exposure;
- negative prompt covering object duplication, morphing, cuts, zoom, motion blur,
  depth-of-field changes, text and watermarks.

Suggested prompt suffix:

> Static environment and completely rigid objects. Slow smooth camera orbit,
> constant focal length, constant exposure, no cuts, no zoom, no moving objects,
> no deformation, physically consistent perspective, photorealistic materials.

This endpoint does not return calibrated camera matrices. Its camera wording is
guidance, not ground truth. After generation the pipeline must estimate camera
poses and depth, track persistent instance masks and reject inconsistent frames.

## Frame-selection baseline

1. Decode at the native frame rate but retain candidate frames at 2 fps.
2. Estimate poses for all candidates.
3. Remove blurred frames and frames whose pose cannot be registered.
4. Keep one anchor with the largest visible mask for each object.
5. Choose the remaining views by farthest-point sampling on recovered camera
   directions.
6. Cap the first WorldSculpt run at eight views per object.

## Hard rejection checks

- scene cut or discontinuous camera jump;
- object count or persistent ID changes;
- topology/shape changes between views;
- strong shadows or highlights that move independently of the camera;
- focal-length drift beyond the pose estimator's tolerance;
- object mask below 0.1% of the full image;
- recovered reprojection error above the calibration threshold;
- inconsistent metric scale between adjacent frames.

## E2E validation order

1. WorldSculpt's UE-MeshyScene sample: validates installation and official data
   contract with known cameras, masks and boxes.
2. An Unreal-rendered synthetic scene: validates our coordinate conversion and
   PCG manifest against exact ground truth.
3. WaveSpeed image-to-video scene: validates the difficult uncalibrated-input
   path without confusing model quality with an installation error.

WaveSpeed Hunyuan3D/Hi3D multi-view endpoints can serve as per-object fallback
or comparison baselines, especially for PBR texturing, but they do not replace
WorldSculpt's shared-world transforms and scene composition contract.
