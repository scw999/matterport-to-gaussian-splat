# Matterport → Gaussian Splat Pipeline

Convert a Matterport 3D tour into a Gaussian Splat training dataset using
ground-truth sweep poses + mesh-derived initial point cloud — no SfM required.

This is built on top of [rebane2001/matterport-dl](https://github.com/rebane2001/matterport-dl)
(included here). The original tool downloads the Matterport assets; the new
scripts below convert those assets into Brush / Nerfstudio-compatible
transforms.json + perspective images + initial point cloud.

## Pipeline overview

```
Matterport URL
   ↓  matterport-dl.py            (3rd-party, included)
downloads/<model>/                cube-face tiles + sweep poses (graph_GetShowcaseSweeps.json) + mesh GLBs
   ↓  stitch_matterport.py
perspective_images_2k/<model>/    6 stitched cube faces per sweep (2048×2048)
   ↓  brush_dense_input.py
brush_dense_input_<model>/        transforms.json + perspective views + points_init.ply
   ↓  zip
brush_dense_input_<model>_brush.zip   → Drop into Brush UI
```

## New scripts (the GS pipeline contribution)

| File | Purpose |
| ---- | ------- |
| `stitch_matterport.py`     | Stitches Matterport's cube-face tile mosaics into 6 face JPGs per sweep |
| `dense_perspective.py`     | Extracts perspective views from cube faces via cubemap → equirect → e2p. Has `--diagnose` for cube-face mapping verification |
| `matterport_to_colmap.py`  | Builds COLMAP / Nerfstudio camera poses from sweep position + rotation; samples mesh surface for init points |
| `brush_dense_input.py`     | End-to-end: cube faces + perspective views + transforms.json (OpenGL c2w) + zip for Brush |
| `gs_trainer.py`            | Custom gsplat-based trainer (abandoned — Windows JIT compile issues) |
| `test_face_mapping.py`     | Diagnostic to compare candidate cube-face → direction mappings |

## Lessons learned (the part that took the most debugging)

### Cube-face index convention (Matterport)

After empirical verification on multiple sweeps + visual equirect inspection:

```
face0 = Up (ceiling)
face1 = Right (+X in sweep-local)
face2 = Back (-Y)
face3 = Left (-X)
face4 = Forward (+Y)
face5 = Down (floor)
```

A naïve `0=F, 1=R, 2=B, 3=L, 4=D, 5=U` mapping is wrong — it places ceiling
content at the equator of the equirect and floor content at the north pole,
producing broken perspective views with "ceiling-on-the-wall" artifacts at
cube-face seams.

The polar swap (`0=U, 5=D`) is necessary. The sides ordering matters too:
both `0:U, 1:F, 2:R, 3:B, 4:L, 5:D` (M1) and `0:U, 1:R, 2:B, 3:L, 4:F, 5:D`
(M2) produce visually clean equirects but represent 90°-rotated frames in
sweep-local coordinates. M2 preserves the side-ordering verified by mesh
reprojection and gives substantially better GS convergence.

### Coordinate conventions

- Matterport sweep poses: world is Z-up.
- Matterport mesh GLBs: glTF default is Y-up. Apply `MESH_TO_WORLD` rotation
  (swap Y↔Z) to align mesh with sweep coordinates.
- Sweep-local frame: +X right, +Y forward, +Z up.
- OpenCV camera: +X right, +Y down, +Z forward.
- OpenGL camera (Nerfstudio / Brush): +X right, +Y up, +Z back.
  Convert via `c2w_gl = c2w_cv @ diag(1, -1, -1)`.

### Why no SfM

Matterport's `graph_GetShowcaseSweeps` GraphQL response gives exact camera
position + rotation for every sweep. Combined with the mesh-derived initial
point cloud, this gives Gaussian Splatting a strong geometric prior and skips
the entire COLMAP feature-matching step.

### What still needs work

- M2 vs M3/M4 (90° / 180° / 270° rotations of side faces) — empirical pick.
  Try alternates if first GS result is still rotated relative to mesh.
- Dense perspective views at elevation 0° only. ±30° elevations introduce
  py360convert artifacts near the cube-face seams; would need direct
  cube-to-perspective ray-casting to fix cleanly.

## Quick start

```bash
# 1. Download the Matterport model
python run.py <MATTERPORT_URL>

# 2. Stitch cube faces (2k or 4k)
python stitch_matterport.py --resolution 2k

# 3. Build Brush dataset (with mesh init points)
python brush_dense_input.py \
  --model-dir downloads/<MODEL_ID> \
  --image-dir perspective_images_2k/<MODEL_ID> \
  --output-dir brush_dense_input_<MODEL_ID> \
  --elevations 0

# 4. Drop brush_dense_input_<MODEL_ID>_brush.zip into Brush
```

## Acknowledgement

The original `matterport-dl.py` and surrounding tooling are by
[@rebane2001](https://github.com/rebane2001) and
[@mitchcapper](https://github.com/mitchcapper), released as public domain (Unlicense).
