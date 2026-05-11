#!/usr/bin/env python3
"""Extract dense perspective views from Matterport panoramas with computed
camera poses, output as Brush/Nerfstudio-compatible transforms.json + images + ply.

For each Matterport sweep, we have 6 cube faces with the sweep's known position
and rotation. We build the equirectangular panorama and extract N perspective
views at evenly-spaced (azimuth, elevation) pairs. Each view's camera pose is
computed analytically (no SfM needed).

The output convention is OpenGL camera-to-world (Nerfstudio default) which is
what Brush expects.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import py360convert
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# Reuse logic from existing scripts
sys.path.insert(0, str(Path(__file__).parent))
from matterport_to_colmap import (
    Sweep,
    load_sweeps,
    load_combined_mesh,
    sample_mesh_surface,
    write_points3d_ply,
    discover_face_images,
)
from dense_perspective import FACE_TO_DIR, FACE_ROTATION_90, load_cube_faces, cubemap_to_equirect


def look_at_R_opencv(az_deg: float, el_deg: float) -> np.ndarray:
    """Camera→sweep-local rotation for view at (azimuth, elevation), OpenCV camera.

    Sweep-local convention: +X right, +Y forward, +Z up.
    Camera (OpenCV): +X right, +Y down, +Z forward.
    """
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    sa, ca = math.sin(az), math.cos(az)
    se, ce = math.sin(el), math.cos(el)
    # look = camera +Z forward in sweep-local
    look = np.array([sa * ce, ca * ce, se])
    # right = camera +X in sweep-local
    right = np.array([ca, -sa, 0.0])
    # down = camera +Y in sweep-local (image y axis goes down)
    down = np.cross(look, right)  # right-handed: forward × right = down (image down)
    R_cam2sweep = np.column_stack([right, down, look])
    return R_cam2sweep


def opencv_to_opengl_c2w(R_c2w: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4x4 camera-to-world in OpenGL convention from OpenCV (R, t)."""
    # OpenCV cam (X right, Y down, Z forward) → OpenGL cam (X right, Y up, Z back).
    # Equivalent to negating cam Y and Z axes: c2w_gl = c2w_cv @ diag(1, -1, -1, 1)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R_c2w @ np.diag([1.0, -1.0, -1.0])
    c2w[:3, 3] = t
    return c2w


def render_perspective(equi: np.ndarray, fov: float, az: float, el: float, out_size: int) -> np.ndarray:
    p = py360convert.e2p(equi, fov_deg=fov, u_deg=az, v_deg=el,
                         out_hw=(out_size, out_size), mode="bilinear")
    return np.clip(p, 0, 255).astype(np.uint8)


def render_perspective_from_cube(
    cube_arrs: dict[int, np.ndarray],
    face_R_cam2sweep: dict[int, np.ndarray],
    fov: float,
    az: float,
    el: float,
    out_size: int,
    blend_deg: float = 10.0,
) -> np.ndarray:
    """Direct cube → perspective via ray casting with cross-face blending.

    Each output pixel's ray is computed in sweep-local. Faces whose look
    direction is within (45° + blend_deg) of the ray contribute, weighted by
    cosine of angular distance to face center past the (45° − blend_deg)
    threshold. Boundaries between cube faces are smoothed over a `blend_deg`
    wide angular band — fixes the diagonal seams that appear when a view
    crosses cube corners (e.g., az~315°, el~−30°) due to Matterport's
    per-face calibration drift.

    Note: cube corners where 3 faces meet are at acos(1/√3) ≈ 54.74° from any
    face center, so `blend_deg` must be ≥ 10° to keep those corner pixels
    covered (otherwise black triangles appear).
    """
    fx = fy = out_size / (2.0 * math.tan(math.radians(fov) / 2.0))
    cx = cy = out_size / 2.0
    j_grid, i_grid = np.meshgrid(np.arange(out_size), np.arange(out_size))
    x = ((j_grid - cx) / fx).astype(np.float32)
    y = ((i_grid - cy) / fy).astype(np.float32)
    z = np.ones_like(x, dtype=np.float32)
    rays_cam = np.stack([x, y, z], axis=-1)
    rays_cam /= np.linalg.norm(rays_cam, axis=-1, keepdims=True)

    R_cam2sweep = look_at_R_opencv(az, el).astype(np.float32)
    rays_sweep = rays_cam @ R_cam2sweep.T  # H, W, 3

    cos_cutoff = math.cos(math.radians(45.0 + blend_deg))
    H, W = out_size, out_size

    # Vectorize cosines across all 6 faces in one matmul
    look_stack = np.stack([face_R_cam2sweep[fi][:, 2] for fi in range(6)]).astype(np.float32)  # 6, 3
    cosines_all = rays_sweep @ look_stack.T  # H, W, 6
    weights_all = np.maximum(0, cosines_all - cos_cutoff)  # H, W, 6

    accum_color = np.zeros((H, W, 3), dtype=np.float32)
    accum_weight = np.zeros((H, W), dtype=np.float32)

    for fi in range(6):
        face_arr = cube_arrs.get(fi)
        if face_arr is None:
            continue
        weight = weights_all[..., fi]
        if not (weight > 0).any():
            continue

        # Project ray into face camera frame: rays_face = R_f2s.T @ ray ↔ rays_sweep @ R_f2s
        R_f2s = face_R_cam2sweep[fi].astype(np.float32)
        rays_face = rays_sweep @ R_f2s  # H, W, 3
        rz = rays_face[..., 2]
        rz_safe = np.where(rz > 1e-9, rz, np.float32(1e-9))
        u = np.clip(rays_face[..., 0] / rz_safe, -1.0, 1.0)
        v = np.clip(rays_face[..., 1] / rz_safe, -1.0, 1.0)

        fh, fw = face_arr.shape[:2]
        px = (u + 1.0) * 0.5 * (fw - 1)
        py = (v + 1.0) * 0.5 * (fh - 1)
        # map_coordinates expects (row, col) order
        coords = np.stack([py.ravel(), px.ravel()])

        sample = np.empty((H * W, 3), dtype=np.float32)
        for c in range(3):
            sample[:, c] = map_coordinates(face_arr[..., c], coords, order=1, mode="nearest")
        sample = sample.reshape(H, W, 3)

        accum_color += sample * weight[..., None]
        accum_weight += weight

    safe_w = np.where(accum_weight > 1e-9, accum_weight, 1.0)[..., None]
    color = accum_color / safe_w
    return np.clip(color, 0, 255).astype(np.uint8)


def main(args: argparse.Namespace) -> int:
    model_dir = Path(args.model_dir).resolve()
    image_dir = Path(args.image_dir).resolve()
    out_root = Path(args.output_dir).resolve()
    images_out = out_root / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    # Load sweep poses
    sweeps_json = model_dir / "api" / "mp" / "models" / "graph_GetShowcaseSweeps.json"
    sweeps = load_sweeps(sweeps_json)
    print(f"Sweeps: {len(sweeps)}")

    # Index existing cube face images by sweep_short
    face_idx = discover_face_images(image_dir)
    matched = [s for s in sweeps if s.sweep_short in face_idx]
    print(f"Matched with images: {len(matched)}")

    # View grid
    azimuths = list(range(0, 360, args.azimuth_step))
    elevations = [float(e) for e in args.elevations.split(",")]
    print(f"Views per sweep: {len(azimuths)} azimuths × {len(elevations)} elevations = {len(azimuths) * len(elevations)}")

    fov = args.fov
    out_size = args.resolution
    fx = fy = out_size / (2.0 * math.tan(math.radians(fov) / 2.0))
    cx = cy = out_size / 2.0
    print(f"Camera: PINHOLE {out_size}x{out_size}, fx={fx:.1f}, fov={fov}°")

    # Face orientation in sweep-local for cube faces — matches FACE_TO_DIR in
    # dense_perspective.py: 0=U, 1=R, 2=B, 3=L, 4=F, 5=D. Preserves the
    # mesh-overlay-verified sides ordering from the original code (1=R, 2=B,
    # 3=L); face0=U and face5=D fixes the polar swap discovered via
    # test_face_mapping.py.
    cube_face_look_up = {
        0: ((0.0, 0.0, 1.0),  (0.0, -1.0, 0.0)),  # U
        1: ((1.0, 0.0, 0.0),  (0.0, 0.0, 1.0)),   # R
        2: ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),   # B
        3: ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),   # L
        4: ((0.0, 1.0, 0.0),  (0.0, 0.0, 1.0)),   # F
        5: ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),   # D
    }

    def cube_face_R_cam2sweep(face_idx: int) -> np.ndarray:
        look = np.array(cube_face_look_up[face_idx][0], dtype=np.float64)
        up = np.array(cube_face_look_up[face_idx][1], dtype=np.float64)
        right = np.cross(look, up)                # camera +X
        down = -up                                # camera +Y
        forward = look                            # camera +Z
        return np.column_stack([right, down, forward])

    # Process each sweep
    frames: list[dict] = []
    bar = tqdm(matched, unit="sweep")
    cube_fov = 90.0
    cube_size = 2048  # Matterport native cube face resolution
    cube_fx = cube_fy = cube_size / (2.0 * math.tan(math.radians(cube_fov) / 2.0))
    cube_cx = cube_cy = cube_size / 2.0

    for sweep_idx, sw in enumerate(bar, start=1):
        bar.set_postfix_str(sw.sweep_short[:8])
        face_files = face_idx[sw.sweep_short]
        R_sweep = sw.rotation_matrix

        # 1) Original 6 cube faces — no equirect intermediate, so no polar seams
        if not args.skip_cube_faces:
            for face_i in range(6):
                src = face_files.get(face_i)
                if src is None:
                    continue
                fname = f"scan{sweep_idx:03d}_{sw.sweep_short[:8]}_cube{face_i}.jpg"
                fpath = images_out / fname
                if not fpath.is_file() or not args.skip_existing:
                    # Just copy/symlink the existing cube face JPG
                    import shutil
                    shutil.copy2(src, fpath)
                R_face2sweep = cube_face_R_cam2sweep(face_i)
                R_c2w_cv = R_sweep @ R_face2sweep
                c2w_gl = opencv_to_opengl_c2w(R_c2w_cv, sw.position)
                frames.append({
                    "file_path": f"images/{fname}",
                    "transform_matrix": c2w_gl.tolist(),
                    "fl_x": cube_fx, "fl_y": cube_fy,
                    "cx": cube_cx, "cy": cube_cy,
                    "w": cube_size, "h": cube_size,
                })

        # 2) Dense perspective views — direct cube-to-perspective (no equirect)
        try:
            cube_arrs: dict[int, np.ndarray] = {}
            for fi in range(6):
                src = face_files.get(fi)
                if src is None:
                    continue
                with Image.open(src) as im:
                    cube_arrs[fi] = np.array(im.convert("RGB"))
        except Exception as e:
            print(f"\n  ⚠ {sw.sweep_short[:8]}: cube load 실패 — perspective 뷰 스킵 ({e})", file=sys.stderr)
            continue

        face_R_cam2sweep = {fi: cube_face_R_cam2sweep(fi) for fi in range(6)}

        for el in elevations:
            for az in azimuths:
                fname = f"scan{sweep_idx:03d}_{sw.sweep_short[:8]}_az{az:03d}_el{int(el):+03d}.jpg"
                fpath = images_out / fname
                if not fpath.is_file() or not args.skip_existing:
                    persp = render_perspective_from_cube(
                        cube_arrs, face_R_cam2sweep, fov, az, el, out_size,
                    )
                    Image.fromarray(persp).save(fpath, "JPEG", quality=92)
                R_face2sweep = look_at_R_opencv(az, el)
                R_c2w_cv = R_sweep @ R_face2sweep
                c2w_gl = opencv_to_opengl_c2w(R_c2w_cv, sw.position)
                frames.append({
                    "file_path": f"images/{fname}",
                    "transform_matrix": c2w_gl.tolist(),
                })

    print(f"\nGenerated frames: {len(frames)}")

    # Sample mesh for init points (reuse logic from matterport_to_colmap)
    if not args.skip_points:
        mesh_dirs = list((model_dir / "models").glob("*/assets/mesh_tiles/~"))
        if mesh_dirs:
            print("\nMesh-derived initial points...")
            verts, faces = load_combined_mesh(mesh_dirs[0])
            pts = sample_mesh_surface(verts, faces, args.num_points)
            write_points3d_ply(out_root / "points_init.ply", pts)
            print(f"Wrote points_init.ply: {len(pts):,} points")
        else:
            print("⚠ mesh_tiles 폴더 없음 — points_init.ply 생성 건너뜀", file=sys.stderr)

    # Write transforms.json (OpenGL convention)
    transforms = {
        "camera_model": "PINHOLE",
        "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy,
        "w": out_size, "h": out_size,
        "k1": 0, "k2": 0, "p1": 0, "p2": 0,
        "frames": frames,
        "ply_file_path": "points_init.ply",
        "applied_transform": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        "applied_scale": 1.0,
    }
    (out_root / "transforms.json").write_text(json.dumps(transforms, indent=2), encoding="utf-8")
    print(f"Wrote transforms.json")

    # Build Brush-compatible zip
    if args.zip:
        zip_path = out_root.parent / f"{out_root.name}_brush.zip"
        print(f"\nBuilding zip: {zip_path}")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            for f in out_root.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(out_root.parent))
        print(f"Zip ready: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")

    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--fov", type=float, default=60.0)
    p.add_argument("--azimuth-step", type=int, default=30)
    p.add_argument("--elevations", default="-30,0,30")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--num-points", type=int, default=200000)
    p.add_argument("--skip-points", action="store_true")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip rendering images that already exist on disk")
    p.add_argument("--zip", action="store_true", default=True)
    p.add_argument("--skip-cube-faces", action="store_true",
                   help="Skip including the original 6 cube faces (just dense perspective)")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(main(parse_args()))
