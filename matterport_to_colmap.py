#!/usr/bin/env python3
"""Convert Matterport download (poses + mesh + cube faces) to COLMAP for PostShot.

Inputs:
  - matterport-dl model dir: contains api/mp/models/graph_GetShowcaseSweeps.json
    and models/<hash>/assets/mesh_tiles/~/{0..3}/*.glb (Draco-compressed glTF tiles).
  - cube face dir: perspective_images_2k/<model_id>/scan{NNN}_{sweep_short}_face{N}.jpg
    produced by stitch_matterport.py (2048x2048, 90° FOV).

Output (COLMAP "binary text" format consumable by PostShot/Nerfstudio):
  postshot_input/<model_id>/
    images/<scan{NNN}_face{N}.jpg> -- copies (or symlinks) of input cube faces
    sparse/0/cameras.txt
    sparse/0/images.txt
    sparse/0/points3D.ply  (initial point cloud sampled from mesh)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pygltflib
import DracoPy
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# --- Coordinate conventions -----------------------------------------------
# Matterport JSON poses: Z-up world (Z is gravity-aligned, span 1.2~7.3m vertical).
# glTF mesh content: Y-up (default per glTF spec). Convert via X-axis rotation:
#   world_v = MESH_TO_WORLD @ glb_v    where MESH_TO_WORLD swaps glTF Y↔Z.
# This rotation has been verified by overlaying mesh bounds onto sweep position
# bounds (mesh fits inside sweep extent after the rotation).
MESH_TO_WORLD: np.ndarray = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ]
)

# --- Cube face → camera orientation ---------------------------------------
# Face mapping verified visually with dense_perspective.py --diagnose:
#   face0=Front  face1=Right  face2=Back  face3=Left  face4=Down  face5=Up
# Each face is a 90° FOV pinhole camera looking in a direction in the sweep's
# local frame. Sweep-local axes: +X right, +Y forward, +Z up (matches world
# axes when sweep rotation is identity).
#
# COLMAP camera convention: +X right, +Y down, +Z forward (looking direction).
# We define each face's (look, image_up) in sweep-local; then build R_face
# (camera→sweep-local) with columns: [look×up, -up, look].
FACE_LOOK_UP_LOCAL: dict[int, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    0: ((0.0, 0.0, 1.0),  (0.0, -1.0, 0.0)),  # U: up, image-up toward sweep -Y (back)
    1: ((1.0, 0.0, 0.0),  (0.0, 0.0, 1.0)),   # R: right, up = +Z
    2: ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),   # B: back, up = +Z
    3: ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),   # L: left, up = +Z
    4: ((0.0, 1.0, 0.0),  (0.0, 0.0, 1.0)),   # F: forward, up = +Z
    5: ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),   # D: down, image-up toward sweep +Y (front)
}

CUBE_FACE_FOV_DEG = 90.0
CUBE_FACE_PIXELS = 2048  # 2k cube faces produced by stitch_matterport.py

SWEEP_FILE_RE = re.compile(r"^scan(\d+)_([0-9a-fA-F]+)_face(\d)\.jpg$")


# --- Data structures ------------------------------------------------------
@dataclass
class Sweep:
    sweep_uuid: str          # full UUID, no hyphens
    sweep_short: str         # first 8 chars
    position: np.ndarray     # (3,) world position
    rotation: np.ndarray     # quaternion as (qx, qy, qz, qw) for scipy

    @property
    def rotation_matrix(self) -> np.ndarray:
        return R.from_quat(self.rotation).as_matrix()


# --- Sweep loading --------------------------------------------------------
def load_sweeps(graphql_json: Path) -> list[Sweep]:
    """Read sweep position + rotation from graph_GetShowcaseSweeps.json."""
    data = json.loads(graphql_json.read_text(encoding="utf-8"))
    locations = data["data"]["model"]["locations"]
    sweeps: list[Sweep] = []
    for loc in locations:
        pano = loc["pano"]
        uuid_clean = pano["sweepUuid"].replace("-", "")
        pos = pano["position"]
        rot = pano["rotation"]
        sweeps.append(
            Sweep(
                sweep_uuid=uuid_clean,
                sweep_short=uuid_clean[:8],
                position=np.array([pos["x"], pos["y"], pos["z"]], dtype=np.float64),
                # JSON has explicit x/y/z/w keys → scipy expects (x, y, z, w)
                rotation=np.array([rot["x"], rot["y"], rot["z"], rot["w"]], dtype=np.float64),
            )
        )
    return sweeps


# --- Face-image discovery -------------------------------------------------
def discover_face_images(image_root: Path) -> dict[str, dict[int, Path]]:
    """Return {sweep_short: {face_idx: jpg_path}} for all cube faces under image_root."""
    out: dict[str, dict[int, Path]] = {}
    for jpg in image_root.glob("*.jpg"):
        m = SWEEP_FILE_RE.match(jpg.name)
        if not m:
            continue
        sweep_short = m.group(2)
        face_idx = int(m.group(3))
        out.setdefault(sweep_short, {})[face_idx] = jpg
    return out


# --- GLB mesh loading -----------------------------------------------------
def _decode_draco_primitive(gltf: pygltflib.GLTF2, prim, blob: bytes) -> dict | None:
    ext = (prim.extensions or {}).get("KHR_draco_mesh_compression")
    if ext is None:
        return None
    bv = gltf.bufferViews[ext["bufferView"]]
    data = bytes(blob[bv.byteOffset : bv.byteOffset + bv.byteLength])
    decoded = DracoPy.decode(data)
    return {
        "vertices": np.array(decoded.points, dtype=np.float32).reshape(-1, 3),
        "faces": np.array(decoded.faces, dtype=np.int64).reshape(-1, 3),
    }


def load_glb_mesh(glb_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load all primitives in a GLB; return (vertices Nx3, faces Mx3 indices)."""
    gltf = pygltflib.GLTF2().load(str(glb_path))
    blob = gltf.binary_blob() or b""
    all_v: list[np.ndarray] = []
    all_f: list[np.ndarray] = []
    v_offset = 0
    for mesh in gltf.meshes:
        for prim in mesh.primitives:
            d = _decode_draco_primitive(gltf, prim, blob)
            if d is None:
                continue
            all_v.append(d["vertices"])
            all_f.append(d["faces"] + v_offset)
            v_offset += len(d["vertices"])
    if not all_v:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int64)
    return np.concatenate(all_v, axis=0), np.concatenate(all_f, axis=0)


def load_combined_mesh(mesh_tiles_dir: Path, lod_max: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Load all GLBs up to lod_max and combine into one mesh in world coords (Z-up)."""
    glbs: list[Path] = []
    for lod in range(lod_max + 1):
        glbs.extend(sorted((mesh_tiles_dir / str(lod)).glob(f"lod{lod}_*.glb")))
    if not glbs:
        raise FileNotFoundError(f"GLB 메시 파일을 찾지 못했습니다: {mesh_tiles_dir}")

    all_v: list[np.ndarray] = []
    all_f: list[np.ndarray] = []
    v_offset = 0
    for glb in tqdm(glbs, desc="GLB 로드", unit="file"):
        v, f = load_glb_mesh(glb)
        if len(v) == 0:
            continue
        all_v.append(v)
        all_f.append(f + v_offset)
        v_offset += len(v)

    verts = np.concatenate(all_v, axis=0).astype(np.float64)
    faces = np.concatenate(all_f, axis=0).astype(np.int64)
    # Apply glTF Y-up → world Z-up rotation.
    verts = verts @ MESH_TO_WORLD.T
    return verts, faces


def sample_mesh_surface(
    verts: np.ndarray,
    faces: np.ndarray,
    n_points: int,
    rng_seed: int = 0,
) -> np.ndarray:
    """Sample n_points uniformly on the surface (area-weighted)."""
    rng = np.random.default_rng(rng_seed)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross = np.cross(edge1, edge2)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    if areas.sum() <= 0:
        return verts[rng.integers(0, len(verts), size=n_points)]
    probs = areas / areas.sum()
    tri_idx = rng.choice(len(faces), size=n_points, p=probs)
    u = rng.random(n_points)
    v = rng.random(n_points)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    return v0[tri_idx] + u[:, None] * edge1[tri_idx] + v[:, None] * edge2[tri_idx]


# --- Camera pose math -----------------------------------------------------
def face_camera_to_sweep(face_idx: int) -> np.ndarray:
    """Build R_face: maps camera-local axes to sweep-local axes."""
    look = np.array(FACE_LOOK_UP_LOCAL[face_idx][0], dtype=np.float64)
    up = np.array(FACE_LOOK_UP_LOCAL[face_idx][1], dtype=np.float64)
    right = np.cross(look, up)              # camera +X = right
    down = -up                              # camera +Y = down (image y axis)
    forward = look                          # camera +Z = forward
    return np.column_stack([right, down, forward])


def world_to_camera_quat_t(
    sweep_pos: np.ndarray,
    sweep_R: np.ndarray,
    face_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (qw, qx, qy, qz, tx, ty, tz) world-to-camera per COLMAP convention."""
    R_face = face_camera_to_sweep(face_idx)             # camera→sweep-local
    R_cam2world = sweep_R @ R_face                       # sweep-local→world chained
    R_world2cam = R_cam2world.T
    t_world2cam = -R_world2cam @ sweep_pos
    quat_xyzw = R.from_matrix(R_world2cam).as_quat()    # scipy returns (x,y,z,w)
    qw = quat_xyzw[3]
    qx, qy, qz = quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]
    return np.array([qw, qx, qy, qz]), t_world2cam


# --- COLMAP file writers --------------------------------------------------
def write_cameras_txt(out_path: Path, fx: float, fy: float, cx: float, cy: float, w: int, h: int) -> None:
    """Write COLMAP cameras.txt with a single PINHOLE camera (id=1)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {w} {h} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")


def write_images_txt(out_path: Path, rows: list[tuple[int, np.ndarray, np.ndarray, str]]) -> None:
    """Write COLMAP images.txt. Each row: (image_id, qwxyz, txyz, name)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID) -- empty here\n")
        f.write(f"# Number of images: {len(rows)}\n")
        for image_id, q, t, name in rows:
            f.write(
                f"{image_id} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
                f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} 1 {name}\n"
            )
            f.write("\n")  # empty POINTS2D line


def write_points3d_ply(out_path: Path, points: np.ndarray, color_rgb: tuple[int, int, int] = (180, 180, 180)) -> None:
    """Write a binary little-endian PLY usable by COLMAP/PostShot as initial point cloud."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with out_path.open("wb") as f:
        f.write(header.encode("ascii"))
        body_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                               ("r", "u1"), ("g", "u1"), ("b", "u1")])
        body = np.zeros(n, dtype=body_dtype)
        body["x"] = points[:, 0].astype(np.float32)
        body["y"] = points[:, 1].astype(np.float32)
        body["z"] = points[:, 2].astype(np.float32)
        body["r"] = color_rgb[0]
        body["g"] = color_rgb[1]
        body["b"] = color_rgb[2]
        f.write(body.tobytes())


def write_points3d_txt(
    out_path: Path,
    points: np.ndarray,
    color_rgb: tuple[int, int, int] = (180, 180, 180),
    fake_track_image_id: int = 1,
) -> None:
    """Write COLMAP points3D.txt. Each point gets one fake track entry — some
    importers (PostShot) reject points that have no observations."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}, mean track length: 1\n")
        r, g, b = color_rgb
        for i, p in enumerate(points, start=1):
            f.write(
                f"{i} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {r} {g} {b} 1.0 "
                f"{fake_track_image_id} 0\n"
            )


def write_points3d_bin(
    out_path: Path,
    points: np.ndarray,
    color_rgb: tuple[int, int, int] = (180, 180, 180),
    fake_track_image_id: int = 1,
) -> None:
    """Write COLMAP binary points3D.bin with one fake track per point."""
    import struct
    r, g, b = color_rgb
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(struct.pack("<Q", len(points)))  # num points (uint64)
        for i, p in enumerate(points, start=1):
            f.write(struct.pack("<Q", i))                          # point3D_id
            f.write(struct.pack("<ddd", float(p[0]), float(p[1]), float(p[2])))  # XYZ
            f.write(struct.pack("<BBB", r, g, b))                   # RGB
            f.write(struct.pack("<d", 1.0))                         # error
            f.write(struct.pack("<Q", 1))                           # track length
            f.write(struct.pack("<II", fake_track_image_id, 0))     # one track entry


def write_cameras_bin(out_path: Path, fx: float, fy: float, cx: float, cy: float, w: int, h: int) -> None:
    """Write COLMAP binary cameras.bin with a single PINHOLE camera (id=1, model_id=1)."""
    import struct
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(struct.pack("<Q", 1))                  # num cameras
        f.write(struct.pack("<I", 1))                  # camera_id
        f.write(struct.pack("<i", 1))                  # model_id (PINHOLE)
        f.write(struct.pack("<QQ", w, h))              # width, height
        f.write(struct.pack("<dddd", fx, fy, cx, cy))  # PINHOLE params: fx fy cx cy


def write_images_bin(out_path: Path, rows: list[tuple[int, np.ndarray, np.ndarray, str]]) -> None:
    """Write COLMAP binary images.bin. POINTS2D is empty per image."""
    import struct
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(struct.pack("<Q", len(rows)))
        for image_id, q, t, name in rows:
            f.write(struct.pack("<I", image_id))
            f.write(struct.pack("<dddd", float(q[0]), float(q[1]), float(q[2]), float(q[3])))
            f.write(struct.pack("<ddd", float(t[0]), float(t[1]), float(t[2])))
            f.write(struct.pack("<I", 1))                # camera_id
            name_bytes = name.encode("utf-8") + b"\x00"  # null-terminated
            f.write(name_bytes)
            f.write(struct.pack("<Q", 0))                # num_points2D = 0


def write_transforms_json(
    out_path: Path,
    rows: list[tuple[int, np.ndarray, np.ndarray, str]],
    fx: float, fy: float, cx: float, cy: float, w: int, h: int,
    images_dirname: str = "images",
) -> None:
    """Write Nerfstudio-style transforms.json (camera-to-world matrices, OpenCV/COLMAP convention)."""
    import json as _json
    frames = []
    for image_id, q_w2c, t_w2c, name in rows:
        # Convert world-to-camera quat (qw, qx, qy, qz) → 4x4 c2w matrix
        qw, qx, qy, qz = q_w2c
        rot = R.from_quat([qx, qy, qz, qw]).as_matrix()  # world-to-camera rotation
        # camera-to-world = inverse
        Rwc = rot.T
        twc = -Rwc @ t_w2c
        T = np.eye(4)
        T[:3, :3] = Rwc
        T[:3, 3] = twc
        frames.append({
            "file_path": f"{images_dirname}/{name}",
            "transform_matrix": T.tolist(),
        })
    payload = {
        "camera_model": "OPENCV",
        "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy,
        "w": w, "h": h,
        "k1": 0, "k2": 0, "p1": 0, "p2": 0,
        "frames": frames,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_json.dumps(payload, indent=2), encoding="utf-8")


# --- Validation visualization ---------------------------------------------
def render_validation(
    sweeps: list[Sweep],
    mesh_points: np.ndarray,
    out_dir: Path,
    highlight_sweep_short: str | None = None,
) -> list[Path]:
    """Save matplotlib top-down + side + 3D views; return list of paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)

    P = np.array([s.position for s in sweeps])
    paths: list[Path] = []

    if highlight_sweep_short is None:
        highlight_sweep_short = sweeps[0].sweep_short
    target = next((s for s in sweeps if s.sweep_short == highlight_sweep_short), sweeps[0])

    # Compute face cameras for the highlighted sweep
    face_origins = []
    face_dirs = []
    face_colors = ["red", "green", "blue", "magenta", "cyan", "yellow"]  # 0..5
    for face_idx in range(6):
        q, t = world_to_camera_quat_t(target.position, target.rotation_matrix, face_idx)
        # camera-to-world rotation
        R_w2c = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        R_c2w = R_w2c.T
        # camera position = -R_c2w @ t = sweep position (sanity)
        cam_pos = -R_c2w @ t
        # camera looking direction in world = R_c2w @ (0,0,1)
        cam_fwd = R_c2w[:, 2]
        face_origins.append(cam_pos)
        face_dirs.append(cam_fwd)

    # Top-down (XY plane)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(mesh_points[:, 0], mesh_points[:, 1], s=0.3, c="lightgray", alpha=0.3, label=f"mesh ({len(mesh_points):,} pts)")
    ax.scatter(P[:, 0], P[:, 1], s=14, c="black", label=f"sweeps ({len(sweeps)})")
    ax.scatter(target.position[0], target.position[1], s=100, c="orange", marker="*", label=f"target {target.sweep_short}", zorder=5)
    arrow_len = 1.0
    for i in range(6):
        o = face_origins[i]
        d = face_dirs[i]
        ax.arrow(o[0], o[1], d[0] * arrow_len, d[1] * arrow_len,
                 head_width=0.15, color=face_colors[i], length_includes_head=True, label=f"face{i}")
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_title("Top-down view (XY)")
    ax.legend(loc="upper right", fontsize=8)
    p1 = out_dir / "validate_topdown.png"
    fig.savefig(p1, dpi=110, bbox_inches="tight")
    plt.close(fig)
    paths.append(p1)

    # Side view (XZ plane)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(mesh_points[:, 0], mesh_points[:, 2], s=0.3, c="lightgray", alpha=0.3)
    ax.scatter(P[:, 0], P[:, 2], s=14, c="black")
    ax.scatter(target.position[0], target.position[2], s=100, c="orange", marker="*")
    for i in range(6):
        o = face_origins[i]
        d = face_dirs[i]
        ax.arrow(o[0], o[2], d[0] * arrow_len, d[2] * arrow_len,
                 head_width=0.15, color=face_colors[i], length_includes_head=True)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m, vertical)"); ax.set_title("Side view (XZ)")
    p2 = out_dir / "validate_side_xz.png"
    fig.savefig(p2, dpi=110, bbox_inches="tight")
    plt.close(fig)
    paths.append(p2)

    # 3D view focused on the target sweep
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    span = 4.0
    cx, cy, cz = target.position
    sub = mesh_points[
        (np.abs(mesh_points[:, 0] - cx) < span)
        & (np.abs(mesh_points[:, 1] - cy) < span)
        & (np.abs(mesh_points[:, 2] - cz) < span)
    ]
    if len(sub) > 5000:
        idx = np.random.choice(len(sub), 5000, replace=False)
        sub = sub[idx]
    ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=2, c="lightgray", alpha=0.3)
    ax.scatter(*target.position, s=120, c="orange", marker="*")
    arrow_len = 0.8
    for i in range(6):
        o = face_origins[i]
        d = face_dirs[i]
        ax.quiver(o[0], o[1], o[2], d[0] * arrow_len, d[1] * arrow_len, d[2] * arrow_len,
                  color=face_colors[i], linewidth=2)
        ax.text(o[0] + d[0] * arrow_len * 1.1, o[1] + d[1] * arrow_len * 1.1,
                o[2] + d[2] * arrow_len * 1.1, f"face{i}", color=face_colors[i], fontsize=10)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"3D — target sweep {target.sweep_short}, 6 face directions")
    p3 = out_dir / "validate_3d.png"
    fig.savefig(p3, dpi=110, bbox_inches="tight")
    plt.close(fig)
    paths.append(p3)

    return paths


# --- Main entry -----------------------------------------------------------
def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Matterport → COLMAP 변환 (PostShot 입력용)")
    p.add_argument("--model-dir", required=True,
                   help="matterport-dl 모델 폴더 (downloads/{model_id})")
    p.add_argument("--image-dir", required=True,
                   help="큐브 페이스 폴더 (perspective_images_2k/{model_id})")
    p.add_argument("--output-dir", required=True,
                   help="COLMAP 출력 폴더 (postshot_input/{model_id})")
    p.add_argument("--num-points", type=int, default=100000,
                   help="초기 포인트 클라우드 샘플 수 (기본 100,000)")
    p.add_argument("--lod-max", type=int, default=3,
                   help="로드할 최대 LOD 단계 (기본 3 = 최고 디테일까지)")
    p.add_argument("--validate", action="store_true",
                   help="시각화만 생성하고 COLMAP 출력은 안 함")
    p.add_argument("--validate-sweep", default=None,
                   help="검증 시각화에 강조할 sweep_short (기본 첫 sweep)")
    p.add_argument("--copy-images", action="store_true",
                   help="이미지를 복사 (기본은 hardlink/symlink)")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    model_dir = Path(args.model_dir).resolve()
    image_dir = Path(args.image_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    # Locate JSON + mesh
    sweeps_json = model_dir / "api" / "mp" / "models" / "graph_GetShowcaseSweeps.json"
    if not sweeps_json.is_file():
        print(f"오류: sweep JSON 없음: {sweeps_json}", file=sys.stderr); return 1
    mesh_tiles_dirs = list((model_dir / "models").glob("*/assets/mesh_tiles/~"))
    if not mesh_tiles_dirs:
        print(f"오류: mesh_tiles 폴더를 찾지 못함 ({model_dir}/models/*/assets/mesh_tiles/~)", file=sys.stderr); return 1
    mesh_tiles_dir = mesh_tiles_dirs[0]

    print(f"Sweeps:  {sweeps_json}")
    print(f"Mesh:    {mesh_tiles_dir}")
    print(f"Images:  {image_dir}")
    print(f"Output:  {output_dir}")
    print()

    sweeps = load_sweeps(sweeps_json)
    print(f"로드된 sweep 수: {len(sweeps)}")
    face_index = discover_face_images(image_dir)
    print(f"이미지 발견된 sweep 수: {len(face_index)}")

    # Match sweeps to images by sweep_short
    matched: list[tuple[Sweep, dict[int, Path]]] = []
    for sw in sweeps:
        if sw.sweep_short in face_index:
            matched.append((sw, face_index[sw.sweep_short]))
    print(f"매칭된 sweep 수: {len(matched)}")
    if not matched:
        print("오류: sweep과 이미지가 하나도 매칭되지 않습니다.", file=sys.stderr); return 1

    print("\n메시 로드 중...")
    verts, faces = load_combined_mesh(mesh_tiles_dir, lod_max=args.lod_max)
    print(f"메시: {len(verts):,} vertices, {len(faces):,} faces")
    print(f"  bounds X[{verts[:,0].min():.2f},{verts[:,0].max():.2f}] "
          f"Y[{verts[:,1].min():.2f},{verts[:,1].max():.2f}] "
          f"Z[{verts[:,2].min():.2f},{verts[:,2].max():.2f}]")

    print(f"\n메시 표면에서 {args.num_points:,}개 포인트 샘플링...")
    points = sample_mesh_surface(verts, faces, args.num_points)

    if args.validate:
        validate_dir = output_dir / "_validate"
        print(f"\n검증 시각화 생성: {validate_dir}")
        # Use sweep matched with image when possible
        target_short = args.validate_sweep or matched[0][0].sweep_short
        paths = render_validation(
            [s for s, _ in matched],
            points,
            validate_dir,
            highlight_sweep_short=target_short,
        )
        print("저장된 PNG:")
        for p in paths:
            print(f"  {p}")
        return 0

    # Full COLMAP export
    sparse_dir = output_dir / "sparse" / "0"
    images_dir = output_dir / "images"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # cameras — single PINHOLE, write both text and binary (PostShot prefers binary)
    fx = fy = CUBE_FACE_PIXELS / (2.0 * math.tan(math.radians(CUBE_FACE_FOV_DEG) / 2.0))
    cx = cy = CUBE_FACE_PIXELS / 2.0
    write_cameras_txt(sparse_dir / "cameras.txt", fx, fy, cx, cy, CUBE_FACE_PIXELS, CUBE_FACE_PIXELS)
    write_cameras_bin(sparse_dir / "cameras.bin", fx, fy, cx, cy, CUBE_FACE_PIXELS, CUBE_FACE_PIXELS)

    # images.txt + copy/link images
    rows: list[tuple[int, np.ndarray, np.ndarray, str]] = []
    image_id = 0
    skipped = 0
    bar = tqdm(matched, desc="이미지 변환", unit="sweep")
    for sweep_idx, (sw, faces_paths) in enumerate(bar, start=1):
        sweep_R = sw.rotation_matrix
        for face_idx in range(6):
            src = faces_paths.get(face_idx)
            if src is None:
                skipped += 1
                continue
            image_id += 1
            dst_name = f"scan{sweep_idx:03d}_{sw.sweep_short}_face{face_idx}.jpg"
            dst = images_dir / dst_name
            if not dst.exists():
                if args.copy_images:
                    shutil.copy2(src, dst)
                else:
                    try:
                        dst.hardlink_to(src)
                    except (OSError, NotImplementedError):
                        shutil.copy2(src, dst)
            q, t = world_to_camera_quat_t(sw.position, sweep_R, face_idx)
            rows.append((image_id, q, t, dst_name))
    write_images_txt(sparse_dir / "images.txt", rows)
    write_images_bin(sparse_dir / "images.bin", rows)
    if skipped:
        print(f"  ⚠ 누락된 face 이미지: {skipped}개 (스킵됨)")

    # points3D — only as a separate .ply at project root.
    # PostShot detects points3D.txt/bin in sparse/0/ as a point cloud AND any .ply
    # at project root → conflict ("multiple point clouds"). Keeping only the .ply.
    print(f"초기 포인트 클라우드 저장 ({len(points):,} points)...")
    write_points3d_ply(output_dir / "points_init.ply", points)

    # Nerfstudio fallback (in case COLMAP import fails)
    write_transforms_json(
        output_dir / "transforms.json", rows,
        fx, fy, cx, cy, CUBE_FACE_PIXELS, CUBE_FACE_PIXELS,
    )

    print()
    print("=" * 60)
    print("완료 — COLMAP 출력")
    print("=" * 60)
    print(f"  cameras.txt/.bin : {sparse_dir / 'cameras.*'}")
    print(f"  images.txt/.bin  : {sparse_dir / 'images.*'} ({len(rows)} 이미지)")
    print(f"  images/          : {images_dir} ({len(rows)} 파일)")
    print(f"  points_init.ply  : {output_dir / 'points_init.ply'} ({len(points):,} points)")
    print(f"  transforms.json  : {output_dir / 'transforms.json'} (Nerfstudio 폴백)")
    print()
    print("PostShot 임포트:")
    print(f"  1) PostShot에서 'New Project' → 폴더 드래그: {output_dir}")
    print(f"  2) Profile: 3DGS Splat3, Position Optimization: OFF (포즈가 ground truth)")
    print(f"  3) 카메라: PINHOLE, fx=fy={fx:.1f}, cx=cy={cx:.1f}, {CUBE_FACE_PIXELS}×{CUBE_FACE_PIXELS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
