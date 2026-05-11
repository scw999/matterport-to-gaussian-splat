#!/usr/bin/env python3
"""Minimal 3D Gaussian Splatting trainer using gsplat.

Inputs:
  - transforms.json (Nerfstudio camera-to-world matrices, OPENCV intrinsics)
  - points_init.ply (initial gaussians, e.g. mesh-sampled)
  - images/ folder referenced by transforms.json frames

Output:
  - splat.ply (final gaussians, SuperSplat-compatible)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from plyfile import PlyData, PlyElement
from tqdm import tqdm


# --- IO helpers -----------------------------------------------------------
def load_transforms(transforms_json: Path) -> tuple[dict, list[dict]]:
    data = json.loads(transforms_json.read_text(encoding="utf-8"))
    return data, data["frames"]


def load_init_ply(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"].data
    pts = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    if all(k in v.dtype.names for k in ("red", "green", "blue")):
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=-1).astype(np.float32) / 255.0
    else:
        rgb = np.full_like(pts, 0.5, dtype=np.float32)
    return pts, rgb


def load_image(path: Path, max_size: int | None = None) -> torch.Tensor:
    img = iio.imread(str(path))
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    if max_size and (img.shape[0] > max_size or img.shape[1] > max_size):
        from PIL import Image
        pil = Image.fromarray(img)
        pil.thumbnail((max_size, max_size))
        img = np.array(pil)
    t = torch.from_numpy(img).float() / 255.0  # (H, W, 3) in [0,1]
    return t


# --- Gaussian model -------------------------------------------------------
def init_scales_from_knn(points: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Heuristic init: per-point scale = mean distance to k nearest neighbors."""
    n = len(points)
    if n > 50000:
        # subsample for speed; assign nearest-subsample scale to each point
        idx = torch.randperm(n, device=points.device)[:50000]
        sub = points[idx]
    else:
        sub = points
    # batched squared-distance computation to avoid OOM on large N
    out = torch.empty(n, device=points.device)
    batch = 2048
    for i in range(0, n, batch):
        end = min(i + batch, n)
        d2 = ((points[i:end, None, :] - sub[None, :, :]) ** 2).sum(-1)  # (B, M)
        d2_topk, _ = d2.topk(k + 1, largest=False)
        # exclude self (first column near-zero) when sub == points
        d2_avg = d2_topk[:, 1:k + 1].mean(-1)
        out[i:end] = d2_avg.sqrt()
    out = out.clamp(min=1e-4)
    return out  # (N,) — isotropic init scale


class GaussianModel(torch.nn.Module):
    def __init__(self, points: np.ndarray, rgb: np.ndarray, sh_degree: int = 0):
        super().__init__()
        n = len(points)
        device = "cuda"
        means = torch.from_numpy(points).float().to(device)
        # Initial scales from kNN
        with torch.no_grad():
            scale_init = init_scales_from_knn(means)
            log_scales = scale_init.log()[:, None].repeat(1, 3)
        # Rotations: identity quaternion (w, x, y, z) - gsplat uses (w, x, y, z)
        quats = torch.zeros(n, 4, device=device)
        quats[:, 0] = 1.0
        # Opacities: sigmoid logit corresponding to alpha=0.1
        opacities = torch.full((n,), -2.197, device=device)  # sigmoid(-2.197)≈0.1
        # Colors as SH DC term: convert linear RGB to SH DC via *0.28209... ~ 1/sqrt(4pi)
        # Simpler: store raw RGB as features, apply sigmoid to clamp
        # We use SH degree 0: just RGB stored as DC coefficient
        # SH0 = (color - 0.5) / 0.28209479177387814
        sh_dc = (torch.from_numpy(rgb).float().to(device) - 0.5) / 0.2820947917738781

        self.means = torch.nn.Parameter(means)
        self.log_scales = torch.nn.Parameter(log_scales)
        self.quats = torch.nn.Parameter(quats)
        self.opacities_raw = torch.nn.Parameter(opacities)
        self.sh_dc = torch.nn.Parameter(sh_dc)  # (N, 3)

        self.sh_degree = sh_degree
        if sh_degree > 0:
            n_extra = (sh_degree + 1) ** 2 - 1
            self.sh_extra = torch.nn.Parameter(torch.zeros(n, n_extra, 3, device=device))
        else:
            self.register_parameter("sh_extra", None)

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[0]

    def get_colors(self) -> torch.Tensor:
        if self.sh_extra is None:
            return self.sh_dc[:, None, :]  # (N, 1, 3)
        return torch.cat([self.sh_dc[:, None, :], self.sh_extra], dim=1)

    def get_opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.opacities_raw)

    def get_scales(self) -> torch.Tensor:
        return self.log_scales.exp()

    def get_quats(self) -> torch.Tensor:
        return F.normalize(self.quats, dim=-1)


# --- Coordinate convention ------------------------------------------------
# Nerfstudio transforms.json frames are camera-to-world in OpenGL convention
# (camera looks at -Z, +Y up). gsplat's rasterization expects OpenCV
# convention (camera looks at +Z, +Y down) for viewmats (world-to-camera).
# We convert: flip Y and Z axes of the camera frame.
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def c2w_to_w2c_opencv(c2w_opengl: np.ndarray) -> np.ndarray:
    """Camera-to-world (OpenGL/Blender) → world-to-camera (OpenCV)."""
    c2w_opencv = c2w_opengl @ OPENGL_TO_OPENCV
    w2c = np.linalg.inv(c2w_opencv)
    return w2c.astype(np.float32)


def detect_convention(transforms: dict) -> str:
    """Best-effort detection of transforms.json camera convention."""
    # Our matterport_to_colmap.py writes COLMAP world-to-camera quats/translations
    # converted to camera-to-world matrices via inverse. The original is OpenCV
    # (we built the camera→sweep rotation in OpenCV convention).
    # So our transforms.json is OPENCV camera-to-world.
    # Nerfstudio default is OpenGL/Blender. transforms.json from instant-NGP / nerfstudio
    # examples are OpenGL.
    # If "applied_transform" or other markers exist, use those. Otherwise default OpenCV.
    cam_model = transforms.get("camera_model", "OPENCV")
    if cam_model in ("OPENCV", "PINHOLE"):
        return "opencv"
    return "opengl"


# --- Training -------------------------------------------------------------
def main(args: argparse.Namespace) -> int:
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    proj = Path(args.data).resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    transforms_json = proj / "transforms.json"
    if not transforms_json.is_file():
        print(f"오류: transforms.json 없음: {transforms_json}", file=sys.stderr)
        return 1
    transforms, frames = load_transforms(transforms_json)
    convention = detect_convention(transforms)
    print(f"transforms.json: {len(frames)} frames, camera_model={transforms.get('camera_model', '?')}, convention={convention}")

    fx = float(transforms["fl_x"])
    fy = float(transforms["fl_y"])
    cx = float(transforms["cx"])
    cy = float(transforms["cy"])
    img_w = int(transforms["w"])
    img_h = int(transforms["h"])

    # Scale factor if downsampling
    scale = args.image_downscale
    if scale != 1.0:
        fx, fy, cx, cy = fx / scale, fy / scale, cx / scale, cy / scale
        img_w = int(img_w / scale)
        img_h = int(img_h / scale)
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device)
    print(f"Image: {img_w}x{img_h}, intrinsics fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    # Build cameras list: world-to-camera 4x4 + image path
    cameras = []
    for f in frames:
        c2w = np.array(f["transform_matrix"], dtype=np.float32)
        if convention == "opengl":
            w2c = c2w_to_w2c_opencv(c2w)
        else:
            w2c = np.linalg.inv(c2w).astype(np.float32)
        img_path = proj / f["file_path"]
        if not img_path.is_file():
            continue
        cameras.append({"w2c": torch.from_numpy(w2c).to(device), "image": img_path})
    print(f"발견된 사용 가능 카메라: {len(cameras)}")
    if not cameras:
        print("오류: 매칭되는 이미지 0개", file=sys.stderr)
        return 1

    # Init points
    ply_path = proj / "points_init.ply"
    if not ply_path.is_file():
        print(f"오류: points_init.ply 없음: {ply_path}", file=sys.stderr)
        return 1
    points, rgb = load_init_ply(ply_path)
    print(f"초기 포인트: {len(points):,}")

    model = GaussianModel(points, rgb, sh_degree=args.sh_degree).to(device)

    # Optimizer with parameter-specific learning rates (mimic 3DGS paper)
    optimizer = torch.optim.Adam([
        {"params": [model.means], "lr": 1.6e-4 * args.lr_scale, "name": "means"},
        {"params": [model.log_scales], "lr": 5e-3 * args.lr_scale, "name": "scales"},
        {"params": [model.quats], "lr": 1e-3 * args.lr_scale, "name": "quats"},
        {"params": [model.opacities_raw], "lr": 5e-2 * args.lr_scale, "name": "opacities"},
        {"params": [model.sh_dc], "lr": 2.5e-3 * args.lr_scale, "name": "sh_dc"},
    ])

    # --- Train loop ---
    print(f"\n학습 시작: {args.iterations} iterations")
    t0 = time.time()
    rng = np.random.default_rng(0)
    cam_idx = list(range(len(cameras)))
    pbar = tqdm(range(args.iterations), desc="train", unit="it")

    cached_imgs: dict[int, torch.Tensor] = {}

    for step in pbar:
        # Sample one camera
        i = int(rng.integers(0, len(cameras)))
        cam = cameras[i]

        # Load image (cache)
        if i in cached_imgs:
            target = cached_imgs[i]
        else:
            t = load_image(cam["image"], max_size=img_w if scale == 1.0 else max(img_w, img_h))
            if scale != 1.0:
                # Resize via PIL
                from PIL import Image
                pil = Image.fromarray((t.numpy() * 255).astype(np.uint8))
                pil = pil.resize((img_w, img_h), Image.BILINEAR)
                t = torch.from_numpy(np.array(pil)).float() / 255.0
            target = t.to(device)
            if len(cached_imgs) < args.image_cache_max:
                cached_imgs[i] = target

        viewmats = cam["w2c"][None, ...]  # (1, 4, 4)
        Ks = K[None, ...]                  # (1, 3, 3)

        renders, alphas, info = rasterization(
            means=model.means,
            quats=model.get_quats(),
            scales=model.get_scales(),
            opacities=model.get_opacities(),
            colors=model.get_colors(),
            viewmats=viewmats,
            Ks=Ks,
            width=img_w,
            height=img_h,
            sh_degree=args.sh_degree,
            packed=False,
        )
        rendered = renders[0]  # (H, W, 3)

        # Loss: L1 + 0.2 * (1 - SSIM)
        l1 = (rendered - target).abs().mean()
        # Simple SSIM approximation (Gaussian filter on luminance) — using just L1 here
        # Could add SSIM later for quality
        loss = l1

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            psnr = -10.0 * (((rendered - target) ** 2).mean().clamp(min=1e-12)).log10()
            elapsed = time.time() - t0
            its = (step + 1) / max(elapsed, 1e-6)
            pbar.set_postfix_str(f"loss={loss.item():.4f} psnr={psnr.item():.2f} N={model.num_gaussians:,} {its:.1f}it/s")

    # --- Save final PLY ---
    print("\nPLY 저장 중...")
    save_ply(model, out_dir / "splat.ply")
    elapsed_total = time.time() - t0
    print(f"\n완료: {elapsed_total/60:.1f}분, {model.num_gaussians:,} gaussians")
    print(f"출력: {out_dir / 'splat.ply'}")
    return 0


def save_ply(model: GaussianModel, path: Path) -> None:
    """SuperSplat-compatible PLY: positions, normals, opacity, scale_*, rot_*, f_dc_*, f_rest_*."""
    means = model.means.detach().cpu().numpy()
    log_scales = model.log_scales.detach().cpu().numpy()
    quats = F.normalize(model.quats, dim=-1).detach().cpu().numpy()
    opac = model.opacities_raw.detach().cpu().numpy()
    sh_dc = model.sh_dc.detach().cpu().numpy()
    n = len(means)

    dtype_list = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    if model.sh_extra is not None:
        sh_extra = model.sh_extra.detach().cpu().numpy().reshape(n, -1)
        for i in range(sh_extra.shape[1]):
            dtype_list.append((f"f_rest_{i}", "f4"))
    arr = np.zeros(n, dtype=dtype_list)
    arr["x"], arr["y"], arr["z"] = means[:, 0], means[:, 1], means[:, 2]
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = sh_dc[:, 0], sh_dc[:, 1], sh_dc[:, 2]
    arr["opacity"] = opac
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = log_scales[:, 0], log_scales[:, 1], log_scales[:, 2]
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    if model.sh_extra is not None:
        for i in range(sh_extra.shape[1]):
            arr[f"f_rest_{i}"] = sh_extra[:, i]
    el = PlyElement.describe(arr, "vertex")
    PlyData([el]).write(str(path))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Project folder with transforms.json + images/ + points_init.ply")
    p.add_argument("--output", required=True, help="Output folder")
    p.add_argument("--iterations", type=int, default=10000)
    p.add_argument("--sh-degree", type=int, default=0, help="Spherical harmonics degree (0=RGB only, 3=full)")
    p.add_argument("--lr-scale", type=float, default=1.0)
    p.add_argument("--image-downscale", type=float, default=2.0, help="Downscale images by this factor")
    p.add_argument("--image-cache-max", type=int, default=400)
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(main(parse_args()))
