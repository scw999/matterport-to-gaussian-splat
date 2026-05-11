#!/usr/bin/env python3
"""Build a seamless equirectangular panorama from 6 cube faces using
Laplacian-pyramid multi-band blending.

Each face is projected onto the equirect with a soft cosine weight mask; the
6 projections are then combined per pyramid level so that low frequencies
(color, exposure) blend smoothly across cube boundaries while high
frequencies (edges, fine detail) take from a single face -- avoiding both
the seams of hard assignment and the ghosting of plain alpha blending on
parallax-shifted close-up content.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, zoom


def _gauss_filter(img: np.ndarray, sigma: float) -> np.ndarray:
    """Channel-wise Gaussian blur for 2D or 3D arrays."""
    if img.ndim == 2:
        return gaussian_filter(img, sigma)
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        out[..., c] = gaussian_filter(img[..., c], sigma)
    return out


def _down(img: np.ndarray) -> np.ndarray:
    """Blur + 2x downsample."""
    blurred = _gauss_filter(img, 1.0)
    return blurred[::2, ::2].copy()


def _up_to(img: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Upsample img to match target_shape's first two dims via bilinear zoom."""
    th, tw = target_shape[:2]
    sh, sw = img.shape[:2]
    if (sh, sw) == (th, tw):
        return img
    fy = th / sh
    fx = tw / sw
    if img.ndim == 2:
        return zoom(img, (fy, fx), order=1)
    out = np.empty((th, tw, img.shape[2]), dtype=img.dtype)
    for c in range(img.shape[2]):
        out[..., c] = zoom(img[..., c], (fy, fx), order=1)
    return out


def _gaussian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    pyr = [img]
    for _ in range(levels - 1):
        pyr.append(_down(pyr[-1]))
    return pyr


def _laplacian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    g = _gaussian_pyramid(img, levels)
    lap = []
    for k in range(levels - 1):
        lap.append(g[k] - _up_to(g[k + 1], g[k].shape))
    lap.append(g[-1])
    return lap


def _collapse_pyramid(lap: Sequence[np.ndarray]) -> np.ndarray:
    out = lap[-1]
    for k in range(len(lap) - 2, -1, -1):
        out = _up_to(out, lap[k].shape) + lap[k]
    return out


def _equirect_rays(h: int, w: int) -> np.ndarray:
    """For each (i, j) in an h×w equirect, return the unit ray in sweep-local.

    Convention matches py360convert's e2p: center column (j = w/2) is the
    "front" direction (+Y in our sweep-local). j = 0 and j = w-1 are the
    back (-Y). Row 0 is north pole (+Z), row h-1 is south pole (-Z).
    """
    j = np.arange(w, dtype=np.float32) + 0.5
    az = (j / w - 0.5) * 2 * np.pi  # -pi..+pi, center column -> 0 (front)
    el = (0.5 - (np.arange(h, dtype=np.float32) + 0.5) / h) * np.pi
    cos_el = np.cos(el)[:, None]
    sin_el = np.sin(el)[:, None]
    sin_az = np.sin(az)[None, :]
    cos_az = np.cos(az)[None, :]
    rays = np.stack([sin_az * cos_el,
                     cos_az * cos_el,
                     np.broadcast_to(sin_el, (h, w))], axis=-1)
    return rays.astype(np.float32)


def project_face_to_equirect(
    face_arr: np.ndarray,
    R_face2sweep: np.ndarray,
    rays: np.ndarray,
    mask_power: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a single cube face onto an h×w equirect.

    Returns (color_HxWx3 float32, weight_HxW float32). Pixels for which this
    face cannot contribute (ray points away from face) get weight 0.
    """
    h, w = rays.shape[:2]
    look_face = R_face2sweep[:, 2]
    cosines = rays @ look_face  # H, W
    # Soft mask sharply peaked at face center; raised to a power so cube
    # corners (cos ~ 0.577) get small but nonzero weight.
    mask = np.maximum(cosines, 0.0) ** mask_power

    rays_face = rays @ R_face2sweep  # H, W, 3 (sweep -> face camera coords)
    rz = rays_face[..., 2]
    rz_safe = np.where(rz > 1e-9, rz, np.float32(1e-9))
    u = rays_face[..., 0] / rz_safe
    v = rays_face[..., 1] / rz_safe

    fh, fw = face_arr.shape[:2]
    # Clip rays falling slightly past the face boundary to the edge so
    # bilinear interp samples the edge pixel rather than wrapping.
    u_clip = np.clip(u, -1.0, 1.0)
    v_clip = np.clip(v, -1.0, 1.0)
    px = (u_clip + 1.0) * 0.5 * (fw - 1)
    py = (v_clip + 1.0) * 0.5 * (fh - 1)
    coords = np.stack([py.ravel(), px.ravel()])

    color = np.empty((h * w, 3), dtype=np.float32)
    for c in range(3):
        color[:, c] = map_coordinates(face_arr[..., c], coords, order=1, mode="nearest")
    color = color.reshape(h, w, 3)

    # Zero out where ray points away from face (rz <= 0)
    valid = rz > 0
    mask = mask * valid
    return color, mask.astype(np.float32)


def build_seamless_equirect(
    cube_arrs: dict[int, np.ndarray],
    face_R_cam2sweep: dict[int, np.ndarray],
    h: int = 2048,
    w: int = 4096,
    levels: int = 6,
) -> np.ndarray:
    """Build a seamless equirect from 6 cube faces via Burt-Adelson Laplacian
    multi-band blending.

    Input mask is BINARY (winner-take-all per pixel, the face with highest
    cosine alignment wins) — this is critical to avoid ghosting at cube
    corners where 3 faces overlap. The Gaussian pyramid naturally smooths
    the binary mask at lower-frequency levels, so high frequencies (edges)
    come from a single face while low frequencies (color, exposure) blend
    smoothly. Returns uint8 (h, w, 3).
    """
    rays = _equirect_rays(h, w)

    face_colors: list[np.ndarray] = []
    cos_stack: list[np.ndarray] = []
    valid_stack: list[np.ndarray] = []
    for fi in sorted(cube_arrs.keys()):
        color, _ = project_face_to_equirect(
            cube_arrs[fi], face_R_cam2sweep[fi], rays, mask_power=1.0,
        )
        face_colors.append(color)
        look_face = face_R_cam2sweep[fi][:, 2]
        cos_stack.append(rays @ look_face)
        rays_face = rays @ face_R_cam2sweep[fi]
        valid_stack.append(rays_face[..., 2] > 0)

    cos_stack_arr = np.stack(cos_stack)  # F, H, W
    valid_stack_arr = np.stack(valid_stack)
    # Set invalid faces to -inf so they never win the argmax
    cos_stack_arr = np.where(valid_stack_arr, cos_stack_arr, -np.inf)
    winner = np.argmax(cos_stack_arr, axis=0)  # H, W with face index

    # Binary masks per face (1 where that face wins)
    masks_binary = [
        (winner == fi).astype(np.float32) for fi in range(len(face_colors))
    ]

    color_pyramids = [_laplacian_pyramid(c, levels) for c in face_colors]
    mask_pyramids = [_gaussian_pyramid(m, levels) for m in masks_binary]

    blended_pyr: list[np.ndarray] = []
    for k in range(levels):
        acc = np.zeros_like(color_pyramids[0][k])
        for fi in range(len(face_colors)):
            m = mask_pyramids[fi][k][..., None]
            acc = acc + color_pyramids[fi][k] * m
        blended_pyr.append(acc)

    final = _collapse_pyramid(blended_pyr)
    return np.clip(final, 0, 255).astype(np.uint8)
