#!/usr/bin/env python3
"""Test multiple FACE_TO_DIR mappings on scan106 (which clearly has
face0=ceiling-only, face5=floor-only). Output equirect + cardinals for each
candidate so we can visually pick the correct one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import py360convert
from PIL import Image, ImageDraw, ImageFont


SCAN_DIR = Path(r"C:\Users\scw99\work\matterport_Export\yangpyong\matterport-dl\brush_dense_input_g7EGi53S2b8\images")
OUT_DIR = Path(r"C:\Users\scw99\work\matterport_Export\yangpyong\matterport-dl\dense_views\g7EGi53S2b8\face_mapping_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_face(face_idx: int, sweep_short: str) -> np.ndarray:
    path = SCAN_DIR / f"scan{106 if sweep_short=='f1072716' else 12:03d}_{sweep_short}_cube{face_idx}.jpg"
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def get_font(size):
    for name in ("arial.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def label_image(img, text):
    draw = ImageDraw.Draw(img)
    font = get_font(max(20, img.width // 16))
    pad = 6
    bbox = draw.textbbox((0, 0), text, font=font)
    box = [10, 10, 10 + (bbox[2] - bbox[0]) + 2 * pad,
           10 + (bbox[3] - bbox[1]) + 2 * pad]
    draw.rectangle(box, fill=(0, 0, 0))
    draw.text((box[0] + pad, box[1] + pad), text, fill=(255, 255, 255), font=font)
    return img


def make_equirect(face_to_dir: dict[int, str], sweep_short: str) -> np.ndarray:
    cube = {face_to_dir[i]: load_face(i, sweep_short) for i in range(6)}
    fs = next(iter(cube.values())).shape[0]
    equi = py360convert.c2e(cube, fs * 2, fs * 4, mode="bilinear", cube_format="dict")
    return np.clip(equi, 0, 255).astype(np.uint8)


def make_cardinals_grid(equi: np.ndarray, label: str, cell_size: int = 384) -> Image.Image:
    cardinals = [
        ("F (az=0,el=0)", 0, 0),
        ("R (az=90)", 90, 0),
        ("B (az=180)", 180, 0),
        ("L (az=270)", 270, 0),
        ("U (el=+89)", 0, 89),
        ("D (el=-89)", 0, -89),
    ]
    cols, rows = 3, 2
    grid = Image.new("RGB", (cell_size * cols, cell_size * rows + 60), (30, 30, 30))
    draw = ImageDraw.Draw(grid)
    draw.text((10, 10), label, fill=(255, 255, 0), font=get_font(28))
    for i, (txt, az, el) in enumerate(cardinals):
        c, r = i % cols, i // cols
        persp = py360convert.e2p(equi, fov_deg=90.0, u_deg=az, v_deg=el,
                                 out_hw=(cell_size, cell_size), mode="bilinear")
        persp = np.clip(persp, 0, 255).astype(np.uint8)
        thumb = label_image(Image.fromarray(persp), txt)
        grid.paste(thumb, (c * cell_size, r * cell_size + 60))
    return grid


CANDIDATES = [
    ("M0_current(0F1R2B3L4D5U)", {0: "F", 1: "R", 2: "B", 3: "L", 4: "D", 5: "U"}),
    ("M1_swap04(0U1F2R3B4L5D)", {0: "U", 1: "F", 2: "R", 3: "B", 4: "L", 5: "D"}),
    ("M2_swap05(0U1R2B3L4D5D)", {0: "U", 1: "R", 2: "B", 3: "L", 4: "F", 5: "D"}),
    ("M3_v3(0U1B2L3F4R5D)", {0: "U", 1: "B", 2: "L", 3: "F", 4: "R", 5: "D"}),
    ("M4_v4(0U1L2F3R4B5D)", {0: "U", 1: "L", 2: "F", 3: "R", 4: "B", 5: "D"}),
]


def main():
    sweep_short = "f1072716"
    print(f"Testing {len(CANDIDATES)} mappings on scan106_{sweep_short}")
    grids = []
    for name, mapping in CANDIDATES:
        try:
            equi = make_equirect(mapping, sweep_short)
            equi_thumb = Image.fromarray(equi)
            equi_thumb.thumbnail((1600, 800))
            equi_thumb.save(OUT_DIR / f"equi_{name}.jpg", quality=85)

            grid = make_cardinals_grid(equi, name, cell_size=384)
            grid.save(OUT_DIR / f"cardinals_{name}.jpg", quality=85)
            grids.append(grid)
            print(f"  OK {name}")
        except Exception as e:
            print(f"  FAIL {name}: {e}")

    # Stack all grids vertically into a single comparison image
    if grids:
        w = grids[0].width
        h_each = grids[0].height
        combined = Image.new("RGB", (w, h_each * len(grids)), (0, 0, 0))
        for i, g in enumerate(grids):
            combined.paste(g, (0, i * h_each))
        # Resize to fit easier viewing
        combined.thumbnail((1600, 5000))
        combined.save(OUT_DIR / "ALL_compared.jpg", quality=85)
        print(f"\nSaved comparison to {OUT_DIR / 'ALL_compared.jpg'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
