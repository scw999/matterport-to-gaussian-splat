#!/usr/bin/env python3
"""Extract dense perspective views from Matterport cube faces for SfM."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import py360convert
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

FACE_COUNT = 6
JPEG_QUALITY = 95

# --- Matterport face → cubemap direction mapping --------------------------
# py360convert uses keys F/R/B/L/U/D = Front/Right/Back/Left/Up/Down.
# Verified by visual inspection of cube faces from multiple sweeps
# (scan001/cca8786e, scan012/acd9f18f, scan106/f1072716): face0 always shows
# pure ceiling content, face5 always shows pure floor content (with tripod
# mask), faces1-4 show side walls. Equirect built with this mapping has
# uniform ceiling at top, walls at horizon, floor at bottom.
FACE_TO_DIR: dict[int, str] = {
    0: "U",
    1: "R",
    2: "B",
    3: "L",
    4: "F",
    5: "D",
}

# Per-face rotation in 90-degree steps (counter-clockwise) applied before
# stitching. Some skybox formats need U/D faces rotated to match side faces.
FACE_ROTATION_90: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

# --- File naming ----------------------------------------------------------
SWEEP_FILE_RE = re.compile(r"^scan(\d+)_([0-9a-fA-F]+)_face(\d)\.jpg$")

DEFAULT_INPUT = "./perspective_2k"
DEFAULT_OUTPUT = "./dense_views"
DEFAULT_FOV = 60.0
DEFAULT_AZ_STEP = 30.0
DEFAULT_ELEVATIONS = "-30,0,30"
DEFAULT_RESOLUTION = 1024


def load_cube_faces(face_files: dict[int, Path]) -> dict[str, np.ndarray]:
    """Load 6 face images into a py360convert-compatible cubemap dict."""
    cube: dict[str, np.ndarray] = {}
    for face_idx, dir_key in FACE_TO_DIR.items():
        path = face_files[face_idx]
        with Image.open(path) as im:
            arr = np.array(im.convert("RGB"))
        rot_steps = FACE_ROTATION_90.get(face_idx, 0) % 4
        if rot_steps:
            arr = np.rot90(arr, k=rot_steps)
        cube[dir_key] = arr
    return cube


def cubemap_to_equirect(cube: dict[str, np.ndarray]) -> np.ndarray:
    """Convert a dict-form cubemap to a uint8 equirectangular array."""
    face_size = next(iter(cube.values())).shape[0]
    h = 2 * face_size
    w = 4 * face_size
    equi = py360convert.c2e(cube, h, w, mode="bilinear", cube_format="dict")
    if equi.dtype != np.uint8:
        equi = np.clip(equi, 0, 255).astype(np.uint8)
    return equi


def extract_perspective(
    equi: np.ndarray,
    fov_deg: float,
    azimuth_deg: float,
    elevation_deg: float,
    out_size: int,
) -> np.ndarray:
    """Extract one square perspective view from an equirectangular image."""
    persp = py360convert.e2p(
        equi,
        fov_deg=fov_deg,
        u_deg=azimuth_deg,
        v_deg=elevation_deg,
        out_hw=(out_size, out_size),
        mode="bilinear",
    )
    if persp.dtype != np.uint8:
        persp = np.clip(persp, 0, 255).astype(np.uint8)
    return persp


def discover_sweeps(
    input_root: Path,
) -> list[tuple[str, str, str, dict[int, Path]]]:
    """Walk input_root → list of (model_id, scan_id, sweep_short, faces)."""
    if not input_root.is_dir():
        return []
    out: list[tuple[str, str, str, dict[int, Path]]] = []
    for model_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        groups: dict[tuple[str, str], dict[int, Path]] = {}
        for jpg in model_dir.glob("*.jpg"):
            m = SWEEP_FILE_RE.match(jpg.name)
            if not m:
                continue
            scan_id = f"scan{m.group(1)}"
            sweep_short = m.group(2)
            face_idx = int(m.group(3))
            groups.setdefault((scan_id, sweep_short), {})[face_idx] = jpg
        for (scan_id, sweep_short), faces in sorted(groups.items()):
            if len(faces) == FACE_COUNT:
                out.append((model_dir.name, scan_id, sweep_short, faces))
            else:
                print(
                    f"  ⚠ {model_dir.name}/{scan_id}_{sweep_short}: "
                    f"6면 중 {len(faces)}면만 — 스킵",
                    file=sys.stderr,
                )
    return out


def find_sweep(sweeps, query: str):
    """Return entries whose sweep_short or scan_id contains the query string."""
    return [s for s in sweeps if query in s[2] or query in s[1]]


def parse_elevations(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def get_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def label_thumbnail(img: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    font = get_font(max(20, img.width // 12))
    pad = 6
    bbox = draw.textbbox((0, 0), text, font=font)
    box = [10, 10, 10 + (bbox[2] - bbox[0]) + 2 * pad, 10 + (bbox[3] - bbox[1]) + 2 * pad]
    draw.rectangle(box, fill=(0, 0, 0))
    draw.text((box[0] + pad, box[1] + pad), text, fill=(255, 255, 255), font=font)
    return img


def diagnose(query: str, sweeps, output_root: Path, cell_size: int = 512) -> int:
    """Output labeled face grid + equirect + center perspective for verification."""
    matches = find_sweep(sweeps, query)
    if not matches:
        print(f"오류: '{query}' 매칭 sweep 없음", file=sys.stderr)
        return 1
    if len(matches) > 1:
        print(f"오류: '{query}' 다중 매칭. 더 구체적으로 입력하세요:", file=sys.stderr)
        for m in matches[:8]:
            print(f"  - {m[1]}_{m[2]}", file=sys.stderr)
        return 1
    model_id, scan_id, sweep_short, faces = matches[0]
    out_dir = output_root / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{scan_id}_{sweep_short}"

    # 1) 6-face labeled grid (3 columns × 2 rows)
    cols, rows = 3, 2
    grid = Image.new("RGB", (cell_size * cols, cell_size * rows), (40, 40, 40))
    for i in range(FACE_COUNT):
        c, r = i % cols, i // cols
        with Image.open(faces[i]) as im:
            thumb = im.convert("RGB").resize((cell_size, cell_size))
        thumb = label_thumbnail(thumb, f"face{i}")
        grid.paste(thumb, (c * cell_size, r * cell_size))
    grid_path = out_dir / f"_diagnose_faces_{tag}.png"
    grid.save(grid_path)

    # 2) Equirectangular preview using current FACE_TO_DIR mapping
    cube = load_cube_faces(faces)
    equi = cubemap_to_equirect(cube)
    equi_path = out_dir / f"_diagnose_equirect_{tag}.png"
    equi_preview = Image.fromarray(equi)
    if equi_preview.width > 4096:
        equi_preview = equi_preview.resize((4096, equi_preview.height * 4096 // equi_preview.width))
    equi_preview.save(equi_path)

    # 3) Front-center sanity perspective
    persp = extract_perspective(equi, 90.0, 0.0, 0.0, 1024)
    persp_path = out_dir / f"_diagnose_persp_az0_el0_{tag}.png"
    Image.fromarray(persp).save(persp_path)

    print(f"\n진단 출력 ({tag}):")
    print(f"  1) {grid_path}")
    print(f"     → face0~face5 라벨 격자 (3×2)")
    print(f"  2) {equi_path}")
    print(f"     → 현재 FACE_TO_DIR 매핑으로 만든 equirectangular 파노라마")
    print(f"  3) {persp_path}")
    print(f"     → az=0°, el=0° perspective (정면 방향 추정, FOV=90°)")
    print()
    print("확인 항목:")
    print("  - face0~5 각각이 어느 방향인지 식별 (앞/오른/뒤/왼/위/아래)")
    print("  - equirect 이미지에 봉합선·뒤집힌 영역·이상한 비대칭이 있는지")
    print("  - perspective 이미지가 의미있는 정면 시야인지")
    print()
    print(f"현재 매핑 (수정 전): {FACE_TO_DIR}")
    return 0


def compare_overlap(
    query: str,
    sweeps,
    output_root: Path,
    fov: float,
    az_step: float,
    out_size: int,
) -> int:
    """Generate side-by-side az=0/az=step PNG to verify overlap visually."""
    matches = find_sweep(sweeps, query)
    if not matches:
        print(f"오류: '{query}' 매칭 sweep 없음", file=sys.stderr)
        return 1
    model_id, scan_id, sweep_short, faces = matches[0]
    cube = load_cube_faces(faces)
    equi = cubemap_to_equirect(cube)
    p0 = extract_perspective(equi, fov, 0.0, 0.0, out_size)
    p1 = extract_perspective(equi, fov, az_step, 0.0, out_size)

    expected = max(0.0, 1.0 - az_step / fov) * 100
    margin, header = 10, 60
    combo = Image.new("RGB", (out_size * 2 + margin, out_size + header), (20, 20, 20))
    combo.paste(Image.fromarray(p0), (0, header))
    combo.paste(Image.fromarray(p1), (out_size + margin, header))
    draw = ImageDraw.Draw(combo)
    font = get_font(28)
    draw.text((10, 14), f"az=0  fov={fov}", fill=(255, 255, 255), font=font)
    draw.text(
        (out_size + margin + 10, 14),
        f"az={az_step}  expected overlap ~{expected:.0f}%",
        fill=(255, 255, 255), font=font,
    )

    out_dir = output_root / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"_compare_overlap_{scan_id}_{sweep_short}.png"
    combo.save(out_path)
    print(f"오버랩 비교 출력: {out_path}")
    print(f"FOV={fov}°, az step={az_step}° → 이론 오버랩 ~{expected:.0f}%")
    return 0


def extract_all(
    sweeps,
    output_root: Path,
    fov: float,
    az_step: float,
    elevations: list[float],
    out_size: int,
) -> int:
    """Extract dense views for every sweep, resumable, with progress bar."""
    n_az = int(round(360.0 / az_step))
    azimuths = [round(i * az_step, 6) for i in range(n_az)]
    views_per_sweep = len(azimuths) * len(elevations)
    print(f"Sweep당 {views_per_sweep}개 뷰 (azimuth {len(azimuths)} × elevation {len(elevations)})")

    total_made = 0
    total_skipped = 0
    sweep_iter = tqdm(sweeps, unit="sweep")
    for model_id, scan_id, sweep_short, faces in sweep_iter:
        sweep_iter.set_postfix_str(f"{scan_id}_{sweep_short}")
        out_dir = output_root / model_id
        out_dir.mkdir(parents=True, exist_ok=True)

        targets = []
        for el in elevations:
            for az in azimuths:
                fn = f"{scan_id}_{sweep_short}_az{int(az):03d}_el{int(el):+03d}.jpg"
                targets.append((az, el, out_dir / fn))

        if all(p.is_file() for _, _, p in targets):
            total_skipped += views_per_sweep
            continue

        cube = load_cube_faces(faces)
        equi = cubemap_to_equirect(cube)
        for az, el, path in targets:
            if path.is_file():
                total_skipped += 1
                continue
            persp = extract_perspective(equi, fov, az, el, out_size)
            Image.fromarray(persp).save(path, "JPEG", quality=JPEG_QUALITY)
            total_made += 1

    focal_px = out_size / (2.0 * math.tan(math.radians(fov) / 2.0))
    print()
    print("=" * 60)
    print("완료 요약")
    print("=" * 60)
    print(f"  처리한 sweep 수    : {len(sweeps)}")
    print(f"  생성된 뷰 수       : {total_made}")
    if total_skipped:
        print(f"  이미 존재해서 스킵 : {total_skipped}")
    print(f"  뷰 해상도          : {out_size} × {out_size}")
    print(f"  FOV                : {fov}°")
    print(f"  azimuth 스텝       : {az_step}° ({len(azimuths)}개)")
    print(f"  elevation 스텝     : {elevations}")
    print(f"  출력 폴더          : {output_root.resolve()}")
    print()
    print("RealityScan 권장 카메라 설정:")
    print(f"  - 이미지 크기 : {out_size} × {out_size} px")
    print(f"  - 시야각(FOV) : {fov}°")
    print(f"  - 초점거리    : {focal_px:.0f} px (FOV → focal = side/2/tan(FOV/2))")
    return 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Matterport 큐브 페이스에서 dense perspective 뷰를 추출합니다.",
    )
    p.add_argument("--input", default=DEFAULT_INPUT,
                   help=f"입력 폴더 (기본: {DEFAULT_INPUT})")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help=f"출력 폴더 (기본: {DEFAULT_OUTPUT})")
    p.add_argument("--fov", type=float, default=DEFAULT_FOV,
                   help=f"시야각 (기본: {DEFAULT_FOV}°)")
    p.add_argument("--azimuth-step", type=float, default=DEFAULT_AZ_STEP,
                   help=f"수평 간격 (기본: {DEFAULT_AZ_STEP}°)")
    p.add_argument("--elevation-steps", default=DEFAULT_ELEVATIONS,
                   help=f"콤마 구분 elevation 리스트 (기본: '{DEFAULT_ELEVATIONS}')")
    p.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION,
                   help=f"출력 한 변 픽셀 (기본: {DEFAULT_RESOLUTION})")
    p.add_argument("--diagnose", default=None, metavar="SWEEP_QUERY",
                   help="해당 sweep을 라벨링된 격자/equirect/perspective로 출력")
    p.add_argument("--compare-overlap", default=None, metavar="SWEEP_QUERY",
                   help="오버랩 검증용 az=0 / az=step 사이드바이사이드 출력")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    sweeps = discover_sweeps(input_root)
    if not sweeps:
        print(f"오류: {input_root} 안에서 sweep을 찾지 못했습니다.", file=sys.stderr)
        return 1
    print(f"입력: {input_root}")
    print(f"발견된 sweep: {len(sweeps)}개")

    if args.diagnose:
        return diagnose(args.diagnose, sweeps, output_root)
    if args.compare_overlap:
        return compare_overlap(
            args.compare_overlap, sweeps, output_root,
            args.fov, args.azimuth_step, args.resolution,
        )

    elevations = parse_elevations(args.elevation_steps)
    return extract_all(sweeps, output_root, args.fov, args.azimuth_step, elevations, args.resolution)


if __name__ == "__main__":
    sys.exit(main())
