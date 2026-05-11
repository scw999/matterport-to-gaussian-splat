#!/usr/bin/env python3
"""Stitch (or rename) Matterport cube-face images into RealityScan-ready inputs.

Two input layouts are supported and auto-detected per model:

1) Tile-mosaic layout (matterport-dl `MAIN_ASSET_DOWNLOAD` output):
     {model}/tiles/{sweep_uuid}/{res}_face{N}_{x}_{y}.jpg
   Tiles are 512x512 each; grid: 512=1x1, 1k=2x2, 2k=4x4, 4k=8x8.

2) Pre-stitched skybox layout (matterport-dl Advanced Assets output):
     {model}/.../assets/pan/{low|high|2k|4k}/[~/]{sweep_id}_skybox{N}.jpg
   Each face is a single complete image; only renaming/copying is needed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

from PIL import Image
from tqdm import tqdm

TILE_PIXELS = 512
FACE_COUNT = 6

# Tile-mosaic layout: tiles per side per resolution.
RESOLUTION_GRID: dict[str, int] = {
    "512": 1,
    "1k": 2,
    "2k": 4,
    "4k": 8,
}

# Skybox layout: maps requested resolution to the folder name on disk.
RESOLUTION_SKYBOX_DIR: dict[str, str] = {
    "512": "low",
    "1k": "high",
    "2k": "2k",
    "4k": "4k",
}

# Final image side length in pixels for each resolution choice (and skybox layout dim).
RESOLUTION_SIDE_PX: dict[str, int] = {
    "512": 512,
    "1k": 1024,
    "2k": 2048,
    "4k": 4096,
}

JPEG_QUALITY = 95

ModelFormat = Literal["tiles", "skybox"]


@dataclass
class ModelInput:
    """One detected Matterport model in the search root."""

    name: str
    fmt: ModelFormat
    # For "tiles" format: path of `{model}/tiles/`.
    # For "skybox" format: path of `.../assets/pan/{res}/` (or its `~/` child).
    source_dir: Path


@dataclass
class StitchStats:
    models: int = 0
    sweeps: int = 0
    images: int = 0
    skipped_existing: int = 0
    skipped_missing: int = 0
    formats_used: set[ModelFormat] = field(default_factory=set)


def _find_skybox_dir(model_dir: Path, resolution: str) -> Path | None:
    """Locate `.../assets/pan/{res}/` (or its `~/` subdir) under a model dir."""
    res_folder = RESOLUTION_SKYBOX_DIR[resolution]
    # matterport-dl puts panos under: models/<hash>/assets/pan/<res>/[~/]
    for pan_dir in model_dir.glob("models/*/assets/pan"):
        candidate = pan_dir / res_folder
        if not candidate.is_dir():
            continue
        # Files may live directly in {res}/ or in {res}/~/ depending on TILDE flag.
        tilde_dir = candidate / "~"
        if tilde_dir.is_dir() and any(tilde_dir.glob("*_skybox*.jpg")):
            return tilde_dir
        if any(candidate.glob("*_skybox*.jpg")):
            return candidate
    return None


def find_models(
    root: Path,
    model_filter: str | None,
    resolution: str,
) -> list[ModelInput]:
    """Detect models in `root`, preferring the tile-mosaic layout when present."""
    if model_filter:
        candidates = [root / model_filter]
        if not candidates[0].is_dir():
            print(f"오류: 모델 폴더를 찾을 수 없습니다: {candidates[0]}", file=sys.stderr)
            return []
    else:
        candidates = [p for p in sorted(root.iterdir()) if p.is_dir()]

    found: list[ModelInput] = []
    for model_dir in candidates:
        tiles_dir = model_dir / "tiles"
        if tiles_dir.is_dir() and any(tiles_dir.iterdir()):
            found.append(ModelInput(model_dir.name, "tiles", tiles_dir))
            continue
        skybox_dir = _find_skybox_dir(model_dir, resolution)
        if skybox_dir is not None:
            found.append(ModelInput(model_dir.name, "skybox", skybox_dir))

    return found


def find_tile_sweeps(tiles_dir: Path) -> list[Path]:
    """Return sweep UUID directories inside a model's `tiles/` folder."""
    return sorted(p for p in tiles_dir.iterdir() if p.is_dir())


def find_skybox_sweeps(skybox_dir: Path) -> list[str]:
    """Return unique sweep IDs found in a skybox directory."""
    sweep_ids: set[str] = set()
    for jpg in skybox_dir.glob("*_skybox*.jpg"):
        # File: "<sweep_id>_skybox<N>.jpg"
        stem = jpg.stem
        idx = stem.rfind("_skybox")
        if idx > 0:
            sweep_ids.add(stem[:idx])
    return sorted(sweep_ids)


def stitch_face(
    sweep_dir: Path,
    resolution: str,
    grid: int,
    face: int,
) -> Image.Image | None:
    """Compose one cube face from its tiles. Returns None if any tile is missing."""
    side_px = grid * TILE_PIXELS
    canvas = Image.new("RGB", (side_px, side_px))
    missing: list[str] = []

    for x in range(grid):
        for y in range(grid):
            tile_name = f"{resolution}_face{face}_{x}_{y}.jpg"
            tile_path = sweep_dir / tile_name
            if not tile_path.is_file():
                missing.append(tile_name)
                continue
            with Image.open(tile_path) as tile:
                # Matterport tile coords: x = column index, y = row index.
                canvas.paste(tile, (x * TILE_PIXELS, y * TILE_PIXELS))

    if missing:
        sample = ", ".join(missing[:3])
        more = f" 외 {len(missing) - 3}개" if len(missing) > 3 else ""
        print(
            f"  ⚠ 경고: {sweep_dir.name} face{face} 타일 누락 ({sample}{more}) — 이 면 건너뜀",
            file=sys.stderr,
        )
        return None

    return canvas


def emit_sweep_from_tiles(
    sweep_dir: Path,
    output_dir: Path,
    scan_index: int,
    resolution: str,
    grid: int,
    stats: StitchStats,
) -> None:
    """Stitch and save all 6 faces for one sweep from its tile mosaic."""
    sweep_short = sweep_dir.name[:8]
    for face in range(FACE_COUNT):
        out_path = output_dir / f"scan{scan_index:03d}_{sweep_short}_face{face}.jpg"
        if out_path.is_file():
            stats.skipped_existing += 1
            continue

        face_img = stitch_face(sweep_dir, resolution, grid, face)
        if face_img is None:
            stats.skipped_missing += 1
            continue

        face_img.save(out_path, "JPEG", quality=JPEG_QUALITY)
        stats.images += 1


def emit_sweep_from_skybox(
    skybox_dir: Path,
    sweep_id: str,
    output_dir: Path,
    scan_index: int,
    expected_side_px: int,
    stats: StitchStats,
) -> None:
    """Copy/rename all 6 pre-stitched skybox faces for one sweep."""
    sweep_short = sweep_id[:8]
    for face in range(FACE_COUNT):
        out_path = output_dir / f"scan{scan_index:03d}_{sweep_short}_face{face}.jpg"
        if out_path.is_file():
            stats.skipped_existing += 1
            continue

        src = skybox_dir / f"{sweep_id}_skybox{face}.jpg"
        if not src.is_file():
            print(
                f"  ⚠ 경고: {sweep_short} face{face} skybox 파일 없음 ({src.name}) — 건너뜀",
                file=sys.stderr,
            )
            stats.skipped_missing += 1
            continue

        # Verify dimensions; if smaller than expected (e.g. partial download), warn.
        with Image.open(src) as img:
            if img.size != (expected_side_px, expected_side_px):
                print(
                    f"  ⚠ 경고: {src.name} 크기가 {img.size}, 예상은 "
                    f"{expected_side_px}x{expected_side_px} — 그대로 저장합니다",
                    file=sys.stderr,
                )
            img.convert("RGB").save(out_path, "JPEG", quality=JPEG_QUALITY)
        stats.images += 1


def write_realityscan_hint(output_dir: Path, side_px: int) -> None:
    """Write a short note describing camera intrinsics for RealityScan import."""
    focal_px = side_px // 2  # 90° FOV → focal = side/2
    content = (
        "RealityScan import hint\n"
        "=======================\n"
        "All images in this folder share the same camera (Matterport cube face).\n"
        "Identical intrinsics:\n"
        f"  - Image size: {side_px} x {side_px} px\n"
        f"  - Field of view: 90 deg (square pinhole)\n"
        f"  - Focal length: {focal_px} px (in pixel units)\n"
        f"  - Principal point: image center ({side_px / 2:.1f}, {side_px / 2:.1f})\n"
        "\n"
        "In RealityScan: select all images -> Image group -> set Prior calibration\n"
        "to a single shared group with the focal length above (Fixed during alignment).\n"
    )
    (output_dir / "realityscan_hint.txt").write_text(content, encoding="utf-8")


def process_model(
    model: ModelInput,
    output_root: Path,
    resolution: str,
    stats: StitchStats,
) -> Path:
    """Process all sweeps in one model and write images + hint file."""
    out_dir = output_root / model.name
    out_dir.mkdir(parents=True, exist_ok=True)
    side_px = RESOLUTION_SIDE_PX[resolution]
    stats.formats_used.add(model.fmt)

    if model.fmt == "tiles":
        grid = RESOLUTION_GRID[resolution]
        sweeps = find_tile_sweeps(model.source_dir)
        if not sweeps:
            print(f"  ⚠ {model.name}: tile sweep 폴더가 비어있습니다", file=sys.stderr)
            return out_dir
        stats.sweeps += len(sweeps)
        bar = tqdm(sweeps, desc=f"  {model.name} [tiles]", unit="sweep")
        for idx, sweep in enumerate(bar, start=1):
            bar.set_postfix_str(sweep.name[:8])
            emit_sweep_from_tiles(sweep, out_dir, idx, resolution, grid, stats)
    else:
        sweep_ids = find_skybox_sweeps(model.source_dir)
        if not sweep_ids:
            print(f"  ⚠ {model.name}: skybox 파일을 찾을 수 없습니다", file=sys.stderr)
            return out_dir
        stats.sweeps += len(sweep_ids)
        bar = tqdm(sweep_ids, desc=f"  {model.name} [skybox]", unit="sweep")
        for idx, sweep_id in enumerate(bar, start=1):
            bar.set_postfix_str(sweep_id[:8])
            emit_sweep_from_skybox(model.source_dir, sweep_id, out_dir, idx, side_px, stats)

    write_realityscan_hint(out_dir, side_px)
    return out_dir


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Matterport 큐브 데이터를 perspective 이미지로 준비합니다 "
        "(타일 모자이크는 스티칭, 스카이박스는 리네임).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="처리할 특정 모델 ID. 생략 시 root 안의 모든 모델을 처리합니다.",
    )
    parser.add_argument(
        "--resolution",
        default="1k",
        choices=sorted(RESOLUTION_GRID.keys()),
        help="출력 해상도 (기본값: 1k).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="출력 폴더 경로 (기본값: <root>/perspective_images/).",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="모델 폴더가 있는 루트 디렉토리. 생략 시 ./downloads 가 있으면 그쪽, "
        "없으면 현재 디렉토리.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def resolve_root(root_arg: str | None) -> Path:
    if root_arg is not None:
        return Path(root_arg).resolve()
    cwd = Path.cwd()
    downloads = cwd / "downloads"
    return downloads.resolve() if downloads.is_dir() else cwd.resolve()


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    root = resolve_root(args.root)
    resolution = args.resolution
    side_px = RESOLUTION_SIDE_PX[resolution]

    output_root = Path(args.output).resolve() if args.output else (Path.cwd() / "perspective_images").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"검색 루트: {root}")
    models = find_models(root, args.model, resolution)
    if not models:
        print(
            f"오류: {root} 안에서 처리할 모델을 찾지 못했습니다. "
            f"'tiles/' 또는 'models/*/assets/pan/{RESOLUTION_SKYBOX_DIR[resolution]}/' 가 필요합니다.",
            file=sys.stderr,
        )
        return 1

    print(f"모델 {len(models)}개 처리 시작 (해상도: {resolution})")
    stats = StitchStats()
    last_out_dir = output_root
    for model in models:
        print(f"\n[{model.name}]  형식: {model.fmt}")
        last_out_dir = process_model(model, output_root, resolution, stats)
        stats.models += 1

    print("\n" + "=" * 60)
    print("완료 요약")
    print("=" * 60)
    print(f"  처리한 모델 수      : {stats.models}")
    print(f"  처리한 스캔 위치 수 : {stats.sweeps}")
    print(f"  생성된 이미지 수    : {stats.images}")
    if stats.skipped_existing:
        print(f"  이미 존재해서 스킵  : {stats.skipped_existing}")
    if stats.skipped_missing:
        print(f"  누락으로 스킵       : {stats.skipped_missing}")
    print(f"  사용된 입력 형식    : {', '.join(sorted(stats.formats_used))}")
    print(f"  출력 폴더           : {output_root}")
    print()
    print("RealityScan 권장 카메라 설정:")
    print(f"  - 이미지 크기 : {side_px} × {side_px} px")
    print(f"  - 시야각(FOV) : 90°")
    print(f"  - 초점거리    : {side_px // 2} px (90° FOV 가정)")
    print()
    print("다음 단계: RealityScan 임포트")
    print(f"  1) {last_out_dir if stats.models == 1 else output_root} 안의 이미지를 RealityScan에 추가")
    print("  2) 같은 폴더의 realityscan_hint.txt 참고하여 prior calibration 설정")
    print("  3) Align Images 실행")
    return 0


if __name__ == "__main__":
    sys.exit(main())
