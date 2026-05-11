#!/usr/bin/env python3
"""End-to-end Matterport to Gaussian Splat dataset pipeline.

Takes a Matterport URL (or model ID) and produces a Brush / Nerfstudio
compatible zip ready to drop into Brush for training. Each stage is resumable:
re-running with the same arguments skips work that's already on disk.

Stages:
  1. matterport-dl  - downloads tour assets to ./downloads/<MODEL>/
  2. stitch         - stitches cube-face tile mosaics into 6 JPGs per sweep
  3. brush dataset  - renders perspective views with direct cube ray-cast
                      (cross-face blended) + mesh-sampled initial points + zip

Usage:
  python pipeline.py <MATTERPORT_URL_OR_MODEL_ID>

  Examples:
    python pipeline.py https://my.matterport.com/show/?m=g7EGi53S2b8
    python pipeline.py g7EGi53S2b8
    python pipeline.py g7EGi53S2b8 --elevations 0          # only horizontal
    python pipeline.py g7EGi53S2b8 --resume                # skip done stages
    python pipeline.py g7EGi53S2b8 --skip-download         # already downloaded
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
PYTHON = sys.executable
MATTERPORT_ID_RE = re.compile(r"[A-Za-z0-9]{11}")


def extract_model_id(url_or_id: str) -> str:
    """Pull the 11-char model ID out of a Matterport URL or accept a bare ID."""
    if "matterport.com" in url_or_id:
        m = re.search(r"[?&]m=([A-Za-z0-9]+)", url_or_id)
        if not m:
            raise SystemExit(f"Could not find ?m=<id> in URL: {url_or_id}")
        return m.group(1)
    if MATTERPORT_ID_RE.fullmatch(url_or_id):
        return url_or_id
    raise SystemExit(
        f"Not a Matterport URL or 11-char model ID: {url_or_id!r}"
    )


def banner(text: str) -> None:
    print()
    print("=" * 70)
    print(f"  {text}")
    print("=" * 70)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=cwd or ROOT)
    if result.returncode != 0:
        raise SystemExit(f"Step failed (exit {result.returncode}): {cmd[0]}")
    print(f"\n[ok] {(time.time() - t0):.1f}s")


def stage_download(model_id: str, args: argparse.Namespace) -> Path:
    """Run matterport-dl via run.py. Skip if model dir already populated."""
    model_dir = ROOT / "downloads" / model_id
    poses_json = model_dir / "api" / "mp" / "models" / "graph_GetShowcaseSweeps.json"
    if args.skip_download or (args.resume and poses_json.is_file()):
        print(f"[skip download] {model_dir} already has sweep poses")
        return model_dir
    banner(f"1/3  Downloading Matterport model {model_id}")
    run([PYTHON, str(ROOT / "run.py"), args.url_or_id])
    if not poses_json.is_file():
        raise SystemExit(
            f"Download finished but {poses_json} is missing - model may have "
            "failed to download fully."
        )
    return model_dir


def stage_stitch(model_id: str, args: argparse.Namespace) -> Path:
    """Stitch cube-face tile mosaics into 6 face JPGs per sweep."""
    image_dir = ROOT / f"perspective_images_{args.resolution}" / model_id
    # Heuristic: directory exists with at least 6 face JPGs to already stitched
    if args.resume and image_dir.is_dir() and len(list(image_dir.glob("*_face*.jpg"))) >= 6:
        print(f"[skip stitch] {image_dir} already has stitched faces")
        return image_dir
    banner(f"2/3  Stitching cube faces at {args.resolution}")
    run([
        PYTHON, str(ROOT / "stitch_matterport.py"),
        "--model", model_id,
        "--resolution", args.resolution,
    ])
    if not image_dir.is_dir():
        raise SystemExit(f"Stitch finished but {image_dir} missing")
    return image_dir


def stage_dataset(model_id: str, model_dir: Path, image_dir: Path,
                  args: argparse.Namespace) -> Path:
    """Build Brush dataset (perspective views + mesh ply + transforms.json + zip)."""
    output_dir = ROOT / f"brush_dense_input_{model_id}"
    zip_path = output_dir.parent / f"{output_dir.name}_brush.zip"
    if args.resume and zip_path.is_file() and not args.force_dataset:
        print(f"[skip dataset] {zip_path} already exists ({zip_path.stat().st_size / 1024 / 1024:.0f} MB)")
        return zip_path
    banner("3/3  Building Brush dataset (cube ray-cast + blending + mesh init)")
    cmd = [
        PYTHON, str(ROOT / "brush_dense_input.py"),
        "--model-dir", str(model_dir),
        "--image-dir", str(image_dir),
        "--output-dir", str(output_dir),
        "--elevations", args.elevations,
        "--azimuth-step", str(args.azimuth_step),
        "--num-points", str(args.num_points),
        "--fov", str(args.fov),
        "--resolution", str(args.view_resolution),
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    run(cmd)
    if not zip_path.is_file():
        raise SystemExit(f"Dataset finished but {zip_path} missing")
    return zip_path


def cleanup_intermediates(model_id: str) -> None:
    """Remove large intermediate dirs once the zip is built."""
    targets = [
        ROOT / "perspective_images_2k" / model_id,
        ROOT / "perspective_images_4k" / model_id,
        ROOT / f"brush_dense_input_{model_id}" / "images",
    ]
    for p in targets:
        if p.exists():
            size_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 / 1024
            shutil.rmtree(p)
            print(f"  removed {p} ({size_mb:.0f} MB)")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Matterport URL to Brush-ready Gaussian Splat dataset zip.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("url_or_id", help="Matterport URL (https://my.matterport.com/show/?m=ID) "
                                     "or just the 11-char model ID")
    p.add_argument("--resolution", default="2k", choices=["1k", "2k", "4k", "512"],
                   help="Cube-face stitch resolution (default: 2k)")
    p.add_argument("--elevations", default="-30,0,30",
                   help="Comma-separated elevations in degrees (default: -30,0,30)")
    p.add_argument("--azimuth-step", type=int, default=30,
                   help="Azimuth step in degrees (default: 30 to 12 views per elevation)")
    p.add_argument("--num-points", type=int, default=2_000_000,
                   help="Mesh-sampled initial point count (default: 2M)")
    p.add_argument("--fov", type=float, default=60.0,
                   help="Perspective view FOV in degrees (default: 60)")
    p.add_argument("--view-resolution", type=int, default=1024,
                   help="Perspective view output side length (default: 1024)")
    p.add_argument("--skip-download", action="store_true",
                   help="Assume model already downloaded; skip run.py")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip re-rendering perspectives that already exist (default: on)")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                   help="Force re-render all perspectives even if files exist")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Skip stages whose outputs already exist (default: on)")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="Force run every stage even if outputs exist")
    p.add_argument("--force-dataset", action="store_true",
                   help="Re-run dataset stage even if zip exists")
    p.add_argument("--cleanup", action="store_true",
                   help="Remove perspective_images_* and images/ folder after zip built")
    args = p.parse_args()

    model_id = extract_model_id(args.url_or_id)
    print(f"Model ID: {model_id}")
    print(f"Resolution: {args.resolution}  Elevations: {args.elevations}  "
          f"Azimuth step: {args.azimuth_step} deg  Points: {args.num_points:,}")

    t0 = time.time()
    model_dir = stage_download(model_id, args)
    image_dir = stage_stitch(model_id, args)
    zip_path = stage_dataset(model_id, model_dir, image_dir, args)
    elapsed = time.time() - t0

    if args.cleanup:
        banner("Cleanup intermediates")
        cleanup_intermediates(model_id)

    banner("Done")
    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"\n  Brush zip:  {zip_path}")
    print(f"  Size:       {size_mb:.0f} MB")
    print(f"  Total time: {elapsed / 60:.1f} min")
    print()
    print("  Next steps:")
    print("    1. Open Brush, drag the zip in, train")
    print("    2. Export trained scene as .ply")
    print("    3. Upload .ply to SuperSplat for cleanup / hosting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
