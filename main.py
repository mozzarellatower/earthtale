#!/usr/bin/env python3
"""
EarthTale - Generate a Hytale world from NASA elevation data.

Usage:
    python3 main.py                              # Interactive mode
    python3 main.py grand_canyon                 # Use a preset
    python3 main.py grand_canyon --small         # Small 1x1 chunk test
    python3 main.py --help                       # Show help

Examples:
    python3 main.py --name "Earth" --min-lat -90 --max-lat 90 --min-lon -180 --max-lon 180 --scale 5000
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import os

# Add src to path so we can import without installing
sys.path.insert(0, str(Path(__file__).parent / "src"))


# Preset locations
PRESET_LOCATIONS = {
    "grand_canyon": {"min_lat": 35.9, "max_lat": 36.3, "min_lon": -112.3, "max_lon": -111.8},
    "mount_everest": {"min_lat": 27.85, "max_lat": 28.05, "min_lon": 86.85, "max_lon": 87.05},
    "alps_matterhorn": {"min_lat": 45.9, "max_lat": 46.1, "min_lon": 7.6, "max_lon": 7.8},
    "death_valley": {"min_lat": 36.1, "max_lat": 36.5, "min_lon": -117.2, "max_lon": -116.7},
    "hawaii_mauna_kea": {"min_lat": 19.75, "max_lat": 19.95, "min_lon": -155.55, "max_lon": -155.35},
    "iceland": {"min_lat": 63.55, "max_lat": 63.7, "min_lon": -19.7, "max_lon": -19.4},
}


def run_conversion(
    name: str,
    bounds: dict,
    scale: float = 5000.0,
    exaggeration: float = 1.0,
    output_dir: str = "output",
    max_chunks: int = 0,
    seed: Optional[int] = None,
    skip_download: bool = False,
    use_blue_marble: bool = True,
    blue_marble_resolution: str = "world_8km",
    ore_config: Optional[str] = None,
    parallel: bool = False,
    workers: Optional[int] = None,
    resume: bool = False,
    nasa_username: Optional[str] = None,
    nasa_password: Optional[str] = None,
    download_cache_mb: Optional[int] = None,
    autosave_every: int = 25,
):
    """Generate a Hytale world using SRTM + Blue Marble data."""
    from earthtale.config import ConversionConfig, BoundingBox
    from earthtale.converter import ConversionPipeline

    bbox = BoundingBox(
        min_lat=bounds["min_lat"],
        max_lat=bounds["max_lat"],
        min_lon=bounds["min_lon"],
        max_lon=bounds["max_lon"],
    )
    config = ConversionConfig(
        name=name,
        bounds=bbox,
        scale=scale,
        output_dir=Path(output_dir),
        vertical_exaggeration=exaggeration,
        seed=seed,
        skip_download=skip_download,
        use_blue_marble=use_blue_marble,
        blue_marble_resolution=blue_marble_resolution,
        ore_config_path=Path(ore_config) if ore_config else None,
        max_chunks=max_chunks or None,
        parallel=parallel,
        parallel_workers=workers,
        resume=resume,
        nasa_username=nasa_username,
        nasa_password=nasa_password,
        download_cache_mb=download_cache_mb,
        autosave_every=autosave_every,
    )
    pipeline = ConversionPipeline(config)
    def progress_callback(progress):
        if progress.phase == "generate":
            if progress.current == 1 or progress.current % 10 == 0 or progress.current == progress.total:
                print(f"[generate] {progress.current}/{progress.total} {progress.message}", flush=True)
        elif progress.message:
            print(f"[{progress.phase}] {progress.message}", flush=True)
        else:
            print(f"[{progress.phase}] {progress.current}/{progress.total}", flush=True)

    return pipeline.run(progress_callback)


def interactive_mode():
    """Run in interactive mode, prompting for input."""
    print("=" * 60)
    print("  EarthTale - NASA Earth Map to Hytale World Converter")
    print("=" * 60)
    print()

    # Show presets
    print("Available presets:")
    preset_names = list(PRESET_LOCATIONS.keys())
    for i, name in enumerate(preset_names, 1):
        print(f"  {i}. {name}")
    print(f"  {len(preset_names) + 1}. Custom coordinates")
    print()

    # Get choice
    choice = input("Select option (1-7) or preset name: ").strip()

    # Parse choice
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(preset_names):
            preset_name = preset_names[idx]
            bounds = PRESET_LOCATIONS[preset_name]
        elif idx == len(preset_names):
            # Custom coordinates
            print()
            bounds = {
                "min_lat": float(input("Minimum latitude: ")),
                "max_lat": float(input("Maximum latitude: ")),
                "min_lon": float(input("Minimum longitude: ")),
                "max_lon": float(input("Maximum longitude: ")),
            }
            preset_name = "custom"
        else:
            print("Invalid choice")
            return None
    elif choice.lower().replace(" ", "_") in PRESET_LOCATIONS:
        preset_name = choice.lower().replace(" ", "_")
        bounds = PRESET_LOCATIONS[preset_name]
    else:
        print(f"Unknown preset: {choice}")
        return None

    # Get world name
    print()
    default_name = preset_name.replace("_", " ").title()
    name = input(f"World name [{default_name}]: ").strip() or default_name

    # Get scale
    scale_str = input("Scale (meters per block) [5000]: ").strip()
    scale = float(scale_str) if scale_str else 5000.0

    # Get exaggeration
    exag_str = input("Vertical exaggeration [1.0]: ").strip()
    exaggeration = float(exag_str) if exag_str else 1.0

    return name, bounds, scale, exaggeration


def main():
    """Main entry point."""
    if len(sys.argv) == 1:
        result = interactive_mode()
        if result is None:
            return
        name, bounds, scale, exaggeration = result
        seed = None
        skip_download = False
        use_blue_marble = True
        blue_marble_resolution = "world_8km"
        ore_config = None
        parallel = False
        workers = None
        resume = False
        small_mode = False
    else:
        parser = argparse.ArgumentParser(
            description="Generate a Hytale world from NASA SRTM + Blue Marble data.",
        )
        parser.add_argument("preset", nargs="?", help="Preset name (optional)")
        parser.add_argument("--min-lat", type=float, help="Minimum latitude")
        parser.add_argument("--max-lat", type=float, help="Maximum latitude")
        parser.add_argument("--min-lon", type=float, help="Minimum longitude")
        parser.add_argument("--max-lon", type=float, help="Maximum longitude")
        parser.add_argument("--scale", type=float, default=5000.0, help="Meters per block")
        parser.add_argument("--exaggeration", type=float, default=1.0, help="Vertical exaggeration")
        parser.add_argument("--name", help="World name")
        parser.add_argument("--seed", type=int, help="World seed")
        parser.add_argument("--skip-download", action="store_true", help="Skip SRTM download")
        parser.add_argument("--no-blue-marble", dest="use_blue_marble", action="store_false")
        parser.set_defaults(use_blue_marble=True)
        parser.add_argument(
            "--blue-marble-resolution",
            default="world_8km",
            help="Blue Marble resolution (world_8km/world_4km/world_2km)",
        )
        parser.add_argument("--ore-config", help="Path to ore config JSON")
        parser.add_argument("--parallel", action="store_true", help="Generate chunks in parallel")
        parser.add_argument(
            "--workers",
            type=int,
            help="Parallel worker count (defaults to all cores)",
        )
        parser.add_argument("--resume", action="store_true", help="Resume generation if output exists")
        parser.add_argument("--small", action="store_true", help="Generate a 1x1 chunk test world")
        parser.add_argument("--nasa-user", help="NASA Earthdata username")
        parser.add_argument("--nasa-pass", help="NASA Earthdata password")
        parser.add_argument("--download-cache-mb", type=int, help="Max SRTM cache size before pausing downloads")
        parser.add_argument("--autosave-every", type=int, default=25, help="Autosave every N chunks (0 disables)")
        args = parser.parse_args()

        bounds = None
        preset_key = None
        if args.preset:
            preset_key = args.preset.lower().replace(" ", "_").replace("-", "_")
            if preset_key not in PRESET_LOCATIONS:
                print(f"Unknown preset: {args.preset}")
                print("Available presets:", ", ".join(PRESET_LOCATIONS.keys()))
                print("Or pass --min-lat/--max-lat/--min-lon/--max-lon for custom bounds.")
                return
            bounds = PRESET_LOCATIONS[preset_key]
        else:
            if all(
                value is not None
                for value in (args.min_lat, args.max_lat, args.min_lon, args.max_lon)
            ):
                bounds = {
                    "min_lat": args.min_lat,
                    "max_lat": args.max_lat,
                    "min_lon": args.min_lon,
                    "max_lon": args.max_lon,
                }
            else:
                print("Available presets:", ", ".join(PRESET_LOCATIONS.keys()))
                print("Or pass --min-lat/--max-lat/--min-lon/--max-lon for custom bounds.")
                return

        name = args.name or (preset_key.replace("_", " ").title() if preset_key else "Custom World")
        scale = args.scale
        exaggeration = args.exaggeration
        seed = args.seed
        skip_download = args.skip_download
        use_blue_marble = args.use_blue_marble
        blue_marble_resolution = args.blue_marble_resolution
        ore_config = args.ore_config
        parallel = args.parallel
        workers = args.workers
        resume = args.resume
        small_mode = args.small
        nasa_username = args.nasa_user or os.getenv("EARTHDATA_USER") or os.getenv("NASA_USERNAME")
        nasa_password = args.nasa_pass or os.getenv("EARTHDATA_PASS") or os.getenv("NASA_PASSWORD")
        download_cache_mb = args.download_cache_mb
        autosave_every = args.autosave_every

    print()
    max_chunks = 1 if small_mode else 0
    run_conversion(
        name,
        bounds,
        scale,
        exaggeration,
        max_chunks=max_chunks,
        seed=seed,
        skip_download=skip_download,
        use_blue_marble=use_blue_marble,
        blue_marble_resolution=blue_marble_resolution,
        ore_config=ore_config,
        parallel=parallel,
        workers=workers,
        resume=resume,
        nasa_username=nasa_username,
        nasa_password=nasa_password,
        download_cache_mb=download_cache_mb,
        autosave_every=autosave_every,
    )


if __name__ == "__main__":
    main()
