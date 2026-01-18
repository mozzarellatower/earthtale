#!/usr/bin/env python3
"""
EarthTale - Generate a Hytale world from NASA elevation data.

Usage:
    python3 main.py                              # Interactive mode
    python3 main.py grand_canyon                 # Use a preset
    python3 main.py grand_canyon --small         # Small 1x1 chunk test
    python3 main.py --help                       # Show help

Options:
    --small                     Generate a small 1x1 chunk world for testing
    --scale <meters>            Meters per block (default: 5000)
    --exaggeration <n>          Vertical exaggeration factor (default: 1.0)
    --name <name>               World name
    --seed <n>                  World seed
    --skip-download             Skip SRTM download
    --no-blue-marble            Disable Blue Marble imagery for biomes
    --blue-marble-resolution    Blue Marble resolution (world_8km/world_4km/world_2km)
    --ore-config <path>         Path to ore config JSON
    --parallel                 Generate chunks in parallel
    --workers <n>              Parallel worker count (defaults to all cores)
    --resume                   Resume generation if output exists
"""

import sys
from pathlib import Path
from typing import Optional

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
    )
    pipeline = ConversionPipeline(config)
    return pipeline.run()


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
    args = sys.argv[1:]

    # Check for help
    if "--help" in args or "-h" in args:
        print(__doc__)
        print("Presets:", ", ".join(PRESET_LOCATIONS.keys()))
        return

    # Check for --small flag
    small_mode = "--small" in args
    if small_mode:
        args = [a for a in args if a != "--small"]

    # Check for preset argument
    if args:
        preset_key = args[0].lower().replace(" ", "_").replace("-", "_")
        if preset_key in PRESET_LOCATIONS:
            bounds = PRESET_LOCATIONS[preset_key]
            name = preset_key.replace("_", " ").title()
            scale = 5000.0
            exaggeration = 1.0
            seed = None
            skip_download = False
            use_blue_marble = True
            blue_marble_resolution = "world_8km"
            ore_config = None
            parallel = False
            workers = None
            resume = False

            # Parse optional args
            i = 1
            while i < len(args):
                if args[i] == "--scale" and i + 1 < len(args):
                    scale = float(args[i + 1])
                    i += 2
                elif args[i] == "--exaggeration" and i + 1 < len(args):
                    exaggeration = float(args[i + 1])
                    i += 2
                elif args[i] == "--name" and i + 1 < len(args):
                    name = args[i + 1]
                    i += 2
                elif args[i] == "--seed" and i + 1 < len(args):
                    seed = int(args[i + 1])
                    i += 2
                elif args[i] == "--skip-download":
                    skip_download = True
                    i += 1
                elif args[i] == "--no-blue-marble":
                    use_blue_marble = False
                    i += 1
                elif args[i] == "--blue-marble-resolution" and i + 1 < len(args):
                    blue_marble_resolution = args[i + 1]
                    i += 2
                elif args[i] == "--ore-config" and i + 1 < len(args):
                    ore_config = args[i + 1]
                    i += 2
                elif args[i] == "--parallel":
                    parallel = True
                    i += 1
                elif args[i] == "--workers" and i + 1 < len(args):
                    workers = int(args[i + 1])
                    i += 2
                elif args[i] == "--resume":
                    resume = True
                    i += 1
                else:
                    i += 1
        else:
            print(f"Unknown preset: {args[0]}")
            print("Available presets:", ", ".join(PRESET_LOCATIONS.keys()))
            return
    else:
        # Interactive mode
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
    )


if __name__ == "__main__":
    main()
