#!/usr/bin/env python3
"""
EarthTale - Generate a Hytale world from NASA elevation data.

Usage:
    python3 main.py                              # Interactive mode
    python3 main.py grand_canyon                 # Use a preset
    python3 main.py grand_canyon --small         # Small 4x4 chunk test
    python3 main.py --help                       # Show help

Options:
    --small              Generate a small 4x4 chunk world for testing
    --scale <meters>     Meters per block (default: 30)
    --exaggeration <n>   Vertical exaggeration factor (default: 1.0)
    --name <name>        World name
"""

import sys
import os
import math
import random
from pathlib import Path

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


def generate_world(name: str, bounds: dict, scale: float = 30.0,
                   exaggeration: float = 1.0, output_dir: str = "output",
                   max_chunks: int = 0):
    """Generate a Hytale world from the given parameters.

    Args:
        max_chunks: If > 0, limit world to max_chunks x max_chunks (for testing)
    """

    # Import Hytale modules (these have no external dependencies)
    from earthtale.hytale.world import World, WorldConfig
    from earthtale.hytale.chunk import Chunk
    from earthtale.hytale.constants import CHUNK_SIZE, WORLD_HEIGHT
    from earthtale.terrain.elevation import ElevationMapper, ElevationConfig
    from earthtale.terrain.biome import BiomeClassifier, BiomeType
    from earthtale.terrain.blocks import BlockColumnGenerator

    print(f"Generating world: {name}")
    print(f"  Bounds: ({bounds['min_lat']:.2f}, {bounds['min_lon']:.2f}) to ({bounds['max_lat']:.2f}, {bounds['max_lon']:.2f})")
    print(f"  Scale: {scale} m/block")

    # Calculate world size
    EARTH_RADIUS = 6_371_000
    lat_meters_per_degree = 2 * math.pi * EARTH_RADIUS / 360
    center_lat = (bounds['min_lat'] + bounds['max_lat']) / 2
    lon_meters_per_degree = lat_meters_per_degree * math.cos(math.radians(center_lat))

    width_degrees = bounds['max_lon'] - bounds['min_lon']
    height_degrees = bounds['max_lat'] - bounds['min_lat']

    width_meters = width_degrees * lon_meters_per_degree
    height_meters = height_degrees * lat_meters_per_degree

    width_blocks = int(width_meters / scale)
    height_blocks = int(height_meters / scale)

    width_chunks = max(1, (width_blocks + CHUNK_SIZE - 1) // CHUNK_SIZE)
    height_chunks = max(1, (height_blocks + CHUNK_SIZE - 1) // CHUNK_SIZE)

    # Limit size if requested
    if max_chunks > 0:
        width_chunks = min(width_chunks, max_chunks)
        height_chunks = min(height_chunks, max_chunks)
        print(f"  World size: {width_chunks}x{height_chunks} chunks (limited for testing)")
    else:
        print(f"  World size: {width_chunks}x{height_chunks} chunks ({width_blocks}x{height_blocks} blocks)")

    # Initialize mappers
    elevation_config = ElevationConfig(
        min_elevation=-500.0,
        max_elevation=8849.0,
        sea_level_y=64,
        vertical_exaggeration=exaggeration,
    )
    elevation_mapper = ElevationMapper(elevation_config)
    biome_classifier = BiomeClassifier()
    block_generator = BlockColumnGenerator()

    # Create world with void generator (no terrain outside our chunks)
    from earthtale.hytale.world import WorldGenConfig
    world_gen = WorldGenConfig(type="Void", name="Void")
    world = World(name, WorldConfig(
        seed=random.randint(0, 2**32),
        world_gen=world_gen,
        save_new_chunks=False,  # Don't save empty chunks
    ))

    # Generate chunks
    total_chunks = width_chunks * height_chunks
    chunk_count = 0

    print(f"\nGenerating {total_chunks} chunks...")

    for chunk_z in range(height_chunks):
        for chunk_x in range(width_chunks):
            chunk = Chunk(chunk_x, chunk_z)

            # Generate terrain for this chunk
            for local_x in range(CHUNK_SIZE):
                for local_z in range(CHUNK_SIZE):
                    block_x = chunk_x * CHUNK_SIZE + local_x
                    block_z = chunk_z * CHUNK_SIZE + local_z

                    # Calculate lat/lon for this block
                    lat_offset = (block_z * scale) / lat_meters_per_degree
                    lon_offset = (block_x * scale) / lon_meters_per_degree
                    lat = bounds['min_lat'] + lat_offset
                    lon = bounds['min_lon'] + lon_offset

                    # Generate procedural elevation (sine waves for demo)
                    # In production, this would read from SRTM data
                    base_elevation = 1000  # Base elevation in meters

                    # Add some terrain variation
                    freq1, freq2, freq3 = 0.01, 0.03, 0.07
                    elevation = base_elevation
                    elevation += 500 * math.sin(block_x * freq1) * math.cos(block_z * freq1)
                    elevation += 200 * math.sin(block_x * freq2 + 1.5) * math.cos(block_z * freq2 + 0.7)
                    elevation += 100 * math.sin(block_x * freq3 + 3.0) * math.cos(block_z * freq3 + 2.1)

                    # Add some random variation
                    elevation += random.uniform(-50, 50)

                    # Map to Y coordinate
                    surface_y = elevation_mapper.get_y(elevation * exaggeration)
                    surface_y = max(1, min(WORLD_HEIGHT - 1, surface_y))

                    # Get biome
                    latitude_factor = biome_classifier.get_latitude_factor(lat)
                    biome = biome_classifier.classify_from_elevation_only(surface_y, latitude_factor)

                    # Generate block column
                    column = block_generator.generate_column(
                        x=block_x,
                        z=block_z,
                        surface_y=surface_y,
                        biome=biome,
                        seed=block_x * 31 + block_z
                    )

                    # Set blocks in chunk
                    for y, block_id in enumerate(column.blocks):
                        if y < WORLD_HEIGHT and block_id != 0:
                            chunk.set_block(local_x, y, local_z, block_id)

            world.add_chunk(chunk)
            chunk_count += 1

            # Progress
            pct = 100 * chunk_count // total_chunks
            print(f"\r  Progress: {pct}% ({chunk_count}/{total_chunks} chunks)", end="", flush=True)

    print()

    # Save world
    print(f"\nSaving world...")
    output_path = Path(output_dir)
    world_path = world.save(output_path)

    print(f"\nWorld saved to: {world_path}")
    return world_path


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
    scale_str = input("Scale (meters per block) [30]: ").strip()
    scale = float(scale_str) if scale_str else 30.0

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
            scale = 30.0
            exaggeration = 1.0

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

    print()
    max_chunks = 4 if small_mode else 0
    generate_world(name, bounds, scale, exaggeration, max_chunks=max_chunks)


if __name__ == "__main__":
    main()
