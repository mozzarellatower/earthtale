"""Command-line interface for EarthTale."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from . import __version__
from .config import ConversionConfig, BoundingBox, get_preset, list_presets
from .converter import ConversionPipeline, ConversionProgress

app = typer.Typer(
    name="earthtale",
    help="Convert NASA elevation data into Hytale world files.",
    no_args_is_help=True,
)
console = Console()


def version_callback(value: bool):
    if value:
        console.print(f"EarthTale version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
):
    """EarthTale: NASA Earth Map to Hytale World Converter."""
    pass


@app.command()
def convert(
    name: str = typer.Argument(..., help="World name"),
    min_lat: float = typer.Option(..., "--min-lat", help="Minimum latitude"),
    max_lat: float = typer.Option(..., "--max-lat", help="Maximum latitude"),
    min_lon: float = typer.Option(..., "--min-lon", help="Minimum longitude"),
    max_lon: float = typer.Option(..., "--max-lon", help="Maximum longitude"),
    scale: float = typer.Option(5000.0, "--scale", "-s", help="Meters per block"),
    output: Path = typer.Option(Path("output"), "--output", "-o", help="Output directory"),
    cache: Path = typer.Option(Path("cache"), "--cache", "-c", help="Cache directory"),
    exaggeration: float = typer.Option(1.0, "--exaggeration", "-e", help="Vertical exaggeration"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed"),
    skip_download: bool = typer.Option(False, "--skip-download", help="Skip SRTM download"),
    use_blue_marble: bool = typer.Option(True, "--blue-marble/--no-blue-marble", help="Use Blue Marble imagery for biomes"),
    blue_marble_resolution: str = typer.Option("world_8km", "--blue-marble-resolution", help="Blue Marble resolution"),
    ore_config: Optional[Path] = typer.Option(None, "--ore-config", help="Path to ore config JSON"),
    parallel: bool = typer.Option(False, "--parallel/--no-parallel", help="Generate chunks in parallel"),
    workers: Optional[int] = typer.Option(None, "--workers", help="Parallel worker count (defaults to all cores)"),
    resume: bool = typer.Option(False, "--resume/--no-resume", help="Resume generation if output exists"),
):
    """Convert a geographic region to a Hytale world.

    Example:
        earthtale convert "GrandCanyon" --min-lat 35.9 --max-lat 36.3 --min-lon -112.3 --max-lon -111.8
    """
    bounds = BoundingBox(
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
    )

    try:
        bounds.validate()
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    config = ConversionConfig(
        name=name,
        bounds=bounds,
        scale=scale,
        output_dir=output,
        cache_dir=cache,
        vertical_exaggeration=exaggeration,
        seed=seed,
        skip_download=skip_download,
        use_blue_marble=use_blue_marble,
        blue_marble_resolution=blue_marble_resolution,
        ore_config_path=ore_config,
        parallel=parallel,
        parallel_workers=workers,
        resume=resume,
    )

    console.print(f"[bold]Converting region to Hytale world: {name}[/bold]")
    console.print(f"  Bounds: ({min_lat}, {min_lon}) to ({max_lat}, {max_lon})")
    console.print(f"  Scale: {scale} meters/block")
    console.print(f"  Output: {output}")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        current_task = None

        def progress_callback(p: ConversionProgress):
            nonlocal current_task
            if current_task is None or progress.tasks[current_task].description != p.phase:
                if current_task is not None:
                    progress.update(current_task, completed=progress.tasks[current_task].total)
                current_task = progress.add_task(f"[cyan]{p.phase}[/cyan]: {p.message}", total=p.total)
            progress.update(current_task, completed=p.current, description=f"[cyan]{p.phase}[/cyan]: {p.message}")

        pipeline = ConversionPipeline(config)
        world_path = pipeline.run(progress_callback)

    console.print()
    console.print(f"[green]Success![/green] World saved to: {world_path}")


@app.command()
def preset(
    name: str = typer.Argument(..., help="Preset name or world name"),
    preset_name: Optional[str] = typer.Option(None, "--preset", "-p", help="Preset location name"),
    scale: float = typer.Option(5000.0, "--scale", "-s", help="Meters per block"),
    output: Path = typer.Option(Path("output"), "--output", "-o", help="Output directory"),
    cache: Path = typer.Option(Path("cache"), "--cache", "-c", help="Cache directory"),
    exaggeration: float = typer.Option(1.0, "--exaggeration", "-e", help="Vertical exaggeration"),
    use_blue_marble: bool = typer.Option(True, "--blue-marble/--no-blue-marble", help="Use Blue Marble imagery for biomes"),
    blue_marble_resolution: str = typer.Option("world_8km", "--blue-marble-resolution", help="Blue Marble resolution"),
    ore_config: Optional[Path] = typer.Option(None, "--ore-config", help="Path to ore config JSON"),
    parallel: bool = typer.Option(False, "--parallel/--no-parallel", help="Generate chunks in parallel"),
    workers: Optional[int] = typer.Option(None, "--workers", help="Parallel worker count (defaults to all cores)"),
    resume: bool = typer.Option(False, "--resume/--no-resume", help="Resume generation if output exists"),
):
    """Convert a preset location to a Hytale world.

    Example:
        earthtale preset "MyGrandCanyon" --preset grand_canyon
    """
    # If no preset specified, try to use name as preset
    actual_preset = preset_name or name

    bounds = get_preset(actual_preset)
    if bounds is None:
        console.print(f"[red]Error:[/red] Unknown preset: {actual_preset}")
        console.print("Available presets:")
        for p in list_presets():
            console.print(f"  - {p}")
        raise typer.Exit(1)

    config = ConversionConfig(
        name=name,
        bounds=bounds,
        scale=scale,
        output_dir=output,
        cache_dir=cache,
        vertical_exaggeration=exaggeration,
        use_blue_marble=use_blue_marble,
        blue_marble_resolution=blue_marble_resolution,
        ore_config_path=ore_config,
        parallel=parallel,
        parallel_workers=workers,
        resume=resume,
    )

    console.print(f"[bold]Converting preset '{actual_preset}' to Hytale world: {name}[/bold]")
    console.print(f"  Bounds: ({bounds.min_lat}, {bounds.min_lon}) to ({bounds.max_lat}, {bounds.max_lon})")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        current_task = None

        def progress_callback(p: ConversionProgress):
            nonlocal current_task
            if current_task is None or progress.tasks[current_task].description != p.phase:
                if current_task is not None:
                    progress.update(current_task, completed=progress.tasks[current_task].total)
                current_task = progress.add_task(f"[cyan]{p.phase}[/cyan]: {p.message}", total=p.total)
            progress.update(current_task, completed=p.current, description=f"[cyan]{p.phase}[/cyan]: {p.message}")

        pipeline = ConversionPipeline(config)
        world_path = pipeline.run(progress_callback)

    console.print()
    console.print(f"[green]Success![/green] World saved to: {world_path}")


@app.command("list-presets")
def list_presets_cmd():
    """List available preset locations."""
    table = Table(title="Available Presets")
    table.add_column("Name", style="cyan")
    table.add_column("Location")
    table.add_column("Bounds")

    presets = {
        "grand_canyon": "Grand Canyon, Arizona",
        "mount_everest": "Mount Everest, Nepal/Tibet",
        "alps_matterhorn": "Matterhorn, Swiss Alps",
        "death_valley": "Death Valley, California",
        "hawaii_mauna_kea": "Mauna Kea, Hawaii",
        "iceland_eyjafjallajokull": "Eyjafjallajokull, Iceland",
    }

    for name in list_presets():
        bounds = get_preset(name)
        location = presets.get(name, "")
        bounds_str = f"({bounds.min_lat:.2f}, {bounds.min_lon:.2f}) to ({bounds.max_lat:.2f}, {bounds.max_lon:.2f})"
        table.add_row(name, location, bounds_str)

    console.print(table)


@app.command()
def info(
    min_lat: float = typer.Option(..., "--min-lat", help="Minimum latitude"),
    max_lat: float = typer.Option(..., "--max-lat", help="Maximum latitude"),
    min_lon: float = typer.Option(..., "--min-lon", help="Minimum longitude"),
    max_lon: float = typer.Option(..., "--max-lon", help="Maximum longitude"),
    scale: float = typer.Option(5000.0, "--scale", "-s", help="Meters per block"),
):
    """Show information about a region without converting.

    Example:
        earthtale info --min-lat 35.9 --max-lat 36.3 --min-lon -112.3 --max-lon -111.8
    """
    import math

    bounds = BoundingBox(min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon)

    # Calculate sizes
    EARTH_RADIUS = 6_371_000
    lat_meters_per_degree = 2 * math.pi * EARTH_RADIUS / 360
    lon_meters_per_degree = lat_meters_per_degree * math.cos(math.radians(bounds.center_lat))

    width_km = bounds.width_degrees * lon_meters_per_degree / 1000
    height_km = bounds.height_degrees * lat_meters_per_degree / 1000

    width_blocks = int(width_km * 1000 / scale)
    height_blocks = int(height_km * 1000 / scale)

    width_chunks = (width_blocks + 31) // 32
    height_chunks = (height_blocks + 31) // 32

    # Get required tiles
    from .nasa import get_required_tiles
    tiles = get_required_tiles(min_lat, max_lat, min_lon, max_lon)

    console.print("[bold]Region Information[/bold]")
    console.print()
    console.print(f"[cyan]Geographic Bounds:[/cyan]")
    console.print(f"  Latitude:  {min_lat:.4f} to {max_lat:.4f}")
    console.print(f"  Longitude: {min_lon:.4f} to {max_lon:.4f}")
    console.print(f"  Center:    ({bounds.center_lat:.4f}, {bounds.center_lon:.4f})")
    console.print()
    console.print(f"[cyan]Real-world Size:[/cyan]")
    console.print(f"  Width:  {width_km:.2f} km")
    console.print(f"  Height: {height_km:.2f} km")
    console.print()
    console.print(f"[cyan]Hytale World Size (at {scale}m/block):[/cyan]")
    console.print(f"  Blocks: {width_blocks} x {height_blocks}")
    console.print(f"  Chunks: {width_chunks} x {height_chunks} ({width_chunks * height_chunks} total)")
    console.print()
    console.print(f"[cyan]Required SRTM Tiles:[/cyan] {len(tiles)}")
    for tile in tiles:
        console.print(f"  - {tile}")


if __name__ == "__main__":
    app()
