"""Main conversion pipeline from NASA data to Hytale worlds."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Callable, Any, Dict, Tuple
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import ConversionConfig, BoundingBox
from ..nasa import (
    SRTMCache,
    SRTMDownloader,
    BlueMarbleDownloader,
    get_required_tiles,
    lat_lon_to_block,
    block_to_lat_lon,
    meters_to_degrees_lat,
    meters_to_degrees_lon,
)
from ..terrain import (
    ElevationMapper,
    ElevationConfig,
    BiomeClassifier,
    BiomeType,
    BlockColumnGenerator,
    ColumnConfig,
    load_ore_configs,
)
from ..hytale import (
    World,
    WorldConfig,
    Chunk,
    CHUNK_SIZE,
    WORLD_HEIGHT,
)
from ..hytale.region import read_region_chunk_indexes, read_region_chunks_raw

try:
    from PIL import Image
except Exception:
    Image = None


@dataclass
class ConversionProgress:
    """Progress information for conversion."""
    phase: str
    current: int
    total: int
    message: str = ""

    @property
    def percent(self) -> float:
        if self.total == 0:
            return 0.0
        return 100.0 * self.current / self.total


ProgressCallback = Callable[[ConversionProgress], None]


class ConversionPipeline:
    """Main pipeline for converting NASA data to Hytale worlds."""

    def __init__(self, config: ConversionConfig):
        """Initialize the pipeline.

        Args:
            config: Conversion configuration
        """
        self.config = config
        config.validate()

        # Set up directories
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._downloader = SRTMDownloader(
            cache_dir=self.config.cache_dir / "srtm",
            username=self.config.nasa_username,
            password=self.config.nasa_password,
        )
        self._srtm_cache = SRTMCache(self.config.cache_dir / "srtm")
        self._blue_marble_downloader = BlueMarbleDownloader(self.config.cache_dir / "blue_marble")
        self._blue_marble_pixels = None
        self._blue_marble_size = None

        # Initialize mappers
        self._elevation_mapper = ElevationMapper(ElevationConfig(
            min_elevation=self.config.min_elevation,
            max_elevation=self.config.max_elevation,
            sea_level_y=self.config.sea_level_y,
            vertical_exaggeration=self.config.vertical_exaggeration,
        ))

        self._biome_classifier = BiomeClassifier()
        ore_configs = load_ore_configs(self.config.ore_config_path)
        self._block_generator = BlockColumnGenerator(
            config=ColumnConfig(ore_configs=ore_configs)
        )

    def _init_blue_marble(self, progress_callback: Optional[ProgressCallback] = None) -> None:
        """Load Blue Marble imagery for biome classification."""
        if not self.config.use_blue_marble or Image is None:
            return
        if progress_callback:
            progress_callback(ConversionProgress("blue_marble", 0, 1, "Loading Blue Marble imagery..."))
        path = None
        if not self.config.skip_download:
            import asyncio
            path = asyncio.run(
                self._blue_marble_downloader.download_image(
                    resolution=self.config.blue_marble_resolution
                )
            )
        if path is None:
            filename = f"blue_marble_{self.config.blue_marble_resolution}.jpg"
            candidate = self._blue_marble_downloader.cache_dir / filename
            if candidate.exists():
                path = candidate
        if path is None:
            return
        image = Image.open(path).convert("RGB")
        self._blue_marble_pixels = image.load()
        self._blue_marble_size = image.size
        if progress_callback:
            progress_callback(ConversionProgress("blue_marble", 1, 1, "Blue Marble imagery ready"))

    def _get_color_at_lat_lon(self, lat: float, lon: float) -> Optional[tuple[int, int, int]]:
        """Sample Blue Marble image for a lat/lon."""
        if self._blue_marble_pixels is None or self._blue_marble_size is None:
            return None
        width, height = self._blue_marble_size
        x = int(((lon + 180.0) / 360.0) * (width - 1))
        y = int(((90.0 - lat) / 180.0) * (height - 1))
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        r, g, b = self._blue_marble_pixels[x, y]
        return (int(r), int(g), int(b))

    def _calculate_world_size(self) -> tuple[int, int, int, int]:
        """Calculate world size in blocks and chunks.

        Returns:
            Tuple of (width_blocks, height_blocks, width_chunks, height_chunks)
        """
        bounds = self.config.bounds

        # Calculate size in meters using approximate conversion
        EARTH_RADIUS = 6_371_000
        lat_meters_per_degree = 2 * math.pi * EARTH_RADIUS / 360
        lon_meters_per_degree = lat_meters_per_degree * math.cos(math.radians(bounds.center_lat))

        width_meters = bounds.width_degrees * lon_meters_per_degree
        height_meters = bounds.height_degrees * lat_meters_per_degree

        # Convert to blocks
        width_blocks = int(width_meters / self.config.scale)
        height_blocks = int(height_meters / self.config.scale)

        # Convert to chunks
        width_chunks = (width_blocks + CHUNK_SIZE - 1) // CHUNK_SIZE
        height_chunks = (height_blocks + CHUNK_SIZE - 1) // CHUNK_SIZE
        if self.config.max_chunks:
            width_chunks = min(width_chunks, self.config.max_chunks)
            height_chunks = min(height_chunks, self.config.max_chunks)
            width_blocks = width_chunks * CHUNK_SIZE
            height_blocks = height_chunks * CHUNK_SIZE

        return (width_blocks, height_blocks, width_chunks, height_chunks)

    def _get_required_tiles(self) -> List[str]:
        """Get list of SRTM tiles needed for this conversion."""
        return get_required_tiles(
            self.config.bounds.min_lat,
            self.config.bounds.max_lat,
            self.config.bounds.min_lon,
            self.config.bounds.max_lon,
        )

    def _get_elevation_at_block(self, block_x: int, block_z: int) -> float:
        """Get real-world elevation at a block position.

        Args:
            block_x: Block X coordinate
            block_z: Block Z coordinate

        Returns:
            Elevation in meters
        """
        # Convert block to lat/lon
        lat, lon = block_to_lat_lon(
            block_x, block_z,
            self.config.bounds.min_lat,
            self.config.bounds.min_lon,
            self.config.scale
        )

        # Get elevation from SRTM
        elevation = self._srtm_cache.get_elevation(lat, lon)
        if elevation is None:
            return 0.0  # Default to sea level if no data
        return elevation

    def _generate_chunk(
        self,
        chunk_x: int,
        chunk_z: int
    ) -> Chunk:
        """Generate a single chunk.

        Args:
            chunk_x: Chunk X coordinate
            chunk_z: Chunk Z coordinate

        Returns:
            Generated Chunk
        """
        chunk = Chunk(chunk_x, chunk_z)

        # Generate each column in the chunk
        for local_x in range(CHUNK_SIZE):
            for local_z in range(CHUNK_SIZE):
                block_x = chunk_x * CHUNK_SIZE + local_x
                block_z = chunk_z * CHUNK_SIZE + local_z

                # Get elevation
                elevation_m = self._get_elevation_at_block(block_x, block_z)
                surface_y = self._elevation_mapper.get_y(elevation_m)

                # Clamp to valid range
                surface_y = max(1, min(WORLD_HEIGHT - 1, surface_y))

                # Get latitude for biome calculation
                lat, lon = block_to_lat_lon(
                    block_x, block_z,
                    self.config.bounds.min_lat,
                    self.config.bounds.min_lon,
                    self.config.scale
                )
                latitude_factor = self._biome_classifier.get_latitude_factor(lat)
                color = self._get_color_at_lat_lon(lat, lon)
                if color is None:
                    biome = self._biome_classifier.classify_from_elevation_only(
                        surface_y, latitude_factor
                    )
                else:
                    r, g, b = color
                    biome = self._biome_classifier.classify(
                        r, g, b, surface_y, latitude_factor
                    )

                # Generate block column
                column = self._block_generator.generate_column(
                    x=block_x,
                    z=block_z,
                    surface_y=surface_y,
                    biome=biome,
                    seed=self.config.seed
                )

                # Set blocks in chunk
                for y, block_id in enumerate(column.blocks):
                    if y < WORLD_HEIGHT and block_id != 0:
                        chunk.set_block(local_x, y, local_z, block_id)

        return chunk

    def _load_existing_indexes(
        self,
        chunks_dir: Path,
        width_chunks: int,
        height_chunks: int,
    ) -> Dict[Tuple[int, int], set[int]]:
        """Load existing chunk indexes per region for resume."""
        existing = {}
        for chunk_z in range(height_chunks):
            for chunk_x in range(width_chunks):
                region_x = chunk_x >> 5
                region_z = chunk_z >> 5
                key = (region_x, region_z)
                if key in existing:
                    continue
                path = chunks_dir / f"{region_x}.{region_z}.region.bin"
                indexes = read_region_chunk_indexes(path)
                if indexes:
                    existing[key] = indexes
        return existing

    def run(
        self,
        progress_callback: Optional[ProgressCallback] = None
    ) -> Path:
        """Run the full conversion pipeline.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            Path to the generated world directory
        """
        def report_progress(phase: str, current: int, total: int, message: str = ""):
            if progress_callback:
                progress_callback(ConversionProgress(phase, current, total, message))

        # Phase 1: Download SRTM data
        report_progress("download", 0, 1, "Checking required SRTM tiles...")
        required_tiles = self._get_required_tiles()
        missing_tiles = self._downloader.get_missing_tiles(
            self.config.bounds.min_lat,
            self.config.bounds.max_lat,
            self.config.bounds.min_lon,
            self.config.bounds.max_lon,
        )

        if missing_tiles and not self.config.skip_download:
            report_progress("download", 0, len(missing_tiles), f"Downloading {len(missing_tiles)} tiles...")
            self._downloader.download_region(
                self.config.bounds.min_lat,
                self.config.bounds.max_lat,
                self.config.bounds.min_lon,
                self.config.bounds.max_lon,
            )
            report_progress("download", len(missing_tiles), len(missing_tiles), "Download complete")
        else:
            report_progress("download", 1, 1, f"All {len(required_tiles)} tiles available")

        # Phase 1b: Load Blue Marble imagery (optional)
        self._init_blue_marble(progress_callback)

        # Phase 2: Calculate world dimensions
        report_progress("calculate", 0, 1, "Calculating world dimensions...")
        width_blocks, height_blocks, width_chunks, height_chunks = self._calculate_world_size()
        total_chunks = width_chunks * height_chunks
        report_progress("calculate", 1, 1,
                       f"World size: {width_chunks}x{height_chunks} chunks ({width_blocks}x{height_blocks} blocks)")

        # Phase 3: Create world
        report_progress("world", 0, 1, "Creating world structure...")
        world = World(
            self.config.name,
            WorldConfig(seed=self.config.seed or 0)
        )

        # Phase 4: Generate chunks
        report_progress("generate", 0, total_chunks, "Generating terrain...")
        chunk_count = 0
        chunks_dir = (self.config.output_dir / self.config.name / "chunks")
        existing_indexes = {}
        if self.config.resume:
            existing_indexes = self._load_existing_indexes(chunks_dir, width_chunks, height_chunks)

        chunk_coords = []
        for chunk_z in range(height_chunks):
            for chunk_x in range(width_chunks):
                if self.config.resume:
                    region_key = (chunk_x >> 5, chunk_z >> 5)
                    idx = (chunk_z & 0x1F) * 32 + (chunk_x & 0x1F)
                    if idx in existing_indexes.get(region_key, set()):
                        chunk_count += 1
                        report_progress("generate", chunk_count, total_chunks,
                                      f"Chunk ({chunk_x}, {chunk_z}) [skipped]")
                        continue
                chunk_coords.append((chunk_x, chunk_z))

        if self.config.parallel:
            workers = self.config.parallel_workers or os.cpu_count() or 1
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(self._generate_chunk, x, z): (x, z) for x, z in chunk_coords}
                for future in as_completed(futures):
                    x, z = futures[future]
                    chunk = future.result()
                    world.add_chunk(chunk)
                    chunk_count += 1
                    report_progress("generate", chunk_count, total_chunks, f"Chunk ({x}, {z})")
        else:
            for chunk_x, chunk_z in chunk_coords:
                chunk = self._generate_chunk(chunk_x, chunk_z)
                world.add_chunk(chunk)
                chunk_count += 1
                report_progress("generate", chunk_count, total_chunks,
                              f"Chunk ({chunk_x}, {chunk_z})")

        # Phase 5: Save world
        report_progress("save", 0, 1, "Saving world files...")
        existing_raw_by_region = None
        if self.config.resume and chunks_dir.exists():
            existing_raw_by_region = {}
            for region_key in {((x >> 5), (z >> 5)) for x, z in chunk_coords}:
                path = chunks_dir / f"{region_key[0]}.{region_key[1]}.region.bin"
                raw = read_region_chunks_raw(path)
                if raw:
                    existing_raw_by_region[region_key] = raw
        world_path = world.save(self.config.output_dir, existing_raw_by_region=existing_raw_by_region)
        report_progress("save", 1, 1, f"World saved to {world_path}")

        return world_path


def convert_region(
    name: str,
    bounds: BoundingBox,
    output_dir: Path,
    scale: float = 5000.0,
    cache_dir: Optional[Path] = None,
    progress_callback: Optional[ProgressCallback] = None,
    **kwargs
) -> Path:
    """Convenience function to convert a geographic region.

    Args:
        name: World name
        bounds: Geographic bounding box
        output_dir: Output directory
        scale: Meters per block
        cache_dir: Cache directory for NASA data
        progress_callback: Optional progress callback
        **kwargs: Additional ConversionConfig options

    Returns:
        Path to generated world
    """
    config = ConversionConfig(
        name=name,
        bounds=bounds,
        scale=scale,
        output_dir=output_dir,
        cache_dir=cache_dir or Path("cache"),
        **kwargs
    )

    pipeline = ConversionPipeline(config)
    return pipeline.run(progress_callback)
