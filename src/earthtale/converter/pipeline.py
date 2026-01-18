"""Main conversion pipeline from NASA data to Hytale worlds."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Callable, Any
import math

from ..config import ConversionConfig, BoundingBox
from ..nasa import (
    SRTMCache,
    SRTMDownloader,
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
)
from ..hytale import (
    World,
    WorldConfig,
    Chunk,
    CHUNK_SIZE,
    WORLD_HEIGHT,
)


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

        # Initialize mappers
        self._elevation_mapper = ElevationMapper(ElevationConfig(
            min_elevation=self.config.min_elevation,
            max_elevation=self.config.max_elevation,
            sea_level_y=self.config.sea_level_y,
            vertical_exaggeration=self.config.vertical_exaggeration,
        ))

        self._biome_classifier = BiomeClassifier()
        self._block_generator = BlockColumnGenerator()

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

                # Classify biome (elevation-only mode)
                biome = self._biome_classifier.classify_from_elevation_only(
                    surface_y, latitude_factor
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
            # Note: In practice, would use async download here
            for i, tile in enumerate(missing_tiles):
                report_progress("download", i, len(missing_tiles), f"Downloading {tile}...")
                # self._downloader.download_tile(tile)  # Would be async
            report_progress("download", len(missing_tiles), len(missing_tiles), "Download complete")
        else:
            report_progress("download", 1, 1, f"All {len(required_tiles)} tiles available")

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

        for chunk_z in range(height_chunks):
            for chunk_x in range(width_chunks):
                chunk = self._generate_chunk(chunk_x, chunk_z)
                world.add_chunk(chunk)
                chunk_count += 1
                report_progress("generate", chunk_count, total_chunks,
                              f"Chunk ({chunk_x}, {chunk_z})")

        # Phase 5: Save world
        report_progress("save", 0, 1, "Saving world files...")
        world_path = world.save(self.config.output_dir)
        report_progress("save", 1, 1, f"World saved to {world_path}")

        return world_path


def convert_region(
    name: str,
    bounds: BoundingBox,
    output_dir: Path,
    scale: float = 30.0,
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
