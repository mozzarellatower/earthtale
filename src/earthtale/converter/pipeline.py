"""Main conversion pipeline from NASA data to Hytale worlds."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Callable, Any, Dict, Tuple
import math
import os
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import deque
from queue import Queue, Empty
import asyncio
import threading
import time

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
        self._blue_marble_ready = threading.Event()

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
            self._blue_marble_ready.set()
            return
        if progress_callback:
            progress_callback(ConversionProgress("blue_marble", 0, 1, "Loading Blue Marble imagery..."))
        path = None
        if not self.config.skip_download:
            import asyncio
            def on_download(progress) -> None:
                if progress.error and progress_callback:
                    progress_callback(
                        ConversionProgress(
                            "blue_marble",
                            0,
                            1,
                            f"Blue Marble download error: {progress.error}",
                        )
                    )
            path = asyncio.run(
                self._blue_marble_downloader.download_image(
                    resolution=self.config.blue_marble_resolution,
                    progress_callback=on_download,
                )
            )
        if path is None:
            filename = f"blue_marble_{self.config.blue_marble_resolution}.jpg"
            candidate = self._blue_marble_downloader.cache_dir / filename
            if candidate.exists():
                path = candidate
        if path is None:
            if progress_callback:
                progress_callback(
                    ConversionProgress(
                        "blue_marble",
                        1,
                        1,
                        "Blue Marble not available; continuing without imagery",
                    )
                )
            self._blue_marble_ready.set()
            return
        image = Image.open(path).convert("RGB")
        self._blue_marble_pixels = image.load()
        self._blue_marble_size = image.size
        if progress_callback:
            progress_callback(ConversionProgress("blue_marble", 1, 1, "Blue Marble imagery ready"))
        self._blue_marble_ready.set()

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
            return self._estimate_missing_elevation(lat, lon)
        return elevation

    def _estimate_missing_elevation(self, lat: float, lon: float) -> float:
        """Estimate elevation for missing tiles by sampling neighbors + noise."""
        samples = []
        offsets = (
            (0.0, 0.25),
            (0.0, -0.25),
            (0.25, 0.0),
            (-0.25, 0.0),
            (0.25, 0.25),
            (-0.25, -0.25),
        )
        for dlat, dlon in offsets:
            elev = self._srtm_cache.get_elevation(lat + dlat, lon + dlon)
            if elev is not None:
                samples.append(elev)
        if samples:
            return sum(samples) / len(samples)
        # Deterministic gentle noise when no neighbors are available.
        seed = int((lat + 90.0) * 1000) ^ (int((lon + 180.0) * 1000) << 1)
        noise = ((seed * 2654435761) & 0xFFFFFFFF) / 0xFFFFFFFF
        return (noise - 0.5) * 10.0

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

                # Estimate slope in Y by sampling neighbors
                neighbor_offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
                max_slope = 0
                for dx, dz in neighbor_offsets:
                    neighbor_elev = self._get_elevation_at_block(block_x + dx, block_z + dz)
                    neighbor_y = self._elevation_mapper.get_y(neighbor_elev)
                    max_slope = max(max_slope, abs(neighbor_y - surface_y))

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
                    slope_y=max_slope,
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
        self._downloader.mark_unavailable_tiles(required_tiles)
        missing_tiles = [tile for tile in required_tiles if not self._downloader.is_tile_available(tile)]
        def tile_sort_key(name: str) -> tuple[int, int]:
            lat = int(name[1:3]) * (1 if name[0] == "N" else -1)
            lon = int(name[4:7]) * (1 if name[3] == "E" else -1)
            return (-lat, lon)
        missing_tiles.sort(key=tile_sort_key)
        tile_queue: Queue[Optional[str]] = Queue()
        download_done = threading.Event()
        download_error: list[Optional[BaseException]] = [None]
        download_cache_bytes = (
            int(self.config.download_cache_mb) * 1024 * 1024
            if self.config.download_cache_mb
            else 0
        )
        generation_started = threading.Event()

        def cache_size_bytes() -> int:
            total = 0
            for path in (self.config.cache_dir / "srtm").glob("*.hgt"):
                try:
                    total += path.stat().st_size
                except FileNotFoundError:
                    continue
            return total

        if missing_tiles and not self.config.skip_download:
            report_progress("download", 0, len(missing_tiles), f"Downloading {len(missing_tiles)} tiles...")

            def download_worker() -> None:
                start_times = {}
                completed = 0
                lock = threading.Lock()
                failures: list[str] = []
                started: set[str] = set()

                def on_progress(progress) -> None:
                    nonlocal completed
                    if progress.filename not in start_times:
                        start_times[progress.filename] = time.perf_counter()
                    if progress.downloaded_bytes > 0 and progress.filename not in started:
                        started.add(progress.filename)
                        report_progress(
                            "download",
                            completed,
                            len(missing_tiles),
                            f"Downloading {progress.filename}...",
                        )
                    if progress.error:
                        tile_name = progress.filename.split(".", 1)[0]
                        with lock:
                            failures.append(f"{tile_name}: {progress.error}")
                            report_progress(
                                "download",
                                completed,
                                len(missing_tiles),
                                f"Tile {tile_name} failed: {progress.error}",
                            )
                    if progress.complete and not progress.error:
                        tile_name = progress.filename.split(".", 1)[0]
                        tile_queue.put(tile_name)
                        elapsed = time.perf_counter() - start_times.get(progress.filename, time.perf_counter())
                        with lock:
                            completed += 1
                            if completed == 1:
                                report_progress(
                                    "download",
                                    completed,
                                    len(missing_tiles),
                                    f"Tile {tile_name} downloaded in {elapsed:.2f}s",
                                )
                            if completed % 10 == 0 or completed == len(missing_tiles):
                                report_progress(
                                    "download",
                                    completed,
                                    len(missing_tiles),
                                    f"Downloaded {completed}/{len(missing_tiles)} tiles",
                                )

                try:
                    async def download_all() -> None:
                        semaphore = asyncio.Semaphore(self.config.download_concurrency)

                        async def download_one(tile: str) -> None:
                            async with semaphore:
                                if download_cache_bytes:
                                    while cache_size_bytes() >= download_cache_bytes and generation_started.is_set():
                                        await asyncio.sleep(0.25)
                                await self._downloader.download_tile(tile, progress_callback=on_progress)
                                if (
                                    not self._downloader.is_tile_cached(tile)
                                    and self._downloader.is_tile_available(tile)
                                ):
                                    tile_queue.put(tile)

                        tasks = [asyncio.create_task(download_one(tile)) for tile in missing_tiles]
                        for task in tasks:
                            await task

                    asyncio.run(download_all())
                    if failures:
                        raise RuntimeError("SRTM download failed: " + "; ".join(failures))
                except BaseException as exc:
                    download_error[0] = exc
                finally:
                    download_done.set()
                    tile_queue.put(None)

            threading.Thread(target=download_worker, daemon=True).start()
        else:
            if missing_tiles and self.config.skip_download:
                report_progress("download", 0, len(missing_tiles), f"Skipping download; {len(missing_tiles)} tiles missing")
            else:
                report_progress("download", 1, 1, f"All {len(required_tiles)} tiles available")
            download_done.set()

        # Phase 2: Calculate world dimensions
        report_progress("calculate", 0, 1, "Calculating world dimensions...")
        width_blocks, height_blocks, width_chunks, height_chunks = self._calculate_world_size()
        total_chunks = width_chunks * height_chunks
        report_progress("calculate", 1, 1,
                       f"World size: {width_chunks}x{height_chunks} chunks ({width_blocks}x{height_blocks} blocks)")

        # Phase 1b: Load Blue Marble imagery (optional)
        if self.config.use_blue_marble and Image is not None:
            if total_chunks <= 4096:
                self._init_blue_marble(progress_callback)
            else:
                threading.Thread(
                    target=self._init_blue_marble,
                    args=(progress_callback,),
                    daemon=True,
                ).start()

        # Phase 3: Create world
        report_progress("world", 0, 1, "Creating world structure...")
        world_config = WorldConfig(seed=self.config.seed or 0)
        world_config.world_gen.type = "Void"
        world_config.world_gen.name = "Void"
        world_config.save_new_chunks = False
        world = World(self.config.name, world_config)

        # Phase 4: Generate chunks
        report_progress("generate", 0, total_chunks, "Generating terrain...")
        chunk_count = 0
        chunks_dir = (self.config.output_dir / self.config.name / "chunks")
        existing_indexes = {}
        if self.config.resume:
            existing_indexes = self._load_existing_indexes(chunks_dir, width_chunks, height_chunks)

        chunk_coords = []
        for chunk_z in range(height_chunks - 1, -1, -1):
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

        origin_lat = self.config.bounds.min_lat
        origin_lon = self.config.bounds.min_lon
        available_tiles = {tile for tile in required_tiles if self._downloader.is_tile_available(tile)}
        chunk_requirements = {}
        tile_to_chunks: Dict[str, list[tuple[int, int]]] = {}
        tile_remaining = {}
        pending_chunks = set()
        ready_queue: deque[tuple[int, int]] = deque()

        def required_tiles_for_chunk(chunk_x: int, chunk_z: int) -> set[str]:
            block_x0 = chunk_x * CHUNK_SIZE
            block_z0 = chunk_z * CHUNK_SIZE
            block_x1 = block_x0 + CHUNK_SIZE
            block_z1 = block_z0 + CHUNK_SIZE
            lat0, lon0 = block_to_lat_lon(block_x0, block_z0, origin_lat, origin_lon, self.config.scale)
            lat1, lon1 = block_to_lat_lon(block_x1, block_z1, origin_lat, origin_lon, self.config.scale)
            min_lat = min(lat0, lat1)
            max_lat = max(lat0, lat1)
            min_lon = min(lon0, lon1)
            max_lon = max(lon0, lon1)
            return set(get_required_tiles(min_lat, max_lat, min_lon, max_lon))

        for chunk_x, chunk_z in chunk_coords:
            required = required_tiles_for_chunk(chunk_x, chunk_z)
            chunk_requirements[(chunk_x, chunk_z)] = required
            pending_chunks.add((chunk_x, chunk_z))
            for tile in required:
                tile_to_chunks.setdefault(tile, []).append((chunk_x, chunk_z))
        # Prefetch tiles for the first chunk to start generation ASAP.
        first_chunk = (0, height_chunks - 1)
        first_chunk_tiles = chunk_requirements.get(first_chunk, set())
        if first_chunk_tiles:
            prefetch = [t for t in missing_tiles if t in first_chunk_tiles]
            rest = [t for t in missing_tiles if t not in first_chunk_tiles]
            missing_tiles = prefetch + rest
        for tile, chunks in tile_to_chunks.items():
            tile_remaining[tile] = len(chunks)

        def enqueue_ready_chunks() -> None:
            for chunk in list(pending_chunks):
                if chunk_requirements[chunk].issubset(available_tiles):
                    pending_chunks.remove(chunk)
                    ready_queue.append(chunk)

        def enqueue_for_tile(tile: str) -> None:
            for chunk in tile_to_chunks.get(tile, []):
                if chunk in pending_chunks and chunk_requirements[chunk].issubset(available_tiles):
                    pending_chunks.remove(chunk)
                    ready_queue.append(chunk)

        def release_tiles_for_chunk(chunk: tuple[int, int]) -> None:
            for tile in chunk_requirements.get(chunk, set()):
                remaining = tile_remaining.get(tile, 0) - 1
                tile_remaining[tile] = remaining
                if remaining <= 0:
                    tile_path = self.config.cache_dir / "srtm" / f"{tile}.hgt"
                    try:
                        tile_path.unlink()
                    except FileNotFoundError:
                        pass
                    self._srtm_cache._cache.pop(tile, None)

        enqueue_ready_chunks()
        if progress_callback:
            report_progress(
                "generate",
                0,
                total_chunks,
                f"Initial ready chunks: {len(ready_queue)}",
            )
        download_reported = False

        autosave_every = self.config.autosave_every or 0
        world_path = self.config.output_dir / self.config.name
        existing_raw_by_region = None
        if self.config.resume and chunks_dir.exists():
            existing_raw_by_region = {}
            for region_key in {((x >> 5), (z >> 5)) for x, z in chunk_coords}:
                path = chunks_dir / f"{region_key[0]}.{region_key[1]}.region.bin"
                raw = read_region_chunks_raw(path)
                if raw:
                    existing_raw_by_region[region_key] = raw

        def autosave(force: bool = False) -> None:
            if autosave_every <= 0 and not force:
                return
            if not force and chunk_count % autosave_every != 0:
                return
            report_progress("save", 0, 1, "Autosaving world files...")
            world.save(self.config.output_dir, existing_raw_by_region=existing_raw_by_region)
            report_progress("save", 1, 1, "Autosave complete")

        try:
            if self.config.parallel:
                workers = self.config.parallel_workers or os.cpu_count() or 1
                max_in_flight = max(1, workers * 2)
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    start_times = {}
                    futures = {}

                    def submit_ready() -> None:
                        while ready_queue and len(futures) < max_in_flight:
                            x, z = ready_queue.popleft()
                            start_times[(x, z)] = time.perf_counter()
                            futures[executor.submit(self._generate_chunk, x, z)] = (x, z)

                    while pending_chunks or futures or ready_queue:
                        while True:
                            try:
                                tile = tile_queue.get_nowait()
                            except Empty:
                                break
                            if tile is None:
                                download_done.set()
                            else:
                                available_tiles.add(tile)
                                enqueue_for_tile(tile)

                        if download_done.is_set() and missing_tiles and not download_reported and not self.config.skip_download:
                            report_progress("download", len(missing_tiles), len(missing_tiles), "Download complete")
                            download_reported = True

                        if download_done.is_set() and pending_chunks and not ready_queue and not futures:
                            for chunk in list(pending_chunks):
                                pending_chunks.remove(chunk)
                                ready_queue.append(chunk)

                        submit_ready()

                        if futures:
                            done, _ = wait(futures, timeout=0.1, return_when=FIRST_COMPLETED)
                            for future in done:
                                x, z = futures.pop(future)
                                chunk = future.result()
                                world.add_chunk(chunk)
                                chunk_count += 1
                                if chunk_count == 1:
                                    generation_started.set()
                                release_tiles_for_chunk((x, z))
                                elapsed = time.perf_counter() - start_times[(x, z)]
                                report_progress(
                                    "generate",
                                    chunk_count,
                                    total_chunks,
                                    f"Chunk ({x}, {z}) done in {elapsed:.2f}s",
                                )
                                autosave()
                        else:
                            if not pending_chunks:
                                break
                            try:
                                tile = tile_queue.get(timeout=0.1)
                            except Empty:
                                continue
                            if tile is None:
                                download_done.set()
                            else:
                                available_tiles.add(tile)
                                enqueue_for_tile(tile)
            else:
                while pending_chunks or ready_queue:
                    if download_done.is_set() and missing_tiles and not download_reported and not self.config.skip_download:
                        report_progress("download", len(missing_tiles), len(missing_tiles), "Download complete")
                        download_reported = True

                    if ready_queue:
                        chunk_x, chunk_z = ready_queue.popleft()
                        start_time = time.perf_counter()
                        chunk = self._generate_chunk(chunk_x, chunk_z)
                        world.add_chunk(chunk)
                        chunk_count += 1
                        if chunk_count == 1:
                            generation_started.set()
                        release_tiles_for_chunk((chunk_x, chunk_z))
                        elapsed = time.perf_counter() - start_time
                        report_progress(
                            "generate",
                            chunk_count,
                            total_chunks,
                            f"Chunk ({chunk_x}, {chunk_z}) done in {elapsed:.2f}s",
                        )
                        autosave()
                        continue

                    if download_done.is_set() and pending_chunks:
                        for chunk in list(pending_chunks):
                            pending_chunks.remove(chunk)
                            ready_queue.append(chunk)
                        continue

                    try:
                        tile = tile_queue.get(timeout=0.1)
                    except Empty:
                        continue
                    if tile is None:
                        download_done.set()
                    else:
                        available_tiles.add(tile)
                        enqueue_for_tile(tile)
        except KeyboardInterrupt:
            report_progress("save", 0, 1, "Interrupted, saving partial world...")
            autosave(force=True)
            report_progress("save", 1, 1, "Partial world saved")
            return world_path

        if download_error[0] is not None:
            raise download_error[0]

        # Phase 5: Save world
        report_progress("save", 0, 1, "Saving world files...")
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
