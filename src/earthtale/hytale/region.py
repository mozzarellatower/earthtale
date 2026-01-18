"""Region file handling for Hytale world format.

Region files use the "HytaleIndexedStorage" format:
- Header: "HytaleIndexedStorage" (20 bytes)
- Version: 4 bytes (big-endian, value: 1)
- Index count: 4 bytes (big-endian, value: 1024 = 32x32)
- Segment size: 4 bytes (big-endian, value: 4096)
- Index table: 1024 x 4-byte segment start indices (1-based, big-endian)
- Chunk data: Compressed chunks with 8-byte header + zstd data, aligned to segment size

Each region file contains 32x32 chunks. Chunk position within a region
is calculated as: (chunk_x % 32, chunk_z % 32)

Chunk data format:
- 4 bytes: uncompressed size (big-endian)
- 4 bytes: flags/metadata (big-endian)
- N bytes: zstd compressed BSON data
"""

import struct
import math
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Iterable

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

from .constants import (
    REGION_SIZE,
    INDEXED_STORAGE_HEADER,
    INDEXED_STORAGE_VERSION,
    INDEXED_STORAGE_INDEX_COUNT,
    INDEXED_STORAGE_SEGMENT_SIZE,
)
from .chunk import Chunk


def chunk_to_region_coords(chunk_x: int, chunk_z: int) -> Tuple[int, int]:
    """Convert chunk coordinates to region coordinates."""
    region_x = chunk_x >> 5  # // 32
    region_z = chunk_z >> 5
    return region_x, region_z


def chunk_to_local_coords(chunk_x: int, chunk_z: int) -> Tuple[int, int]:
    """Convert chunk coordinates to local coordinates within a region."""
    local_x = chunk_x & 0x1F  # % 32
    local_z = chunk_z & 0x1F
    return local_x, local_z


def local_to_index(local_x: int, local_z: int) -> int:
    """Convert local chunk coordinates to index in region file."""
    return local_z * REGION_SIZE + local_x


class Region:
    """A region containing 32x32 chunks.

    Attributes:
        x: Region X coordinate
        z: Region Z coordinate
        chunks: Dictionary mapping (local_x, local_z) to Chunk
    """

    def __init__(self, x: int, z: int):
        self.x = x
        self.z = z
        self.chunks: Dict[Tuple[int, int], Chunk] = {}

    def get_chunk(self, chunk_x: int, chunk_z: int) -> Optional[Chunk]:
        """Get chunk by world chunk coordinates."""
        local_x, local_z = chunk_to_local_coords(chunk_x, chunk_z)
        return self.chunks.get((local_x, local_z))

    def set_chunk(self, chunk: Chunk) -> None:
        """Add or replace a chunk in this region."""
        local_x, local_z = chunk_to_local_coords(chunk.x, chunk.z)
        self.chunks[(local_x, local_z)] = chunk

    def get_filename(self) -> str:
        """Get the region filename."""
        return f"{self.x}.{self.z}.region.bin"


class RegionWriter:
    """Writes region files in Hytale format."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_region(
        self,
        region: Region,
        block_migration_version: int = 0,
        existing_raw: Optional[Dict[int, bytes]] = None,
    ) -> Path:
        """Write a region to disk.

        Args:
            region: Region to write
            block_migration_version: Block migration version for serialization

        Returns:
            Path to the written file
        """
        filepath = self.output_dir / region.get_filename()

        # Initialize zstd compressor
        if HAS_ZSTD:
            cctx = zstd.ZstdCompressor(level=3)
        else:
            cctx = None

        # Serialize and compress all chunks
        chunk_data: Dict[int, bytes] = {}
        if existing_raw:
            chunk_data.update(existing_raw)
        for (local_x, local_z), chunk in region.chunks.items():
            idx = local_to_index(local_x, local_z)
            bson_data = chunk.to_bson(block_migration_version)

            # Compress with zstd and add 8-byte header
            if cctx:
                compressed = cctx.compress(bson_data)
            else:
                # Fallback: no compression (may not work with Hytale)
                compressed = bson_data

            # Build chunk entry: 4 bytes uncompressed size + 4 bytes compressed size + compressed data
            chunk_entry = bytearray()
            chunk_entry.extend(struct.pack(">I", len(bson_data)))  # Uncompressed size
            chunk_entry.extend(struct.pack(">I", len(compressed)))  # Compressed size
            chunk_entry.extend(compressed)

            chunk_data[idx] = bytes(chunk_entry)

        # Index table stores the start segment index (1-based). 0 means no chunk.
        chunk_offsets = [0] * INDEXED_STORAGE_INDEX_COUNT
        segment_offset = 0
        for idx in range(INDEXED_STORAGE_INDEX_COUNT):
            if idx in chunk_data:
                chunk_offsets[idx] = segment_offset + 1
                data = chunk_data[idx]
                aligned_size = math.ceil(len(data) / INDEXED_STORAGE_SEGMENT_SIZE) * INDEXED_STORAGE_SEGMENT_SIZE
                segment_offset += aligned_size // INDEXED_STORAGE_SEGMENT_SIZE

        # Build the file
        with open(filepath, "wb") as f:
            # Header
            f.write(INDEXED_STORAGE_HEADER)

            # Version (big-endian)
            f.write(struct.pack(">I", INDEXED_STORAGE_VERSION))

            # Index count (big-endian)
            f.write(struct.pack(">I", INDEXED_STORAGE_INDEX_COUNT))

            # Segment size (big-endian)
            f.write(struct.pack(">I", INDEXED_STORAGE_SEGMENT_SIZE))

            # Index table (big-endian 1-based segment offsets)
            for offset in chunk_offsets:
                f.write(struct.pack(">I", offset))

            # Chunk data (in index order, padded to segment boundaries)
            for idx in range(INDEXED_STORAGE_INDEX_COUNT):
                if idx in chunk_data:
                    data = chunk_data[idx]
                    f.write(data)
                    # Pad to segment boundary
                    remainder = len(data) % INDEXED_STORAGE_SEGMENT_SIZE
                    if remainder > 0:
                        padding = INDEXED_STORAGE_SEGMENT_SIZE - remainder
                        f.write(b"\x00" * padding)

        return filepath


class RegionReader:
    """Reads region files in Hytale format."""

    def __init__(self, chunks_dir: Path):
        self.chunks_dir = chunks_dir

    def read_region(self, region_x: int, region_z: int) -> Optional[Region]:
        """Read a region from disk.

        Args:
            region_x: Region X coordinate
            region_z: Region Z coordinate

        Returns:
            Region object or None if file doesn't exist
        """
        filename = f"{region_x}.{region_z}.region.bin"
        filepath = self.chunks_dir / filename

        if not filepath.exists():
            return None

        region = Region(region_x, region_z)

        with open(filepath, "rb") as f:
            # Read header
            header = f.read(20)
            if header != INDEXED_STORAGE_HEADER:
                raise ValueError(f"Invalid header: {header}")

            # Read version
            version = struct.unpack(">I", f.read(4))[0]
            if version != INDEXED_STORAGE_VERSION:
                raise ValueError(f"Unsupported version: {version}")

            # Read index count
            index_count = struct.unpack(">I", f.read(4))[0]
            if index_count != INDEXED_STORAGE_INDEX_COUNT:
                raise ValueError(f"Unexpected index count: {index_count}")

            # Read segment size
            segment_size = struct.unpack(">I", f.read(4))[0]

            # Read index table
            segment_offsets = []
            for _ in range(index_count):
                offset = struct.unpack(">I", f.read(4))[0]
                segment_offsets.append(offset)

            # Data starts after header + version + counts + index table
            data_start = 20 + 4 + 4 + 4 + (index_count * 4)

            # TODO: Actually parse BSON chunk data
            # For now, we just return the region structure

        return region


def read_region_chunk_indexes(filepath: Path) -> Optional[set[int]]:
    """Read the set of chunk indexes present in a region file."""
    if not filepath.exists():
        return None
    with open(filepath, "rb") as f:
        header = f.read(20)
        if header != INDEXED_STORAGE_HEADER:
            raise ValueError(f"Invalid header: {header}")
        version = struct.unpack(">I", f.read(4))[0]
        if version != INDEXED_STORAGE_VERSION:
            raise ValueError(f"Unsupported version: {version}")
        index_count = struct.unpack(">I", f.read(4))[0]
        if index_count != INDEXED_STORAGE_INDEX_COUNT:
            raise ValueError(f"Unexpected index count: {index_count}")
        f.read(4)  # segment size
        indexes = set()
        for i in range(index_count):
            entry = struct.unpack(">I", f.read(4))[0]
            if entry:
                indexes.add(i)
    return indexes


def read_region_chunks_raw(filepath: Path) -> Optional[Dict[int, bytes]]:
    """Read raw chunk entries from a region file (compressed, with 8-byte header)."""
    if not filepath.exists():
        return None
    with open(filepath, "rb") as f:
        header = f.read(20)
        if header != INDEXED_STORAGE_HEADER:
            raise ValueError(f"Invalid header: {header}")
        version = struct.unpack(">I", f.read(4))[0]
        if version != INDEXED_STORAGE_VERSION:
            raise ValueError(f"Unsupported version: {version}")
        index_count = struct.unpack(">I", f.read(4))[0]
        if index_count != INDEXED_STORAGE_INDEX_COUNT:
            raise ValueError(f"Unexpected index count: {index_count}")
        segment_size = struct.unpack(">I", f.read(4))[0]
        index_entries = [struct.unpack(">I", f.read(4))[0] for _ in range(index_count)]
        data_start = 20 + 4 + 4 + 4 + (index_count * 4)
        raw = {}
        data = f.read()
        for idx, entry in enumerate(index_entries):
            if not entry:
                continue
            start = (entry - 1) * segment_size
            pos = data_start + start
            if pos + 8 > data_start + len(data):
                continue
            compressed_size = struct.unpack(">I", data[pos + 4:pos + 8])[0]
            length = 8 + compressed_size
            raw[idx] = data[pos:pos + length]
        return raw


def write_regions(
    chunks: List[Chunk],
    output_dir: Path,
    block_migration_version: int = 0,
    existing_raw_by_region: Optional[Dict[Tuple[int, int], Dict[int, bytes]]] = None,
) -> List[Path]:
    """Write multiple chunks to region files.

    Args:
        chunks: List of chunks to write
        output_dir: Directory for region files
        block_migration_version: Block migration version

    Returns:
        List of written file paths
    """
    # Group chunks by region
    regions: Dict[Tuple[int, int], Region] = {}

    for chunk in chunks:
        region_x, region_z = chunk_to_region_coords(chunk.x, chunk.z)
        key = (region_x, region_z)
        if key not in regions:
            regions[key] = Region(region_x, region_z)
        regions[key].set_chunk(chunk)

    # Write all regions
    writer = RegionWriter(output_dir)
    written_files = []

    for region in regions.values():
        existing_raw = None
        if existing_raw_by_region:
            existing_raw = existing_raw_by_region.get((region.x, region.z))
        filepath = writer.write_region(region, block_migration_version, existing_raw=existing_raw)
        written_files.append(filepath)

    return written_files
