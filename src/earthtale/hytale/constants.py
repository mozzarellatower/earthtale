"""Constants for Hytale world format derived from ChunkUtil.java."""

# Chunk dimensions
CHUNK_BITS = 5
CHUNK_SIZE = 32  # 32x32 blocks horizontally
CHUNK_SIZE_MASK = 31
CHUNK_SIZE_SQUARED = 1024  # 32*32 columns per chunk

# Section/Height
SECTION_HEIGHT = 32  # Blocks per section vertically
SECTIONS_PER_CHUNK = 10  # 10 sections of 32 = 320 total
WORLD_HEIGHT = 320  # Total height in blocks
WORLD_HEIGHT_MASK = 511  # (highestOneBit(320) << 1) - 1

# Block counts
BLOCKS_PER_SECTION = 32768  # 32*32*32
BLOCKS_PER_COLUMN = 327680  # 32*32*320

# Y bounds
MIN_Y = 0
MAX_Y = 319
MIN_ENTITY_Y = -32
MIN_SECTION = 0

# Region file format
REGION_SIZE = 32  # 32x32 chunks per region
INDEXED_STORAGE_HEADER = b"HytaleIndexedStorage"
INDEXED_STORAGE_VERSION = 1
INDEXED_STORAGE_INDEX_COUNT = 1024  # 32*32 chunks
INDEXED_STORAGE_SEGMENT_SIZE = 4096

# World config
WORLD_CONFIG_VERSION = 4
BLOCK_SECTION_VERSION = 6


def index_column(x: int, z: int) -> int:
    """Calculate column index from local x,z coordinates."""
    return (z & 0x1F) << 5 | (x & 0x1F)


def x_from_column(index: int) -> int:
    """Extract x coordinate from column index."""
    return index & 0x1F


def z_from_column(index: int) -> int:
    """Extract z coordinate from column index."""
    return (index >> 5) & 0x1F


def index_section(y: int) -> int:
    """Calculate section index from y coordinate."""
    return y >> 5


def index_block(x: int, y: int, z: int) -> int:
    """Calculate block index within a section from local x,y,z coordinates.

    This is the primary indexing function for block data within sections.
    Y occupies bits 10-14, Z occupies bits 5-9, X occupies bits 0-4.
    """
    return (y & 0x1F) << 10 | (z & 0x1F) << 5 | (x & 0x1F)


def index_block_from_column(column: int, y: int) -> int:
    """Calculate block index from column index and y coordinate."""
    return (y & 0x1F) << 10 | (column & 0x3FF)


def x_from_index(index: int) -> int:
    """Extract x coordinate from block index."""
    return index & 0x1F


def y_from_index(index: int) -> int:
    """Extract y coordinate from block index."""
    return (index >> 10) & 0x1F


def z_from_index(index: int) -> int:
    """Extract z coordinate from block index."""
    return (index >> 5) & 0x1F


def chunk_coordinate(block: int) -> int:
    """Calculate chunk coordinate from block coordinate."""
    return block >> 5


def min_block(chunk_index: int) -> int:
    """Get minimum block coordinate for a chunk index."""
    return chunk_index << 5


def max_block(chunk_index: int) -> int:
    """Get maximum block coordinate for a chunk index."""
    return (chunk_index << 5) + 31


def is_within_local_chunk(x: int, z: int) -> bool:
    """Check if coordinates are within a local chunk (0-31)."""
    return 0 <= x < 32 and 0 <= z < 32


def is_border_block(x: int, z: int) -> bool:
    """Check if a block is on the border of a chunk."""
    return x == 0 or z == 0 or x == 31 or z == 31
