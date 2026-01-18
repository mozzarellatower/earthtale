"""Chunk data management and BSON serialization for Hytale world format.

A chunk is a 32x32x320 column of blocks divided into 10 sections.
Chunks are stored in region files using BSON serialization.
"""

import struct
import base64
from typing import List, Optional, Dict, Any

from .constants import (
    CHUNK_SIZE,
    SECTIONS_PER_CHUNK,
    WORLD_HEIGHT,
    BLOCK_SECTION_VERSION,
    index_block,
    index_section,
)
from .section import BlockSection, create_section_from_column

try:
    from . import template_data as _template_data
except Exception:
    _template_data = None


def _decode_template_data() -> Dict[str, Optional[bytes]]:
    if _template_data is None:
        return {"block": None, "env": None, "health": None}
    try:
        block = base64.b64decode(_template_data.BLOCK_CHUNK_DATA_B64.encode("ascii"))
        env = base64.b64decode(_template_data.ENVIRONMENT_CHUNK_DATA_B64.encode("ascii"))
        health = base64.b64decode(_template_data.BLOCK_HEALTH_DATA_B64.encode("ascii"))
        return {"block": block, "env": env, "health": health}
    except Exception:
        return {"block": None, "env": None, "health": None}


_TEMPLATE_DATA = _decode_template_data()


def encode_bson_document(doc: Dict[str, Any]) -> bytes:
    """Encode a dictionary as BSON.

    This is a minimal BSON encoder for chunk data serialization.
    """
    # BSON format: int32 total_size + elements + \x00
    buf = bytearray()

    for key, value in doc.items():
        key_bytes = key.encode("utf-8") + b"\x00"

        if isinstance(value, bytes):
            # Binary data: type 0x05
            buf.append(0x05)
            buf.extend(key_bytes)
            # int32 length + byte subtype + data
            buf.extend(struct.pack("<i", len(value)))
            buf.append(0x00)  # Generic binary subtype
            buf.extend(value)
        elif isinstance(value, int):
            # Int32: type 0x10
            buf.append(0x10)
            buf.extend(key_bytes)
            buf.extend(struct.pack("<i", value))
        elif isinstance(value, str):
            # String: type 0x02
            buf.append(0x02)
            buf.extend(key_bytes)
            str_bytes = value.encode("utf-8") + b"\x00"
            buf.extend(struct.pack("<i", len(str_bytes)))
            buf.extend(str_bytes)
        elif isinstance(value, dict):
            # Embedded document: type 0x03
            buf.append(0x03)
            buf.extend(key_bytes)
            buf.extend(encode_bson_document(value))
        elif isinstance(value, list):
            # Array: type 0x04
            buf.append(0x04)
            buf.extend(key_bytes)
            # Convert list to dict with string indices
            array_doc = {str(i): v for i, v in enumerate(value)}
            buf.extend(encode_bson_document(array_doc))
        elif isinstance(value, bool):
            # Boolean: type 0x08
            buf.append(0x08)
            buf.extend(key_bytes)
            buf.append(0x01 if value else 0x00)

    # Null terminator
    buf.append(0x00)

    # Prepend total size (including the size field itself)
    total_size = len(buf) + 4
    return struct.pack("<i", total_size) + buf


class Chunk:
    """A chunk containing 10 sections of blocks.

    Attributes:
        x: Chunk X coordinate
        z: Chunk Z coordinate
        sections: List of 10 BlockSections
    """

    def __init__(self, x: int, z: int):
        self.x = x
        self.z = z
        self.sections: List[BlockSection] = [BlockSection() for _ in range(SECTIONS_PER_CHUNK)]

    def get_section(self, section_index: int) -> BlockSection:
        """Get a section by index (0-9)."""
        if not 0 <= section_index < SECTIONS_PER_CHUNK:
            raise ValueError(f"Section index must be 0-{SECTIONS_PER_CHUNK-1}")
        return self.sections[section_index]

    def get_block(self, x: int, y: int, z: int) -> int:
        """Get block at local coordinates (0-31, 0-319, 0-31)."""
        if not (0 <= x < CHUNK_SIZE and 0 <= z < CHUNK_SIZE and 0 <= y < WORLD_HEIGHT):
            raise ValueError(f"Coordinates out of range: ({x}, {y}, {z})")
        section_idx = index_section(y)
        local_y = y & 0x1F
        return self.sections[section_idx].get(x, local_y, z)

    def set_block(self, x: int, y: int, z: int, block_id: int, rotation: int = 0, filler: int = 0) -> None:
        """Set block at local coordinates (0-31, 0-319, 0-31)."""
        if not (0 <= x < CHUNK_SIZE and 0 <= z < CHUNK_SIZE and 0 <= y < WORLD_HEIGHT):
            raise ValueError(f"Coordinates out of range: ({x}, {y}, {z})")
        section_idx = index_section(y)
        local_y = y & 0x1F
        self.sections[section_idx].set(x, local_y, z, block_id, rotation, filler)

    def set_column(self, x: int, z: int, blocks: List[int]) -> None:
        """Set an entire column of blocks (up to 320 blocks)."""
        if not (0 <= x < CHUNK_SIZE and 0 <= z < CHUNK_SIZE):
            raise ValueError(f"Coordinates out of range: ({x}, {z})")
        for y, block_id in enumerate(blocks[:WORLD_HEIGHT]):
            self.set_block(x, y, z, block_id)

    def is_empty(self) -> bool:
        """Check if all sections are empty (air only)."""
        return all(section.is_empty() for section in self.sections)

    def serialize_sections(self, block_migration_version: int = 0) -> List[bytes]:
        """Serialize all sections to bytes."""
        return [section.serialize(block_migration_version) for section in self.sections]

    def to_bson(self, block_migration_version: int = 0) -> bytes:
        """Serialize chunk to BSON format matching Hytale's component structure.

        The chunk is serialized with a Components wrapper containing:
        - WorldChunk: empty (marker component)
        - BlockChunk: Version + Data (heightmap and tint data)
        - ChunkColumn: Sections array with Block components
        - EntityChunk: Entities array (empty)
        - EnvironmentChunk: Data (biome/environment data)
        - BlockComponentChunk: BlockComponents (block entity data)
        - BlockHealthChunk: Data (block damage state)
        """
        # Build sections array for ChunkColumn
        sections_list = []
        for i, section in enumerate(self.sections):
            section_data = section.serialize(block_migration_version)
            section_doc = {
                "Components": {
                    "ChunkSection": {},  # Marker component
                    "Block": {
                        "Version": BLOCK_SECTION_VERSION,
                        "Data": section_data
                    }
                }
            }
            sections_list.append(section_doc)

        block_chunk_data = _TEMPLATE_DATA["block"]
        if block_chunk_data is None:
            # Fallback: heightmap + tint layout (may not match server expectations).
            block_chunk_data = bytearray()
            block_chunk_data.extend(struct.pack(">i", 3))  # Version 3
            for _ in range(1024):
                block_chunk_data.extend(struct.pack(">h", 64))
            for _ in range(1024):
                block_chunk_data.extend(struct.pack(">i", 0))
            block_chunk_data = bytes(block_chunk_data)

        env_data = _TEMPLATE_DATA["env"]
        if env_data is None:
            env_data = bytes(32 * 32 * 10)

        block_health_data = _TEMPLATE_DATA["health"]
        if block_health_data is None:
            block_health_data = bytes(1)

        # Build the full Components document
        doc = {
            "Components": {
                "WorldChunk": {},
                "BlockChunk": {
                    "Version": 3,
                    "Data": block_chunk_data
                },
                "ChunkColumn": {
                    "Sections": sections_list
                },
                "EntityChunk": {
                    "Entities": []
                },
                "EnvironmentChunk": {
                    "Data": env_data
                },
                "BlockComponentChunk": {
                    "BlockComponents": {}
                },
                "BlockHealthChunk": {
                    "Data": block_health_data
                }
            }
        }

        return encode_bson_document(doc)


def create_chunk_from_columns(
    chunk_x: int,
    chunk_z: int,
    columns: List[List[int]]
) -> Chunk:
    """Create a chunk from a 2D array of block columns.

    Args:
        chunk_x: Chunk X coordinate
        chunk_z: Chunk Z coordinate
        columns: Flat list of 1024 columns (32x32), each column is a list of blocks

    Returns:
        Chunk with all sections populated
    """
    if len(columns) != CHUNK_SIZE * CHUNK_SIZE:
        raise ValueError(f"Expected {CHUNK_SIZE * CHUNK_SIZE} columns")

    chunk = Chunk(chunk_x, chunk_z)

    # Reorganize columns into 2D array
    columns_2d = [[None for _ in range(CHUNK_SIZE)] for _ in range(CHUNK_SIZE)]
    for i, col in enumerate(columns):
        x = i % CHUNK_SIZE
        z = i // CHUNK_SIZE
        columns_2d[x][z] = col

    # Create each section
    for section_y in range(SECTIONS_PER_CHUNK):
        chunk.sections[section_y] = create_section_from_column(columns_2d, section_y)

    return chunk
