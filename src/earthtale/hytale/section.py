"""BlockSection serialization for Hytale world format (Version 6).

A BlockSection represents a 32x32x32 cube of blocks within a chunk.
Each chunk has 10 sections (totaling 320 blocks high).

Serialization format (v6):
1. Block migration version (int, 4 bytes)
2. Palette type (byte) + palette data
3. If not empty: ticking blocks (count as short, length as short, BitSet as longs)
4. Filler palette type (byte) + data
5. Rotation palette type (byte) + data
6. Local light data
7. Global light data
8. Change counters (2 shorts)
"""

import struct
from typing import List, Optional

from .constants import BLOCKS_PER_SECTION, BLOCK_SECTION_VERSION
from .palette import (
    SectionPalette,
    EmptyPalette,
    HalfBytePalette,
    BytePalette,
    ShortPalette,
    PaletteType,
    create_palette_from_blocks,
)
from .blocks import get_default_registry


class ChunkLightData:
    """Light data for a section (local or global)."""

    EMPTY = None  # Will be set after class definition

    def __init__(self, data: Optional[bytes] = None, change_id: int = 0):
        self.data = data or b""
        self.change_id = change_id

    def serialize(self) -> bytes:
        """Serialize light data."""
        buf = bytearray()
        if self.data:
            buf.extend(struct.pack(">H", len(self.data)))
            buf.extend(self.data)
        else:
            buf.extend(struct.pack(">H", 0))
        buf.extend(struct.pack(">H", self.change_id))
        return bytes(buf)

    @classmethod
    def empty(cls) -> 'ChunkLightData':
        """Create empty light data."""
        return cls(None, 0)


ChunkLightData.EMPTY = ChunkLightData.empty()


def serialize_block_key(block_id: int) -> bytes:
    """Serialize a block ID as a key (UTF string format).

    In Hytale, block keys are stored as UTF strings with the actual block name.
    We look up the Hytale block name from the registry.
    """
    # Get the Hytale block name from registry
    registry = get_default_registry()
    block_name = registry.get_name(block_id)

    # Write as UTF string: 2-byte length (big-endian) + string bytes
    key_bytes = block_name.encode("utf-8")
    return struct.pack(">H", len(key_bytes)) + key_bytes


def serialize_filler_key(filler: int) -> bytes:
    """Serialize a filler value (unsigned short)."""
    return struct.pack(">H", filler)


def serialize_rotation_key(rotation: int) -> bytes:
    """Serialize a rotation value (unsigned byte)."""
    return struct.pack("B", rotation)


class BlockSection:
    """A 32x32x32 section of blocks within a chunk.

    Attributes:
        chunk_section: The main block palette
        filler_section: Filler data palette (for large blocks)
        rotation_section: Block rotation palette
        local_light: Local lighting data
        global_light: Global lighting data
    """

    def __init__(self):
        self.chunk_section: SectionPalette = EmptyPalette()
        self.filler_section: SectionPalette = EmptyPalette()
        self.rotation_section: SectionPalette = EmptyPalette()
        self.ticking_blocks: set = set()
        self.local_light = ChunkLightData.EMPTY
        self.global_light = ChunkLightData.EMPTY
        self.local_change_counter = 0
        self.global_change_counter = 0

    def get(self, x: int, y: int, z: int) -> int:
        """Get block ID at local coordinates."""
        from .constants import index_block
        return self.chunk_section.get(index_block(x, y, z))

    def get_by_index(self, index: int) -> int:
        """Get block ID by linear index."""
        return self.chunk_section.get(index)

    def set(self, x: int, y: int, z: int, block_id: int, rotation: int = 0, filler: int = 0) -> None:
        """Set block at local coordinates."""
        from .constants import index_block
        idx = index_block(x, y, z)
        self.chunk_section = self.chunk_section.set(idx, block_id)
        if rotation != 0:
            self.rotation_section = self.rotation_section.set(idx, rotation)
        if filler != 0:
            self.filler_section = self.filler_section.set(idx, filler)

    def set_by_index(self, index: int, block_id: int, rotation: int = 0, filler: int = 0) -> None:
        """Set block by linear index."""
        self.chunk_section = self.chunk_section.set(index, block_id)
        if rotation != 0:
            self.rotation_section = self.rotation_section.set(index, rotation)
        if filler != 0:
            self.filler_section = self.filler_section.set(index, filler)

    def set_from_blocks(self, blocks: List[int]) -> None:
        """Set all blocks from a list of block IDs."""
        self.chunk_section = create_palette_from_blocks(blocks)

    def is_empty(self) -> bool:
        """Check if section is entirely air."""
        return self.chunk_section.is_solid(0)

    def serialize(self, block_migration_version: int = 0) -> bytes:
        """Serialize the section to bytes in Hytale format (v6)."""
        buf = bytearray()

        # 1. Block migration version (int)
        buf.extend(struct.pack(">i", block_migration_version))

        # 2. Palette type + data
        buf.append(self.chunk_section.palette_type.value)
        buf.extend(self.chunk_section.serialize(serialize_block_key))

        # 3. Ticking blocks (if not empty palette)
        if self.chunk_section.palette_type != PaletteType.EMPTY:
            # Ticking blocks count (short)
            buf.extend(struct.pack(">H", len(self.ticking_blocks)))
            # BitSet as longs
            if self.ticking_blocks:
                # Convert set to BitSet representation
                max_index = max(self.ticking_blocks) if self.ticking_blocks else -1
                num_longs = (max_index // 64) + 1 if max_index >= 0 else 0
                longs = [0] * num_longs
                for idx in self.ticking_blocks:
                    longs[idx // 64] |= (1 << (idx % 64))
                buf.extend(struct.pack(">H", num_longs))
                for lng in longs:
                    buf.extend(struct.pack(">Q", lng))
            else:
                buf.extend(struct.pack(">H", 0))

        # 4. Filler palette type + data
        buf.append(self.filler_section.palette_type.value)
        buf.extend(self.filler_section.serialize(serialize_filler_key))

        # 5. Rotation palette type + data
        buf.append(self.rotation_section.palette_type.value)
        buf.extend(self.rotation_section.serialize(serialize_rotation_key))

        # 6. Local light data
        buf.extend(self.local_light.serialize())

        # 7. Global light data
        buf.extend(self.global_light.serialize())

        # 8. Change counters
        buf.extend(struct.pack(">h", self.local_change_counter))
        buf.extend(struct.pack(">h", self.global_change_counter))

        return bytes(buf)


def create_section_from_column(
    blocks: List[List[int]],
    section_y: int
) -> BlockSection:
    """Create a section from a 2D array of block columns.

    Args:
        blocks: 32x32 array of block columns (each column is a list of 320 blocks)
        section_y: Section index (0-9)

    Returns:
        BlockSection with the appropriate blocks set
    """
    from .constants import index_block

    section = BlockSection()
    y_start = section_y * 32
    y_end = y_start + 32

    section_blocks = [0] * BLOCKS_PER_SECTION
    for x in range(32):
        for z in range(32):
            column = blocks[x][z]
            for local_y in range(32):
                world_y = y_start + local_y
                if world_y < len(column):
                    block_id = column[world_y]
                    idx = index_block(x, local_y, z)
                    section_blocks[idx] = block_id

    section.set_from_blocks(section_blocks)
    return section
