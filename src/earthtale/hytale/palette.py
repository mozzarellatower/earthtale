"""Section palette compression for Hytale blocks.

Hytale uses a palette system to compress block data:
- Empty: Section is entirely one block type (usually air)
- HalfByte: Up to 16 unique blocks, 4 bits per block (16KB)
- Byte: Up to 256 unique blocks, 8 bits per block (32KB)
- Short: Up to 65536 unique blocks, 16 bits per block (64KB)

The palette maps internal IDs (compact) to external IDs (actual block IDs).
"""

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
import struct

from .constants import BLOCKS_PER_SECTION


class PaletteType(IntEnum):
    """Palette type enumeration matching Hytale's PaletteType."""
    EMPTY = 0
    HALF_BYTE = 1
    BYTE = 2
    SHORT = 3


class SectionPalette(ABC):
    """Abstract base class for section palettes."""

    @property
    @abstractmethod
    def palette_type(self) -> PaletteType:
        """Get the palette type."""
        pass

    @abstractmethod
    def get(self, index: int) -> int:
        """Get external block ID at index."""
        pass

    @abstractmethod
    def set(self, index: int, block_id: int) -> 'SectionPalette':
        """Set block at index, potentially returning a promoted palette."""
        pass

    @abstractmethod
    def serialize(self, key_serializer) -> bytes:
        """Serialize palette to bytes."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return number of unique block types."""
        pass

    def is_solid(self, block_id: int) -> bool:
        """Check if the section is entirely one block type."""
        return False


class EmptyPalette(SectionPalette):
    """Palette for sections that are entirely one block type (air by default)."""

    def __init__(self, block_id: int = 0):
        self._block_id = block_id

    @property
    def palette_type(self) -> PaletteType:
        return PaletteType.EMPTY

    def get(self, index: int) -> int:
        return self._block_id

    def set(self, index: int, block_id: int) -> SectionPalette:
        if block_id == self._block_id:
            return self
        # Need to promote to HalfByte palette
        palette = HalfBytePalette()
        # Fill with current block ID
        if self._block_id != 0:
            for i in range(BLOCKS_PER_SECTION):
                palette._set_internal(i, palette._get_or_create_internal(self._block_id))
        # Set the new block
        return palette.set(index, block_id)

    def serialize(self, key_serializer) -> bytes:
        # Empty palette has no data - it's just the type byte
        return b""

    def count(self) -> int:
        return 1

    def is_solid(self, block_id: int) -> bool:
        return self._block_id == block_id


class HalfBytePalette(SectionPalette):
    """Palette using 4 bits per block, supporting up to 16 unique blocks."""

    MAX_UNIQUE = 16

    def __init__(self):
        # 4 bits per block = 16384 bytes for 32768 blocks
        self._blocks = bytearray(BLOCKS_PER_SECTION // 2)
        self._external_to_internal: Dict[int, int] = {0: 0}
        self._internal_to_external: Dict[int, int] = {0: 0}
        self._internal_counts: Dict[int, int] = {0: BLOCKS_PER_SECTION}
        self._next_internal = 1

    @property
    def palette_type(self) -> PaletteType:
        return PaletteType.HALF_BYTE

    def _get_nibble(self, index: int) -> int:
        """Get 4-bit value at index."""
        byte_idx = index // 2
        if index % 2 == 0:
            return self._blocks[byte_idx] & 0x0F
        else:
            return (self._blocks[byte_idx] >> 4) & 0x0F

    def _set_nibble(self, index: int, value: int) -> None:
        """Set 4-bit value at index."""
        byte_idx = index // 2
        if index % 2 == 0:
            self._blocks[byte_idx] = (self._blocks[byte_idx] & 0xF0) | (value & 0x0F)
        else:
            self._blocks[byte_idx] = (self._blocks[byte_idx] & 0x0F) | ((value & 0x0F) << 4)

    def _get_or_create_internal(self, external_id: int) -> Optional[int]:
        """Get or create internal ID for external ID."""
        if external_id in self._external_to_internal:
            return self._external_to_internal[external_id]
        if len(self._external_to_internal) >= self.MAX_UNIQUE:
            return None  # Need promotion
        internal = self._next_internal
        self._next_internal += 1
        self._external_to_internal[external_id] = internal
        self._internal_to_external[internal] = external_id
        self._internal_counts[internal] = 0
        return internal

    def _set_internal(self, index: int, internal_id: int) -> None:
        """Set internal ID at index."""
        old_internal = self._get_nibble(index)
        if old_internal == internal_id:
            return
        # Update counts
        self._internal_counts[old_internal] -= 1
        self._internal_counts[internal_id] = self._internal_counts.get(internal_id, 0) + 1
        # Set new value
        self._set_nibble(index, internal_id)

    def get(self, index: int) -> int:
        internal = self._get_nibble(index)
        return self._internal_to_external.get(internal, 0)

    def set(self, index: int, block_id: int) -> SectionPalette:
        internal = self._get_or_create_internal(block_id)
        if internal is None:
            # Need to promote to Byte palette
            return self._promote().set(index, block_id)
        self._set_internal(index, internal)
        return self

    def _promote(self) -> 'BytePalette':
        """Promote to BytePalette."""
        palette = BytePalette()
        palette._external_to_internal = dict(self._external_to_internal)
        palette._internal_to_external = dict(self._internal_to_external)
        palette._internal_counts = dict(self._internal_counts)
        palette._next_internal = self._next_internal
        for i in range(BLOCKS_PER_SECTION):
            palette._blocks[i] = self._get_nibble(i)
        return palette

    def serialize(self, key_serializer) -> bytes:
        """Serialize palette to bytes."""
        buf = bytearray()
        # Palette size (2 bytes, big endian)
        buf.extend(struct.pack(">H", len(self._internal_to_external)))
        # Palette entries
        for internal_id, external_id in self._internal_to_external.items():
            buf.append(internal_id)
            buf.extend(key_serializer(external_id))
            # Handle signed short overflow: 32768 becomes -32768
            count = self._internal_counts.get(internal_id, 0)
            if count > 32767:
                count = count - 65536
            buf.extend(struct.pack(">h", count))
        # Block data
        buf.extend(self._blocks)
        return bytes(buf)

    def count(self) -> int:
        return len(self._internal_to_external)

    def is_solid(self, block_id: int) -> bool:
        if block_id not in self._external_to_internal:
            return False
        internal = self._external_to_internal[block_id]
        return self._internal_counts.get(internal, 0) == BLOCKS_PER_SECTION


class BytePalette(SectionPalette):
    """Palette using 8 bits per block, supporting up to 256 unique blocks."""

    MAX_UNIQUE = 256
    DEMOTE_THRESHOLD = 14

    def __init__(self):
        self._blocks = bytearray(BLOCKS_PER_SECTION)
        self._external_to_internal: Dict[int, int] = {0: 0}
        self._internal_to_external: Dict[int, int] = {0: 0}
        self._internal_counts: Dict[int, int] = {0: BLOCKS_PER_SECTION}
        self._next_internal = 1

    @property
    def palette_type(self) -> PaletteType:
        return PaletteType.BYTE

    def _get_or_create_internal(self, external_id: int) -> Optional[int]:
        """Get or create internal ID for external ID."""
        if external_id in self._external_to_internal:
            return self._external_to_internal[external_id]
        if len(self._external_to_internal) >= self.MAX_UNIQUE:
            return None  # Need promotion
        internal = self._next_internal
        self._next_internal += 1
        self._external_to_internal[external_id] = internal
        self._internal_to_external[internal] = external_id
        self._internal_counts[internal] = 0
        return internal

    def get(self, index: int) -> int:
        internal = self._blocks[index]
        return self._internal_to_external.get(internal, 0)

    def set(self, index: int, block_id: int) -> SectionPalette:
        internal = self._get_or_create_internal(block_id)
        if internal is None:
            # Need to promote to Short palette
            return self._promote().set(index, block_id)
        old_internal = self._blocks[index]
        if old_internal != internal:
            self._internal_counts[old_internal] -= 1
            self._internal_counts[internal] = self._internal_counts.get(internal, 0) + 1
            self._blocks[index] = internal
        return self

    def _promote(self) -> 'ShortPalette':
        """Promote to ShortPalette."""
        palette = ShortPalette()
        palette._external_to_internal = dict(self._external_to_internal)
        palette._internal_to_external = dict(self._internal_to_external)
        palette._internal_counts = dict(self._internal_counts)
        palette._next_internal = self._next_internal
        for i in range(BLOCKS_PER_SECTION):
            palette._blocks[i] = self._blocks[i]
        return palette

    def serialize(self, key_serializer) -> bytes:
        """Serialize palette to bytes."""
        buf = bytearray()
        # Palette size (2 bytes, big endian)
        buf.extend(struct.pack(">H", len(self._internal_to_external)))
        # Palette entries
        for internal_id, external_id in self._internal_to_external.items():
            buf.append(internal_id)
            buf.extend(key_serializer(external_id))
            # Handle signed short overflow: 32768 becomes -32768
            count = self._internal_counts.get(internal_id, 0)
            if count > 32767:
                count = count - 65536
            buf.extend(struct.pack(">h", count))
        # Block data
        buf.extend(self._blocks)
        return bytes(buf)

    def count(self) -> int:
        return len(self._internal_to_external)

    def is_solid(self, block_id: int) -> bool:
        if block_id not in self._external_to_internal:
            return False
        internal = self._external_to_internal[block_id]
        return self._internal_counts.get(internal, 0) == BLOCKS_PER_SECTION


class ShortPalette(SectionPalette):
    """Palette using 16 bits per block, supporting up to 65536 unique blocks."""

    MAX_UNIQUE = 65536
    DEMOTE_THRESHOLD = 251

    def __init__(self):
        self._blocks = [0] * BLOCKS_PER_SECTION  # shorts stored as ints
        self._external_to_internal: Dict[int, int] = {0: 0}
        self._internal_to_external: Dict[int, int] = {0: 0}
        self._internal_counts: Dict[int, int] = {0: BLOCKS_PER_SECTION}
        self._next_internal = 1

    @property
    def palette_type(self) -> PaletteType:
        return PaletteType.SHORT

    def _get_or_create_internal(self, external_id: int) -> Optional[int]:
        """Get or create internal ID for external ID."""
        if external_id in self._external_to_internal:
            return self._external_to_internal[external_id]
        if len(self._external_to_internal) >= self.MAX_UNIQUE:
            raise ValueError("Exceeded maximum unique blocks for short palette")
        internal = self._next_internal
        self._next_internal += 1
        self._external_to_internal[external_id] = internal
        self._internal_to_external[internal] = external_id
        self._internal_counts[internal] = 0
        return internal

    def get(self, index: int) -> int:
        internal = self._blocks[index]
        return self._internal_to_external.get(internal, 0)

    def set(self, index: int, block_id: int) -> SectionPalette:
        internal = self._get_or_create_internal(block_id)
        old_internal = self._blocks[index]
        if old_internal != internal:
            self._internal_counts[old_internal] -= 1
            self._internal_counts[internal] = self._internal_counts.get(internal, 0) + 1
            self._blocks[index] = internal
        return self

    def serialize(self, key_serializer) -> bytes:
        """Serialize palette to bytes."""
        buf = bytearray()
        # Palette size (2 bytes, big endian)
        buf.extend(struct.pack(">H", len(self._internal_to_external)))
        # Palette entries
        for internal_id, external_id in self._internal_to_external.items():
            buf.extend(struct.pack(">H", internal_id & 0xFFFF))
            buf.extend(key_serializer(external_id))
            # Handle signed short overflow: 32768 becomes -32768
            count = self._internal_counts.get(internal_id, 0)
            if count > 32767:
                count = count - 65536
            buf.extend(struct.pack(">h", count))
        # Block data
        for block in self._blocks:
            buf.extend(struct.pack(">H", block & 0xFFFF))
        return bytes(buf)

    def count(self) -> int:
        return len(self._internal_to_external)

    def is_solid(self, block_id: int) -> bool:
        if block_id not in self._external_to_internal:
            return False
        internal = self._external_to_internal[block_id]
        return self._internal_counts.get(internal, 0) == BLOCKS_PER_SECTION


def create_palette_from_blocks(blocks: List[int]) -> SectionPalette:
    """Create an appropriate palette from a list of block IDs.

    Args:
        blocks: List of BLOCKS_PER_SECTION block IDs

    Returns:
        The most compact palette that can store the blocks
    """
    if len(blocks) != BLOCKS_PER_SECTION:
        raise ValueError(f"Expected {BLOCKS_PER_SECTION} blocks, got {len(blocks)}")

    # Count unique blocks
    unique = set(blocks)

    # Check if all one block
    if len(unique) == 1:
        return EmptyPalette(blocks[0])

    # Choose palette type based on unique count
    if len(unique) <= HalfBytePalette.MAX_UNIQUE:
        palette = HalfBytePalette()
    elif len(unique) <= BytePalette.MAX_UNIQUE:
        palette = BytePalette()
    else:
        palette = ShortPalette()

    # Set all blocks
    for i, block_id in enumerate(blocks):
        palette = palette.set(i, block_id)

    return palette
