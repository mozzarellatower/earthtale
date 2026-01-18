"""Block ID registry for Hytale blocks.

This module provides a mapping between block names and their numeric IDs.
Block IDs are used internally in the palette system.

IMPORTANT: Block names must match Hytale's actual asset names exactly.
Common Hytale block names:
- "Empty" (air/void)
- "Rock_Basalt", "Rock_Granite", "Rock_Limestone", etc.
- "Dirt_Grassland", "Dirt_Desert", etc.
- "Grass_Grassland", "Grass_Forest", etc.
- "Sand_Beach", "Sand_Desert", etc.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class BlockType:
    """Represents a block type with its properties."""
    id: int
    name: str  # Must match Hytale's exact asset name
    is_solid: bool = True
    is_transparent: bool = False


class BlockRegistry:
    """Registry of known block types.

    Maps internal numeric IDs to Hytale block names.
    """

    def __init__(self):
        self._by_id: Dict[int, BlockType] = {}
        self._by_name: Dict[str, BlockType] = {}
        self._next_id = 1  # 0 is reserved for Empty (air)

        # Register Empty (air) as block 0 - Hytale uses "Empty" not "Air"
        self._register_block(BlockType(0, "Empty", is_solid=False, is_transparent=True))

    def _register_block(self, block: BlockType) -> None:
        """Register a block type."""
        self._by_id[block.id] = block
        self._by_name[block.name] = block

    def register(self, name: str, is_solid: bool = True, is_transparent: bool = False) -> BlockType:
        """Register a new block type and return it."""
        block = BlockType(self._next_id, name, is_solid, is_transparent)
        self._register_block(block)
        self._next_id += 1
        return block

    def get_by_id(self, block_id: int) -> Optional[BlockType]:
        """Get a block type by its ID."""
        return self._by_id.get(block_id)

    def get_by_name(self, name: str) -> Optional[BlockType]:
        """Get a block type by its name."""
        return self._by_name.get(name)

    def get_id(self, name: str) -> int:
        """Get block ID by name, returning 0 (Empty) if not found."""
        block = self._by_name.get(name)
        return block.id if block else 0

    def get_name(self, block_id: int) -> str:
        """Get block name by ID, returning 'Empty' if not found."""
        block = self._by_id.get(block_id)
        return block.name if block else "Empty"


# Default block registry with Hytale-compatible block names
DEFAULT_REGISTRY = BlockRegistry()

# Register common terrain blocks using Hytale's actual asset names
# Block ID 0 is "Empty" (already registered)
BLOCK_AIR = DEFAULT_REGISTRY.get_by_id(0)

# Rock types (ID 1-3)
BLOCK_STONE = DEFAULT_REGISTRY.register("Rock_Basalt")
BLOCK_GRANITE = DEFAULT_REGISTRY.register("Rock_Stone")
BLOCK_LIMESTONE = DEFAULT_REGISTRY.register("Rock_Marble")

# Dirt types (ID 4-5)
BLOCK_DIRT = DEFAULT_REGISTRY.register("Soil_Dirt")
BLOCK_DIRT_DESERT = DEFAULT_REGISTRY.register("Soil_Dirt_Burnt")

# Surface blocks (ID 6-9)
BLOCK_GRASS = DEFAULT_REGISTRY.register("Soil_Grass")
BLOCK_SAND = DEFAULT_REGISTRY.register("Soil_Gravel_Sand")
BLOCK_GRAVEL = DEFAULT_REGISTRY.register("Soil_Gravel")
BLOCK_SNOW = DEFAULT_REGISTRY.register("Soil_Snow")

# Special blocks (ID 10-12)
BLOCK_WATER = DEFAULT_REGISTRY.register("Water_Source", is_solid=False, is_transparent=True)
BLOCK_ICE = DEFAULT_REGISTRY.register("Rock_Ice_Blue", is_transparent=True)
BLOCK_BEDROCK = DEFAULT_REGISTRY.register("Rock_Bedrock")

# Ores (Basalt variants align with default stone)
BLOCK_ORE_COPPER = DEFAULT_REGISTRY.register("Ore_Copper_Basalt")
BLOCK_ORE_IRON = DEFAULT_REGISTRY.register("Ore_Iron_Basalt")
BLOCK_ORE_THORIUM = DEFAULT_REGISTRY.register("Ore_Thorium_Basalt")
BLOCK_ORE_COBALT = DEFAULT_REGISTRY.register("Ore_Cobalt_Basalt")
BLOCK_ORE_ADAMANTITE = DEFAULT_REGISTRY.register("Ore_Adamantite_Basalt")

# Legacy name aliases for compatibility with terrain generator
# Maps old names to new Hytale names
LEGACY_NAME_MAP = {
    "Air": "Empty",
    "Stone": "Rock_Basalt",
    "Dirt": "Soil_Dirt",
    "Grass": "Soil_Grass",
    "Sand": "Soil_Gravel_Sand",
    "Gravel": "Soil_Gravel",
    "Water": "Water_Source",
    "Snow": "Soil_Snow",
    "Ice": "Rock_Ice_Blue",
    "Sandstone": "Rock_Sandstone",
    "Clay": "Soil_Clay",
    "Bedrock": "Rock_Bedrock",
}


def get_default_registry() -> BlockRegistry:
    """Get the default block registry."""
    return DEFAULT_REGISTRY


def get_hytale_name(legacy_name: str) -> str:
    """Convert a legacy block name to Hytale's actual name."""
    return LEGACY_NAME_MAP.get(legacy_name, legacy_name)
