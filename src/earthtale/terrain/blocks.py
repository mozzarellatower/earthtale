"""Block column generation for terrain.

This module generates vertical columns of blocks based on:
- Surface elevation
- Biome type
- Depth layers (surface, subsurface, rock, bedrock)
"""

from dataclasses import dataclass, field
from typing import List, Optional
import random

from .biome import BiomeType, get_surface_block, get_subsurface_block
from ..hytale.blocks import BlockRegistry, get_default_registry, get_hytale_name


@dataclass
class ColumnConfig:
    """Configuration for block column generation."""
    # Layer depths
    surface_depth: int = 1  # Surface block layer
    subsurface_depth: int = 3  # Dirt/sand layer below surface
    transition_depth: int = 4  # Mixed stone/subsurface layer
    bedrock_depth: int = 3  # Bedrock layer at bottom

    # Block variation
    add_stone_variation: bool = True  # Mix in gravel, etc.
    variation_chance: float = 0.1  # Chance of variation blocks

    # Water
    water_level: int = 64  # Sea level Y coordinate
    fill_water: bool = True  # Fill below water level with water


@dataclass
class BlockColumn:
    """A vertical column of blocks."""
    x: int
    z: int
    blocks: List[int] = field(default_factory=list)
    surface_y: int = 0
    biome: BiomeType = BiomeType.GRASSLAND

    def get_block(self, y: int) -> int:
        """Get block ID at Y coordinate."""
        if 0 <= y < len(self.blocks):
            return self.blocks[y]
        return 0  # Air

    def set_block(self, y: int, block_id: int) -> None:
        """Set block ID at Y coordinate."""
        # Extend list if needed
        while len(self.blocks) <= y:
            self.blocks.append(0)
        self.blocks[y] = block_id


class BlockColumnGenerator:
    """Generates block columns from terrain data."""

    def __init__(
        self,
        registry: Optional[BlockRegistry] = None,
        config: Optional[ColumnConfig] = None
    ):
        """Initialize the generator.

        Args:
            registry: Block registry for name to ID mapping
            config: Configuration for column generation
        """
        self.registry = registry or get_default_registry()
        self.config = config or ColumnConfig()

        # Cache block IDs - map legacy names to Hytale names first
        self._block_ids = {
            "Air": 0,  # "Empty" in Hytale
            "Stone": self.registry.get_id(get_hytale_name("Stone")),
            "Dirt": self.registry.get_id(get_hytale_name("Dirt")),
            "Grass": self.registry.get_id(get_hytale_name("Grass")),
            "Sand": self.registry.get_id(get_hytale_name("Sand")),
            "Sandstone": self.registry.get_id(get_hytale_name("Sandstone")),
            "Gravel": self.registry.get_id(get_hytale_name("Gravel")),
            "Water": self.registry.get_id(get_hytale_name("Water")),
            "Snow": self.registry.get_id(get_hytale_name("Snow")),
            "Ice": self.registry.get_id(get_hytale_name("Ice")),
            "Clay": self.registry.get_id(get_hytale_name("Clay")),
            "Bedrock": self.registry.get_id(get_hytale_name("Bedrock")),
        }

    def _get_block_id(self, name: str) -> int:
        """Get block ID from name, with caching."""
        if name in self._block_ids:
            return self._block_ids[name]
        block_id = self.registry.get_id(name)
        self._block_ids[name] = block_id
        return block_id

    def generate_column(
        self,
        x: int,
        z: int,
        surface_y: int,
        biome: BiomeType,
        seed: Optional[int] = None
    ) -> BlockColumn:
        """Generate a block column.

        Args:
            x: Column X coordinate
            z: Column Z coordinate
            surface_y: Y coordinate of the surface
            biome: Biome type for block selection
            seed: Optional random seed for variation

        Returns:
            BlockColumn with blocks filled in
        """
        column = BlockColumn(x=x, z=z, surface_y=surface_y, biome=biome)

        # Initialize random for variation
        if seed is not None:
            rng = random.Random(seed + x * 31 + z * 17)
        else:
            rng = random.Random(x * 31 + z * 17)

        # Get block types for this biome
        surface_block = self._get_block_id(get_surface_block(biome))
        subsurface_block = self._get_block_id(get_subsurface_block(biome))
        stone_block = self._get_block_id("Stone")
        bedrock_block = self._get_block_id("Bedrock")
        gravel_block = self._get_block_id("Gravel")
        water_block = self._get_block_id("Water")

        # Special handling for water biomes
        is_water_biome = biome in (BiomeType.OCEAN, BiomeType.RIVER, BiomeType.LAKE)

        # Build the column from bottom to top
        max_y = max(surface_y, self.config.water_level) + 1

        for y in range(max_y):
            if y < self.config.bedrock_depth:
                # Bedrock layer
                block = bedrock_block
            elif y < surface_y - self.config.transition_depth - self.config.subsurface_depth:
                # Deep stone
                if self.config.add_stone_variation and rng.random() < self.config.variation_chance:
                    block = gravel_block
                else:
                    block = stone_block
            elif y < surface_y - self.config.subsurface_depth:
                # Transition layer (mixed stone and subsurface)
                if rng.random() < 0.5:
                    block = stone_block
                else:
                    block = subsurface_block
            elif y < surface_y:
                # Subsurface layer
                block = subsurface_block
            elif y == surface_y:
                # Surface layer
                if is_water_biome and y < self.config.water_level:
                    block = subsurface_block  # Underwater surface
                else:
                    block = surface_block
            elif y <= self.config.water_level and self.config.fill_water:
                # Water above surface (if below water level)
                block = water_block
            else:
                # Air
                block = 0

            column.set_block(y, block)

        return column

    def generate_columns_for_chunk(
        self,
        chunk_x: int,
        chunk_z: int,
        elevations: List[List[int]],
        biomes: List[List[BiomeType]],
        seed: Optional[int] = None
    ) -> List[List[BlockColumn]]:
        """Generate all columns for a chunk.

        Args:
            chunk_x: Chunk X coordinate
            chunk_z: Chunk Z coordinate
            elevations: 32x32 array of surface Y coordinates
            biomes: 32x32 array of biome types
            seed: Optional random seed

        Returns:
            32x32 array of BlockColumns
        """
        columns = []

        for local_x in range(32):
            row = []
            for local_z in range(32):
                world_x = chunk_x * 32 + local_x
                world_z = chunk_z * 32 + local_z

                column = self.generate_column(
                    x=world_x,
                    z=world_z,
                    surface_y=elevations[local_x][local_z],
                    biome=biomes[local_x][local_z],
                    seed=seed
                )
                row.append(column)
            columns.append(row)

        return columns

    def columns_to_block_lists(
        self,
        columns: List[List[BlockColumn]]
    ) -> List[List[int]]:
        """Convert BlockColumns to simple block ID lists.

        Args:
            columns: 32x32 array of BlockColumns

        Returns:
            Flat list of 1024 block columns (each is a list of block IDs)
        """
        result = []
        for x in range(32):
            for z in range(32):
                result.append(columns[x][z].blocks)
        return result


def generate_flat_world(
    width_chunks: int,
    height_chunks: int,
    surface_y: int = 64,
    biome: BiomeType = BiomeType.GRASSLAND,
    registry: Optional[BlockRegistry] = None
) -> List[BlockColumn]:
    """Generate a flat world of block columns.

    Args:
        width_chunks: Width in chunks
        height_chunks: Height in chunks
        surface_y: Surface Y level
        biome: Biome type for all blocks
        registry: Optional block registry

    Returns:
        List of all BlockColumns
    """
    generator = BlockColumnGenerator(registry)
    columns = []

    for chunk_x in range(width_chunks):
        for chunk_z in range(height_chunks):
            for local_x in range(32):
                for local_z in range(32):
                    world_x = chunk_x * 32 + local_x
                    world_z = chunk_z * 32 + local_z
                    column = generator.generate_column(
                        x=world_x,
                        z=world_z,
                        surface_y=surface_y,
                        biome=biome
                    )
                    columns.append(column)

    return columns
