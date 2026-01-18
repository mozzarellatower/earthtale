"""Ore distribution configuration for Minecraft-like generation."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import json


@dataclass
class OreConfig:
    """Configuration for a single ore type."""
    name: str
    block_name: str
    min_y: int
    max_y: int
    chance: float
    vein_min: int
    vein_max: int


DEFAULT_ORE_CONFIGS: List[OreConfig] = [
    OreConfig(
        name="Copper",
        block_name="Ore_Copper_Basalt",
        min_y=20,
        max_y=180,
        chance=0.02,
        vein_min=3,
        vein_max=7,
    ),
    OreConfig(
        name="Iron",
        block_name="Ore_Iron_Basalt",
        min_y=0,
        max_y=200,
        chance=0.015,
        vein_min=3,
        vein_max=6,
    ),
    OreConfig(
        name="Thorium",
        block_name="Ore_Thorium_Basalt",
        min_y=0,
        max_y=140,
        chance=0.008,
        vein_min=2,
        vein_max=5,
    ),
    OreConfig(
        name="Cobalt",
        block_name="Ore_Cobalt_Basalt",
        min_y=0,
        max_y=90,
        chance=0.004,
        vein_min=2,
        vein_max=4,
    ),
    OreConfig(
        name="Adamantite",
        block_name="Ore_Adamantite_Basalt",
        min_y=0,
        max_y=50,
        chance=0.002,
        vein_min=1,
        vein_max=3,
    ),
]


def load_ore_configs(path: Optional[Path]) -> List[OreConfig]:
    """Load ore configs from JSON, falling back to defaults."""
    if path is None:
        return list(DEFAULT_ORE_CONFIGS)
    path = Path(path)
    if not path.exists():
        return list(DEFAULT_ORE_CONFIGS)
    with open(path) as f:
        data = json.load(f)
    configs = []
    for entry in data:
        configs.append(
            OreConfig(
                name=entry["name"],
                block_name=entry["block_name"],
                min_y=entry["min_y"],
                max_y=entry["max_y"],
                chance=entry["chance"],
                vein_min=entry["vein_min"],
                vein_max=entry["vein_max"],
            )
        )
    return configs
