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
    depth_bias: str = "neutral"  # neutral/surface/deep
    bias_power: float = 1.5
    patch_chance: float = 0.08
    patch_multiplier: float = 3.0
    single_multiplier: float = 1.0


DEFAULT_ORE_CONFIGS: List[OreConfig] = [
    OreConfig(
        name="Copper",
        block_name="Ore_Copper_Basalt",
        min_y=40,
        max_y=240,
        chance=0.008,
        vein_min=3,
        vein_max=6,
        depth_bias="surface",
        bias_power=1.7,
        patch_chance=0.10,
        patch_multiplier=3.0,
        single_multiplier=0.7,
    ),
    OreConfig(
        name="Iron",
        block_name="Ore_Iron_Basalt",
        min_y=10,
        max_y=200,
        chance=0.006,
        vein_min=3,
        vein_max=6,
        depth_bias="surface",
        bias_power=1.5,
        patch_chance=0.08,
        patch_multiplier=2.5,
        single_multiplier=0.6,
    ),
    OreConfig(
        name="Thorium",
        block_name="Ore_Thorium_Basalt",
        min_y=0,
        max_y=140,
        chance=0.003,
        vein_min=2,
        vein_max=5,
        depth_bias="deep",
        bias_power=1.4,
        patch_chance=0.06,
        patch_multiplier=2.5,
        single_multiplier=0.5,
    ),
    OreConfig(
        name="Cobalt",
        block_name="Ore_Cobalt_Basalt",
        min_y=0,
        max_y=90,
        chance=0.0015,
        vein_min=2,
        vein_max=4,
        depth_bias="deep",
        bias_power=1.6,
        patch_chance=0.05,
        patch_multiplier=2.0,
        single_multiplier=0.4,
    ),
    OreConfig(
        name="Adamantite",
        block_name="Ore_Adamantite_Basalt",
        min_y=0,
        max_y=50,
        chance=0.0006,
        vein_min=1,
        vein_max=3,
        depth_bias="deep",
        bias_power=1.8,
        patch_chance=0.04,
        patch_multiplier=2.0,
        single_multiplier=0.3,
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
                depth_bias=entry.get("depth_bias", "neutral"),
                bias_power=entry.get("bias_power", 1.5),
                patch_chance=entry.get("patch_chance", 0.08),
                patch_multiplier=entry.get("patch_multiplier", 3.0),
                single_multiplier=entry.get("single_multiplier", 1.0),
            )
        )
    return configs
