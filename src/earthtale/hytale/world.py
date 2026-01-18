"""World configuration and structure management for Hytale worlds.

World structure:
universe/worlds/{name}/
    config.json          # World configuration (Version 4)
    chunks/
        {X}.{Z}.region.bin  # Region files
    resources/           # World resources (Time.json, etc.)
"""

import json
import uuid
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from .constants import WORLD_CONFIG_VERSION
from .chunk import Chunk
from .region import write_regions


@dataclass
class WorldGenConfig:
    """World generation configuration."""
    type: str = "Hytale"
    name: str = "Default"


@dataclass
class ClientEffects:
    """Visual effects configuration."""
    sun_height_percent: float = 100.0
    sun_angle_degrees: float = 0.0
    bloom_intensity: float = 0.3
    bloom_power: float = 8.0
    sun_intensity: float = 0.25
    sunshaft_intensity: float = 0.3
    sunshaft_scale_factor: float = 4.0


@dataclass
class WorldConfig:
    """World configuration matching config.json structure."""
    version: int = WORLD_CONFIG_VERSION
    uuid: Optional[str] = None
    seed: int = 0
    world_gen: WorldGenConfig = field(default_factory=WorldGenConfig)
    is_ticking: bool = True
    is_block_ticking: bool = True
    is_pvp_enabled: bool = False
    is_fall_damage_enabled: bool = True
    is_game_time_paused: bool = False
    game_time: str = "0001-01-01T05:30:00Z"
    client_effects: ClientEffects = field(default_factory=ClientEffects)
    is_spawning_npc: bool = True
    is_spawn_markers_enabled: bool = True
    is_all_npc_frozen: bool = False
    gameplay_config: str = "Default"
    is_compass_updating: bool = True
    is_saving_players: bool = True
    is_saving_chunks: bool = True
    save_new_chunks: bool = True
    is_unloading_chunks: bool = True
    is_objective_markers_enabled: bool = True
    delete_on_universe_start: bool = False
    delete_on_remove: bool = False

    def __post_init__(self):
        if self.uuid is None:
            self.uuid = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary matching Hytale config.json format."""
        # Generate UUID as BSON binary format
        uuid_bytes = uuid.UUID(self.uuid).bytes
        uuid_b64 = base64.b64encode(uuid_bytes).decode("ascii")

        return {
            "Version": self.version,
            "UUID": {
                "$binary": uuid_b64,
                "$type": "04"
            },
            "Seed": self.seed,
            "WorldGen": {
                "Type": self.world_gen.type,
                "Name": self.world_gen.name
            },
            "WorldMap": {
                "Type": "WorldGen"
            },
            "ChunkStorage": {
                "Type": "Hytale"
            },
            "ChunkConfig": {},
            "IsTicking": self.is_ticking,
            "IsBlockTicking": self.is_block_ticking,
            "IsPvpEnabled": self.is_pvp_enabled,
            "IsFallDamageEnabled": self.is_fall_damage_enabled,
            "IsGameTimePaused": self.is_game_time_paused,
            "GameTime": self.game_time,
            "ClientEffects": {
                "SunHeightPercent": self.client_effects.sun_height_percent,
                "SunAngleDegrees": self.client_effects.sun_angle_degrees,
                "BloomIntensity": self.client_effects.bloom_intensity,
                "BloomPower": self.client_effects.bloom_power,
                "SunIntensity": self.client_effects.sun_intensity,
                "SunshaftIntensity": self.client_effects.sunshaft_intensity,
                "SunshaftScaleFactor": self.client_effects.sunshaft_scale_factor
            },
            "RequiredPlugins": {},
            "IsSpawningNPC": self.is_spawning_npc,
            "IsSpawnMarkersEnabled": self.is_spawn_markers_enabled,
            "IsAllNPCFrozen": self.is_all_npc_frozen,
            "GameplayConfig": self.gameplay_config,
            "IsCompassUpdating": self.is_compass_updating,
            "IsSavingPlayers": self.is_saving_players,
            "IsSavingChunks": self.is_saving_chunks,
            "SaveNewChunks": self.save_new_chunks,
            "IsUnloadingChunks": self.is_unloading_chunks,
            "IsObjectiveMarkersEnabled": self.is_objective_markers_enabled,
            "DeleteOnUniverseStart": self.delete_on_universe_start,
            "DeleteOnRemove": self.delete_on_remove,
            "ResourceStorage": {
                "Type": "Hytale"
            },
            "Plugin": {}
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class World:
    """A Hytale world containing chunks and configuration.

    Attributes:
        name: World name (used for directory)
        config: World configuration
        chunks: Dictionary of chunks by (x, z) coordinates
    """

    def __init__(self, name: str, config: Optional[WorldConfig] = None):
        self.name = name
        self.config = config or WorldConfig()
        self.chunks: Dict[tuple, Chunk] = {}

    def add_chunk(self, chunk: Chunk) -> None:
        """Add a chunk to the world."""
        self.chunks[(chunk.x, chunk.z)] = chunk

    def get_chunk(self, x: int, z: int) -> Optional[Chunk]:
        """Get a chunk by coordinates."""
        return self.chunks.get((x, z))

    def save(self, output_path: Path, block_migration_version: int = 0) -> Path:
        """Save the world to disk.

        Args:
            output_path: Base path for universe/worlds/
            block_migration_version: Block migration version for serialization

        Returns:
            Path to the world directory
        """
        # Create directory structure (output directly to output_path/name/)
        world_dir = output_path / self.name
        chunks_dir = world_dir / "chunks"
        resources_dir = world_dir / "resources"

        world_dir.mkdir(parents=True, exist_ok=True)
        chunks_dir.mkdir(exist_ok=True)
        resources_dir.mkdir(exist_ok=True)

        # Write config.json
        config_path = world_dir / "config.json"
        with open(config_path, "w") as f:
            f.write(self.config.to_json())

        # Write Time.json in resources
        time_path = resources_dir / "Time.json"
        time_data = {
            "Version": 1,
            "GameTime": self.config.game_time
        }
        with open(time_path, "w") as f:
            json.dump(time_data, f, indent=2)

        # Write region files
        if self.chunks:
            write_regions(
                list(self.chunks.values()),
                chunks_dir,
                block_migration_version
            )

        return world_dir


def create_world(
    name: str,
    seed: Optional[int] = None,
    **config_kwargs
) -> World:
    """Create a new world with the given configuration.

    Args:
        name: World name
        seed: Random seed for world generation
        **config_kwargs: Additional WorldConfig parameters

    Returns:
        New World instance
    """
    if seed is None:
        import random
        seed = random.randint(0, 2**48 - 1)

    config = WorldConfig(seed=seed, **config_kwargs)
    return World(name, config)
