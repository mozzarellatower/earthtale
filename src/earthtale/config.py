"""Configuration classes for EarthTale conversion."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import json


@dataclass
class BoundingBox:
    """Geographic bounding box."""
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def validate(self) -> None:
        """Validate the bounding box."""
        if not (-90 <= self.min_lat <= 90):
            raise ValueError(f"Invalid min_lat: {self.min_lat}")
        if not (-90 <= self.max_lat <= 90):
            raise ValueError(f"Invalid max_lat: {self.max_lat}")
        if not (-180 <= self.min_lon <= 180):
            raise ValueError(f"Invalid min_lon: {self.min_lon}")
        if not (-180 <= self.max_lon <= 180):
            raise ValueError(f"Invalid max_lon: {self.max_lon}")
        if self.min_lat >= self.max_lat:
            raise ValueError("min_lat must be less than max_lat")
        if self.min_lon >= self.max_lon:
            raise ValueError("min_lon must be less than max_lon")

    @property
    def center_lat(self) -> float:
        """Get center latitude."""
        return (self.min_lat + self.max_lat) / 2

    @property
    def center_lon(self) -> float:
        """Get center longitude."""
        return (self.min_lon + self.max_lon) / 2

    @property
    def width_degrees(self) -> float:
        """Get width in degrees longitude."""
        return self.max_lon - self.min_lon

    @property
    def height_degrees(self) -> float:
        """Get height in degrees latitude."""
        return self.max_lat - self.min_lat


@dataclass
class ConversionConfig:
    """Configuration for world conversion."""
    # World name
    name: str

    # Geographic bounds
    bounds: BoundingBox

    # Scale (meters per block)
    scale: float = 5000.0

    # Output directory
    output_dir: Path = field(default_factory=lambda: Path("output"))

    # Cache directory for NASA data
    cache_dir: Path = field(default_factory=lambda: Path("cache"))

    # Elevation settings
    min_elevation: float = -500.0  # Meters below sea level
    max_elevation: float = 8849.0  # Everest
    sea_level_y: int = 64
    vertical_exaggeration: float = 1.0

    # World generation seed
    seed: Optional[int] = None

    # Processing options
    parallel_chunks: int = 4
    skip_download: bool = False
    max_chunks: Optional[int] = None
    parallel: bool = False
    parallel_workers: Optional[int] = None
    resume: bool = False
    download_cache_mb: Optional[int] = None
    autosave_every: int = 25

    # Blue Marble imagery (optional, for biome classification)
    use_blue_marble: bool = True
    blue_marble_resolution: str = "world_8km"

    # Ore generation config
    ore_config_path: Optional[Path] = None

    # NASA credentials (optional)
    nasa_username: Optional[str] = None
    nasa_password: Optional[str] = None

    def validate(self) -> None:
        """Validate the configuration."""
        self.bounds.validate()
        if self.scale <= 0:
            raise ValueError(f"Scale must be positive: {self.scale}")
        if self.sea_level_y < 0 or self.sea_level_y > 319:
            raise ValueError(f"sea_level_y must be 0-319: {self.sea_level_y}")
        if self.vertical_exaggeration <= 0:
            raise ValueError(f"vertical_exaggeration must be positive: {self.vertical_exaggeration}")
        if self.parallel_workers is not None and self.parallel_workers <= 0:
            raise ValueError(f"parallel_workers must be positive: {self.parallel_workers}")
        if self.download_cache_mb is not None and self.download_cache_mb <= 0:
            raise ValueError(f"download_cache_mb must be positive: {self.download_cache_mb}")
        if self.autosave_every < 0:
            raise ValueError(f"autosave_every must be >= 0: {self.autosave_every}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "bounds": {
                "min_lat": self.bounds.min_lat,
                "max_lat": self.bounds.max_lat,
                "min_lon": self.bounds.min_lon,
                "max_lon": self.bounds.max_lon,
            },
            "scale": self.scale,
            "output_dir": str(self.output_dir),
            "cache_dir": str(self.cache_dir),
            "min_elevation": self.min_elevation,
            "max_elevation": self.max_elevation,
            "sea_level_y": self.sea_level_y,
            "vertical_exaggeration": self.vertical_exaggeration,
            "seed": self.seed,
            "parallel_chunks": self.parallel_chunks,
            "max_chunks": self.max_chunks,
            "parallel": self.parallel,
            "parallel_workers": self.parallel_workers,
            "resume": self.resume,
            "use_blue_marble": self.use_blue_marble,
            "blue_marble_resolution": self.blue_marble_resolution,
            "ore_config_path": str(self.ore_config_path) if self.ore_config_path else None,
        }

    def save(self, filepath: Path) -> None:
        """Save configuration to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "ConversionConfig":
        """Load configuration from JSON file."""
        with open(filepath) as f:
            data = json.load(f)

        bounds = BoundingBox(
            min_lat=data["bounds"]["min_lat"],
            max_lat=data["bounds"]["max_lat"],
            min_lon=data["bounds"]["min_lon"],
            max_lon=data["bounds"]["max_lon"],
        )

        return cls(
            name=data["name"],
            bounds=bounds,
            scale=data.get("scale", 5000.0),
            output_dir=Path(data.get("output_dir", "output")),
            cache_dir=Path(data.get("cache_dir", "cache")),
            min_elevation=data.get("min_elevation", -500.0),
            max_elevation=data.get("max_elevation", 8849.0),
            sea_level_y=data.get("sea_level_y", 64),
            vertical_exaggeration=data.get("vertical_exaggeration", 1.0),
            seed=data.get("seed"),
            parallel_chunks=data.get("parallel_chunks", 4),
            max_chunks=data.get("max_chunks"),
            parallel=data.get("parallel", False),
            parallel_workers=data.get("parallel_workers"),
            resume=data.get("resume", False),
            use_blue_marble=data.get("use_blue_marble", True),
            blue_marble_resolution=data.get("blue_marble_resolution", "world_8km"),
            ore_config_path=Path(data["ore_config_path"]) if data.get("ore_config_path") else None,
        )


# Preset locations for common conversions
PRESET_LOCATIONS = {
    "grand_canyon": BoundingBox(
        min_lat=35.9,
        max_lat=36.3,
        min_lon=-112.3,
        max_lon=-111.8,
    ),
    "mount_everest": BoundingBox(
        min_lat=27.85,
        max_lat=28.05,
        min_lon=86.85,
        max_lon=87.05,
    ),
    "alps_matterhorn": BoundingBox(
        min_lat=45.9,
        max_lat=46.1,
        min_lon=7.6,
        max_lon=7.8,
    ),
    "death_valley": BoundingBox(
        min_lat=36.1,
        max_lat=36.5,
        min_lon=-117.2,
        max_lon=-116.7,
    ),
    "hawaii_mauna_kea": BoundingBox(
        min_lat=19.75,
        max_lat=19.95,
        min_lon=-155.55,
        max_lon=-155.35,
    ),
    "iceland_eyjafjallajokull": BoundingBox(
        min_lat=63.55,
        max_lat=63.7,
        min_lon=-19.7,
        max_lon=-19.4,
    ),
}


def get_preset(name: str) -> Optional[BoundingBox]:
    """Get a preset bounding box by name."""
    return PRESET_LOCATIONS.get(name.lower().replace("-", "_").replace(" ", "_"))


def list_presets() -> list[str]:
    """Get list of available preset names."""
    return list(PRESET_LOCATIONS.keys())
