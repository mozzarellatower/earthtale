"""Elevation mapping from real-world meters to Hytale Y coordinates.

This module handles the conversion of elevation data (in meters) to
Hytale world Y coordinates (0-319).

The default mapping strategy:
- Sea level (0m) maps to Y=64
- Lowest point (-500m, e.g., Dead Sea) maps to Y=0
- Highest point (8849m, Everest) maps to Y=319
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math


@dataclass
class ElevationConfig:
    """Configuration for elevation mapping."""
    # Minimum elevation to map (meters below sea level)
    min_elevation: float = -500.0
    # Maximum elevation to map (meters above sea level)
    max_elevation: float = 8849.0
    # Y coordinate for sea level
    sea_level_y: int = 64
    # Minimum Y coordinate
    min_y: int = 0
    # Maximum Y coordinate
    max_y: int = 319
    # Whether to use linear or logarithmic scaling for high elevations
    use_log_scale: bool = False
    # Exaggeration factor for terrain (1.0 = realistic)
    vertical_exaggeration: float = 1.0


def elevation_to_y(
    meters: float,
    config: Optional[ElevationConfig] = None
) -> int:
    """Convert elevation in meters to Hytale Y coordinate.

    Args:
        meters: Elevation in meters (can be negative for below sea level)
        config: Optional configuration for the mapping

    Returns:
        Y coordinate in range [min_y, max_y]
    """
    if config is None:
        config = ElevationConfig()

    # Apply vertical exaggeration
    if config.vertical_exaggeration != 1.0:
        # Exaggerate relative to sea level
        meters = meters * config.vertical_exaggeration

    # Handle out-of-range values
    meters = max(config.min_elevation, min(config.max_elevation, meters))

    if config.use_log_scale and meters > 0:
        # Logarithmic scaling for high elevations
        # Below sea level uses linear scaling
        below_sea_range = config.sea_level_y - config.min_y
        above_sea_range = config.max_y - config.sea_level_y

        if meters < 0:
            # Linear for below sea level
            normalized = (meters - config.min_elevation) / abs(config.min_elevation)
            return int(config.min_y + normalized * below_sea_range)
        else:
            # Logarithmic for above sea level
            # log(1) = 0, log(max+1) = max_log
            max_log = math.log1p(config.max_elevation)
            normalized = math.log1p(meters) / max_log
            return int(config.sea_level_y + normalized * above_sea_range)
    else:
        # Linear scaling
        total_range = config.max_elevation - config.min_elevation
        y_range = config.max_y - config.min_y

        normalized = (meters - config.min_elevation) / total_range
        return int(config.min_y + normalized * y_range)


def y_to_elevation(
    y: int,
    config: Optional[ElevationConfig] = None
) -> float:
    """Convert Hytale Y coordinate to elevation in meters.

    Args:
        y: Y coordinate
        config: Optional configuration for the mapping

    Returns:
        Elevation in meters
    """
    if config is None:
        config = ElevationConfig()

    # Clamp Y to valid range
    y = max(config.min_y, min(config.max_y, y))

    if config.use_log_scale:
        below_sea_range = config.sea_level_y - config.min_y
        above_sea_range = config.max_y - config.sea_level_y

        if y < config.sea_level_y:
            # Linear for below sea level
            normalized = (y - config.min_y) / below_sea_range
            meters = config.min_elevation + normalized * abs(config.min_elevation)
        else:
            # Inverse logarithmic for above sea level
            max_log = math.log1p(config.max_elevation)
            normalized = (y - config.sea_level_y) / above_sea_range
            meters = math.expm1(normalized * max_log)
    else:
        # Linear scaling
        total_range = config.max_elevation - config.min_elevation
        y_range = config.max_y - config.min_y

        normalized = (y - config.min_y) / y_range
        meters = config.min_elevation + normalized * total_range

    # Undo vertical exaggeration
    if config.vertical_exaggeration != 1.0:
        meters = meters / config.vertical_exaggeration

    return meters


class ElevationMapper:
    """Maps real-world elevations to Hytale Y coordinates."""

    def __init__(self, config: Optional[ElevationConfig] = None):
        """Initialize the mapper.

        Args:
            config: Configuration for elevation mapping
        """
        self.config = config or ElevationConfig()

    def get_y(self, meters: float) -> int:
        """Get Y coordinate for elevation in meters."""
        return elevation_to_y(meters, self.config)

    def get_elevation(self, y: int) -> float:
        """Get elevation in meters for Y coordinate."""
        return y_to_elevation(y, self.config)

    def get_column_heights(
        self,
        elevations: list[float],
        fill_below: bool = True
    ) -> list[Tuple[int, int]]:
        """Convert a list of elevations to (surface_y, bedrock_y) tuples.

        Args:
            elevations: List of elevations in meters
            fill_below: If True, return (surface_y, 0), otherwise (surface_y, surface_y)

        Returns:
            List of (surface_y, bottom_y) tuples
        """
        result = []
        for elev in elevations:
            surface_y = self.get_y(elev)
            bottom_y = 0 if fill_below else surface_y
            result.append((surface_y, bottom_y))
        return result


def create_mapper_for_region(
    min_elevation: float,
    max_elevation: float,
    sea_level_y: int = 64,
    min_y: int = 0,
    max_y: int = 319
) -> ElevationMapper:
    """Create an elevation mapper optimized for a specific region.

    This creates a mapper that uses the full Y range for the given
    elevation range, maximizing vertical detail.

    Args:
        min_elevation: Minimum elevation in the region (meters)
        max_elevation: Maximum elevation in the region (meters)
        sea_level_y: Y coordinate for sea level (0m)
        min_y: Minimum Y coordinate to use
        max_y: Maximum Y coordinate to use

    Returns:
        Configured ElevationMapper
    """
    config = ElevationConfig(
        min_elevation=min_elevation,
        max_elevation=max_elevation,
        sea_level_y=sea_level_y,
        min_y=min_y,
        max_y=max_y
    )
    return ElevationMapper(config)
