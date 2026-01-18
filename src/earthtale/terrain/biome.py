"""Biome classification from satellite imagery colors.

This module classifies terrain into biomes based on RGB colors from
satellite imagery (e.g., Blue Marble) and elevation data.

Biome types are simplified for Hytale terrain generation.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple
import math


class BiomeType(Enum):
    """Simplified biome types for terrain generation."""
    OCEAN = auto()
    BEACH = auto()
    DESERT = auto()
    SAVANNA = auto()
    GRASSLAND = auto()
    FOREST = auto()
    TAIGA = auto()
    TUNDRA = auto()
    SNOW = auto()
    MOUNTAIN = auto()
    RIVER = auto()
    LAKE = auto()


@dataclass
class BiomeConfig:
    """Configuration for biome classification."""
    # Elevation thresholds (in Y coordinates)
    snow_line: int = 250  # Above this is snow
    mountain_line: int = 200  # Above this is mountain
    tree_line: int = 280  # No trees above this

    # Color thresholds (RGB ranges)
    water_threshold: int = 100  # Blue dominance for water
    sand_threshold: int = 200  # High R+G, low B for desert/beach
    green_threshold: int = 80  # Green dominance for vegetation

    # Temperature gradient (based on latitude, 0-1)
    cold_threshold: float = 0.7  # Above this is cold biome
    hot_threshold: float = 0.3  # Below this is hot biome


@dataclass
class ColorHSV:
    """HSV color representation."""
    h: float  # Hue (0-360)
    s: float  # Saturation (0-1)
    v: float  # Value (0-1)


def rgb_to_hsv(r: int, g: int, b: int) -> ColorHSV:
    """Convert RGB (0-255) to HSV."""
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0

    max_c = max(r_norm, g_norm, b_norm)
    min_c = min(r_norm, g_norm, b_norm)
    diff = max_c - min_c

    # Hue calculation
    if diff == 0:
        h = 0
    elif max_c == r_norm:
        h = 60 * (((g_norm - b_norm) / diff) % 6)
    elif max_c == g_norm:
        h = 60 * (((b_norm - r_norm) / diff) + 2)
    else:
        h = 60 * (((r_norm - g_norm) / diff) + 4)

    # Saturation
    s = 0 if max_c == 0 else diff / max_c

    # Value
    v = max_c

    return ColorHSV(h, s, v)


class BiomeClassifier:
    """Classifies terrain into biomes based on color and elevation."""

    def __init__(self, config: Optional[BiomeConfig] = None):
        """Initialize the classifier.

        Args:
            config: Configuration for biome classification
        """
        self.config = config or BiomeConfig()

    def classify(
        self,
        r: int,
        g: int,
        b: int,
        elevation_y: int,
        latitude_factor: float = 0.5
    ) -> BiomeType:
        """Classify a point into a biome.

        Args:
            r: Red component (0-255)
            g: Green component (0-255)
            b: Blue component (0-255)
            elevation_y: Y coordinate of the surface
            latitude_factor: 0=equator, 1=pole

        Returns:
            BiomeType classification
        """
        hsv = rgb_to_hsv(r, g, b)

        # Check for water (high blue dominance, low saturation with blue hue)
        if b > r and b > g:
            if hsv.h > 180 and hsv.h < 260:  # Blue-ish hue
                if hsv.s > 0.3:  # Saturated blue = ocean
                    return BiomeType.OCEAN
                else:  # Desaturated blue = ice/snow
                    if latitude_factor > self.config.cold_threshold:
                        return BiomeType.SNOW
                    return BiomeType.LAKE

        # Check elevation-based biomes first
        if elevation_y >= self.config.snow_line:
            return BiomeType.SNOW
        if elevation_y >= self.config.mountain_line:
            return BiomeType.MOUNTAIN

        # Check for desert/sand (yellow-ish, high brightness)
        if hsv.h < 60 or hsv.h > 300:  # Red to yellow range
            if hsv.s < 0.5 and hsv.v > 0.6:
                if elevation_y < 70:  # Near sea level
                    return BiomeType.BEACH
                return BiomeType.DESERT
        # Drylands (brown/orange, low green dominance)
        green_ratio = g / max(1.0, (r + g + b))
        if 10 <= hsv.h <= 90 and hsv.s > 0.15 and hsv.v > 0.25:
            if green_ratio < 0.38:
                return BiomeType.DESERT
        if r >= g >= b and (r - g) > 5 and (g - b) > 5 and green_ratio < 0.40:
            return BiomeType.DESERT

        # Semi-arid drylands hiding in greenish hues
        if 50 < hsv.h < 140:
            if hsv.s < 0.35 and hsv.v > 0.35:
                if latitude_factor < self.config.hot_threshold + 0.1:
                    return BiomeType.DESERT
            if green_ratio < 0.35:
                return BiomeType.DESERT

        # Green vegetation
        if 60 < hsv.h < 180:  # Green range
            if hsv.s > 0.3 and green_ratio >= 0.4:  # Strong green
                # Temperature determines type
                if latitude_factor > self.config.cold_threshold:
                    if elevation_y > 150:
                        return BiomeType.TUNDRA
                    return BiomeType.TAIGA
                elif latitude_factor < self.config.hot_threshold:
                    if hsv.v < 0.4:  # Dark green
                        return BiomeType.FOREST
                    return BiomeType.SAVANNA
                else:
                    if hsv.s > 0.4:
                        return BiomeType.FOREST
                    return BiomeType.GRASSLAND

        # Default fallbacks based on latitude
        if latitude_factor > self.config.cold_threshold:
            return BiomeType.TUNDRA
        elif latitude_factor < self.config.hot_threshold:
            return BiomeType.SAVANNA
        else:
            return BiomeType.GRASSLAND

    def classify_from_elevation_only(
        self,
        elevation_y: int,
        latitude_factor: float = 0.5
    ) -> BiomeType:
        """Classify based on elevation when no color data is available.

        Args:
            elevation_y: Y coordinate of the surface
            latitude_factor: 0=equator, 1=pole

        Returns:
            BiomeType classification
        """
        # Below sea level
        if elevation_y < 64:
            return BiomeType.OCEAN

        # Snow line
        if elevation_y >= self.config.snow_line:
            return BiomeType.SNOW

        # Mountain line
        if elevation_y >= self.config.mountain_line:
            return BiomeType.MOUNTAIN

        # Beach/coast
        if elevation_y < 70:
            return BiomeType.BEACH

        # Temperature-based biomes
        if latitude_factor > self.config.cold_threshold:
            if elevation_y > 150:
                return BiomeType.TUNDRA
            return BiomeType.TAIGA
        elif latitude_factor < self.config.hot_threshold:
            if elevation_y < 100:
                return BiomeType.DESERT
            return BiomeType.SAVANNA
        else:
            if elevation_y > 150:
                return BiomeType.FOREST
            return BiomeType.GRASSLAND

    def get_latitude_factor(self, latitude: float) -> float:
        """Convert latitude to a 0-1 factor (0=equator, 1=pole).

        Args:
            latitude: Latitude in degrees (-90 to 90)

        Returns:
            Factor from 0 (equator) to 1 (pole)
        """
        return abs(latitude) / 90.0


# Biome to primary surface block mapping
BIOME_SURFACE_BLOCKS = {
    BiomeType.OCEAN: "Water",
    BiomeType.BEACH: "Sand",
    BiomeType.DESERT: "Sand",
    BiomeType.SAVANNA: "Grass",
    BiomeType.GRASSLAND: "Grass",
    BiomeType.FOREST: "Grass",
    BiomeType.TAIGA: "Grass",
    BiomeType.TUNDRA: "Dirt",
    BiomeType.SNOW: "Snow",
    BiomeType.MOUNTAIN: "Stone",
    BiomeType.RIVER: "Water",
    BiomeType.LAKE: "Water",
}

# Biome to subsurface block mapping
BIOME_SUBSURFACE_BLOCKS = {
    BiomeType.OCEAN: "Sand",
    BiomeType.BEACH: "Sand",
    BiomeType.DESERT: "Sandstone",
    BiomeType.SAVANNA: "Dirt",
    BiomeType.GRASSLAND: "Dirt",
    BiomeType.FOREST: "Dirt",
    BiomeType.TAIGA: "Dirt",
    BiomeType.TUNDRA: "Stone",
    BiomeType.SNOW: "Stone",
    BiomeType.MOUNTAIN: "Stone",
    BiomeType.RIVER: "Gravel",
    BiomeType.LAKE: "Clay",
}


def get_surface_block(biome: BiomeType) -> str:
    """Get the surface block name for a biome."""
    return BIOME_SURFACE_BLOCKS.get(biome, "Grass")


def get_subsurface_block(biome: BiomeType) -> str:
    """Get the subsurface block name for a biome."""
    return BIOME_SUBSURFACE_BLOCKS.get(biome, "Dirt")
