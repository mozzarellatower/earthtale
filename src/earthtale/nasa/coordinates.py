"""Coordinate conversion utilities for NASA data.

SRTM tiles are named by their southwest corner coordinates.
For example, N34W119.hgt covers latitudes 34-35N and longitudes 118-119W.
"""

from dataclasses import dataclass
from typing import Tuple
import math


@dataclass
class LatLon:
    """Latitude/longitude coordinate pair."""
    lat: float
    lon: float

    def to_tuple(self) -> Tuple[float, float]:
        return (self.lat, self.lon)


@dataclass
class TileCoord:
    """SRTM tile coordinate (integer lat/lon of southwest corner)."""
    lat: int  # South-west latitude
    lon: int  # South-west longitude


def get_srtm_tile_name(lat: float, lon: float) -> str:
    """Get the SRTM tile name for a given coordinate.

    SRTM tiles are named by their southwest corner.
    Example: N34W119 for the tile containing (34.5, -118.5)

    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees

    Returns:
        Tile name like "N34W119" or "S12E045"
    """
    tile_lat = int(math.floor(lat))
    tile_lon = int(math.floor(lon))

    if tile_lat >= 0:
        lat_str = f"N{tile_lat:02d}"
    else:
        lat_str = f"S{abs(tile_lat):02d}"

    if tile_lon >= 0:
        lon_str = f"E{tile_lon:03d}"
    else:
        lon_str = f"W{abs(tile_lon):03d}"

    return f"{lat_str}{lon_str}"


def get_srtm_tile_coord(lat: float, lon: float) -> TileCoord:
    """Get the SRTM tile coordinate for a given location.

    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees

    Returns:
        TileCoord of the southwest corner
    """
    return TileCoord(
        lat=int(math.floor(lat)),
        lon=int(math.floor(lon))
    )


def lat_lon_to_tile(lat: float, lon: float) -> Tuple[int, int]:
    """Convert lat/lon to SRTM tile indices.

    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees

    Returns:
        Tuple of (tile_lat, tile_lon) integers
    """
    return (int(math.floor(lat)), int(math.floor(lon)))


def tile_to_lat_lon(tile_lat: int, tile_lon: int) -> Tuple[float, float]:
    """Convert tile indices to lat/lon (southwest corner).

    Args:
        tile_lat: Tile latitude (integer)
        tile_lon: Tile longitude (integer)

    Returns:
        Tuple of (lat, lon) for the southwest corner
    """
    return (float(tile_lat), float(tile_lon))


def lat_lon_to_srtm_pixel(
    lat: float,
    lon: float,
    tile_lat: int,
    tile_lon: int,
    pixels_per_degree: int = 3600
) -> Tuple[int, int]:
    """Convert lat/lon to pixel position within an SRTM tile.

    SRTM tiles have pixels from south to north, west to east.
    The pixel at (0, 0) is at the southwest corner.

    For SRTM1 (1 arc-second): 3601 x 3601 pixels
    For SRTM3 (3 arc-second): 1201 x 1201 pixels

    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
        tile_lat: Tile southwest latitude
        tile_lon: Tile southwest longitude
        pixels_per_degree: Resolution (3600 for SRTM1, 1200 for SRTM3)

    Returns:
        Tuple of (row, col) pixel coordinates
    """
    # Offset from southwest corner
    lat_offset = lat - tile_lat
    lon_offset = lon - tile_lon

    # Convert to pixels
    row = int(lat_offset * pixels_per_degree)
    col = int(lon_offset * pixels_per_degree)

    return (row, col)


def lat_lon_to_block(
    lat: float,
    lon: float,
    origin_lat: float,
    origin_lon: float,
    meters_per_block: float = 5000.0
) -> Tuple[int, int]:
    """Convert lat/lon to block coordinates relative to an origin.

    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
        origin_lat: Origin latitude (usually min_lat of the region)
        origin_lon: Origin longitude (usually min_lon of the region)
        meters_per_block: Scale factor (default 5000m)

    Returns:
        Tuple of (block_x, block_z) coordinates
    """
    # Earth's radius in meters
    EARTH_RADIUS = 6_371_000

    # Latitude: 1 degree ≈ 111,320 meters
    lat_meters_per_degree = 2 * math.pi * EARTH_RADIUS / 360

    # Longitude: varies by latitude
    lon_meters_per_degree = lat_meters_per_degree * math.cos(math.radians(origin_lat))

    # Calculate offsets
    lat_diff = lat - origin_lat
    lon_diff = lon - origin_lon

    meters_north = lat_diff * lat_meters_per_degree
    meters_east = lon_diff * lon_meters_per_degree

    # Convert to blocks (Z is north in Hytale, X is east)
    block_z = int(meters_north / meters_per_block)
    block_x = int(meters_east / meters_per_block)

    return (block_x, block_z)


def block_to_lat_lon(
    block_x: int,
    block_z: int,
    origin_lat: float,
    origin_lon: float,
    meters_per_block: float = 5000.0
) -> Tuple[float, float]:
    """Convert block coordinates to lat/lon.

    Args:
        block_x: Block X coordinate (east)
        block_z: Block Z coordinate (north)
        origin_lat: Origin latitude
        origin_lon: Origin longitude
        meters_per_block: Scale factor

    Returns:
        Tuple of (lat, lon) in decimal degrees
    """
    EARTH_RADIUS = 6_371_000
    lat_meters_per_degree = 2 * math.pi * EARTH_RADIUS / 360
    lon_meters_per_degree = lat_meters_per_degree * math.cos(math.radians(origin_lat))

    meters_east = block_x * meters_per_block
    meters_north = block_z * meters_per_block

    lat = origin_lat + (meters_north / lat_meters_per_degree)
    lon = origin_lon + (meters_east / lon_meters_per_degree)

    return (lat, lon)


def get_required_tiles(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float
) -> list[str]:
    """Get list of SRTM tile names needed to cover a bounding box.

    Args:
        min_lat: Minimum latitude
        max_lat: Maximum latitude
        min_lon: Minimum longitude
        max_lon: Maximum longitude

    Returns:
        List of tile names like ["N34W119", "N34W118", ...]
    """
    tiles = []

    min_tile_lat = int(math.floor(min_lat))
    max_tile_lat = int(math.floor(max_lat))
    min_tile_lon = int(math.floor(min_lon))
    max_tile_lon = int(math.floor(max_lon))

    for lat in range(min_tile_lat, max_tile_lat + 1):
        for lon in range(min_tile_lon, max_tile_lon + 1):
            tiles.append(get_srtm_tile_name(lat, lon))

    return tiles


def meters_to_degrees_lat(meters: float) -> float:
    """Convert meters to degrees latitude."""
    EARTH_RADIUS = 6_371_000
    return meters / (2 * math.pi * EARTH_RADIUS / 360)


def meters_to_degrees_lon(meters: float, lat: float) -> float:
    """Convert meters to degrees longitude at a given latitude."""
    EARTH_RADIUS = 6_371_000
    lat_meters_per_degree = 2 * math.pi * EARTH_RADIUS / 360
    lon_meters_per_degree = lat_meters_per_degree * math.cos(math.radians(lat))
    return meters / lon_meters_per_degree
