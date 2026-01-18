"""SRTM elevation data parser.

SRTM (Shuttle Radar Topography Mission) data format:
- Files are named like N34W119.hgt
- SRTM1: 3601 x 3601 signed 16-bit big-endian integers (about 25MB)
- SRTM3: 1201 x 1201 signed 16-bit big-endian integers (about 2.8MB)
- Values are elevation in meters
- Void data (no measurement) is -32768

Data is stored row by row from north to south, west to east.
The first value is at (north, west) corner, not southwest.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import struct
import numpy as np

from .coordinates import get_srtm_tile_name, lat_lon_to_srtm_pixel


# SRTM void value (no data)
SRTM_VOID = -32768


@dataclass
class SRTMTile:
    """An SRTM elevation tile.

    Attributes:
        name: Tile name (e.g., "N34W119")
        lat: Southwest corner latitude
        lon: Southwest corner longitude
        data: 2D numpy array of elevations (rows from north to south)
        resolution: Pixels per degree (3600 for SRTM1, 1200 for SRTM3)
    """
    name: str
    lat: int
    lon: int
    data: np.ndarray
    resolution: int

    @property
    def size(self) -> int:
        """Number of pixels per side (includes overlap pixel)."""
        return self.data.shape[0]

    def get_elevation(self, lat: float, lon: float) -> Optional[float]:
        """Get elevation at a specific coordinate.

        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees

        Returns:
            Elevation in meters, or None if void/out of range
        """
        # Check bounds
        if not (self.lat <= lat <= self.lat + 1 and self.lon <= lon <= self.lon + 1):
            return None

        # Convert to pixel coordinates
        row, col = lat_lon_to_srtm_pixel(lat, lon, self.lat, self.lon, self.resolution)

        # SRTM data is north-to-south, so invert row
        row = self.size - 1 - row

        # Clamp to valid range
        row = max(0, min(self.size - 1, row))
        col = max(0, min(self.size - 1, col))

        elevation = self.data[row, col]
        if elevation == SRTM_VOID:
            return None
        return float(elevation)

    def get_elevation_bilinear(self, lat: float, lon: float) -> Optional[float]:
        """Get elevation with bilinear interpolation.

        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees

        Returns:
            Interpolated elevation in meters, or None if void
        """
        if not (self.lat <= lat <= self.lat + 1 and self.lon <= lon <= self.lon + 1):
            return None

        # Get fractional pixel position
        lat_offset = lat - self.lat
        lon_offset = lon - self.lon

        row_f = lat_offset * self.resolution
        col_f = lon_offset * self.resolution

        # SRTM data is north-to-south
        row_f = self.size - 1 - row_f

        # Get surrounding pixels
        row0 = int(row_f)
        col0 = int(col_f)
        row1 = min(row0 + 1, self.size - 1)
        col1 = min(col0 + 1, self.size - 1)

        # Fractional parts
        t_row = row_f - row0
        t_col = col_f - col0

        # Get four corners
        v00 = self.data[row0, col0]
        v01 = self.data[row0, col1]
        v10 = self.data[row1, col0]
        v11 = self.data[row1, col1]

        # Check for void values
        if SRTM_VOID in (v00, v01, v10, v11):
            # Fall back to nearest valid
            return self.get_elevation(lat, lon)

        # Bilinear interpolation
        v0 = v00 * (1 - t_col) + v01 * t_col
        v1 = v10 * (1 - t_col) + v11 * t_col
        result = v0 * (1 - t_row) + v1 * t_row

        return float(result)


class SRTMParser:
    """Parser for SRTM .hgt files."""

    @staticmethod
    def detect_resolution(file_size: int) -> int:
        """Detect SRTM resolution from file size.

        Args:
            file_size: Size of the .hgt file in bytes

        Returns:
            Resolution (3600 for SRTM1, 1200 for SRTM3)
        """
        SRTM1_SIZE = 3601 * 3601 * 2  # 25934402 bytes
        SRTM3_SIZE = 1201 * 1201 * 2  # 2884802 bytes

        if file_size == SRTM1_SIZE:
            return 3600
        elif file_size == SRTM3_SIZE:
            return 1200
        else:
            # Try to infer from file size
            pixels = file_size // 2
            side = int(pixels ** 0.5)
            if side * side == pixels:
                return side - 1
            raise ValueError(f"Unknown SRTM file size: {file_size}")

    @staticmethod
    def parse_filename(filename: str) -> tuple[int, int]:
        """Parse lat/lon from SRTM filename.

        Args:
            filename: Filename like "N34W119.hgt"

        Returns:
            Tuple of (lat, lon) as integers
        """
        name = Path(filename).stem.upper()

        # Parse latitude
        if name[0] == 'N':
            lat = int(name[1:3])
        elif name[0] == 'S':
            lat = -int(name[1:3])
        else:
            raise ValueError(f"Invalid SRTM filename: {filename}")

        # Parse longitude
        if name[3] == 'E':
            lon = int(name[4:7])
        elif name[3] == 'W':
            lon = -int(name[4:7])
        else:
            raise ValueError(f"Invalid SRTM filename: {filename}")

        return (lat, lon)

    def parse(self, filepath: Union[str, Path]) -> SRTMTile:
        """Parse an SRTM .hgt file.

        Args:
            filepath: Path to the .hgt file

        Returns:
            SRTMTile with elevation data
        """
        filepath = Path(filepath)

        # Parse coordinates from filename
        lat, lon = self.parse_filename(filepath.name)

        # Detect resolution
        file_size = filepath.stat().st_size
        resolution = self.detect_resolution(file_size)
        side = resolution + 1

        # Read data
        with open(filepath, 'rb') as f:
            # SRTM data is big-endian signed 16-bit integers
            raw_data = f.read()

        # Convert to numpy array
        data = np.frombuffer(raw_data, dtype='>i2')  # big-endian int16
        data = data.reshape((side, side))

        # Create a writable copy
        data = data.astype(np.int16)

        return SRTMTile(
            name=get_srtm_tile_name(lat, lon),
            lat=lat,
            lon=lon,
            data=data,
            resolution=resolution
        )


class SRTMCache:
    """Cache for loaded SRTM tiles."""

    def __init__(self, data_dir: Path):
        """Initialize the cache.

        Args:
            data_dir: Directory containing .hgt files
        """
        self.data_dir = Path(data_dir)
        self._cache: dict[str, SRTMTile] = {}
        self._parser = SRTMParser()

    def get_tile(self, lat: float, lon: float) -> Optional[SRTMTile]:
        """Get the SRTM tile containing a coordinate.

        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees

        Returns:
            SRTMTile or None if not available
        """
        tile_name = get_srtm_tile_name(lat, lon)

        if tile_name in self._cache:
            return self._cache[tile_name]

        # Try to load from disk
        filepath = self.data_dir / f"{tile_name}.hgt"
        if not filepath.exists():
            return None

        tile = self._parser.parse(filepath)
        self._cache[tile_name] = tile
        return tile

    def get_elevation(self, lat: float, lon: float, interpolate: bool = True) -> Optional[float]:
        """Get elevation at a coordinate.

        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees
            interpolate: Whether to use bilinear interpolation

        Returns:
            Elevation in meters, or None if unavailable
        """
        tile = self.get_tile(lat, lon)
        if tile is None:
            return None

        if interpolate:
            return tile.get_elevation_bilinear(lat, lon)
        else:
            return tile.get_elevation(lat, lon)

    def clear(self) -> None:
        """Clear the tile cache."""
        self._cache.clear()
