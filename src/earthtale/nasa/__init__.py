"""NASA data handling for elevation and imagery."""

from .coordinates import (
    LatLon,
    TileCoord,
    lat_lon_to_tile,
    tile_to_lat_lon,
    lat_lon_to_block,
    block_to_lat_lon,
    get_srtm_tile_name,
    get_srtm_tile_coord,
    get_required_tiles,
    lat_lon_to_srtm_pixel,
    meters_to_degrees_lat,
    meters_to_degrees_lon,
)
from .srtm import SRTMParser, SRTMTile, SRTMCache, SRTM_VOID
from .downloader import (
    SRTMDownloader,
    BlueMarbleDownloader,
    DownloadProgress,
    ProgressCallback,
)

__all__ = [
    # Coordinates
    "LatLon",
    "TileCoord",
    "lat_lon_to_tile",
    "tile_to_lat_lon",
    "lat_lon_to_block",
    "block_to_lat_lon",
    "get_srtm_tile_name",
    "get_srtm_tile_coord",
    "get_required_tiles",
    "lat_lon_to_srtm_pixel",
    "meters_to_degrees_lat",
    "meters_to_degrees_lon",
    # SRTM
    "SRTMParser",
    "SRTMTile",
    "SRTMCache",
    "SRTM_VOID",
    # Downloader
    "SRTMDownloader",
    "BlueMarbleDownloader",
    "DownloadProgress",
    "ProgressCallback",
]
