"""NASA data downloader with caching support.

SRTM data sources:
- USGS EarthExplorer: https://earthexplorer.usgs.gov/
- NASA LP DAAC: https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/

Blue Marble imagery:
- https://visibleearth.nasa.gov/collection/1484/blue-marble

Note: Most NASA data requires authentication. This module provides
the infrastructure for downloading, but users need to provide credentials.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Callable
from urllib.parse import urljoin
import hashlib
import json
import gzip

import httpx

from .coordinates import get_required_tiles


@dataclass
class DownloadProgress:
    """Progress information for a download."""
    url: str
    filename: str
    total_bytes: int
    downloaded_bytes: int
    complete: bool = False
    error: Optional[str] = None

    @property
    def percent(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return 100.0 * self.downloaded_bytes / self.total_bytes


ProgressCallback = Callable[[DownloadProgress], None]


class SRTMDownloader:
    """Downloads SRTM elevation data from NASA/USGS servers."""

    # SRTM data URL patterns
    USGS_SRTM_BASE = "https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/"
    PUBLIC_SRTM_BASE = "https://elevation-tiles-prod.s3.amazonaws.com/skadi/"
    BAILU_BASE = "https://bailu.ch/dem3/"
    SRTM_MAX_LAT = 60
    SRTM_MIN_LAT = -56

    def __init__(
        self,
        cache_dir: Path,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        """Initialize the downloader.

        Args:
            cache_dir: Directory to cache downloaded files
            username: NASA Earthdata username (optional)
            password: NASA Earthdata password (optional)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.username = username
        self.password = password
        self._metadata_file = self.cache_dir / "download_metadata.json"
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """Load download metadata from cache."""
        if self._metadata_file.exists():
            with open(self._metadata_file) as f:
                return json.load(f)
        return {"downloads": {}}

    def _save_metadata(self) -> None:
        """Save download metadata to cache."""
        with open(self._metadata_file, "w") as f:
            json.dump(self._metadata, f, indent=2)

    def get_cached_path(self, tile_name: str) -> Optional[Path]:
        """Get the cached path for a tile if it exists.

        Args:
            tile_name: Tile name like "N34W119"

        Returns:
            Path to cached file or None
        """
        filepath = self.cache_dir / f"{tile_name}.hgt"
        if filepath.exists():
            return filepath
        return None

    def is_tile_cached(self, tile_name: str) -> bool:
        """Check if a tile is cached.

        Args:
            tile_name: Tile name like "N34W119"

        Returns:
            True if the tile is cached
        """
        return self.get_cached_path(tile_name) is not None

    def is_tile_available(self, tile_name: str) -> bool:
        """Check if a tile is available or marked unavailable.

        Returns:
            True if the tile exists on disk or is known-unavailable.
        """
        if self.is_tile_cached(tile_name):
            return True
        entry = self._metadata.get("downloads", {}).get(tile_name, {})
        return bool(entry.get("missing"))

    def _parse_tile_lat(self, tile_name: str) -> int:
        """Parse latitude from an SRTM tile name (e.g., N34W119)."""
        name = tile_name.upper()
        if len(name) < 3:
            raise ValueError(f"Invalid tile name: {tile_name}")
        sign = 1 if name[0] == "N" else -1
        return sign * int(name[1:3])

    def mark_unavailable_tiles(self, tile_names: List[str]) -> None:
        """Mark tiles outside SRTM coverage as unavailable."""
        changed = False
        for tile in tile_names:
            try:
                lat = self._parse_tile_lat(tile)
            except Exception:
                continue
            if lat > self.SRTM_MAX_LAT or lat < self.SRTM_MIN_LAT:
                entry = self._metadata.setdefault("downloads", {}).get(tile)
                if not entry or not entry.get("missing"):
                    self._metadata["downloads"][tile] = {
                        "url": "coverage",
                        "size": 0,
                        "missing": True,
                    }
                    changed = True
        if changed:
            self._save_metadata()

    def get_missing_tiles(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float
    ) -> List[str]:
        """Get list of tiles needed that aren't cached.

        Args:
            min_lat: Minimum latitude
            max_lat: Maximum latitude
            min_lon: Minimum longitude
            max_lon: Maximum longitude

        Returns:
            List of tile names that need to be downloaded
        """
        required = get_required_tiles(min_lat, max_lat, min_lon, max_lon)
        return [t for t in required if not self.is_tile_available(t)]

    async def download_tile(
        self,
        tile_name: str,
        progress_callback: Optional[ProgressCallback] = None
    ) -> Optional[Path]:
        """Download a single SRTM tile.

        Args:
            tile_name: Tile name like "N34W119"
            progress_callback: Optional callback for progress updates

        Returns:
            Path to downloaded file or None if failed
        """
        # Check cache first
        cached = self.get_cached_path(tile_name)
        if cached:
            return cached
        if self.is_tile_available(tile_name):
            return None

        filepath = self.cache_dir / f"{tile_name}.hgt"
        gz_filepath = self.cache_dir / f"{tile_name}.hgt.gz"
        zip_filepath = self.cache_dir / f"{tile_name}.hgt.zip"

        async def download_gzip(url: str) -> tuple[bool, Optional[str], bool]:
            progress = DownloadProgress(url=url, filename=tile_name, total_bytes=0, downloaded_bytes=0)
            try:
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    async with client.stream("GET", url) as response:
                        if response.status_code != 200:
                            return False, f"HTTP {response.status_code}", response.status_code == 404
                        progress.total_bytes = int(response.headers.get("content-length", 0))
                        with open(gz_filepath, "wb") as f:
                            async for chunk in response.aiter_bytes(chunk_size=8192):
                                f.write(chunk)
                                progress.downloaded_bytes += len(chunk)
                                if progress_callback:
                                    progress_callback(progress)
                with gzip.open(gz_filepath, "rb") as src, open(filepath, "wb") as dst:
                    dst.write(src.read())
                gz_filepath.unlink(missing_ok=True)
                progress.complete = True
                if progress_callback:
                    progress_callback(progress)
                self._metadata["downloads"][tile_name] = {
                    "url": url,
                    "size": filepath.stat().st_size if filepath.exists() else 0
                }
                self._save_metadata()
                return True, None, False
            except Exception as exc:
                return False, str(exc), False

        async def download_zip(url: str) -> tuple[bool, Optional[str]]:
            progress = DownloadProgress(url=url, filename=tile_name, total_bytes=0, downloaded_bytes=0)
            try:
                auth = None
                if self.username and self.password:
                    auth = httpx.BasicAuth(self.username, self.password)
                async with httpx.AsyncClient(auth=auth, follow_redirects=True) as client:
                    async with client.stream("GET", url) as response:
                        if response.status_code != 200:
                            return False, f"HTTP {response.status_code}"
                        progress.total_bytes = int(response.headers.get("content-length", 0))
                        with open(zip_filepath, "wb") as f:
                            async for chunk in response.aiter_bytes(chunk_size=8192):
                                f.write(chunk)
                                progress.downloaded_bytes += len(chunk)
                                if progress_callback:
                                    progress_callback(progress)
                import zipfile
                with zipfile.ZipFile(zip_filepath) as zf:
                    hgt_files = [n for n in zf.namelist() if n.endswith(".hgt")]
                    if hgt_files:
                        with zf.open(hgt_files[0]) as src, open(filepath, "wb") as dst:
                            dst.write(src.read())
                zip_filepath.unlink(missing_ok=True)
                progress.complete = True
                if progress_callback:
                    progress_callback(progress)
                self._metadata["downloads"][tile_name] = {
                    "url": url,
                    "size": filepath.stat().st_size if filepath.exists() else 0
                }
                self._save_metadata()
                return True, None
            except Exception as exc:
                return False, str(exc)

        # Try public mirror first (no auth required)
        public_url = urljoin(self.PUBLIC_SRTM_BASE, f"{tile_name[:3]}/{tile_name}.hgt.gz")
        ok, err, unavailable = await download_gzip(public_url)
        if ok:
            return filepath

        # Try bailu.ch fallback (global 3-arcsecond)
        lat_dir = tile_name[:3]
        bailu_url = urljoin(self.BAILU_BASE, f"{lat_dir}/{tile_name}.hgt.zip")
        ok_zip, err_zip = await download_zip(bailu_url)
        if ok_zip:
            return filepath

        if unavailable or (err_zip and "HTTP 404" in err_zip):
            self._metadata.setdefault("downloads", {})[tile_name] = {
                "url": bailu_url,
                "size": 0,
                "missing": True,
            }
            self._save_metadata()
            progress = DownloadProgress(
                url=bailu_url,
                filename=tile_name,
                total_bytes=0,
                downloaded_bytes=0,
                complete=False,
                error="HTTP 404 (missing tile)",
            )
            if progress_callback:
                progress_callback(progress)
            return None

        # Fall back to NASA Earthdata if credentials provided
        if self.username and self.password:
            nasa_url = urljoin(self.USGS_SRTM_BASE, f"{tile_name}.SRTMGL1.hgt.zip")
            ok, err = await download_zip(nasa_url)
            if ok:
                return filepath

        self._metadata.setdefault("downloads", {})[tile_name] = {
            "url": bailu_url,
            "size": 0,
            "missing": True,
            "error": err or "Download failed",
        }
        self._save_metadata()

        progress = DownloadProgress(
            url=public_url,
            filename=tile_name,
            total_bytes=0,
            downloaded_bytes=0,
            complete=False,
            error=err or "Download failed",
        )
        if progress_callback:
            progress_callback(progress)
        return None

    async def download_tiles(
        self,
        tile_names: List[str],
        max_concurrent: int = 4,
        progress_callback: Optional[ProgressCallback] = None
    ) -> dict[str, Optional[Path]]:
        """Download multiple tiles concurrently.

        Args:
            tile_names: List of tile names to download
            max_concurrent: Maximum concurrent downloads
            progress_callback: Optional callback for progress updates

        Returns:
            Dict mapping tile names to file paths (or None if failed)
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def download_with_semaphore(tile_name: str) -> tuple[str, Optional[Path]]:
            async with semaphore:
                path = await self.download_tile(tile_name, progress_callback)
                return (tile_name, path)

        tasks = [download_with_semaphore(name) for name in tile_names]
        results = await asyncio.gather(*tasks)

        return dict(results)

    def download_region(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        progress_callback: Optional[ProgressCallback] = None
    ) -> dict[str, Optional[Path]]:
        """Download all tiles needed for a region (synchronous wrapper).

        Args:
            min_lat: Minimum latitude
            max_lat: Maximum latitude
            min_lon: Minimum longitude
            max_lon: Maximum longitude
            progress_callback: Optional callback for progress updates

        Returns:
            Dict mapping tile names to file paths
        """
        required = get_required_tiles(min_lat, max_lat, min_lon, max_lon)
        missing = [t for t in required if not self.is_tile_cached(t)]

        if not missing:
            # All tiles cached
            return {t: self.get_cached_path(t) for t in required}

        # Download missing tiles
        downloaded = asyncio.run(self.download_tiles(missing, progress_callback=progress_callback))

        # Combine with cached
        result = {}
        for tile in required:
            if tile in downloaded:
                result[tile] = downloaded[tile]
            else:
                result[tile] = self.get_cached_path(tile)

        return result


class BlueMarbleDownloader:
    """Downloads Blue Marble satellite imagery."""

    # Blue Marble URLs (fallback list per resolution)
    BLUE_MARBLE_URLS = {
        "world_8km": [
            "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73776/world.200408.3x5400x2700.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/5/56/Blue_Marble_Next_Generation_%2B_topography_%2B_bathymetry.jpg",
        ],
        "world_4km": [
            "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73776/world.200408.3x10800x5400.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/5/56/Blue_Marble_Next_Generation_%2B_topography_%2B_bathymetry.jpg",
        ],
        "world_2km": [
            "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73776/world.200408.3x21600x10800.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/5/56/Blue_Marble_Next_Generation_%2B_topography_%2B_bathymetry.jpg",
        ],
    }

    def __init__(self, cache_dir: Path):
        """Initialize the downloader.

        Args:
            cache_dir: Directory to cache downloaded files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def download_image(
        self,
        resolution: str = "world_8km",
        progress_callback: Optional[ProgressCallback] = None
    ) -> Optional[Path]:
        """Download Blue Marble imagery.

        Args:
            resolution: One of "world_8km", "world_4km", "world_2km"
            progress_callback: Optional callback for progress updates

        Returns:
            Path to downloaded file or None if failed
        """
        if resolution not in self.BLUE_MARBLE_URLS:
            raise ValueError(f"Unknown resolution: {resolution}")

        filename = f"blue_marble_{resolution}.jpg"
        filepath = self.cache_dir / filename

        # Check cache
        if filepath.exists():
            return filepath

        headers = {"User-Agent": "Mozilla/5.0"}
        urls = self.BLUE_MARBLE_URLS[resolution]
        for url in urls:
            progress = DownloadProgress(
                url=url,
                filename=filename,
                total_bytes=0,
                downloaded_bytes=0
            )
            try:
                async with httpx.AsyncClient(follow_redirects=True, headers=headers) as client:
                    async with client.stream("GET", url) as response:
                        if response.status_code != 200:
                            progress.error = f"HTTP {response.status_code}"
                            if progress_callback:
                                progress_callback(progress)
                            continue

                        total_size = int(response.headers.get("content-length", 0))
                        progress.total_bytes = total_size

                        with open(filepath, "wb") as f:
                            async for chunk in response.aiter_bytes(chunk_size=8192):
                                f.write(chunk)
                                progress.downloaded_bytes += len(chunk)
                                if progress_callback:
                                    progress_callback(progress)

                progress.complete = True
                if progress_callback:
                    progress_callback(progress)

                return filepath
            except Exception as e:
                progress.error = str(e)
                if progress_callback:
                    progress_callback(progress)
                continue

        return None
