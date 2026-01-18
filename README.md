# EarthTale

Convert NASA elevation and satellite data into playable Hytale world files.

## Features

- **SRTM Elevation Data**: Downloads and processes NASA SRTM elevation data
- **Blue Marble Surface Cues**: Uses NASA imagery colors for biome/surface hints
- **Hytale World Format**: Generates valid Hytale world files with proper region/chunk structure
- **Biome Zoning**: Assigns biome zones based on elevation + imagery
- **Ore Generation**: Configurable ore tiers (Copper, Iron, Thorium, Cobalt, Adamantite)
- **Preset Locations**: Includes preset coordinates for famous landmarks
- **Parallel + Resume**: Multi-core chunk generation with resume support
- **Progress Tracking**: Real-time progress updates during conversion

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/earthtale.git
cd earthtale
```

## Quick Start

### Convert a preset location

```bash
# Install dependencies for main.py
pip install zstandard httpx pillow

# Convert the Grand Canyon
python3 main.py grand_canyon

# Convert Mount Everest with 2x vertical exaggeration
python3 main.py mount_everest --exaggeration 2.0
```

## NASA Earthdata Credentials

By default, EarthTale uses a public SRTM mirror (no auth required). Some ocean/polar tiles simply do not exist in SRTM; those are treated as missing and will generate sea-level terrain. If a tile isn't available on the public mirror, you can provide NASA Earthdata credentials as a fallback:

```bash
python3 main.py grand_canyon --nasa-user YOUR_USER --nasa-pass YOUR_PASS
```

Or:
```bash
export EARTHDATA_USER=YOUR_USER
export EARTHDATA_PASS=YOUR_PASS
python3 main.py grand_canyon
```

### Convert custom coordinates

```bash
# Convert a specific region
python3 main.py --name "GrandCanyon" \
  --min-lat 35.9 --max-lat 36.3 \
  --min-lon -112.3 --max-lon -111.8 \
  --scale 5000  # meters per block
```

### Convert the whole Earth

```bash
python3 main.py --name "Earth" \
  --min-lat -90 --max-lat 90 \
  --min-lon -180 --max-lon 180 \
  --scale 5000 --parallel
```

### View region information

Optional CLI tools (requires `pip install typer rich`):

```bash
PYTHONPATH=src python3 -m earthtale.cli info --min-lat 35.9 --max-lat 36.3 --min-lon -112.3 --max-lon -111.8
```

### List available presets

```bash
PYTHONPATH=src python3 -m earthtale.cli list-presets
```

## Available Presets

| Name | Location | Description |
|------|----------|-------------|
| `grand_canyon` | Arizona, USA | Grand Canyon National Park |
| `mount_everest` | Nepal/Tibet | World's highest peak |
| `alps_matterhorn` | Switzerland | Famous Alpine peak |
| `death_valley` | California, USA | Lowest point in North America |
| `hawaii_mauna_kea` | Hawaii, USA | Tallest mountain from base |
| `iceland` | Iceland | Famous volcano |

## CLI Options

### `python3 main.py`

| Option | Description | Default |
|--------|-------------|---------|
| `--min-lat` | Minimum latitude (custom bounds) | Required if no preset |
| `--max-lat` | Maximum latitude (custom bounds) | Required if no preset |
| `--min-lon` | Minimum longitude (custom bounds) | Required if no preset |
| `--max-lon` | Maximum longitude (custom bounds) | Required if no preset |
| `--scale` | Meters per block | 5000.0 |
| `--exaggeration` | Vertical exaggeration | 1.0 |
| `--name` | World name | Preset name or "Custom World" |
| `--seed` | Random seed | None |
| `--small` | Generate a 1x1 chunk test world | Disabled |
| `--skip-download` | Skip SRTM download | Disabled |
| `--no-blue-marble` | Disable Blue Marble imagery | Disabled |
| `--blue-marble-resolution` | Blue Marble resolution (world_8km/world_4km/world_2km) | world_8km |
| `--ore-config` | Path to ore configuration JSON | None |
| `--parallel` | Generate chunks in parallel | Disabled |
| `--workers` | Parallel worker count (defaults to all cores) | None |
| `--resume` | Resume generation if output exists | Disabled |
| `--nasa-user` | NASA Earthdata username (or `EARTHDATA_USER`) | None |
| `--nasa-pass` | NASA Earthdata password (or `EARTHDATA_PASS`) | None |
| `--download-cache-mb` | Pause SRTM downloads when cache reaches this size | None |
| `--autosave-every` | Autosave every N chunks (0 disables) | 25 |

## Output Structure

```
output/
└── {world_name}/
    ├── config.json          # World configuration
    ├── chunks/
    │   └── {X}.{Z}.region.bin  # Region files
    └── resources/
        └── Time.json        # World time
```

## Parallel Processing

Use `--parallel` to enable multi-core chunk generation. If you do not set `--workers`, the converter uses all available CPU cores by default.

Example:
```bash
python3 main.py grand_canyon --parallel --workers 8
```

## Resuming a Conversion

If a conversion is interrupted, rerun the same command with `--resume`. Existing chunks already written to region files are preserved and skipped.
Autosaves happen every 25 chunks by default, and a Ctrl+C will save a partial world before exiting.

Example:
```bash
python3 main.py grand_canyon --resume
```

## How It Works

1. **Download Phase**: Downloads required SRTM elevation tiles from NASA servers
2. **Calculate Phase**: Determines world dimensions based on geographic bounds and scale
3. **Generate Phase**: Creates terrain by:
   - Converting real-world elevation to Y coordinates
   - Classifying biome zones using Blue Marble color + elevation
   - Generating block columns with appropriate surface/subsurface blocks
4. **Save Phase**: Writes Hytale world files in the proper format

## Elevation Mapping

The converter maps real-world elevations to Hytale Y coordinates:

- **Sea level (0m)** → Y=64
- **Lowest (-500m)** → Y=0
- **Highest (8849m)** → Y=319

Use `--exaggeration` to make terrain more dramatic (e.g., `--exaggeration 2.0` doubles vertical relief).

## Biome Types

Biomes are assigned based on Blue Marble colors, elevation, and latitude:

- **Ocean**: Below sea level
- **Beach**: Near sea level at coast
- **Desert**: Low elevation in hot regions
- **Grassland**: Moderate elevation in temperate regions
- **Forest**: Higher elevation in temperate regions
- **Mountain**: High elevation
- **Snow**: Very high elevation or polar regions

## Data Sources

- **SRTM Elevation**: [NASA SRTM](https://www2.jpl.nasa.gov/srtm/)
- **Public SRTM Mirror**: [AWS elevation-tiles (Mapzen)](https://registry.opendata.aws/terrain-tiles/)
- **Blue Marble Imagery**: [NASA Blue Marble](https://visibleearth.nasa.gov/collection/1484/blue-marble)

## Data Sizes (Approximate)

- **SRTMGL1 tiles**: ~25MB uncompressed per 1x1 degree tile (smaller when zipped)
- **Blue Marble**:
  - `world_8km`: a few MB
  - `world_4km`: tens of MB
  - `world_2km`: larger still (use only if you need the extra detail)
- **Full-Earth SRTM cache**: 64,800 tiles total (~1.6TB uncompressed). Expect hundreds of GB on disk if you cache everything.

## Output Size Estimates (Very Rough)

These vary a lot based on compression, terrain complexity, and water/air ratios.

- **Whole Earth @ 1:5000**: ~31k chunks. Typical output in the low single‑digit GB range.
- **Whole Earth @ 1:1000**: ~780k chunks. Typical output in the tens of GB range.

## Requirements

- Python 3.10+
- zstandard
- httpx
- pillow
- typer (optional, for `python3 -m earthtale.cli`)
- rich (optional, for `python3 -m earthtale.cli`)

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting a pull request.

## Disclaimer

This tool is for educational and personal use. Please respect NASA data usage policies and Hytale's terms of service.
