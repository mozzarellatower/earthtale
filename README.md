# EarthTale

Convert NASA elevation and satellite data into playable Hytale world files.

## Features

- **SRTM Elevation Data**: Downloads and processes NASA SRTM elevation data
- **Hytale World Format**: Generates valid Hytale world files with proper region/chunk structure
- **Biome Classification**: Automatically assigns biomes based on elevation and latitude
- **Preset Locations**: Includes preset coordinates for famous landmarks
- **Progress Tracking**: Real-time progress updates during conversion

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/earthtale.git
cd earthtale

# Install in development mode
pip install -e .
```

## Quick Start

### Convert a preset location

```bash
# Convert the Grand Canyon
earthtale preset "MyGrandCanyon" --preset grand_canyon

# Convert Mount Everest with 2x vertical exaggeration
earthtale preset "Everest" --preset mount_everest --exaggeration 2.0
```

### Convert custom coordinates

```bash
# Convert a specific region
earthtale convert "GrandCanyon" \
  --min-lat 35.9 --max-lat 36.3 \
  --min-lon -112.3 --max-lon -111.8 \
  --scale 5000  # meters per block
```

### View region information

```bash
earthtale info --min-lat 35.9 --max-lat 36.3 --min-lon -112.3 --max-lon -111.8
```

### List available presets

```bash
earthtale list-presets
```

## Available Presets

| Name | Location | Description |
|------|----------|-------------|
| `grand_canyon` | Arizona, USA | Grand Canyon National Park |
| `mount_everest` | Nepal/Tibet | World's highest peak |
| `alps_matterhorn` | Switzerland | Famous Alpine peak |
| `death_valley` | California, USA | Lowest point in North America |
| `hawaii_mauna_kea` | Hawaii, USA | Tallest mountain from base |
| `iceland_eyjafjallajokull` | Iceland | Famous volcano |

## CLI Options

### `earthtale convert`

| Option | Description | Default |
|--------|-------------|---------|
| `--min-lat` | Minimum latitude | Required |
| `--max-lat` | Maximum latitude | Required |
| `--min-lon` | Minimum longitude | Required |
| `--max-lon` | Maximum longitude | Required |
| `--scale, -s` | Meters per block | 5000.0 |
| `--blue-marble/--no-blue-marble` | Use Blue Marble imagery for biome classification | Enabled |
| `--blue-marble-resolution` | Blue Marble resolution (world_8km/world_4km/world_2km) | world_8km |
| `--ore-config` | Path to ore configuration JSON | None |
| `--parallel/--no-parallel` | Generate chunks in parallel | Disabled |
| `--workers` | Parallel worker count (defaults to all cores) | None |
| `--resume/--no-resume` | Resume generation if output exists | Disabled |
| `--output, -o` | Output directory | `output` |
| `--cache, -c` | Cache directory | `cache` |
| `--exaggeration, -e` | Vertical exaggeration | 1.0 |
| `--seed` | Random seed | None |
| `--skip-download` | Skip SRTM download | False |

## Output Structure

```
output/
└── universe/
    └── worlds/
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
earthtale preset "GrandCanyon" --preset grand_canyon --parallel --workers 8
```

## Resuming a Conversion

If a conversion is interrupted, rerun the same command with `--resume`. Existing chunks already written to region files are preserved and skipped.

Example:
```bash
earthtale preset "GrandCanyon" --preset grand_canyon --resume
```

## How It Works

1. **Download Phase**: Downloads required SRTM elevation tiles from NASA servers
2. **Calculate Phase**: Determines world dimensions based on geographic bounds and scale
3. **Generate Phase**: Creates terrain by:
   - Converting real-world elevation to Y coordinates
   - Classifying biomes based on elevation and latitude
   - Generating block columns with appropriate surface/subsurface blocks
4. **Save Phase**: Writes Hytale world files in the proper format

## Elevation Mapping

The converter maps real-world elevations to Hytale Y coordinates:

- **Sea level (0m)** → Y=64
- **Lowest (-500m)** → Y=0
- **Highest (8849m)** → Y=319

Use `--exaggeration` to make terrain more dramatic (e.g., `--exaggeration 2.0` doubles vertical relief).

## Biome Types

Biomes are assigned based on elevation and latitude:

- **Ocean**: Below sea level
- **Beach**: Near sea level at coast
- **Desert**: Low elevation in hot regions
- **Grassland**: Moderate elevation in temperate regions
- **Forest**: Higher elevation in temperate regions
- **Mountain**: High elevation
- **Snow**: Very high elevation or polar regions

## Data Sources

- **SRTM Elevation**: [NASA SRTM](https://www2.jpl.nasa.gov/srtm/)
- **Blue Marble Imagery**: [NASA Blue Marble](https://visibleearth.nasa.gov/collection/1484/blue-marble)

## Requirements

- Python 3.10+
- numpy
- rasterio (for GeoTIFF support)
- pillow (for image processing)
- httpx (for async downloads)
- typer (CLI framework)
- rich (progress display)
- pymongo (BSON serialization)

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting a pull request.

## Disclaimer

This tool is for educational and personal use. Please respect NASA data usage policies and Hytale's terms of service.
