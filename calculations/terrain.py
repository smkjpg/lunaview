import numpy as np
import rasterio
from rasterio.enums import Resampling


def load_terrain(path="data/terrain/Site01_elevation.tif", max_size=500):
    """
    Load lunar elevation data and downsample it for efficient display.
    Coordinates are in the raster's native projected coordinate system.
    """

    with rasterio.open(path) as src:
        scale = min(1, max_size / max(src.width, src.height))

        width = max(1, int(src.width * scale))
        height = max(1, int(src.height * scale))

        elevation = src.read(
            1,
            out_shape=(height, width),
            masked=True,
            resampling=Resampling.average
        )

        transform = src.transform * src.transform.scale(
            src.width / width,
            src.height / height
        )

        columns = np.arange(width) + 0.5
        rows = np.arange(height) + 0.5

        x = transform.c + columns * transform.a
        y = transform.f + rows * transform.e

        metadata = {
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "bounds": src.bounds,
            "pixel_size": src.res,
        }

    return x, y, elevation, metadata
