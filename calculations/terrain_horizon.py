
import math

import numpy as np
import rasterio
from pyproj import Transformer


MOON_RADIUS_M = 1_737_400.0


def destination_point(latitude, longitude, azimuth_deg, distance_m):
    """Calculate a destination on a spherical Moon."""

    lat1 = math.radians(latitude)
    lon1 = math.radians(longitude)
    bearing = math.radians(azimuth_deg)

    angular_distance = distance_m / MOON_RADIUS_M

    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_distance)
        + math.cos(lat1)
        * math.sin(angular_distance)
        * math.cos(bearing)
    )

    lon2 = lon1 + math.atan2(
        math.sin(bearing)
        * math.sin(angular_distance)
        * math.cos(lat1),
        math.cos(angular_distance)
        - math.sin(lat1) * math.sin(lat2),
    )

    return math.degrees(lat2), math.degrees(lon2)


def bilinear_sample(grid, inverse_transform, x, y):
    """
    Bilinearly interpolate a raster at map coordinates.

    Raster values are assumed to represent pixel-centre samples.
    Returns NaN when interpolation is outside the raster
    or when one of the four required pixels is invalid.
    """

    col_corner, row_corner = inverse_transform * (x, y)

    col_position = col_corner - 0.5
    row_position = row_corner - 0.5

    col0 = math.floor(col_position)
    row0 = math.floor(row_position)

    dx = col_position - col0
    dy = row_position - row0

    height, width = grid.shape

    if (
        row0 < 0
        or col0 < 0
        or row0 + 1 >= height
        or col0 + 1 >= width
    ):
        return float("nan")

    values = np.array([
        grid[row0, col0],
        grid[row0, col0 + 1],
        grid[row0 + 1, col0],
        grid[row0 + 1, col0 + 1],
    ], dtype=float)

    if not np.all(np.isfinite(values)):
        return float("nan")

    top = values[0] * (1.0 - dx) + values[1] * dx
    bottom = values[2] * (1.0 - dx) + values[3] * dx

    return float(top * (1.0 - dy) + bottom * dy)


def calculate_observer_elevation(
    elevation_grid,
    inverse_transform,
    x,
    y,
):
    """Calculate observer elevation using bilinear interpolation."""

    elevation = bilinear_sample(
        elevation_grid,
        inverse_transform,
        x,
        y,
    )

    if not math.isfinite(elevation):
        raise ValueError(
            "Observer elevation is outside valid terrain data."
        )

    return elevation


def calculate_terrain_horizon(
    terrain_path,
    latitude,
    longitude,
    max_distance_m=5000,
    distance_step_m=5,
    azimuth_step_deg=0.25,
):
    """
    Calculate terrain horizon around a lunar observer.

    Uses:
        - Spherical lunar coordinates
        - Raster CRS transformation
        - Bilinear interpolation throughout
        - Spherical lunar curvature geometry
    """

    if max_distance_m <= 0:
        raise ValueError("Maximum distance must be positive.")

    if distance_step_m <= 0:
        raise ValueError("Distance step must be positive.")

    if not 0 < azimuth_step_deg <= 360:
        raise ValueError(
            "Azimuth step must be between 0 and 360 degrees."
        )

    source_crs = "+proj=longlat +R=1737400 +no_defs"

    horizon_profile = []

    with rasterio.open(terrain_path) as src:

        elevation_grid = src.read(1).astype(np.float64)
        inverse_transform = ~src.transform

        transformer = Transformer.from_crs(
            source_crs,
            src.crs,
            always_xy=True,
        )

        observer_x, observer_y = transformer.transform(
            longitude,
            latitude,
        )

        observer_elevation = calculate_observer_elevation(
            elevation_grid,
            inverse_transform,
            observer_x,
            observer_y,
        )

        distances = np.arange(
            distance_step_m,
            max_distance_m + 0.001,
            distance_step_m,
        )

        azimuths = np.arange(
            0.0,
            360.0,
            azimuth_step_deg,
        )

        observer_radius = MOON_RADIUS_M + observer_elevation

        for azimuth in azimuths:

            sample_latitudes = []
            sample_longitudes = []

            for distance in distances:

                sample_lat, sample_lon = destination_point(
                    latitude,
                    longitude,
                    float(azimuth),
                    float(distance),
                )

                sample_latitudes.append(sample_lat)
                sample_longitudes.append(sample_lon)

            projected_x, projected_y = transformer.transform(
                sample_longitudes,
                sample_latitudes,
            )

            terrain_values = np.array([
                bilinear_sample(
                    elevation_grid,
                    inverse_transform,
                    x,
                    y,
                )
                for x, y in zip(projected_x, projected_y)
            ])

            valid = np.isfinite(terrain_values)

            valid_elevations = terrain_values[valid]
            valid_distances = distances[valid]

            if len(valid_elevations) == 0:

                horizon_profile.append({
                    "azimuth_deg": float(azimuth),
                    "horizon_elevation_deg": float("nan"),
                    "samples": 0,
                })

                continue

            central_angle = (
                valid_distances / MOON_RADIUS_M
            )

            terrain_radius = (
                MOON_RADIUS_M + valid_elevations
            )

            horizontal_component = (
                terrain_radius * np.sin(central_angle)
            )

            vertical_component = (
                terrain_radius * np.cos(central_angle)
                - observer_radius
            )

            elevation_angles = np.degrees(
                np.arctan2(
                    vertical_component,
                    horizontal_component,
                )
            )

            peak_index = int(np.argmax(elevation_angles))

            horizon_profile.append({
                "azimuth_deg": float(azimuth),
                "horizon_elevation_deg": float(
                    elevation_angles[peak_index]
                ),
                "samples": int(len(valid_elevations)),
                "peak_distance_m": float(
                    valid_distances[peak_index]
                ),
                "peak_terrain_elevation_m": float(
                    valid_elevations[peak_index]
                ),
            })

    return {
        "observer_elevation_m": observer_elevation,
        "horizon_profile": horizon_profile,
        "max_distance_m": float(max_distance_m),
        "distance_step_m": float(distance_step_m),
        "azimuth_step_deg": float(azimuth_step_deg),
    }


def compare_sun_with_terrain(
    horizon_result,
    sun_azimuth_deg,
    sun_elevation_deg,
):
    """Compare Sun elevation with the interpolated terrain horizon."""

    profile = horizon_result["horizon_profile"]

    valid_profile = [
        item for item in profile
        if math.isfinite(item["horizon_elevation_deg"])
    ]

    if not valid_profile:
        raise ValueError(
            "No valid terrain horizon samples were found."
        )

    azimuths = np.array([
        item["azimuth_deg"] for item in valid_profile
    ])

    horizons = np.array([
        item["horizon_elevation_deg"] for item in valid_profile
    ])

    order = np.argsort(azimuths)

    azimuths = azimuths[order]
    horizons = horizons[order]

    extended_azimuths = np.concatenate((
        [azimuths[-1] - 360.0],
        azimuths,
        [azimuths[0] + 360.0],
    ))

    extended_horizons = np.concatenate((
        [horizons[-1]],
        horizons,
        [horizons[0]],
    ))

    horizon_at_sun = float(np.interp(
        sun_azimuth_deg % 360.0,
        extended_azimuths,
        extended_horizons,
    ))

    clearance = sun_elevation_deg - horizon_at_sun

    return {
        "sun_azimuth_deg": float(sun_azimuth_deg),
        "sun_elevation_deg": float(sun_elevation_deg),
        "terrain_horizon_deg": horizon_at_sun,
        "angular_clearance_deg": float(clearance),
        "visible": bool(clearance > 0),
    }


if __name__ == "__main__":

    TERRAIN_PATH = "data/terrain/Site01_elevation.tif"

    LATITUDE = -89.4632
    LONGITUDE = -137.49

    TEST_SUN_AZIMUTH = 282.37
    TEST_SUN_ELEVATION = 1.69

    print("\nLUNAVIEW TERRAIN HORIZON TEST")
    print("-" * 45)

    result = calculate_terrain_horizon(
        TERRAIN_PATH,
        LATITUDE,
        LONGITUDE,
    )

    print(
        "Observer elevation:",
        round(result["observer_elevation_m"], 3),
        "m",
    )

    print("Radial sampling:", result["distance_step_m"], "m")
    print("Azimuth sampling:", result["azimuth_step_deg"], "degrees")

    comparison = compare_sun_with_terrain(
        result,
        TEST_SUN_AZIMUTH,
        TEST_SUN_ELEVATION,
    )

    print("\nSUN VISIBILITY TEST")
    print("-" * 45)
    print("Sun azimuth:", round(TEST_SUN_AZIMUTH, 2), "degrees")
    print("Sun elevation:", round(TEST_SUN_ELEVATION, 2), "degrees")
    print(
        "Terrain horizon:",
        round(comparison["terrain_horizon_deg"], 3),
        "degrees",
    )
    print(
        "Angular clearance:",
        round(comparison["angular_clearance_deg"], 3),
        "degrees",
    )

    if comparison["visible"]:
        print("RESULT: ABOVE TERRAIN HORIZON")
    else:
        print("RESULT: BLOCKED BY TERRAIN")

    print("\nTERRAIN HORIZON PROFILE")
    print("-" * 45)

    for item in result["horizon_profile"]:

        if abs(item["azimuth_deg"] % 10.0) < 0.001:

            print(
                f"Azimuth {item['azimuth_deg']:6.1f}° | "
                f"Horizon "
                f"{item['horizon_elevation_deg']:7.2f}° | "
                f"Samples {item['samples']:4d} | "
                f"Peak distance "
                f"{item.get('peak_distance_m', float('nan')):7.1f} m"
            )

    print("\nTERRAIN HORIZON TEST COMPLETED")
