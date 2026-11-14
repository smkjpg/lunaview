
from datetime import datetime, timedelta, timezone
from pathlib import Path
import math

import numpy as np
import spiceypy as spice


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
KERNEL_DIR = PROJECT_DIR / "data" / "kernels"

KERNELS = [
    "naif0012.tls",
    "pck00011.tpc",
    "de440s.bsp",
    "moon_pa_de440_200625.bpc",
    "moon_de440_250416.tf",
]

EARTH_RADIUS_KM = 6371.0


# ============================================================
# VECTOR AND COORDINATE HELPERS
# ============================================================

def normalize(vector):
    length = math.sqrt(sum(value * value for value in vector))

    if length == 0:
        raise ValueError("Cannot normalize a zero-length vector.")

    return [value / length for value in vector]


def dot(vector_a, vector_b):
    return sum(
        a * b for a, b in zip(vector_a, vector_b)
    )


def get_local_directions(latitude_deg, longitude_deg, observer, a, c):
    """Calculate local east, north and up directions."""

    latitude = math.radians(latitude_deg)
    longitude = math.radians(longitude_deg)

    up = normalize([
        observer[0] / a**2,
        observer[1] / a**2,
        observer[2] / c**2,
    ])

    east = [
        -math.sin(longitude),
        math.cos(longitude),
        0.0,
    ]

    north = normalize([
        up[1] * east[2] - up[2] * east[1],
        up[2] * east[0] - up[0] * east[2],
        up[0] * east[1] - up[1] * east[0],
    ])

    return east, north, up


def calculate_azimuth_elevation(relative, east, north, up):
    """Calculate azimuth clockwise from north and elevation."""

    east_component = dot(relative, east)
    north_component = dot(relative, north)
    up_component = dot(relative, up)

    azimuth = math.degrees(
        math.atan2(east_component, north_component)
    ) % 360.0

    elevation = math.degrees(
        math.atan2(
            up_component,
            math.hypot(east_component, north_component),
        )
    )

    return azimuth, elevation


# ============================================================
# TIME HELPERS
# ============================================================

def validate_datetime(value, name):
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime object.")

    if value.tzinfo is None:
        raise ValueError(
            f"{name} must include a timezone. "
            "Use timezone.utc for UTC."
        )

    return value.astimezone(timezone.utc)


def format_spice_time(value):
    """Convert an aware datetime to a SPICE-compatible UTC string."""

    return value.strftime("%Y %b %d %H:%M:%S UTC").upper()


# ============================================================
# TERRAIN HORIZON INTERPOLATION
# ============================================================

def get_horizon_elevation(horizon_result, azimuth_deg):
    """
    Interpolate the terrain horizon at a requested azimuth.

    If no terrain profile is supplied, use a geometric
    zero-degree horizon.
    """

    if horizon_result is None:
        return 0.0

    profile = horizon_result.get("horizon_profile", [])

    valid = [
        item for item in profile
        if math.isfinite(
            item.get("horizon_elevation_deg", float("nan"))
        )
    ]

    if len(valid) < 2:
        raise ValueError(
            "Terrain horizon profile has insufficient valid samples."
        )

    azimuths = np.array([
        item["azimuth_deg"] for item in valid
    ], dtype=float)

    horizons = np.array([
        item["horizon_elevation_deg"] for item in valid
    ], dtype=float)

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

    return float(np.interp(
        azimuth_deg % 360.0,
        extended_azimuths,
        extended_horizons,
    ))


# ============================================================
# SOLAR PANEL MODEL
# ============================================================

def calculate_panel_output(
    sun_azimuth_deg,
    sun_elevation_deg,
    panel_rated_power_w,
    panel_tilt_deg=0.0,
    panel_azimuth_deg=0.0,
    sun_visible=True,
):
    """
    Estimate solar-panel electrical output.

    panel_rated_power_w:
        Rated output under normal incidence and reference
        illumination conditions.

    panel_tilt_deg:
        Panel tilt measured upward from the local horizontal.

    panel_azimuth_deg:
        Direction the panel faces, clockwise from north.

    This is a simplified orientation model. It does not
    model temperature, dust, degradation, battery losses,
    or detailed electrical behaviour.
    """

    if panel_rated_power_w < 0:
        raise ValueError("Panel rated power cannot be negative.")

    if not 0 <= panel_tilt_deg <= 180:
        raise ValueError("Panel tilt must be between 0 and 180 degrees.")

    if not sun_visible or sun_elevation_deg <= 0:
        return 0.0

    sun_elevation = math.radians(sun_elevation_deg)
    sun_azimuth = math.radians(sun_azimuth_deg)

    panel_tilt = math.radians(panel_tilt_deg)
    panel_azimuth = math.radians(panel_azimuth_deg)

    incidence_cosine = (
        math.sin(sun_elevation) * math.cos(panel_tilt)
        + math.cos(sun_elevation)
        * math.sin(panel_tilt)
        * math.cos(sun_azimuth - panel_azimuth)
    )

    incidence_factor = max(0.0, incidence_cosine)

    return panel_rated_power_w * incidence_factor


# ============================================================
# MAIN MISSION WINDOW CALCULATION
# ============================================================

def calculate_mission_windows(
    latitude_deg,
    longitude_deg,
    start_utc,
    end_utc,
    interval_minutes=30,
    altitude_km=0.0,
    horizon_result=None,
    antenna_min_elevation_deg=5.0,
    panel_rated_power_w=100.0,
    panel_tilt_deg=0.0,
    panel_azimuth_deg=0.0,
):
    """
    Calculate sunlight and direct-to-Earth communication
    opportunities over a selected UTC period.

    Uses the existing LunaView SPICE kernels.

    Terrain horizon:
        Supply the result from calculate_terrain_horizon()
        to account for local terrain obstruction.

    Communication:
        Requires the Earth centre to be above the local
        terrain horizon plus the antenna minimum elevation.

    Solar power:
        Uses a simplified fixed-panel orientation model.
    """

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    if not -90 <= latitude_deg <= 90:
        raise ValueError("Latitude must be between -90 and 90 degrees.")

    if not -180 <= longitude_deg <= 180:
        raise ValueError("Longitude must be between -180 and 180 degrees.")

    start_utc = validate_datetime(start_utc, "start_utc")
    end_utc = validate_datetime(end_utc, "end_utc")

    if end_utc <= start_utc:
        raise ValueError("End time must be after start time.")

    if interval_minutes <= 0:
        raise ValueError("Interval must be greater than zero.")

    if antenna_min_elevation_deg < 0:
        raise ValueError("Antenna minimum elevation cannot be negative.")

    if panel_rated_power_w < 0:
        raise ValueError("Panel rated power cannot be negative.")

    if altitude_km < 0:
        raise ValueError("Altitude cannot be negative.")

    interval_seconds = interval_minutes * 60

    duration_seconds = (
        end_utc - start_utc
    ).total_seconds()

    sample_count = math.ceil(
        duration_seconds / interval_seconds
    ) + 1

    if sample_count > 5000:
        raise ValueError(
            "This request exceeds 5,000 samples. "
            "Increase interval_minutes or shorten the period."
        )

    # --------------------------------------------------------
    # Load SPICE kernels once for the complete calculation
    # --------------------------------------------------------

    furnished = []

    try:
        for filename in KERNELS:
            kernel_path = KERNEL_DIR / filename

            if not kernel_path.exists():
                raise FileNotFoundError(
                    f"Required SPICE kernel not found: {kernel_path}"
                )

            spice.furnsh(str(kernel_path))
            furnished.append(str(kernel_path))

        radii = spice.bodvrd("MOON", "RADII", 3)[1]

        a = float(radii[0])
        c = float(radii[2])

        flattening = (a - c) / a

        latitude = math.radians(latitude_deg)
        longitude = math.radians(longitude_deg)

        observer = spice.georec(
            longitude,
            latitude,
            altitude_km,
            a,
            flattening,
        )

        east, north, up = get_local_directions(
            latitude_deg,
            longitude_deg,
            observer,
            a,
            c,
        )

        # ----------------------------------------------------
        # Generate time samples
        # ----------------------------------------------------

        timestamps = []
        current = start_utc

        while current <= end_utc:
            timestamps.append(current)
            current += timedelta(seconds=interval_seconds)

        if timestamps[-1] < end_utc:
            timestamps.append(end_utc)

        records = []

        # ----------------------------------------------------
        # Calculate Sun and Earth positions
        # ----------------------------------------------------

        for timestamp in timestamps:

            et = spice.str2et(
                format_spice_time(timestamp)
            )

            sun_state, _ = spice.spkezr(
                "SUN",
                et,
                "MOON_ME",
                "NONE",
                "MOON",
            )

            earth_state, _ = spice.spkezr(
                "EARTH",
                et,
                "MOON_ME",
                "NONE",
                "MOON",
            )

            sun_relative = [
                sun_state[i] - observer[i]
                for i in range(3)
            ]

            earth_relative = [
                earth_state[i] - observer[i]
                for i in range(3)
            ]

            sun_azimuth, sun_elevation = (
                calculate_azimuth_elevation(
                    sun_relative,
                    east,
                    north,
                    up,
                )
            )

            earth_azimuth, earth_elevation = (
                calculate_azimuth_elevation(
                    earth_relative,
                    east,
                    north,
                    up,
                )
            )

            sun_range = spice.vnorm(sun_relative)
            earth_range = spice.vnorm(earth_relative)

            earth_angular_radius = math.degrees(
                math.asin(
                    min(1.0, EARTH_RADIUS_KM / earth_range)
                )
            )

            # ------------------------------------------------
            # Terrain obstruction
            # ------------------------------------------------

            sun_horizon = get_horizon_elevation(
                horizon_result,
                sun_azimuth,
            )

            earth_horizon = get_horizon_elevation(
                horizon_result,
                earth_azimuth,
            )

            sun_clearance = (
                sun_elevation - sun_horizon
            )

            earth_clearance = (
                earth_elevation - earth_horizon
            )

            # Sun centre must be above the terrain horizon.
            sun_visible = sun_clearance > 0.0

            # Physical Earth disk can be partially visible
            # even when its centre is below the horizon.
            earth_disk_visible = (
                earth_elevation + earth_angular_radius
                > earth_horizon
            )

            # For communication, require the Earth centre
            # to clear the terrain and antenna elevation limit.
            communication_visible = (
                earth_clearance
                >= antenna_min_elevation_deg
            )

            panel_power = calculate_panel_output(
                sun_azimuth_deg=sun_azimuth,
                sun_elevation_deg=sun_elevation,
                panel_rated_power_w=panel_rated_power_w,
                panel_tilt_deg=panel_tilt_deg,
                panel_azimuth_deg=panel_azimuth_deg,
                sun_visible=sun_visible,
            )

            records.append({
                "utc_time": timestamp.isoformat(),
                "sun_azimuth_deg": sun_azimuth,
                "sun_elevation_deg": sun_elevation,
                "sun_terrain_horizon_deg": sun_horizon,
                "sun_clearance_deg": sun_clearance,
                "sun_visible": sun_visible,
                "earth_azimuth_deg": earth_azimuth,
                "earth_elevation_deg": earth_elevation,
                "earth_angular_radius_deg": earth_angular_radius,
                "earth_terrain_horizon_deg": earth_horizon,
                "earth_clearance_deg": earth_clearance,
                "earth_disk_visible": earth_disk_visible,
                "communication_visible": communication_visible,
                "panel_power_w": panel_power,
                "sun_range_km": sun_range,
                "earth_range_km": earth_range,
            })

        # ----------------------------------------------------
        # Integrate durations and energy
        # ----------------------------------------------------

        sunlight_seconds = 0.0
        communication_seconds = 0.0
        energy_wh = 0.0

        for index in range(len(records) - 1):

            first = records[index]
            second = records[index + 1]

            elapsed_hours = (
                datetime.fromisoformat(second["utc_time"])
                - datetime.fromisoformat(first["utc_time"])
            ).total_seconds() / 3600.0

            elapsed_seconds = elapsed_hours * 3600.0

            # Trapezoidal integration of binary visibility.
            sunlight_seconds += (
                int(first["sun_visible"])
                + int(second["sun_visible"])
            ) / 2.0 * elapsed_seconds

            communication_seconds += (
                int(first["communication_visible"])
                + int(second["communication_visible"])
            ) / 2.0 * elapsed_seconds

            # Trapezoidal integration of panel power.
            energy_wh += (
                first["panel_power_w"]
                + second["panel_power_w"]
            ) / 2.0 * elapsed_hours

        total_hours = duration_seconds / 3600.0

        summary = {
            "period_start_utc": start_utc.isoformat(),
            "period_end_utc": end_utc.isoformat(),
            "total_period_hours": round(total_hours, 3),
            "sunlight_hours": round(sunlight_seconds / 3600.0, 3),
            "sunlight_percentage": round(
                sunlight_seconds / duration_seconds * 100.0,
                2,
            ),
            "communication_hours": round(
                communication_seconds / 3600.0,
                3,
            ),
            "communication_percentage": round(
                communication_seconds / duration_seconds * 100.0,
                2,
            ),
            "estimated_energy_wh": round(energy_wh, 3),
            "panel_rated_power_w": panel_rated_power_w,
            "antenna_min_elevation_deg": antenna_min_elevation_deg,
            "sample_count": len(records),
            "interval_minutes": interval_minutes,
            "terrain_horizon_used": horizon_result is not None,
        }

        return {
            "summary": summary,
            "records": records,
        }

    finally:
        # Clear the kernels loaded by this calculation.
        spice.kclear()


# ============================================================
# SIMPLE COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    result = calculate_mission_windows(
        latitude_deg=-89.4632,
        longitude_deg=-137.49,
        start_utc=datetime(
            2026, 11, 14, 0, 0,
            tzinfo=timezone.utc,
        ),
        end_utc=datetime(
            2026, 11, 15, 0, 0,
            tzinfo=timezone.utc,
        ),
        interval_minutes=30,
        panel_rated_power_w=100.0,
        panel_tilt_deg=0.0,
        panel_azimuth_deg=0.0,
    )

    print("\nLUNAVIEW MISSION WINDOW TEST")
    print("=" * 50)

    for key, value in result["summary"].items():
        print(f"{key}: {value}")

    print("\nFIRST THREE TIME SAMPLES")
    print("-" * 50)

    for record in result["records"][:3]:
        print(
            record["utc_time"],
            "| Sun:",
            round(record["sun_elevation_deg"], 2),
            "degrees",
            "| Earth:",
            round(record["earth_elevation_deg"], 2),
            "degrees",
            "| Comms:",
            record["communication_visible"],
        )
