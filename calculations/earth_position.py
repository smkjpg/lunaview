from pathlib import Path
import math
import spiceypy as spice

KERNEL_DIR = Path(__file__).resolve().parents[1] / "data" / "kernels"

KERNELS = [
    "naif0012.tls",
    "pck00011.tpc",
    "de440s.bsp",
    "moon_pa_de440_200625.bpc",
    "moon_de440_250416.tf",
]


def normalize(vector):
    length = math.sqrt(sum(x * x for x in vector))
    if length == 0:
        raise ValueError("Cannot normalize zero-length vector.")
    return [x / length for x in vector]


def calculate_earth_position(
    latitude_deg,
    longitude_deg,
    utc_time,
    altitude_km=0.0
):
    if not -90 <= latitude_deg <= 90:
        raise ValueError("Latitude must be between -90 and 90 degrees.")

    if not -180 <= longitude_deg <= 180:
        raise ValueError("Longitude must be between -180 and 180 degrees.")

    try:
        for filename in KERNELS:
            spice.furnsh(str(KERNEL_DIR / filename))

        et = spice.str2et(utc_time)

        radii = spice.bodvrd("MOON", "RADII", 3)[1]
        a, c = radii[0], radii[2]
        flattening = (a - c) / a

        lat = math.radians(latitude_deg)
        lon = math.radians(longitude_deg)

        observer = spice.georec(
            lon, lat, altitude_km, a, flattening
        )

        earth_state, _ = spice.spkezr(
            "EARTH", et, "MOON_ME", "NONE", "MOON"
        )

        relative = [
            earth_state[i] - observer[i] for i in range(3)
        ]

        up = normalize([
            observer[0] / a**2,
            observer[1] / a**2,
            observer[2] / c**2,
        ])

        east = [
            -math.sin(lon),
            math.cos(lon),
            0.0,
        ]

        north = normalize([
            up[1] * east[2] - up[2] * east[1],
            up[2] * east[0] - up[0] * east[2],
            up[0] * east[1] - up[1] * east[0],
        ])

        def dot(v1, v2):
            return sum(x * y for x, y in zip(v1, v2))

        e = dot(relative, east)
        n = dot(relative, north)
        u = dot(relative, up)

        azimuth = math.degrees(math.atan2(e, n)) % 360

        elevation = math.degrees(
            math.atan2(u, math.hypot(e, n))
        )

        distance = spice.vnorm(relative)

        # Approximate apparent angular radius of Earth.
        earth_radius_km = 6371.0
        angular_radius = math.degrees(
            math.asin(min(1.0, earth_radius_km / distance))
        )

        disk_visible = elevation + angular_radius > 0
        full_disk_visible = elevation - angular_radius > 0

        return {
            "latitude_deg": latitude_deg,
            "longitude_deg": longitude_deg,
            "utc_time": utc_time,
            "earth_azimuth_deg": azimuth,
            "earth_elevation_deg": elevation,
            "earth_distance_km": distance,
            "earth_angular_radius_deg": angular_radius,
            "earth_disk_visible": disk_visible,
            "earth_full_disk_visible": full_disk_visible,
        }

    finally:
        spice.kclear()


if __name__ == "__main__":
    result = calculate_earth_position(
        latitude_deg=0.0,
        longitude_deg=0.0,
        utc_time="2026 NOV 12 11:50:00 UTC"
    )

    print("\nLUNAVIEW EARTH POSITION TEST")
    print("---------------------------")

    for key, value in result.items():
        print(f"{key}: {value}")
