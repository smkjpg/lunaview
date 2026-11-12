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
        raise ValueError("Cannot normalize a zero-length vector.")
    return [x / length for x in vector]


def calculate_sun_position(
    latitude_deg,
    longitude_deg,
    utc_time,
    altitude_km=0.0
):
    """Calculate geometric Sun azimuth and elevation at a lunar site.

    Latitude: degrees, north positive.
    Longitude: degrees, east positive.
    Altitude: kilometres above the lunar reference ellipsoid.
    Azimuth: degrees clockwise from local north.
    """

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

        sun_state, _ = spice.spkezr(
            "SUN", et, "MOON_PA", "NONE", "MOON"
        )

        relative = [
            sun_state[i] - observer[i] for i in range(3)
        ]

        # Local ellipsoid surface directions.
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

        return {
            "latitude_deg": latitude_deg,
            "longitude_deg": longitude_deg,
            "utc_time": utc_time,
            "sun_azimuth_deg": azimuth,
            "sun_elevation_deg": elevation,
            "sun_range_km": spice.vnorm(relative),
        }

    finally:
        spice.kclear()


if __name__ == "__main__":
    result = calculate_sun_position(
        latitude_deg=0.0,
        longitude_deg=0.0,
        utc_time="2026 NOV 12 11:50:00 UTC"
    )

    print("\nLUNAVIEW SUN POSITION")
    print("---------------------")

    for key, value in result.items():
        print(f"{key}: {value}")
