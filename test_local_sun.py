from pathlib import Path
import math
import spiceypy as spice

d = Path("data/kernels")

try:
    for filename in [
        "naif0012.tls",
        "pck00011.tpc",
        "de440s.bsp",
        "moon_pa_de440_200625.bpc",
        "moon_de440_250416.tf",
    ]:
        spice.furnsh(str(d / filename))

    # Temporary test coordinates
    lat_deg = 0.0
    lon_deg = 0.0
    utc = "2026 NOV 12 11:50:00 UTC"

    et = spice.str2et(utc)
    radii = spice.bodvrd("MOON", "RADII", 3)[1]
    a, c = radii[0], radii[2]
    flattening = (a - c) / a

    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    observer = spice.georec(lon, lat, 0.0, a, flattening)

    # Sun position relative to Moon center, in Moon-fixed coordinates
    sun_state, _ = spice.spkezr(
        "SUN", et, "MOON_PA", "NONE", "MOON"
    )

    relative = sun_state[:3] - observer

    # Local ellipsoid surface normal
    up = [
        observer[0] / a**2,
        observer[1] / a**2,
        observer[2] / c**2,
    ]

    def normalize(v):
        length = math.sqrt(sum(x*x for x in v))
        return [x / length for x in v]

    up = normalize(up)

    east = [-math.sin(lon), math.cos(lon), 0.0]

    north = [
        up[1] * east[2] - up[2] * east[1],
        up[2] * east[0] - up[0] * east[2],
        up[0] * east[1] - up[1] * east[0],
    ]
    north = normalize(north)

    dot = lambda x, y: sum(a*b for a, b in zip(x, y))

    east_component = dot(relative, east)
    north_component = dot(relative, north)
    up_component = dot(relative, up)

    azimuth = math.degrees(
        math.atan2(east_component, north_component)
    ) % 360

    elevation = math.degrees(
        math.atan2(
            up_component,
            math.hypot(east_component, north_component)
        )
    )

    distance = spice.vnorm(relative)

    print("\nLUNAVIEW LOCAL SUN POSITION")
    print("---------------------------")
    print("UTC:", utc)
    print("Latitude:", lat_deg)
    print("Longitude:", lon_deg)
    print(f"Sun azimuth from north: {azimuth:.4f} degrees")
    print(f"Sun elevation: {elevation:.4f} degrees")
    print(f"Sun distance: {distance:.3f} km")

finally:
    spice.kclear()
