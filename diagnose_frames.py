from pathlib import Path
import spiceypy as spice

d = Path("data/kernels")

files = [
    "naif0012.tls",
    "pck00011.tpc",
    "de440s.bsp",
    "moon_pa_de440_200625.bpc",
    "moon_de440_250416.tf",
]

try:
    for filename in files:
        path = d / filename
        print(f"Loading: {filename}")
        spice.furnsh(str(path))

    print("\nCHECKING FRAME DEFINITIONS")

    for name in ["MOON_PA", "MOON_PA_DE440", "IAU_MOON"]:
        try:
            code = spice.namfrm(name)
            print(f"{name}: frame ID = {code}")
        except Exception as e:
            print(f"{name}: ERROR = {e}")

    print("\nCHECKING ROTATION")

    et = spice.str2et("2026 NOV 12 11:50:00 UTC")

    for name in ["MOON_PA", "MOON_PA_DE440", "IAU_MOON"]:
        try:
            rotation = spice.pxform(name, "J2000", et)
            print(f"{name}: SUCCESS")
            print(rotation)
        except Exception as e:
            print(f"{name}: FAILED - {e}")

finally:
    spice.kclear()
