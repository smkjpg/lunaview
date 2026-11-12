from pathlib import Path
import spiceypy as spice

kernel_dir = Path("data/kernels")

print("Loading lunar SPICE kernels...")

spice.furnsh(str(kernel_dir / "naif0012.tls"))
spice.furnsh(str(kernel_dir / "pck00011.tpc"))
spice.furnsh(str(kernel_dir / "de440s.bsp"))
spice.furnsh(str(kernel_dir / "moon_pa_de440_200625.bpc"))

utc = "2026 NOV 12 11:50:00 UTC"
et = spice.str2et(utc)

try:
    rotation = spice.pxform("MOON_PA", "J2000", et)
    radii = spice.bodvrd("MOON", "RADII", 3)[1]

    print("\nLUNAVIEW LUNAR FRAME TEST")
    print("-------------------------")
    print("Observation time:", utc)
    print("Moon radii (km):", radii)
    print("MOON_PA to J2000 rotation:")
    print(rotation)
    print("\nSUCCESS: Lunar orientation frame is working!")

except Exception as error:
    print("\nSPICE ERROR:", error)

finally:
    spice.kclear()
