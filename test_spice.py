from pathlib import Path
import spiceypy as spice

kernel_dir = Path("data/kernels")

print("Loading NASA SPICE kernels...")

spice.furnsh(str(kernel_dir / "naif0012.tls"))
spice.furnsh(str(kernel_dir / "pck00011.tpc"))
spice.furnsh(str(kernel_dir / "de440s.bsp"))

print("All kernels loaded successfully!")

# Your selected observation time
utc = "2026 NOV 12 11:50:00 UTC"

# Convert UTC to SPICE ephemeris time
et = spice.str2et(utc)

# Calculate Sun position relative to Moon's centre
state, light_time = spice.spkezr(
    "SUN",
    et,
    "J2000",
    "NONE",
    "MOON"
)

distance = spice.vnorm(state[:3])

print("\nLUNAVIEW SPICE TEST")
print("-------------------")
print("Observation time:", utc)
print("Ephemeris time:", et)
print("Sun X:", state[0], "km")
print("Sun Y:", state[1], "km")
print("Sun Z:", state[2], "km")
print("Sun distance:", distance, "km")

spice.kclear()

print("\nSPICE TEST COMPLETED!")
