from pathlib import Path
import requests

KERNEL_DIR = Path("data/kernels")
KERNEL_DIR.mkdir(parents=True, exist_ok=True)

KERNELS = {
    "naif0012.tls":
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",

    "pck00011.tpc":
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc",

    "de440s.bsp":
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp",
}

for filename, url in KERNELS.items():
    destination = KERNEL_DIR / filename

    if destination.exists():
        print(f"Already downloaded: {filename}")
        continue

    print(f"Downloading {filename}...")

    response = requests.get(url, timeout=180)
    response.raise_for_status()

    destination.write_bytes(response.content)
    print(f"Downloaded: {filename}")

print("Kernel download finished!")
