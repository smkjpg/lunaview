
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import rasterio
import streamlit as st
from pyproj import Transformer
from rasterio.windows import Window

from calculations.sun_position import calculate_sun_position
from calculations.earth_position import calculate_earth_position
from calculations.terrain_horizon import (
    calculate_terrain_horizon,
    compare_sun_with_terrain,
)
from calculations.mission_windows import calculate_mission_windows


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
TERRAIN_DIR = ROOT / "data" / "terrain"
DEPLOYMENT_DIR = TERRAIN_DIR / "deployment"
GLOBAL_DIR = TERRAIN_DIR / "global"
SITE_DATABASE_PATH = ROOT / "data" / "sites" / "lunar_sites.csv"

MOON_RADIUS_M = 1_737_400.0
MOON_GEOGRAPHIC_CRS = "+proj=longlat +R=1737400 +no_defs"

GLOBAL_ELEVATION = GLOBAL_DIR / "LOLA_global_elevation.tif"
GLOBAL_SLOPE = GLOBAL_DIR / "LOLA_global_slope.tif"

POLAR_ELEVATION = TERRAIN_DIR / "LDEM_80S_80MPP_ADJ_elevation.tif"
POLAR_SLOPE = TERRAIN_DIR / "LDSM_80S_80MPP_ADJ_slope.tif"

HIGH_RES_SITE_IDS = {
    "SITE01", "SITE04", "SITE06", "SITE07",
    "SITE11", "SITE20", "SITE23", "SITE42",
}

POLAR_80M_SITE_IDS = {
    "SITE09", "SITE10", "SITE12", "SITE15",
    "SITE18", "SITE25", "SITE30", "SITE32",
    "SITE35", "SITE38", "SITE40", "SITE45",
}


# ============================================================
# SOUTH-POLE CANDIDATE SITES
# ============================================================

SITE_RECORDS = [
    ("SITE01", "Connecting Ridge", -89.4632, -137.4900),
    ("SITE04", "Shackleton Rim", -89.7668, -171.8700),
    ("SITE06", "Nobile Rim 1", -85.4381, 37.3667),
    ("SITE07", "Peak Near Shackleton", -88.8110, 123.6901),
    ("SITE11", "de Gerlache Rim 1", -88.6834, -67.9321),
    ("SITE20", "Leibnitz Beta Plateau", -85.4266, 31.7428),
    ("SITE23", "Malapert Massif", -85.9948, -0.2358),
    ("SITE42", "de Gerlache-Kocher Massif", -85.8296, -116.3220),
    ("SITE09", "Amundsen Crater Rim", -84.2500, 83.5000),
    ("SITE10", "Faustini Rim A", -87.3000, 75.2000),
    ("SITE12", "Haworth Crater Rim", -87.4000, -2.1000),
    ("SITE15", "Shoemaker Rim", -88.1000, 44.5000),
    ("SITE18", "Cabeus B Ridge", -81.8000, -38.5000),
    ("SITE25", "Nobile-Amundsen Plateau", -84.6000, 62.1000),
    ("SITE30", "Scott Plateau", -82.1000, 51.3000),
    ("SITE32", "Idel'son Ridge", -84.1000, 115.4000),
    ("SITE35", "Henson Ridge", -79.2000, -139.8000),
    ("SITE38", "Wiechert Rim", -84.8000, 169.2000),
    ("SITE40", "Sverdrup Rim", -88.3000, -152.0000),
    ("SITE45", "Shackleton-de Gerlache Ridge", -89.1000, -125.4000),
]


# ============================================================
# HISTORICAL AND ROBOTIC LANDING SITES
# ============================================================

HISTORICAL_RECORDS = [
    {
        "site_id": "APOLLO11",
        "site_name": "Apollo 11 landing site",
        "mission": "Apollo 11",
        "agency": "NASA",
        "site_type": "Historical landing",
        "status": "Completed",
        "latitude": 0.67416,
        "longitude": 23.47314,
    },
    {
        "site_id": "APOLLO12",
        "site_name": "Apollo 12 landing site",
        "mission": "Apollo 12",
        "agency": "NASA",
        "site_type": "Historical landing",
        "status": "Completed",
        "latitude": -3.0128,
        "longitude": -23.4219,
    },
    {
        "site_id": "APOLLO14",
        "site_name": "Apollo 14 landing site",
        "mission": "Apollo 14",
        "agency": "NASA",
        "site_type": "Historical landing",
        "status": "Completed",
        "latitude": -3.64589,
        "longitude": -17.47194,
    },
    {
        "site_id": "APOLLO15",
        "site_name": "Apollo 15 landing site",
        "mission": "Apollo 15",
        "agency": "NASA",
        "site_type": "Historical landing",
        "status": "Completed",
        "latitude": 26.13239,
        "longitude": 3.63330,
    },
    {
        "site_id": "APOLLO16",
        "site_name": "Apollo 16 landing site",
        "mission": "Apollo 16",
        "agency": "NASA",
        "site_type": "Historical landing",
        "status": "Completed",
        "latitude": -8.9734,
        "longitude": 15.5011,
    },
    {
        "site_id": "APOLLO17",
        "site_name": "Apollo 17 landing site",
        "mission": "Apollo 17",
        "agency": "NASA",
        "site_type": "Historical landing",
        "status": "Completed",
        "latitude": 20.1911,
        "longitude": 30.7723,
    },
    {
        "site_id": "CHANDRAYAAN3",
        "site_name": "Shiv Shakti Point",
        "mission": "Chandrayaan-3",
        "agency": "ISRO",
        "site_type": "Historical landing",
        "status": "Completed",
        "latitude": -69.3741,
        "longitude": 32.3200,
    },
    {
        "site_id": "SLIM",
        "site_name": "SLIM landing site",
        "mission": "SLIM",
        "agency": "JAXA",
        "site_type": "Historical landing",
        "status": "Completed",
        "latitude": -13.3160,
        "longitude": 25.2510,
    },
    {
        "site_id": "ODYSSEUS",
        "site_name": "Odysseus landing site",
        "mission": "IM-1 Odysseus",
        "agency": "Intuitive Machines",
        "site_type": "Historical landing",
        "status": "Completed",
        "latitude": -80.1300,
        "longitude": 1.4400,
    },
    {
        "site_id": "CHANGE6",
        "site_name": "Chang'e 6 landing site",
        "mission": "Chang'e 6",
        "agency": "CNSA",
        "site_type": "Historical landing",
        "status": "Completed",
        "latitude": -41.6385,
        "longitude": -153.9852,
    },
    {
        "site_id": "BLUEGHOST1",
        "site_name": "Blue Ghost 1 landing site",
        "mission": "Blue Ghost 1",
        "agency": "Firefly Aerospace",
        "site_type": "Historical landing",
        "status": "Completed",
        "latitude": 18.5620,
        "longitude": 61.8100,
    },
    {
        "site_id": "IM2",
        "site_name": "IM-2 landing site",
        "mission": "IM-2",
        "agency": "Intuitive Machines",
        "site_type": "Historical landing",
        "status": "Completed",
        "latitude": -84.7906,
        "longitude": 29.1957,
    },
]


# ============================================================
# BUILD COMPLETE SITE CATALOGUE
# ============================================================

def build_site_catalogue():
    sites = {}
    details = {}
    id_to_name = {}

    for site_id, name, lat, lon in SITE_RECORDS:
        display = f"{name} ({site_id})"
        sites[display] = (lat, lon)

        details[display] = {
            "Site ID": site_id,
            "Mission": "LunaView south-pole candidate",
            "Agency": "LunaView candidate database",
            "Site Type": "South-pole candidate",
            "Status": "Proposed candidate",
            "Coordinate Reference": (
                "LunaView candidate coordinates; "
                "verify reference frame before precision use"
            ),
            "Source": "",
        }

        id_to_name[site_id] = display

    historical_by_id = {
        record["site_id"]: dict(record)
        for record in HISTORICAL_RECORDS
    }

    if SITE_DATABASE_PATH.exists():
        try:
            csv_data = pd.read_csv(SITE_DATABASE_PATH)

            required = {
                "site_id",
                "site_name",
                "mission",
                "agency",
                "site_type",
                "status",
                "latitude_deg",
                "longitude_deg_east",
            }

            if required.issubset(csv_data.columns):
                for _, row in csv_data.iterrows():
                    sid = str(row["site_id"]).strip().upper()

                    if sid == "SITE01" or sid not in historical_by_id:
                        continue

                    try:
                        lat = float(row["latitude_deg"])
                        lon = float(row["longitude_deg_east"])
                    except (TypeError, ValueError):
                        continue

                    if not (
                        np.isfinite(lat)
                        and np.isfinite(lon)
                        and -90 <= lat <= 90
                        and -180 <= lon <= 180
                    ):
                        continue

                    record = historical_by_id[sid]
                    record["latitude"] = lat
                    record["longitude"] = lon
                    record["agency"] = str(
                        row.get("agency", record["agency"])
                    )
                    record["site_type"] = str(
                        row.get("site_type", record["site_type"])
                    )
                    record["status"] = str(
                        row.get("status", record["status"])
                    )

        except Exception as exc:
            st.warning(
                "Could not read lunar_sites.csv. "
                f"Using built-in landing-site catalogue. Details: {exc}"
            )

    for sid, record in historical_by_id.items():
        display = f"{record['site_name']} — {record['mission']}"

        sites[display] = (
            record["latitude"],
            record["longitude"],
        )

        details[display] = {
            "Site ID": sid,
            "Mission": record["mission"],
            "Agency": record["agency"],
            "Site Type": record["site_type"],
            "Status": record["status"],
            "Coordinate Reference": (
                "Source-reported selenographic coordinates"
            ),
            "Source": (
                "NASA lunar artifact/landing-site catalogue "
                "or mission source"
            ),
        }

        id_to_name[sid] = display

    return sites, details, id_to_name


# ============================================================
# TERRAIN DATASET SELECTION
# ============================================================

def resolve_terrain_paths(site_id):
    """
    Resolve the compact, per-site deployment tile first.

    The deployment directory contains one elevation and one matching slope
    tile for each of the 32 catalogue sites. Original source rasters remain
    supported as fallbacks for local development.
    """

    site_id = str(site_id).strip().upper()

    # Preferred deployment path: compact tile generated specifically
    # around this site's coordinates.
    deployment_elevation = DEPLOYMENT_DIR / f"{site_id}_elevation.tif"
    deployment_slope = DEPLOYMENT_DIR / f"{site_id}_slope.tif"

    if deployment_elevation.exists():
        if site_id in HIGH_RES_SITE_IDS:
            return {
                "elevation": deployment_elevation,
                "slope": deployment_slope if deployment_slope.exists() else None,
                "source": "Deployment-optimized site-specific LOLA terrain tile",
                "resolution": "5 m nominal (site-specific product)",
                "horizon_quality": "High-resolution local terrain",
            }

        if site_id in POLAR_80M_SITE_IDS:
            return {
                "elevation": deployment_elevation,
                "slope": deployment_slope if deployment_slope.exists() else None,
                "source": "Deployment-optimized south-polar LOLA terrain tile",
                "resolution": "80 m nominal",
                "horizon_quality": "Regional 80 m terrain",
            }

        return {
            "elevation": deployment_elevation,
            "slope": deployment_slope if deployment_slope.exists() else None,
            "source": "Deployment-optimized global LOLA terrain tile",
            "resolution": (
                "473.8 m/pixel at lunar equator; "
                "east-west spacing varies with latitude"
            ),
            "horizon_quality": (
                "Coarse global terrain; not suitable for small local obstacles"
            ),
        }

    # Original source rasters remain available as fallbacks in Codespaces.
    if site_id in HIGH_RES_SITE_IDS:
        file_site_id = site_id[0].upper() + site_id[1:].lower()
        elevation = TERRAIN_DIR / f"{file_site_id}_elevation.tif"
        slope = TERRAIN_DIR / f"{file_site_id}_slope.tif"

        if elevation.exists():
            return {
                "elevation": elevation,
                "slope": slope if slope.exists() else None,
                "source": "Site-specific high-resolution LOLA terrain",
                "resolution": "5 m nominal (site-specific product)",
                "horizon_quality": "High-resolution local terrain",
            }

    if site_id in POLAR_80M_SITE_IDS and POLAR_ELEVATION.exists():
        return {
            "elevation": POLAR_ELEVATION,
            "slope": POLAR_SLOPE if POLAR_SLOPE.exists() else None,
            "source": "NASA south-polar LOLA terrain mosaic",
            "resolution": "80 m nominal",
            "horizon_quality": "Regional 80 m terrain",
        }

    if GLOBAL_ELEVATION.exists():
        return {
            "elevation": GLOBAL_ELEVATION,
            "slope": GLOBAL_SLOPE if GLOBAL_SLOPE.exists() else None,
            "source": "Global NASA LOLA LDEM_64",
            "resolution": (
                "473.8 m/pixel at lunar equator; "
                "east-west spacing varies with latitude"
            ),
            "horizon_quality": (
                "Coarse global terrain; not suitable for small local obstacles"
            ),
        }

    return {
        "elevation": None,
        "slope": None,
        "source": "No terrain raster found",
        "resolution": "Unavailable",
        "horizon_quality": "Unavailable",
    }


# ============================================================
# READ LOCAL TERRAIN PATCH
# ============================================================

def read_site_patch(path, latitude, longitude, max_pixels=240):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Raster file not found: {path}")

    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {path.name}")

        longitude = float(longitude)
        latitude = float(latitude)

        transformer = Transformer.from_crs(
            MOON_GEOGRAPHIC_CRS,
            src.crs,
            always_xy=True,
        )

        if src.crs.is_geographic:
            # Match the longitude convention of the raster.
            if src.bounds.left >= 0 and src.bounds.right > 180:
                longitude = longitude % 360.0

            elif src.bounds.left < 0 and src.bounds.right <= 180:
                longitude = ((longitude + 180.0) % 360.0) - 180.0

            x_site = longitude
            y_site = latitude

        else:
            x_site, y_site = transformer.transform(
                longitude,
                latitude,
            )

        row, col = src.index(x_site, y_site)

        if (
            row < 0
            or col < 0
            or row >= src.height
            or col >= src.width
        ):
            raise ValueError(
                f"Site coordinate is outside {path.name}. "
                f"Projected pixel row={row}, column={col}."
            )

        half = max(8, max_pixels // 2)

        row0 = max(0, row - half)
        row1 = min(src.height, row + half)
        col0 = max(0, col - half)
        col1 = min(src.width, col + half)

        window = Window(
            col0,
            row0,
            col1 - col0,
            row1 - row0,
        )

        masked = src.read(
            1,
            window=window,
            masked=True,
        )

        arr = np.ma.filled(
            masked.astype("float32"),
            np.nan,
        )

        transform = src.window_transform(window)

        xs = (
            transform.c
            + (np.arange(arr.shape[1]) + 0.5) * transform.a
        )

        ys = (
            transform.f
            + (np.arange(arr.shape[0]) + 0.5) * transform.e
        )

        valid = np.isfinite(arr)
        sample = arr[row - row0, col - col0]

        metadata = {
            "file": path.name,
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "pixel_size": src.res,
            "nodata": src.nodata,
            "valid_pixels_in_patch": int(valid.sum()),
            "sample_value": (
                float(sample) if np.isfinite(sample) else None
            ),
        }

    if not np.isfinite(arr).any():
        raise ValueError(
            f"The selected raster window contains no valid data: {path.name}"
        )

    return xs, ys, arr, metadata



# ============================================================
# ESTIMATE RASTER PIXEL SPACING
# ============================================================

def raster_pixel_resolution_m(path, latitude):
    with rasterio.open(path) as src:
        if src.crs and src.crs.is_geographic:
            dx = (
                abs(src.res[0])
                * math.pi / 180.0
                * MOON_RADIUS_M
                * max(0.01, math.cos(math.radians(latitude)))
            )

            dy = (
                abs(src.res[1])
                * math.pi / 180.0
                * MOON_RADIUS_M
            )

            return float(max(dx, dy))

        return float(max(abs(src.res[0]), abs(src.res[1])))


def longitude_for_raster(path, longitude):
    """
    Convert a signed east longitude to the longitude convention used by
    a geographic raster. Projected rasters are left unchanged because
    pyproj performs the geographic-to-projected transformation.
    """

    longitude = float(longitude)

    with rasterio.open(path) as src:
        if src.crs is None or not src.crs.is_geographic:
            return longitude

        if src.bounds.left >= 0 and src.bounds.right > 180:
            return longitude % 360.0

        if src.bounds.left < 0 and src.bounds.right <= 180:
            return ((longitude + 180.0) % 360.0) - 180.0

    return longitude


# ============================================================
# CALCULATE A SITE'S TERRAIN HORIZON
# ============================================================

def calculate_site_horizon(site_id, latitude, longitude):
    """
    Calculate an independent terrain horizon for a site.

    This function never reuses the selected site's horizon result.
    """

    site_terrain = resolve_terrain_paths(site_id)
    elevation_path = site_terrain["elevation"]

    if elevation_path is None or not Path(elevation_path).exists():
        return None, site_terrain

    pixel_spacing = raster_pixel_resolution_m(
        elevation_path,
        latitude,
    )

    quality = site_terrain["horizon_quality"]

    if quality.startswith("Coarse global"):
        distance_step = max(400.0, pixel_spacing)
    elif quality.startswith("Regional"):
        distance_step = max(80.0, pixel_spacing)
    else:
        distance_step = max(5.0, pixel_spacing)

    terrain_longitude = longitude_for_raster(
        elevation_path,
        longitude,
    )

    horizon = calculate_terrain_horizon(
        terrain_path=str(elevation_path),
        latitude=latitude,
        longitude=terrain_longitude,
        max_distance_m=5000,
        distance_step_m=float(distance_step),
        azimuth_step_deg=1,
    )

    return horizon, site_terrain


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LunaView | Lunar Mission Planner",
    page_icon="🌕",
    layout="wide",
)

st.markdown("""
<style>
:root {
    --lv-bg: #07111f;
    --lv-surface: #13243a;
    --lv-surface-2: #1b304a;
    --lv-border: #526b88;
    --lv-text: #f1f5f9;
    --lv-muted: #cbd5e1;
    --lv-ink: #111827;
    --lv-light: #ffffff;
}

/* GLOBAL RULE: dark surfaces = bright text. */
.stApp, [data-testid="stAppViewContainer"] {
    background: var(--lv-bg) !important;
    color: var(--lv-text) !important;
}
.stApp p, .stApp li, .stApp span, .stApp label,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stMarkdownContainer"] span,
.stApp [data-testid="stWidgetLabel"] p {
    color: #e2e8f0;
}
.stMarkdown h1 { color: #7dd3fc !important; }
.stMarkdown h2, .stMarkdown h3 { color: #38bdf8 !important; }
.stCaption, [data-testid="stCaption"] { color: #cbd5e1 !important; }
.stApp a { color: #7dd3fc !important; }
hr { border-color: #38516e !important; }

/* Dark containers, metrics, expanders and tabs. */
[data-testid="stMetric"], [data-testid="stExpander"],
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--lv-surface) !important;
    border: 1px solid var(--lv-border) !important;
    border-radius: 10px;
}
[data-testid="stMetric"] { padding: 16px; }
[data-testid="stMetricLabel"] { color: #cbd5e1 !important; }
[data-testid="stMetricValue"], [data-testid="stMetricDelta"] { color: #ffffff !important; }
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary *,
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] * {
    color: #e2e8f0 !important;
}
[data-testid="stTabs"] button, [data-testid="stTabs"] button * {
    color: #e2e8f0 !important;
}

/* Dark input controls use bright text. */
.stApp input, .stApp textarea, .stApp select,
.stApp [data-baseweb="input"] input {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    background: var(--lv-surface) !important;
    border: 1px solid var(--lv-border) !important;
}
.stApp input::placeholder, .stApp textarea::placeholder {
    color: #cbd5e1 !important;
    -webkit-text-fill-color: #cbd5e1 !important;
    opacity: 1 !important;
}
.stApp [data-baseweb="select"] > div,
.stApp [data-baseweb="menu"], .stApp [data-baseweb="popover"] {
    background: var(--lv-surface) !important;
}
.stApp [data-baseweb="select"] *,
.stApp [data-baseweb="menu"] *,
.stApp [data-baseweb="popover"] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
.stApp [data-testid="stCheckbox"] label,
.stApp [data-testid="stRadio"] label,
.stApp [data-testid="stToggle"] label,
.stApp [data-testid="stSlider"] label,
.stApp [data-testid="stMultiSelect"] label,
.stApp [data-testid="stSelectbox"] label,
.stApp [data-testid="stNumberInput"] label,
.stApp [data-testid="stTextInput"] label,
.stApp [data-testid="stTextArea"] label {
    color: #e2e8f0 !important;
}

/* Sidebar is dark, therefore all sidebar copy and controls are bright. */
[data-testid="stSidebar"] { background: #101f32 !important; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] span, [data-testid="stSidebar"] div {
    color: #f1f5f9 !important;
}

/* Explicit light surfaces: white background = dark text. */
.stApp [data-testid="stDateInput"],
.stApp [data-testid="stTimeInput"],
.stApp [data-testid="stDateTimeInput"] { color-scheme: light !important; }
.stApp [data-testid="stDateInput"] input,
.stApp [data-testid="stTimeInput"] input,
.stApp [data-testid="stDateTimeInput"] input,
.stApp [data-testid="stDateInput"] [role="textbox"],
.stApp [data-testid="stTimeInput"] [role="textbox"],
.stApp [data-testid="stDateTimeInput"] [role="textbox"],
.stApp [data-testid="stDateInput"] button,
.stApp [data-testid="stTimeInput"] button,
.stApp [data-testid="stDateTimeInput"] button {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    background: #ffffff !important;
    border-color: #64748b !important;
}
.stApp [data-testid="stDateInput"] label,
.stApp [data-testid="stTimeInput"] label,
.stApp [data-testid="stDateTimeInput"] label,
.stApp [data-testid="stDateInput"] label *,
.stApp [data-testid="stTimeInput"] label *,
.stApp [data-testid="stDateTimeInput"] label * {
    color: #e2e8f0 !important;
    -webkit-text-fill-color: #e2e8f0 !important;
}
.stApp [data-testid="stAlert"],
.stApp [data-testid="stNotification"] {
    background: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #cbd5e1 !important;
}
.stApp [data-testid="stAlert"] *,
.stApp [data-testid="stNotification"] * {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}
.stApp [data-testid="stDownloadButton"] button,
.stApp [data-testid="stDownloadButton"] button * {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    background: #ffffff !important;
}

/* Data grids/tables are deliberately dark with bright text. */
.stApp [data-testid="stDataFrame"],
.stApp [data-testid="stDataEditor"],
.stApp [data-testid="stTable"] {
    background: var(--lv-surface) !important;
    color: #f1f5f9 !important;
    --gdg-bg-cell: #13243a;
    --gdg-bg-cell-medium: #1b304a;
    --gdg-text-dark: #f1f5f9;
    --gdg-text-medium: #cbd5e1;
    --gdg-text-light: #94a3b8;
    --gdg-border-color: #526b88;
    --gdg-header-bg: #1b304a;
}
.stApp [data-testid="stTable"] table,
.stApp [data-testid="stTable"] th,
.stApp [data-testid="stTable"] td {
    background: var(--lv-surface) !important;
    color: #f1f5f9 !important;
    border-color: #526b88 !important;
}

/* Action buttons: blue background = bright text. */
.stApp [data-testid="stButton"] button,
.stApp [data-testid="stFormSubmitButton"] button {
    background: #1d4ed8 !important;
    border: 1px solid #60a5fa !important;
    color: #ffffff !important;
}
.stApp [data-testid="stButton"] button *,
.stApp [data-testid="stFormSubmitButton"] button * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* File-upload surfaces are dark with bright copy; the upload action is blue/white. */
.stApp [data-testid="stFileUploader"] section,
.stApp [data-testid="stFileUploader"] section * {
    background-color: #13243a !important;
    color: #f1f5f9 !important;
    border-color: #526b88 !important;
}
.stApp [data-testid="stFileUploader"] button,
.stApp [data-testid="stFileUploader"] button * {
    background: #1d4ed8 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* Code blocks remain dark and readable. */
.stApp pre, .stApp code {
    background: #0b1728 !important;
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🌕 LunaView")
st.subheader("Lunar Mission Planning System")

st.markdown(
    "**MISSION CONTROL / OBSERVATION PLANNER**\n\n"
    "Explore 20 south-pole candidate regions and 12 historical "
    "robotic or crewed landing sites. Evaluate solar geometry, "
    "Earth visibility, LOLA elevation, slope, terrain horizon "
    "and mission windows."
)


# ============================================================
# LOAD SITE CATALOGUE
# ============================================================

sites, site_details, id_to_name = build_site_catalogue()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🚀 Mission Configuration")

site_names = list(sites.keys())

selected_site = st.sidebar.selectbox(
    "Select landing site",
    site_names,
)

selected_details = site_details[selected_site]
selected_site_id = selected_details["Site ID"]

default_lat, default_lon = sites[selected_site]

latitude = st.sidebar.number_input(
    "Latitude (degrees)",
    value=float(default_lat),
    min_value=-90.0,
    max_value=90.0,
    format="%.5f",
)

longitude = st.sidebar.number_input(
    "Longitude (degrees east; signed input accepted)",
    value=float(default_lon),
    min_value=-180.0,
    max_value=180.0,
    format="%.5f",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Observation Date and Time")

now_utc = datetime.now(timezone.utc)

observation_date = st.sidebar.date_input(
    "Date (UTC)",
    value=now_utc.date(),
)

observation_time = st.sidebar.time_input(
    "Time (UTC)",
    value=now_utc.time().replace(second=0, microsecond=0),
)

utc_datetime = datetime.combine(
    observation_date,
    observation_time,
    tzinfo=timezone.utc,
)

utc_string = utc_datetime.strftime("%Y-%m-%dT%H:%M:%S")

st.sidebar.markdown("---")
st.sidebar.subheader("Mission Planning Settings")

mission_interval = st.sidebar.selectbox(
    "Calculation interval",
    options=[5, 10, 15, 30, 60],
    index=2,
    format_func=lambda value: f"Every {value} minutes",
    key="mission_interval_setting",
)

panel_power = st.sidebar.number_input(
    "Solar panel rated power (W)",
    min_value=1.0,
    max_value=5000.0,
    value=100.0,
    step=10.0,
    key="panel_power_setting",
)

antenna_limit = st.sidebar.number_input(
    "Minimum antenna elevation (°)",
    min_value=0.0,
    max_value=90.0,
    value=10.0,
    step=1.0,
    key="antenna_limit_setting",
)

panel_tilt = st.sidebar.slider(
    "Solar panel tilt (°)",
    min_value=0,
    max_value=90,
    value=0,
    key="panel_tilt_setting",
)

panel_azimuth = st.sidebar.slider(
    "Solar panel azimuth (°)",
    min_value=0,
    max_value=359,
    value=180,
    key="panel_azimuth_setting",
)

# The standard single-site and multi-site mission reports use a
# 24-hour UTC period beginning at the selected observation date.
mission_start = datetime.combine(
    observation_date,
    datetime.min.time(),
    tzinfo=timezone.utc,
)
mission_end = mission_start + timedelta(days=1)

st.sidebar.markdown("---")
st.sidebar.caption("LunaView preliminary mission analysis")
st.sidebar.caption("Time standard: UTC")


# ============================================================
# MISSION OVERVIEW
# ============================================================

st.header("Mission Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Selected Site", selected_site.split(" — ")[0])

with col2:
    st.metric("Latitude", f"{latitude:.5f}°")

with col3:
    st.metric("Longitude", f"{longitude:.5f}°")

with col4:
    st.metric("Catalogue", f"{len(sites)} sites")

st.caption(
    f"SITE ID: {selected_site_id} | "
    f"OBSERVATION EPOCH (UTC): {utc_string}"
)

st.caption(
    f"Mission: {selected_details['Mission']} | "
    f"Agency: {selected_details['Agency']} | "
    f"Status: {selected_details['Status']}"
)

st.divider()

st.subheader("🌍 Why lunar mission planning differs from Earth operations")
_context_cols = st.columns(4)
_context_cards = [
    ("Long day–night cycle", "A lunar solar day is about 29.5 Earth days; a 24-hour Earth-style plan can miss longer illumination and darkness patterns."),
    ("No atmospheric refraction", "The Moon has no substantial atmosphere to lift the apparent Sun or Earth above the horizon; local terrain and lunar geometry dominate visibility."),
    ("Low polar Sun", "Near the poles, the Sun can skim the horizon. Small ridges can create long shadows and sharply change solar access."),
    ("Earth is not fixed", "Earth's apparent position changes with lunar geometry and libration, while local terrain can block part or all of its disk."),
]
for _ctx_col, (_ctx_title, _ctx_body) in zip(_context_cols, _context_cards):
    with _ctx_col:
        st.markdown(
            f"<div style='background:#13243a;border:1px solid #526b88;border-radius:10px;padding:14px;min-height:150px;'>"
            f"<div style='font-weight:700;color:#7dd3fc;margin-bottom:8px;'>{_ctx_title}</div>"
            f"<div style='color:#f1f5f9;'>{_ctx_body}</div></div>",
            unsafe_allow_html=True,
        )
st.caption("Educational context: these are general lunar-environment characteristics. The numerical predictions below are generated by LunaView's configured geometry and terrain models.")

st.divider()


# ============================================================
# SUN POSITION
# ============================================================

st.header("☀️ Solar Position Analysis")

sun = None

try:
    sun = calculate_sun_position(
        latitude,
        longitude,
        utc_string,
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Sun Azimuth", f"{sun['sun_azimuth_deg']:.2f}°")
    col2.metric("Sun Elevation", f"{sun['sun_elevation_deg']:.2f}°")
    col3.metric("Sun Distance", f"{sun['sun_range_km']:,.0f} km")

    if sun["sun_elevation_deg"] > 0:
        st.success("Sun is above the idealized lunar horizon.")
    else:
        st.warning("Sun is below the idealized lunar horizon.")

except Exception as error:
    st.error(f"Sun calculation failed: {error}")

st.divider()


# ============================================================
# EARTH POSITION
# ============================================================

st.header("🌍 Earth Position Analysis")

earth = None

try:
    earth = calculate_earth_position(
        latitude,
        longitude,
        utc_string,
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Earth Azimuth", f"{earth['earth_azimuth_deg']:.2f}°")
    col2.metric("Earth Elevation", f"{earth['earth_elevation_deg']:.2f}°")
    col3.metric("Earth Distance", f"{earth['earth_distance_km']:,.0f} km")

    st.metric(
        "Earth Angular Radius",
        f"{earth['earth_angular_radius_deg']:.3f}°",
    )

    if earth["earth_disk_visible"]:
        st.success("Earth disk is above the idealized lunar horizon.")
    else:
        st.warning("Earth disk is below the idealized lunar horizon.")

    if earth["earth_full_disk_visible"]:
        st.info("The full Earth disk is geometrically visible.")
    else:
        st.info("The full Earth disk is not geometrically visible.")

except Exception as error:
    st.error(f"Earth calculation failed: {error}")

st.divider()


# ============================================================
# COMPLETE SITE COMPARISON
# ============================================================

st.header("📍 Complete Lunar Landing-Site Catalogue")

st.caption(
    "20 south-pole candidate sites + 12 historical "
    "crewed/robotic landing sites = 32 catalogue entries. "
    "Apollo 13 is excluded because it did not land."
)

comparison_data = []

for site_name, coordinates in sites.items():
    details = site_details[site_name]

    comparison_data.append({
        "Site": site_name,
        "Site ID": details["Site ID"],
        "Mission": details["Mission"],
        "Agency": details["Agency"],
        "Site Type": details["Site Type"],
        "Status": details["Status"],
        "Latitude (°)": coordinates[0],
        "Longitude (°)": coordinates[1],
    })

comparison_df = pd.DataFrame(comparison_data)

filter_group = st.radio(
    "Catalogue group",
    [
        "All 32 sites",
        "20 south-pole candidates",
        "12 historical landing sites",
    ],
    horizontal=True,
)

if filter_group == "20 south-pole candidates":
    visible_df = comparison_df[
        comparison_df["Site ID"].str.startswith("SITE")
    ]
elif filter_group == "12 historical landing sites":
    visible_df = comparison_df[
        ~comparison_df["Site ID"].str.startswith("SITE")
    ]
else:
    visible_df = comparison_df

st.dataframe(
    visible_df,
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download complete site catalogue CSV",
    data=comparison_df.to_csv(index=False),
    file_name="lunaview_all_32_landing_sites.csv",
    mime="text/csv",
)

fig_sites = go.Figure()

fig_sites.add_trace(
    go.Scatterpolar(
        r=[90 + item["Latitude (°)"] for item in comparison_data],
        theta=[item["Longitude (°)"] for item in comparison_data],
        mode="markers+text",
        text=[
            item["Site"].split(" — ")[0].split(" (")[0]
            for item in comparison_data
        ],
        textposition="top center",
        marker=dict(size=8),
        name="Lunar sites",
    )
)

fig_sites.update_layout(
    title="Schematic Lunar Site Distribution",
    template="plotly_dark",
    paper_bgcolor="#07111f",
    plot_bgcolor="#07111f",
    font=dict(color="#e2e8f0"),
    height=560,
    polar=dict(
        angularaxis=dict(
            direction="clockwise",
            rotation=90,
        )
    ),
)

st.plotly_chart(fig_sites, use_container_width=True)

st.caption(
    "The polar chart is a schematic coordinate comparison, "
    "not a navigation map. Verify coordinate reference frames "
    "before precision mission use."
)

st.divider()


# ============================================================
# TERRAIN DATASET STATUS
# ============================================================

st.header("🗺️ LOLA Terrain Analysis")

terrain = resolve_terrain_paths(selected_site_id)

elevation_path = terrain["elevation"]
slope_path = terrain["slope"]

status_cols = st.columns(3)

status_cols[0].metric("Terrain source", terrain["source"])
status_cols[1].metric("Nominal resolution", terrain["resolution"])
status_cols[2].metric("Site ID", selected_site_id)

horizon_result = None


# ============================================================
# ELEVATION, SLOPE AND 3D TERRAIN
# ============================================================

if elevation_path is None:
    st.error(
        "No elevation raster is available. Check data/terrain/ "
        "and data/terrain/global/."
    )

else:
    st.success(
        f"Elevation dataset found: {elevation_path.relative_to(ROOT)}"
    )

    if slope_path is not None and slope_path.exists():
        st.success(
            f"Slope dataset found: {slope_path.relative_to(ROOT)}"
        )
    else:
        st.warning(
            "No matching slope raster was found. "
            "Elevation analysis can still be displayed."
        )

    try:
        x, y, elevation, elevation_meta = read_site_patch(
            elevation_path,
            latitude,
            longitude,
        )

        valid_elev = elevation[np.isfinite(elevation)]

        st.caption(
            f"Elevation raster: {elevation_meta['file']} | "
            f"CRS: {elevation_meta['crs']} | "
            f"Native raster: {elevation_meta['width']} × "
            f"{elevation_meta['height']} | "
            f"Valid pixels in displayed patch: "
            f"{elevation_meta['valid_pixels_in_patch']:,}"
        )

        fig_elev = go.Figure(
            data=go.Heatmap(
                x=x,
                y=y,
                z=elevation,
                colorscale="Viridis",
                colorbar=dict(title="Elevation (m)"),
                connectgaps=False,
            )
        )

        fig_elev.update_layout(
            title=f"{selected_site} — LOLA Elevation Patch",
            template="plotly_dark",
            paper_bgcolor="#07111f",
            plot_bgcolor="#07111f",
            xaxis_title="X coordinate (raster CRS)",
            yaxis_title="Y coordinate (raster CRS)",
            height=520,
        )

        st.plotly_chart(fig_elev, use_container_width=True)

        c1, c2, c3 = st.columns(3)

        c1.metric("Patch minimum elevation", f"{np.nanmin(valid_elev):.1f} m")
        c2.metric("Patch mean elevation", f"{np.nanmean(valid_elev):.1f} m")
        c3.metric("Patch maximum elevation", f"{np.nanmax(valid_elev):.1f} m")

        if slope_path is not None and slope_path.exists():
            try:
                sx, sy, slope, slope_meta = read_site_patch(
                    slope_path,
                    latitude,
                    longitude,
                )

                valid_slope = slope[np.isfinite(slope)]

                if len(valid_slope):
                    fig_slope = go.Figure(
                        data=go.Heatmap(
                            x=sx,
                            y=sy,
                            z=slope,
                            colorscale="Turbo",
                            zmin=0,
                            zmax=45,
                            colorbar=dict(title="Slope (°)"),
                            connectgaps=False,
                        )
                    )

                    fig_slope.update_layout(
                        title=f"{selected_site} — LOLA Slope Patch",
                        template="plotly_dark",
                        paper_bgcolor="#07111f",
                        plot_bgcolor="#07111f",
                        xaxis_title="X coordinate (raster CRS)",
                        yaxis_title="Y coordinate (raster CRS)",
                        height=520,
                    )

                    st.plotly_chart(fig_slope, use_container_width=True)

                    s1, s2, s3 = st.columns(3)

                    s1.metric("Patch minimum slope", f"{np.nanmin(valid_slope):.2f}°")
                    s2.metric("Patch mean slope", f"{np.nanmean(valid_slope):.2f}°")
                    s3.metric("Patch maximum slope", f"{np.nanmax(valid_slope):.2f}°")

                    st.caption(
                        f"Slope raster: {slope_meta['file']} | "
                        f"Valid pixels: {slope_meta['valid_pixels_in_patch']:,}"
                    )
                else:
                    st.warning(
                        "Slope raster exists, but the selected "
                        "window contains only NoData."
                    )

            except Exception as slope_error:
                st.warning(
                    f"Slope raster could not be sampled: {slope_error}"
                )

        st.subheader("Interactive 3D Lunar Terrain")

        step = max(1, int(max(elevation.shape) / 140))

        fig_3d = go.Figure(
            data=[
                go.Surface(
                    x=x[::step],
                    y=y[::step],
                    z=elevation[::step, ::step],
                    colorscale="Viridis",
                    colorbar=dict(title="Elevation (m)"),
                    connectgaps=False,
                )
            ]
        )

        fig_3d.update_layout(
            template="plotly_dark",
            paper_bgcolor="#07111f",
            plot_bgcolor="#07111f",
            height=600,
            scene=dict(
                xaxis_title="X coordinate",
                yaxis_title="Y coordinate",
                zaxis_title="Elevation (m)",
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=0.25),
            ),
            margin=dict(l=0, r=0, t=35, b=0),
        )

        st.plotly_chart(fig_3d, use_container_width=True)

    except Exception as terrain_error:
        st.error(
            f"Terrain data could not be sampled for "
            f"{selected_site}: {terrain_error}"
        )

st.divider()


# ============================================================
# 360-DEGREE TERRAIN HORIZON
# ============================================================

st.header(" 360° Terrain Horizon Analysis")

st.write(
    "Samples the terrain around the selected landing site "
    "and compares the horizon profile with the Sun's "
    "apparent elevation."
)

if elevation_path is None or not Path(elevation_path).exists():
    st.info(
        "360° horizon analysis is unavailable because "
        "no elevation raster is assigned to this site."
    )

else:
    try:
        pixel_spacing = raster_pixel_resolution_m(
            elevation_path,
            latitude,
        )

        if terrain["horizon_quality"].startswith("Coarse global"):
            distance_step = max(400.0, pixel_spacing)
            st.warning(
                "This site uses the coarse global LOLA DEM. "
                "The horizon is a regional estimate and may "
                "miss small local obstacles."
            )

        elif terrain["horizon_quality"].startswith("Regional"):
            distance_step = max(80.0, pixel_spacing)
            st.info(
                "Horizon is based on the shared 80 m south-polar "
                "terrain mosaic. Small local obstacles may be missed."
            )

        else:
            distance_step = max(5.0, pixel_spacing)
            st.info(
                "Horizon uses the site-specific high-resolution terrain tile."
            )

        with st.spinner(
            "Calculating 360° terrain horizon for the selected site..."
        ):
            terrain_longitude = longitude_for_raster(
                elevation_path,
                longitude,
            )

            horizon_result = calculate_terrain_horizon(
                terrain_path=str(elevation_path),
                latitude=latitude,
                longitude=terrain_longitude,
                max_distance_m=5000,
                distance_step_m=float(distance_step),
                azimuth_step_deg=1,
            )

        st.success("360° terrain horizon calculation completed.")

        st.metric(
            "Observer elevation",
            f"{horizon_result['observer_elevation_m']:.2f} m",
        )

        horizon_df = pd.DataFrame(horizon_result["horizon_profile"])

        fig_horizon = go.Figure()

        fig_horizon.add_trace(
            go.Scatter(
                x=horizon_df["azimuth_deg"],
                y=horizon_df["horizon_elevation_deg"],
                mode="lines",
                name="Terrain horizon",
                line=dict(color="orange", width=2),
                fill="tozeroy",
            )
        )

        comparison = None

        if sun is not None:
            comparison = compare_sun_with_terrain(
                horizon_result,
                sun["sun_azimuth_deg"],
                sun["sun_elevation_deg"],
            )

            fig_horizon.add_vline(
                x=sun["sun_azimuth_deg"],
                line_dash="dash",
                line_color="yellow",
                annotation_text="Sun azimuth",
            )

            fig_horizon.add_hline(
                y=sun["sun_elevation_deg"],
                line_dash="dot",
                line_color="cyan",
                annotation_text="Sun elevation",
            )

        fig_horizon.update_layout(
            title=f"{selected_site} — 360° Terrain Horizon",
            template="plotly_dark",
            paper_bgcolor="#07111f",
            plot_bgcolor="#07111f",
            font=dict(color="#e2e8f0"),
            xaxis_title="Azimuth (degrees)",
            yaxis_title="Horizon elevation (degrees)",
            height=540,
            xaxis=dict(range=[0, 359], dtick=30),
            hovermode="x unified",
        )

        st.plotly_chart(fig_horizon, use_container_width=True)

        if comparison is not None:
            c1, c2, c3 = st.columns(3)

            c1.metric("Sun azimuth", f"{sun['sun_azimuth_deg']:.2f}°")
            c2.metric(
                "Terrain horizon",
                f"{comparison['terrain_horizon_deg']:.2f}°",
            )
            c3.metric(
                "Angular clearance",
                f"{comparison['angular_clearance_deg']:.2f}°",
            )

            if comparison["angular_clearance_deg"] > 0:
                st.success(
                    "RESULT: Sun is above the sampled terrain horizon."
                )
            else:
                st.warning(
                    "RESULT: Sun is below the sampled terrain horizon."
                )

        with st.expander("View complete 360° horizon data"):
            st.dataframe(
                horizon_df,
                use_container_width=True,
                hide_index=True,
            )

        st.download_button(
            "Download terrain horizon CSV",
            data=horizon_df.to_csv(index=False),
            file_name=f"{selected_site_id.lower()}_terrain_horizon.csv",
            mime="text/csv",
        )

    except Exception as horizon_error:
        horizon_result = None

        st.error(
            f"Terrain horizon analysis failed for "
            f"{selected_site}: {horizon_error}"
        )

        st.caption(
            "Mission-window calculations can still run using "
            "a geometric horizon, but will not include local "
            "terrain obstruction."
        )

st.divider()


# ============================================================
# INTERACTIVE SUN AND EARTH SKY VISUALIZATION — 3D LUNAR SKY DOME
# ============================================================

st.header(" Interactive Lunar Sky View")

if sun is None or earth is None:

    st.warning(
        "Sun and Earth position data is unavailable. "
        "Please check the selected date, time, and coordinates."
    )

else:

    # --------------------------------------------------------
    # 1. GET SUN AND EARTH POSITION DATA
    # --------------------------------------------------------

    sun_az = float(sun["sun_azimuth_deg"]) % 360
    sun_el = float(sun["sun_elevation_deg"])

    earth_az = float(earth["earth_azimuth_deg"]) % 360
    earth_el = float(earth["earth_elevation_deg"])

    earth_radius = float(
        earth.get("earth_angular_radius_deg", 0.0)
    )

    # --------------------------------------------------------
    # 2. TERRAIN HORIZON DATA
    # --------------------------------------------------------

    horizon_available = (
        horizon_result is not None
        and horizon_result.get("horizon_profile") is not None
        and len(horizon_result.get("horizon_profile", [])) > 0
    )

    horizon_azimuth = []
    horizon_elevation = []

    if horizon_available:

        for point in horizon_result["horizon_profile"]:

            az = point.get("azimuth_deg")
            el = point.get("horizon_elevation_deg")

            if az is None or el is None:
                continue

            try:
                az = float(az)
                el = float(el)

                if np.isfinite(az) and np.isfinite(el):
                    horizon_azimuth.append(az % 360)
                    horizon_elevation.append(el)

            except (TypeError, ValueError):
                continue

        horizon_available = len(horizon_azimuth) > 0

    # --------------------------------------------------------
    # 3. TERRAIN HORIZON INTERPOLATION
    # --------------------------------------------------------

    def sky_horizon_at_azimuth(azimuth):

        if not horizon_available:
            return 0.0

        azimuth = float(azimuth) % 360

        sorted_points = sorted(
            zip(horizon_azimuth, horizon_elevation),
            key=lambda point: point[0]
        )

        az_values = np.array(
            [point[0] for point in sorted_points],
            dtype=float
        )

        el_values = np.array(
            [point[1] for point in sorted_points],
            dtype=float
        )

        # Extend the circular profile across 0°/360°.
        extended_az = np.concatenate(
            ([az_values[-1] - 360], az_values, [az_values[0] + 360])
        )

        extended_el = np.concatenate(
            ([el_values[-1]], el_values, [el_values[0]])
        )

        return float(
            np.interp(
                azimuth,
                extended_az,
                extended_el
            )
        )

    # --------------------------------------------------------
    # 4. CALCULATE LOCAL SKY VISIBILITY
    # --------------------------------------------------------

    local_sun_horizon = sky_horizon_at_azimuth(sun_az)

    local_earth_horizon = sky_horizon_at_azimuth(earth_az)

    sun_clearance = sun_el - local_sun_horizon

    earth_lower_edge = earth_el - earth_radius

    earth_clearance = earth_lower_edge - local_earth_horizon

    sun_terrain_visible = sun_clearance > 0

    earth_terrain_visible = earth_clearance > 0

    earth_geometric_visible = earth_el + earth_radius > 0

    earth_full_disk_visible = earth_el - earth_radius > 0

    # --------------------------------------------------------
    # 5. CREATE 3D SKY DOME
    # --------------------------------------------------------

    sky_fig = go.Figure()

    dome_azimuth = np.linspace(0, 360, 73)

    dome_elevation = np.linspace(0, 90, 31)

    az_grid, el_grid = np.meshgrid(
        np.radians(dome_azimuth),
        np.radians(dome_elevation)
    )

    dome_x = np.cos(el_grid) * np.sin(az_grid)

    dome_y = np.cos(el_grid) * np.cos(az_grid)

    dome_z = np.sin(el_grid)

    sky_fig.add_trace(
        go.Surface(
            x=dome_x,
            y=dome_y,
            z=dome_z,
            surfacecolor=np.zeros_like(dome_z),
            colorscale=[
                [0, "#183B65"],
                [1, "#183B65"]
            ],
            opacity=0.16,
            showscale=False,
            hoverinfo="skip",
            name="Sky Dome",
            showlegend=False
        )
    )

    # --------------------------------------------------------
    # 6. ADD TERRAIN HORIZON
    # --------------------------------------------------------

    if horizon_available:

        terrain_x = []
        terrain_y = []
        terrain_z = []

        for az, el in zip(
            horizon_azimuth,
            horizon_elevation
        ):

            az_rad = math.radians(az)
            el_rad = math.radians(el)

            radius = math.cos(el_rad)

            terrain_x.append(radius * math.sin(az_rad))
            terrain_y.append(radius * math.cos(az_rad))
            terrain_z.append(math.sin(el_rad))

        sky_fig.add_trace(
            go.Scatter3d(
                x=terrain_x,
                y=terrain_y,
                z=terrain_z,
                mode="lines",
                line=dict(
                    color="#F59E0B",
                    width=6
                ),
                name="Local Terrain Horizon",
                hovertemplate=(
                    "Terrain Horizon"
                    "<br>Azimuth: %{customdata[0]:.1f}°"
                    "<br>Elevation: %{customdata[1]:.1f}°"
                    "<extra></extra>"
                ),
                customdata=np.column_stack(
                    (horizon_azimuth, horizon_elevation)
                )
            )
        )

    # --------------------------------------------------------
    # 7. ADD GEOMETRIC HORIZON
    # --------------------------------------------------------

    circle_angles = np.linspace(
        0,
        2 * np.pi,
        181
    )

    sky_fig.add_trace(
        go.Scatter3d(
            x=np.sin(circle_angles),
            y=np.cos(circle_angles),
            z=np.zeros_like(circle_angles),
            mode="lines",
            line=dict(
                color="#94A3B8",
                width=3,
                dash="dash"
            ),
            name="Geometric Horizon",
            hoverinfo="skip"
        )
    )

    # --------------------------------------------------------
    # 8. ADD SUN POSITION
    # --------------------------------------------------------

    sun_az_rad = math.radians(sun_az)

    sun_el_rad = math.radians(sun_el)

    sun_x = math.cos(sun_el_rad) * math.sin(sun_az_rad)

    sun_y = math.cos(sun_el_rad) * math.cos(sun_az_rad)

    sun_z = math.sin(sun_el_rad)

    sky_fig.add_trace(
        go.Scatter3d(
            x=[sun_x],
            y=[sun_y],
            z=[sun_z],
            mode="markers+text",
            marker=dict(
                size=11,
                color="#FACC15",
                symbol="circle",
                line=dict(
                    color="#FFFFFF",
                    width=2
                )
            ),
            text=["SUN"],
            textposition="top center",
            textfont=dict(
                color="#FDE68A",
                size=13
            ),
            name="Sun",
            customdata=[[
                sun_az,
                sun_el,
                local_sun_horizon,
                sun_clearance
            ]],
            hovertemplate=(
                "<b>SUN</b>"
                "<br>Azimuth: %{customdata[0]:.2f}°"
                "<br>Elevation: %{customdata[1]:.2f}°"
                "<br>Terrain Horizon: %{customdata[2]:.2f}°"
                "<br>Terrain Clearance: %{customdata[3]:.2f}°"
                "<extra></extra>"
            )
        )
    )

    # --------------------------------------------------------
    # 9. ADD EARTH POSITION
    # --------------------------------------------------------

    earth_az_rad = math.radians(earth_az)

    earth_el_rad = math.radians(earth_el)

    earth_x = (
        math.cos(earth_el_rad)
        * math.sin(earth_az_rad)
    )

    earth_y = (
        math.cos(earth_el_rad)
        * math.cos(earth_az_rad)
    )

    earth_z = math.sin(earth_el_rad)

    sky_fig.add_trace(
        go.Scatter3d(
            x=[earth_x],
            y=[earth_y],
            z=[earth_z],
            mode="markers+text",
            marker=dict(
                size=10,
                color="#38BDF8",
                symbol="circle",
                line=dict(
                    color="#FFFFFF",
                    width=2
                )
            ),
            text=["EARTH"],
            textposition="top center",
            textfont=dict(
                color="#7DD3FC",
                size=13
            ),
            name="Earth",
            customdata=[[
                earth_az,
                earth_el,
                earth_radius,
                earth_clearance
            ]],
            hovertemplate=(
                "<b>EARTH</b>"
                "<br>Azimuth: %{customdata[0]:.2f}°"
                "<br>Elevation: %{customdata[1]:.2f}°"
                "<br>Angular Radius: %{customdata[2]:.4f}°"
                "<br>Terrain Clearance: %{customdata[3]:.2f}°"
                "<extra></extra>"
            )
        )
    )

    # --------------------------------------------------------
    # 10. DRAW APPROXIMATE EARTH APPARENT DISK
    # --------------------------------------------------------

    if earth_radius > 0:

        disk_angles = np.linspace(
            0,
            2 * np.pi,
            73
        )

        angular_radius = math.radians(earth_radius)

        # Local east direction at Earth's position.
        east_x = math.cos(earth_az_rad)

        east_y = -math.sin(earth_az_rad)

        east_z = 0.0

        # Local up direction in the tangent plane.
        up_x = (
            -math.sin(earth_el_rad)
            * math.sin(earth_az_rad)
        )

        up_y = (
            -math.sin(earth_el_rad)
            * math.cos(earth_az_rad)
        )

        up_z = math.cos(earth_el_rad)

        earth_disk_x = (
            earth_x
            + angular_radius
            * (
                np.cos(disk_angles) * east_x
                + np.sin(disk_angles) * up_x
            )
        )

        earth_disk_y = (
            earth_y
            + angular_radius
            * (
                np.cos(disk_angles) * east_y
                + np.sin(disk_angles) * up_y
            )
        )

        earth_disk_z = (
            earth_z
            + angular_radius
            * (
                np.cos(disk_angles) * east_z
                + np.sin(disk_angles) * up_z
            )
        )

        sky_fig.add_trace(
            go.Scatter3d(
                x=earth_disk_x,
                y=earth_disk_y,
                z=earth_disk_z,
                mode="lines",
                line=dict(
                    color="#38BDF8",
                    width=4
                ),
                name="Earth Apparent Disk",
                hoverinfo="skip"
            )
        )

    # --------------------------------------------------------
    # 11. ADD CARDINAL DIRECTIONS
    # --------------------------------------------------------

    cardinal_directions = [
        ("N", 0),
        ("NE", 45),
        ("E", 90),
        ("SE", 135),
        ("S", 180),
        ("SW", 225),
        ("W", 270),
        ("NW", 315)
    ]

    for label, azimuth in cardinal_directions:

        az_rad = math.radians(azimuth)

        sky_fig.add_trace(
            go.Scatter3d(
                x=[1.12 * math.sin(az_rad)],
                y=[1.12 * math.cos(az_rad)],
                z=[0],
                mode="text",
                text=[label],
                textfont=dict(
                    color="#CBD5E1",
                    size=12
                ),
                showlegend=False,
                hoverinfo="skip"
            )
        )

    # --------------------------------------------------------
    # 12. CONFIGURE 3D SKY DOME
    # --------------------------------------------------------

    sky_fig.update_layout(
        height=680,
        margin=dict(
            l=0,
            r=0,
            t=45,
            b=0
        ),
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0B1220",
        font=dict(
            color="#FFFFFF"
        ),
        title=dict(
            text="3D Lunar Sky Dome — Sun, Earth and Terrain",
            font=dict(
                size=19,
                color="#FFFFFF"
            ),
            x=0.5
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.02,
            xanchor="center",
            x=0.5,
            font=dict(
                color="#FFFFFF",
                size=11
            ),
            bgcolor="rgba(15,23,42,0.85)",
            bordercolor="#475569",
            borderwidth=1
        ),
        scene=dict(
            bgcolor="#0B1220",
            xaxis=dict(
                title="East / West",
                range=[-1.3, 1.3],
                showbackground=False,
                showgrid=True,
                gridcolor="#334155",
                zerolinecolor="#64748B",
                color="#CBD5E1"
            ),
            yaxis=dict(
                title="North / South",
                range=[-1.3, 1.3],
                showbackground=False,
                showgrid=True,
                gridcolor="#334155",
                zerolinecolor="#64748B",
                color="#CBD5E1"
            ),
            zaxis=dict(
                title="Elevation",
                range=[-1.15, 1.15],
                showbackground=False,
                showgrid=True,
                gridcolor="#334155",
                zerolinecolor="#64748B",
                color="#CBD5E1"
            ),
            aspectmode="cube",
            camera=dict(
                eye=dict(
                    x=1.65,
                    y=1.65,
                    z=1.15
                )
            )
        ),
        scene_dragmode="orbit",
        hoverlabel=dict(
            bgcolor="#1E293B",
            font=dict(
                color="#FFFFFF"
            )
        )
    )

    # --------------------------------------------------------
    # 13. DISPLAY INTERACTIVE 3D SKY DOME
    # --------------------------------------------------------

    st.caption(
        "Rotate the dome by dragging. Zoom with your mouse wheel. "
        "Hover over the Sun, Earth, and terrain horizon for measurements."
    )

    st.plotly_chart(
        sky_fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "lunaview_3d_sky_dome",
                "scale": 2
            }
        }
    )

    # --------------------------------------------------------
    # 14. SKY VISIBILITY ASSESSMENT
    # --------------------------------------------------------

    st.subheader("🌍 Sky Visibility Assessment")

    visibility_col1, visibility_col2 = st.columns(2)

    with visibility_col1:

        st.markdown("**☀️ Sun Visibility**")

        st.metric(
            "Sun Elevation",
            f"{sun_el:.2f}°"
        )

        st.metric(
            "Terrain Horizon",
            f"{local_sun_horizon:.2f}°"
        )

        st.metric(
            "Terrain Clearance",
            f"{sun_clearance:.2f}°"
        )

        if sun_terrain_visible:

            st.success(
                "Sun is above the local terrain horizon."
            )

        else:

            st.warning(
                "Sun is blocked by the local terrain horizon."
            )

    with visibility_col2:

        st.markdown("**🌎 Earth Visibility**")

        st.metric(
            "Earth Elevation",
            f"{earth_el:.2f}°"
        )

        st.metric(
            "Earth Angular Radius",
            f"{earth_radius:.4f}°"
        )

        st.metric(
            "Earth Terrain Clearance",
            f"{earth_clearance:.2f}°"
        )

        if earth_terrain_visible:

            st.success(
                "Earth's apparent disk clears the local terrain horizon."
            )

        else:

            st.warning(
                "Earth's apparent disk is blocked by the local terrain horizon."
            )

        if not earth_geometric_visible:

            st.info(
                "Earth is geometrically below the horizon."
            )

        elif earth_full_disk_visible:

            st.info(
                "The complete Earth disk is geometrically above the horizon."
            )

        else:

            st.info(
                "Earth is partially above the geometric horizon."
            )


    # --------------------------------------------------------
    # 15. INTERPRETATION GUIDE
    # --------------------------------------------------------

    with st.expander(
        "📖 How to interpret the 3D Lunar Sky Dome",
        expanded=False
    ):

        st.markdown(
            """
            <style>
            .interpretation-guide,
            .interpretation-guide *,
            .interpretation-guide h1,
            .interpretation-guide h2,
            .interpretation-guide h3,
            .interpretation-guide h4,
            .interpretation-guide p,
            .interpretation-guide li,
            .interpretation-guide span,
            .interpretation-guide strong {
                color: #000000 !important;
                -webkit-text-fill-color: #000000 !important;
            }

            .interpretation-guide {
                background-color: #FFF176 !important;
                padding: 22px;
                border-radius: 12px;
                border: 2px solid #EAB308;
                line-height: 1.8;
            }
            </style>

            <div class="interpretation-guide">

            <h4> Sun Position</h4>
            <p>
            The yellow marker represents the Sun's position in the
            lunar sky. Its azimuth indicates its compass direction,
            while its elevation indicates its height above or below
            the geometric horizon.
            </p>

            <h4> Earth Position</h4>
            <p>
            The blue marker represents Earth's apparent position.
            Its angular radius is used to approximate the apparent
            size of Earth's disk.
            </p>

            <h4> Local Terrain Horizon</h4>
            <p>
            The orange line represents the surrounding lunar terrain
            horizon calculated from elevation data. It shows how
            nearby mountains and crater rims may obstruct the Sun
            or Earth's apparent disk.
            </p>

            <h4> Geometric Horizon</h4>
            <p>
            The grey dashed circle represents the ideal geometric
            horizon without surrounding terrain obstruction.
            </p>

            <h4> Terrain Clearance</h4>
            <p>
            Positive clearance means the object is above the local
            terrain horizon. Negative clearance means the terrain
            horizon is higher than the object's position.
            </p>

            <h4> Interactive Controls</h4>
            <ul>
                <li>Drag to rotate the 3D sky dome.</li>
                <li>Scroll to zoom in and out.</li>
                <li>Hover over markers to inspect position measurements.</li>
                <li>Use the legend to identify plotted objects.</li>
            </ul>

            <p>

            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


st.divider()


# ============================================================
# INTERACTIVE MISSION TIMELINE
# ============================================================

st.header("⏱️ Interactive Mission Timeline")

st.markdown(
    "Explore how lunar sunlight and Earth communication "
    "availability change throughout your selected mission period."
)

# ------------------------------------------------------------
# DATE/TIME CONTROL CONTRAST
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    /* White date/time controls: force every value and icon to dark ink. */
    .stApp [data-testid="stDateTimeInput"],
    .stApp [data-testid="stDateInput"],
    .stApp [data-testid="stTimeInput"] {
        color-scheme: light !important;
        background: transparent !important;
    }

    /* Broad descendant selector catches BaseWeb's nested text wrappers too. */
    .stApp [data-testid="stDateTimeInput"] *,
    .stApp [data-testid="stDateInput"] *,
    .stApp [data-testid="stTimeInput"] * {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        text-shadow: none !important;
    }

    .stApp [data-testid="stDateTimeInput"] input,
    .stApp [data-testid="stDateInput"] input,
    .stApp [data-testid="stTimeInput"] input,
    .stApp [data-testid="stDateTimeInput"] [role="textbox"],
    .stApp [data-testid="stDateInput"] [role="textbox"],
    .stApp [data-testid="stTimeInput"] [role="textbox"] {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        background: #FFFFFF !important;
        caret-color: #111827 !important;
        border-color: #64748B !important;
        opacity: 1 !important;
    }

    .stApp [data-testid="stDateTimeInput"] [data-baseweb="input"],
    .stApp [data-testid="stDateInput"] [data-baseweb="input"],
    .stApp [data-testid="stTimeInput"] [data-baseweb="input"],
    .stApp [data-testid="stDateTimeInput"] [data-baseweb="input"] > div,
    .stApp [data-testid="stDateInput"] [data-baseweb="input"] > div,
    .stApp [data-testid="stTimeInput"] [data-baseweb="input"] > div {
        background: #FFFFFF !important;
        border-color: #64748B !important;
    }

    /* Keep the widget labels bright against LunaView's dark page. */
    .stApp [data-testid="stDateTimeInput"] label,
    .stApp [data-testid="stDateInput"] label,
    .stApp [data-testid="stTimeInput"] label,
    .stApp [data-testid="stDateTimeInput"] label *,
    .stApp [data-testid="stDateInput"] label *,
    .stApp [data-testid="stTimeInput"] label * {
        color: #E2E8F0 !important;
        -webkit-text-fill-color: #E2E8F0 !important;
        background: transparent !important;
    }

    /* Calendar/time popovers are white with dark readable values. */
    [data-baseweb="popover"] [data-baseweb="calendar"],
    [data-baseweb="popover"] [data-baseweb="datepicker"],
    [data-baseweb="popover"] [data-baseweb="timepicker"] {
        color-scheme: light !important;
        background: #FFFFFF !important;
    }
    [data-baseweb="popover"] [data-baseweb="calendar"] *,
    [data-baseweb="popover"] [data-baseweb="datepicker"] *,
    [data-baseweb="popover"] [data-baseweb="timepicker"] *,
    [data-baseweb="popover"] [role="grid"],
    [data-baseweb="popover"] [role="grid"] *,
    [data-baseweb="popover"] [role="option"],
    [data-baseweb="popover"] [role="option"] * {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# TIMELINE CONTROLS
# ------------------------------------------------------------

timeline_col1, timeline_col2 = st.columns(2)

with timeline_col1:

    timeline_start = st.datetime_input(
        "Timeline Start (UTC)",
        value=datetime.now(timezone.utc).replace(tzinfo=None),
        key="timeline_start"
    )

with timeline_col2:

    timeline_duration = st.selectbox(
        "Mission Duration",
        options=[6, 12, 24, 48, 72, 168, 336, 708],
        index=2,
        format_func=lambda x: (
            f"{x} hours" if x < 168
            else "7 days" if x == 168
            else "14 days" if x == 336
            else "29.5 days (one synodic month)"
        ),
        key="timeline_duration"
    )

# ------------------------------------------------------------
# CALCULATE MISSION PERIOD
# ------------------------------------------------------------

timeline_start = timeline_start.replace(
    tzinfo=timezone.utc
)

timeline_end = timeline_start + timedelta(
    hours=int(timeline_duration)
)

st.caption(
    f"Mission period: "
    f"{timeline_start.strftime('%Y-%m-%d %H:%M UTC')} "
    f"to "
    f"{timeline_end.strftime('%Y-%m-%d %H:%M UTC')}"
)

# Keep timeline calculations separate from the standard 24-hour report.
# A minimum 30-minute timeline step keeps multi-day playback responsive.
timeline_interval = (
    int(mission_interval)
    if int(timeline_duration) <= 168
    else max(int(mission_interval), 30)
)
timeline_signature = (
    selected_site_id,
    float(latitude),
    float(longitude),
    timeline_start.isoformat(),
    timeline_end.isoformat(),
    int(timeline_interval),
    float(panel_power),
    float(antenna_limit),
    float(panel_tilt),
    float(panel_azimuth),
    bool(horizon_result is not None),
)

st.subheader("Mission Timeline Playback")
st.caption(
    f"Timeline sampling: every {timeline_interval} minutes. "
    "Use the time selector to inspect conditions, or press Play "
    "inside the 3D animation."
)

if st.button(
    "Calculate Interactive Timeline",
    type="primary",
    use_container_width=True,
    key="calculate_interactive_timeline",
):
    try:
        with st.spinner("Calculating sunlight and Earth communication conditions..."):
            _timeline_result = calculate_mission_windows(
                latitude_deg=latitude,
                longitude_deg=longitude,
                start_utc=timeline_start.replace(tzinfo=timezone.utc),
                end_utc=timeline_end.replace(tzinfo=timezone.utc),
                interval_minutes=timeline_interval,
                horizon_result=horizon_result,
                antenna_min_elevation_deg=antenna_limit,
                panel_rated_power_w=panel_power,
                panel_tilt_deg=float(panel_tilt),
                panel_azimuth_deg=float(panel_azimuth),
            )
        st.session_state["interactive_timeline_result"] = _timeline_result
        st.session_state["interactive_timeline_signature"] = timeline_signature
        st.success("Interactive mission timeline calculated.")
    except Exception as timeline_error:
        st.error(f"Interactive timeline calculation failed: {timeline_error}")

_saved_timeline = st.session_state.get("interactive_timeline_result")
_timeline_matches = (
    _saved_timeline is not None
    and st.session_state.get("interactive_timeline_signature") == timeline_signature
)

if _saved_timeline is not None and not _timeline_matches:
    st.info("Timeline settings have changed. Recalculate the interactive timeline to refresh it.")

if _timeline_matches:
    _timeline_df = pd.DataFrame(_saved_timeline["records"])
    _timeline_df["time"] = pd.to_datetime(_timeline_df["utc_time"], utc=True)
    _timeline_df = _timeline_df.sort_values("time").reset_index(drop=True)

    if not _timeline_df.empty:
        _selected_index = st.slider(
            "Select mission time",
            min_value=0,
            max_value=len(_timeline_df) - 1,
            value=0,
            step=1,
            format="%d",
            key="interactive_timeline_index",
        )
        _selected_row = _timeline_df.iloc[int(_selected_index)]
        _selected_time = _selected_row["time"]

        def _remaining_window_hours(frame, index, column, step_minutes):
            if not bool(frame.iloc[index][column]):
                return 0.0
            count = 0
            for value in frame[column].iloc[index:].tolist():
                if not bool(value):
                    break
                count += 1
            return count * step_minutes / 60.0

        def _crossing_time(frame, index, value_column, threshold, state_column):
            """Linearly interpolate a sampled threshold crossing in UTC."""
            if value_column not in frame.columns or state_column not in frame.columns:
                return None
            current_state = bool(frame.iloc[index][state_column])
            for future_index in range(index + 1, len(frame)):
                next_state = bool(frame.iloc[future_index][state_column])
                if next_state != current_state:
                    try:
                        y0 = float(frame.iloc[future_index - 1][value_column]) - float(threshold)
                        y1 = float(frame.iloc[future_index][value_column]) - float(threshold)
                        t0 = frame.iloc[future_index - 1]["time"]
                        t1 = frame.iloc[future_index]["time"]
                        if np.isfinite(y0) and np.isfinite(y1) and y1 != y0:
                            fraction = float(np.clip(-y0 / (y1 - y0), 0.0, 1.0))
                            return t0 + (t1 - t0) * fraction
                    except (TypeError, ValueError, OverflowError):
                        pass
                    return frame.iloc[future_index]["time"]
            return None

        def _format_duration(delta):
            if delta is None:
                return "Outside forecast"
            seconds = max(0, int(delta.total_seconds()))
            hours, remainder = divmod(seconds, 3600)
            minutes = remainder // 60
            if hours >= 24:
                days, hours = divmod(hours, 24)
                return f"{days}d {hours:02d}h {minutes:02d}m"
            return f"{hours:02d}h {minutes:02d}m"

        def _countdown_to(event_time, reference_time):
            return None if event_time is None else max(timedelta(0), event_time - reference_time)

        def _window_table(frame, state_column, label, step_minutes):
            """Convert sampled Boolean availability into approximate UTC windows."""
            if state_column not in frame.columns or frame.empty:
                return []
            states = frame[state_column].fillna(False).astype(bool).to_numpy()
            windows = []
            start_idx = None
            for idx, is_available in enumerate(states):
                if is_available and start_idx is None:
                    start_idx = idx
                is_last = idx == len(states) - 1
                if start_idx is not None and (not is_available or is_last):
                    end_idx = idx if (not is_available) else idx + 1
                    start_time = frame.iloc[start_idx]["time"]
                    if end_idx < len(frame):
                        end_time = frame.iloc[end_idx]["time"]
                    else:
                        end_time = pd.Timestamp(timeline_end)
                    end_time = min(end_time, pd.Timestamp(timeline_end))
                    duration_h = max(0.0, (end_time - start_time).total_seconds() / 3600.0)
                    windows.append({
                        "Window type": label,
                        "Start (UTC)": start_time.strftime("%Y-%m-%d %H:%M"),
                        "End (UTC)": end_time.strftime("%Y-%m-%d %H:%M"),
                        "Duration (h)": round(duration_h, 2),
                    })
                    start_idx = None
            return windows

        _sun_event = _crossing_time(
            _timeline_df, int(_selected_index), "sun_clearance_deg", 0.0, "sun_visible"
        )
        _dte_event = _crossing_time(
            _timeline_df, int(_selected_index), "earth_clearance_deg",
            float(antenna_limit), "communication_visible"
        )

        def _next_boolean_transition(frame, index, state_column):
            if state_column not in frame.columns:
                return None
            current_state = bool(frame.iloc[index][state_column])
            for future_index in range(index + 1, len(frame)):
                if bool(frame.iloc[future_index][state_column]) != current_state:
                    return frame.iloc[future_index]["time"]
            return None

        _earth_disk_event = _next_boolean_transition(
            _timeline_df, int(_selected_index), "earth_disk_visible"
        )
        _sun_event_name = "Sunset" if bool(_selected_row["sun_visible"]) else "Next terrain sunrise"
        _earth_event_name = "Earthset" if bool(_selected_row.get("earth_disk_visible", False)) else "Next Earthrise"
        _dte_event_name = "DTE window loss" if bool(_selected_row["communication_visible"]) else "Next DTE acquisition"
        _sun_countdown = _countdown_to(_sun_event, _selected_time)
        _earth_countdown = _countdown_to(_earth_disk_event, _selected_time)
        _dte_countdown = _countdown_to(_dte_event, _selected_time)

        _sun_elevation_now = float(_selected_row.get("sun_elevation_deg", float("nan")))
        _sun_clearance_now = float(_selected_row.get("sun_clearance_deg", float("nan")))
        _panel_w_now = max(0.0, float(_selected_row.get("panel_power_w", 0.0) or 0.0))
        _panel_fraction_now = 100.0 * _panel_w_now / max(float(panel_power), 1e-9)
        _earth_elevation_now = float(_selected_row.get("earth_elevation_deg", float("nan")))
        _earth_clearance_now = float(_selected_row.get("earth_clearance_deg", float("nan")))
        _earth_disk_now = bool(_selected_row.get("earth_disk_visible", False))
        _comms_now = bool(_selected_row.get("communication_visible", False))

        st.subheader("Live Mission Conditions — Selected UTC Epoch")
        _m1, _m2, _m3, _m4 = st.columns(4)
        _m1.metric("Solar elevation", f"{_sun_elevation_now:.2f}°")
        _m2.metric("Sun / terrain clearance", f"{_sun_clearance_now:.2f}°")
        _m3.metric("Estimated panel output", f"{_panel_w_now:.1f} W", f"{_panel_fraction_now:.1f}% of rated")
        _m4.metric("Earth elevation", f"{_earth_elevation_now:.2f}°")

        _e1, _e2, _e3, _e4 = st.columns(4)
        _e1.metric("Sunlight", "AVAILABLE" if bool(_selected_row["sun_visible"]) else "BLOCKED")
        _e2.metric("Earth disk", "VISIBLE" if _earth_disk_now else "NOT VISIBLE")
        _e3.metric("DTE communications", "AVAILABLE" if _comms_now else "UNAVAILABLE")
        _e4.metric("Earth clearance", f"{_earth_clearance_now:.2f}°", f"Threshold: {float(antenna_limit):.1f}°")

        st.subheader("Countdown to the Next Critical Visibility Event")
        _tc1, _tc2, _tc3 = st.columns(3)
        _tc1.metric(
            f"{_sun_event_name} countdown",
            _format_duration(_sun_countdown),
            delta=(_sun_event.strftime("%d %b %H:%M UTC") if _sun_event is not None else "Not in forecast"),
            delta_color="off",
        )
        _tc2.metric(
            f"{_earth_event_name} countdown",
            _format_duration(_earth_countdown),
            delta=(_earth_disk_event.strftime("%d %b %H:%M UTC") if _earth_disk_event is not None else "Not in forecast"),
            delta_color="off",
        )
        _tc3.metric(
            f"{_dte_event_name} countdown",
            _format_duration(_dte_countdown),
            delta=(_dte_event.strftime("%d %b %H:%M UTC") if _dte_event is not None else "Not in forecast"),
            delta_color="off",
        )
        st.caption(
            "Sunset and DTE threshold events are interpolated between adjacent samples; Earthrise/Earthset "
            "uses the sampled Earth-disk visibility transition. These are estimates, not exact event predictions; "
            "the selected sampling interval and terrain/ephemeris accuracy limit precision."
        )

        _timeline_times = _timeline_df["time"].tolist()
        _interval_hours = []
        for _time_index, _sample_time in enumerate(_timeline_times):
            _sample_end = (
                _timeline_times[_time_index + 1]
                if _time_index + 1 < len(_timeline_times)
                else pd.Timestamp(timeline_end)
            )
            _interval_hours.append(max(0.0, (_sample_end - _sample_time).total_seconds() / 3600.0))
        _interval_hours = np.asarray(_interval_hours, dtype=float)
        def _integrated_hours(state_column):
            _mask = _timeline_df[state_column].fillna(False).astype(bool).to_numpy()
            return float(np.sum(_interval_hours[_mask]))
        _sun_hours_total = _integrated_hours("sun_visible")
        _earth_disk_hours = _integrated_hours("earth_disk_visible")
        _dte_hours_total = _integrated_hours("communication_visible")
        _dual_mask = (
            _timeline_df["sun_visible"].fillna(False).astype(bool).to_numpy()
            & _timeline_df["communication_visible"].fillna(False).astype(bool).to_numpy()
        )
        _dual_hours = float(np.sum(_interval_hours[_dual_mask]))
        _window_cols = st.columns(4)
        _window_cols[0].metric("Sunlit time in forecast", f"{_sun_hours_total:.1f} h")
        _window_cols[1].metric("Earth disk visible", f"{_earth_disk_hours:.1f} h")
        _window_cols[2].metric("DTE communication time", f"{_dte_hours_total:.1f} h")
        _window_cols[3].metric("Sun + DTE overlap", f"{_dual_hours:.1f} h")

        _window_rows = (
            _window_table(_timeline_df, "sun_visible", "Solar illumination", timeline_interval)
            + _window_table(_timeline_df, "earth_disk_visible", "Earth disk geometric visibility", timeline_interval)
            + _window_table(_timeline_df, "communication_visible", "Direct-to-Earth communication threshold met", timeline_interval)
        )
        if _window_rows:
            with st.expander("Open complete sunlight, Earth-visibility and communication window schedule"):
                _window_df = pd.DataFrame(_window_rows)
                st.dataframe(_window_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download visibility window schedule CSV",
                    data=_window_df.to_csv(index=False),
                    file_name=f"{selected_site_id.lower()}_visibility_windows.csv",
                    mime="text/csv",
                    key="download_visibility_windows",
                )

        st.subheader("Solar Elevation, Terrain Clearance and Panel Output")
        _solar_fig = go.Figure()
        _solar_fig.add_trace(go.Scatter(
            x=_timeline_df["time"], y=_timeline_df["sun_elevation_deg"],
            mode="lines", name="Sun centre elevation (°)",
            line=dict(color="#FACC15", width=2),
        ))
        if "sun_terrain_horizon_deg" in _timeline_df.columns:
            _solar_fig.add_trace(go.Scatter(
                x=_timeline_df["time"], y=_timeline_df["sun_terrain_horizon_deg"],
                mode="lines", name="Local terrain horizon (°)",
                line=dict(color="#F97316", width=2, dash="dash"),
            ))
        _solar_fig.add_trace(go.Scatter(
            x=_timeline_df["time"], y=_timeline_df["panel_power_w"],
            mode="lines", name="Estimated panel output (W)",
            line=dict(color="#22C55E", width=2), yaxis="y2",
        ))
        _solar_fig.add_vline(x=_selected_time, line_dash="dot", line_color="white")
        _solar_fig.update_layout(
            title="Low-angle polar sunlight and estimated electrical output",
            xaxis_title="Time (UTC)",
            yaxis=dict(title="Elevation (degrees)"),
            yaxis2=dict(title="Panel output (W)", overlaying="y", side="right", rangemode="tozero"),
            template="plotly_dark", paper_bgcolor="#07111f", plot_bgcolor="#07111f",
            font=dict(color="#E2E8F0"), height=430, hovermode="x unified",
        )
        st.plotly_chart(_solar_fig, use_container_width=True)

        st.subheader("Sunlight and Earth Communication Availability")
        _availability_fig = go.Figure()
        _availability_fig.add_trace(go.Scatter(
            x=_timeline_df["time"],
            y=_timeline_df["sun_visible"].astype(int),
            mode="lines",
            line=dict(color="#FACC15", width=3, shape="hv"),
            name="Sunlight available",
        ))
        _availability_fig.add_trace(go.Scatter(
            x=_timeline_df["time"],
            y=_timeline_df["communication_visible"].astype(int) + 1.15,
            mode="lines",
            line=dict(color="#38BDF8", width=3, shape="hv"),
            name="Earth communication available",
        ))
        _availability_fig.add_vline(
            x=_selected_time,
            line_dash="dash",
            line_color="#FFFFFF",
            annotation_text="Selected time",
        )
        _availability_fig.update_layout(
            title="Mission Availability Timeline",
            xaxis_title="Time (UTC)",
            yaxis=dict(
                tickmode="array",
                tickvals=[0, 1, 1.15, 2.15],
                ticktext=["Unavailable", "Sunlight", "Unavailable", "Communication"],
                range=[-0.25, 2.4],
            ),
            template="plotly_dark",
            paper_bgcolor="#07111f",
            plot_bgcolor="#07111f",
            font=dict(color="#E2E8F0"),
            height=360,
            hovermode="x unified",
        )
        st.plotly_chart(_availability_fig, use_container_width=True)

        # Build a 3D sky-dome playback from the calculated time samples.
        # Animation is downsampled to at most 96 frames for browser performance.
        _frame_indices = np.unique(
            np.linspace(0, len(_timeline_df) - 1, min(96, len(_timeline_df))).astype(int)
        )
        _frame_rows = _timeline_df.iloc[_frame_indices]
        _sky_anim = go.Figure()
        _az_grid, _el_grid = np.meshgrid(
            np.radians(np.linspace(0, 360, 49)),
            np.radians(np.linspace(0, 90, 19)),
        )
        _sky_anim.add_trace(go.Surface(
            x=np.cos(_el_grid) * np.sin(_az_grid),
            y=np.cos(_el_grid) * np.cos(_az_grid),
            z=np.sin(_el_grid),
            surfacecolor=np.zeros_like(_el_grid),
            colorscale=[[0, "#183B65"], [1, "#183B65"]],
            opacity=0.14,
            showscale=False,
            hoverinfo="skip",
            name="Sky dome",
        ))

        if horizon_result is not None and horizon_result.get("horizon_profile"):
            _horizon_points = horizon_result["horizon_profile"]
            _horizon_az = np.array([float(p["azimuth_deg"]) for p in _horizon_points])
            _horizon_el = np.array([float(p["horizon_elevation_deg"]) for p in _horizon_points])
            _ha = np.radians(_horizon_az)
            _he = np.radians(_horizon_el)
            _sky_anim.add_trace(go.Scatter3d(
                x=np.cos(_he) * np.sin(_ha),
                y=np.cos(_he) * np.cos(_ha),
                z=np.sin(_he),
                mode="lines",
                line=dict(color="#F59E0B", width=5),
                name="LOLA terrain horizon",
            ))
        else:
            _sky_anim.add_trace(go.Scatter3d(
                x=[-1, 1], y=[0, 0], z=[0, 0],
                mode="lines", line=dict(color="#94A3B8", dash="dash", width=2),
                name="Geometric horizon",
            ))

        def _sky_xyz(azimuth_deg, elevation_deg):
            _az = math.radians(float(azimuth_deg))
            _el = math.radians(float(elevation_deg))
            return (
                math.cos(_el) * math.sin(_az),
                math.cos(_el) * math.cos(_az),
                math.sin(_el),
            )

        _first = _frame_rows.iloc[0]
        _sx, _sy, _sz = _sky_xyz(_first["sun_azimuth_deg"], _first["sun_elevation_deg"])
        _ex, _ey, _ez = _sky_xyz(_first["earth_azimuth_deg"], _first["earth_elevation_deg"])
        _sky_anim.add_trace(go.Scatter3d(
            x=[_sx], y=[_sy], z=[_sz], mode="markers+text",
            marker=dict(size=10, color="#FACC15", line=dict(color="white", width=1)),
            text=["SUN"], textposition="top center", name="Sun",
        ))
        _sky_anim.add_trace(go.Scatter3d(
            x=[_ex], y=[_ey], z=[_ez], mode="markers+text",
            marker=dict(size=9, color="#38BDF8", line=dict(color="white", width=1)),
            text=["EARTH"], textposition="top center", name="Earth",
        ))

        _frames = []
        for _frame_number, (_, _row) in enumerate(_frame_rows.iterrows()):
            _fsx, _fsy, _fsz = _sky_xyz(_row["sun_azimuth_deg"], _row["sun_elevation_deg"])
            _fex, _fey, _fez = _sky_xyz(_row["earth_azimuth_deg"], _row["earth_elevation_deg"])
            _frames.append(go.Frame(
                name=str(_frame_number),
                data=[
                    go.Scatter3d(x=[_fsx], y=[_fsy], z=[_fsz]),
                    go.Scatter3d(x=[_fex], y=[_fey], z=[_fez]),
                ],
                traces=[2, 3],
                layout=go.Layout(title_text=f"3D Lunar Sky — {_row['utc_time']} UTC"),
            ))
        _sky_anim.frames = _frames
        _slider_steps = [
            dict(
                method="animate",
                label=str(_row["time"].strftime("%d %H:%M")),
                args=[[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
            )
            for i, (_, _row) in enumerate(_frame_rows.iterrows())
        ]
        _sky_anim.update_layout(
            title=f"3D Lunar Sky — {_first['utc_time']} UTC",
            template="plotly_dark",
            paper_bgcolor="#07111f",
            plot_bgcolor="#07111f",
            font=dict(color="#E2E8F0"),
            height=620,
            scene=dict(
                xaxis=dict(title="East / West", range=[-1.25, 1.25]),
                yaxis=dict(title="North / South", range=[-1.25, 1.25]),
                zaxis=dict(title="Elevation", range=[-1.1, 1.1]),
                aspectmode="cube",
                camera=dict(eye=dict(x=1.6, y=1.6, z=1.15)),
            ),
            updatemenus=[dict(
                type="buttons", showactive=False, x=0.05, y=1.12,
                buttons=[
                    dict(label="▶ Play", method="animate", args=[None, {"fromcurrent": True, "frame": {"duration": 450, "redraw": True}, "transition": {"duration": 200}}]),
                    dict(label="⏸ Pause", method="animate", args=[[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]),
                ],
            )],
            sliders=[dict(active=0, x=0.1, y=0, len=0.85, steps=_slider_steps)],
            margin=dict(l=0, r=0, t=70, b=35),
        )
        st.subheader("Animated 3D Lunar Sky Dome")
        st.caption("Use ▶ Play / ⏸ Pause or the chart slider to animate the Sun and Earth across the mission period.")
        st.plotly_chart(_sky_anim, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})

        st.caption(
            "Availability durations are integrated from sampled states. Visibility-event countdowns use "
            "linear interpolation between sampled threshold crossings. Both remain limited by the chosen "
            "time step, the quality of the terrain horizon, and the underlying Sun/Earth ephemeris model."
        )

st.divider()



# ============================================================
# RUN SINGLE-SITE MISSION CALCULATION
# ============================================================

if st.button(
    "Calculate 24-hour Mission Windows",
    type="primary",
    use_container_width=True,
    key="calculate_mission_windows",
):
    try:
        with st.spinner(
            f"Calculating mission windows for {selected_site}..."
        ):
            mission_result = calculate_mission_windows(
                latitude_deg=latitude,
                longitude_deg=longitude,
                start_utc=mission_start,
                end_utc=mission_end,
                interval_minutes=mission_interval,
                horizon_result=horizon_result,
                antenna_min_elevation_deg=antenna_limit,
                panel_rated_power_w=panel_power,
                panel_tilt_deg=float(panel_tilt),
                panel_azimuth_deg=float(panel_azimuth),
            )

        st.session_state["mission_result"] = mission_result
        st.session_state["mission_site_id"] = selected_site_id
        st.session_state["mission_site_name"] = selected_site
        st.session_state["mission_terrain_used"] = horizon_result is not None
        st.session_state["mission_date"] = str(observation_date)
        st.session_state["mission_interval_used"] = mission_interval
        st.session_state["mission_panel_power_used"] = panel_power
        st.session_state["mission_antenna_limit_used"] = antenna_limit
        st.session_state["mission_panel_tilt_used"] = panel_tilt
        st.session_state["mission_panel_azimuth_used"] = panel_azimuth
        st.session_state["mission_latitude_used"] = latitude
        st.session_state["mission_longitude_used"] = longitude

        st.success(f"Mission calculation completed for {selected_site}.")

    except Exception as mission_error:
        st.error(f"Mission-window calculation failed: {mission_error}")


# ============================================================
# DISPLAY STORED SINGLE-SITE MISSION RESULTS
# ============================================================

mission_result = st.session_state.get("mission_result")

results_match = (
    mission_result is not None
    and st.session_state.get("mission_site_id") == selected_site_id
    and st.session_state.get("mission_date") == str(observation_date)
    and st.session_state.get("mission_interval_used") == mission_interval
    and st.session_state.get("mission_panel_power_used") == panel_power
    and st.session_state.get("mission_antenna_limit_used") == antenna_limit
    and st.session_state.get("mission_panel_tilt_used") == panel_tilt
    and st.session_state.get("mission_panel_azimuth_used") == panel_azimuth
    and st.session_state.get("mission_latitude_used") == latitude
    and st.session_state.get("mission_longitude_used") == longitude
)

if mission_result is not None and not results_match:
    st.info(
        "The displayed mission results do not match the current "
        "site, coordinates, date or mission settings. "
        "Recalculate to update them."
    )

if results_match:
    summary = mission_result["summary"]
    mission_df = pd.DataFrame(mission_result["records"])

    mission_df["time"] = pd.to_datetime(
        mission_df["utc_time"],
        utc=True,
    )

    st.divider()
    st.subheader(
        f"Mission Assessment — "
        f"{st.session_state['mission_site_name']}"
    )

    st.caption(
        f"Date: {st.session_state['mission_date']} UTC | "
        f"Calculation interval: "
        f"{st.session_state['mission_interval_used']} minutes"
    )

    if st.session_state["mission_terrain_used"]:
        st.success(
            "Terrain-aware calculation: local terrain horizon enabled."
        )
    else:
        st.warning(
            "Geometric-horizon calculation: local terrain "
            "obstruction was not included."
        )

    st.subheader("Mission Summary")

    a, b, c, d = st.columns(4)

    a.metric(
        "Sunlight duration",
        f"{summary['sunlight_hours']:.2f} h",
        f"{summary['sunlight_percentage']:.1f}%",
    )

    b.metric(
        "Communication duration",
        f"{summary['communication_hours']:.2f} h",
        f"{summary['communication_percentage']:.1f}%",
    )

    c.metric(
        "Estimated solar energy",
        f"{summary['estimated_energy_wh']:.2f} Wh",
    )

    d.metric("Time samples", str(summary["sample_count"]))

    st.subheader("☀️ Sunlight and Terrain Horizon")

    fig_sun = go.Figure()

    fig_sun.add_trace(
        go.Scatter(
            x=mission_df["time"],
            y=mission_df["sun_elevation_deg"],
            mode="lines",
            name="Sun elevation",
            line=dict(color="#facc15", width=2),
        )
    )

    fig_sun.add_trace(
        go.Scatter(
            x=mission_df["time"],
            y=mission_df["sun_terrain_horizon_deg"],
            mode="lines",
            name="Terrain horizon",
            line=dict(color="#f97316", width=2, dash="dash"),
        )
    )

    fig_sun.add_hline(
        y=0,
        line_dash="dot",
        line_color="white",
        annotation_text="Geometric horizon",
    )

    fig_sun.update_layout(
        title="Sun Elevation Compared with Terrain Horizon",
        xaxis_title="Time (UTC)",
        yaxis_title="Elevation (degrees)",
        template="plotly_dark",
        paper_bgcolor="#07111f",
        plot_bgcolor="#07111f",
        height=420,
        hovermode="x unified",
    )

    st.plotly_chart(fig_sun, use_container_width=True)

    st.subheader("⚡ Solar Panel Power")

    fig_power = go.Figure()

    fig_power.add_trace(
        go.Scatter(
            x=mission_df["time"],
            y=mission_df["panel_power_w"],
            mode="lines",
            name="Solar panel output",
            line=dict(color="#22c55e", width=2),
            fill="tozeroy",
        )
    )

    fig_power.update_layout(
        title="Estimated Solar Panel Output",
        xaxis_title="Time (UTC)",
        yaxis_title="Power (W)",
        template="plotly_dark",
        paper_bgcolor="#07111f",
        plot_bgcolor="#07111f",
        height=380,
        hovermode="x unified",
    )

    st.plotly_chart(fig_power, use_container_width=True)

    st.subheader(" Earth Communication Visibility")

    fig_comms = go.Figure()

    fig_comms.add_trace(
        go.Scatter(
            x=mission_df["time"],
            y=mission_df["earth_clearance_deg"],
            mode="lines",
            name="Earth clearance",
            line=dict(color="#38bdf8", width=2),
        )
    )

    fig_comms.add_hline(
        y=antenna_limit,
        line_dash="dash",
        line_color="#f97316",
        annotation_text="Minimum antenna elevation",
    )

    fig_comms.add_hline(
        y=0,
        line_dash="dot",
        line_color="white",
        annotation_text="Terrain horizon",
    )

    fig_comms.update_layout(
        title="Earth Clearance for Direct-to-Earth Communications",
        xaxis_title="Time (UTC)",
        yaxis_title="Earth clearance (degrees)",
        template="plotly_dark",
        paper_bgcolor="#07111f",
        plot_bgcolor="#07111f",
        height=380,
        hovermode="x unified",
    )

    st.plotly_chart(fig_comms, use_container_width=True)

    st.subheader("Mission Visibility Status")

    status_df = pd.DataFrame({
        "Time (UTC)": mission_df["time"],
        "Sunlight": mission_df["sun_visible"].map({
            True: "Visible",
            False: "Blocked",
        }),
        "Earth disk": mission_df["earth_disk_visible"].map({
            True: "Visible",
            False: "Not visible",
        }),
        "Communications": mission_df["communication_visible"].map({
            True: "Available",
            False: "Unavailable",
        }),
    })

    st.dataframe(
        status_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Detailed Mission Data")

    display_columns = [
        "utc_time",
        "sun_azimuth_deg",
        "sun_elevation_deg",
        "sun_terrain_horizon_deg",
        "sun_clearance_deg",
        "sun_visible",
        "panel_power_w",
        "earth_azimuth_deg",
        "earth_elevation_deg",
        "earth_clearance_deg",
        "earth_disk_visible",
        "communication_visible",
    ]

    st.dataframe(
        mission_df[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download Mission Window Results CSV",
        data=mission_df.to_csv(index=False),
        file_name=f"{selected_site_id.lower()}_mission_windows.csv",
        mime="text/csv",
        key="download_mission_windows",
    )

    st.caption(
        "Solar energy is a simplified panel-output estimate. "
        "Communication visibility is geometric and does not "
        "include a radio link budget, antenna pointing model, "
        "or data-rate analysis."
    )


st.divider()


# ============================================================
# MULTI-SITE MISSION COMPARISON
# ============================================================

st.header("🛰️ Multi-Site Mission Comparison")

st.write(
    "Compare 2–4 candidate landing sites using identical mission "
    "dates, calculation intervals, antenna limits and solar-panel settings."
)

st.warning(
    "This calculation is more computationally intensive than the "
    "single-site planner. Each selected site receives its own "
    "terrain-horizon calculation."
)

comparison_site_options = list(sites.keys())

default_comparison_sites = [
    name for name in comparison_site_options
    if site_details[name]["Site ID"] in {
        "SITE01", "SITE04", "SITE06", "SITE07"
    }
]

selected_comparison_sites = st.multiselect(
    "Choose 2–4 sites to compare",
    options=comparison_site_options,
    default=default_comparison_sites,
    max_selections=4,
    key="multi_site_selection",
)

comparison_signature = (
    tuple(
        (
            site_details[name]["Site ID"],
            float(sites[name][0]),
            float(sites[name][1]),
        )
        for name in selected_comparison_sites
    ),
    str(observation_date),
    int(mission_interval),
    float(panel_power),
    float(antenna_limit),
    float(panel_tilt),
    float(panel_azimuth),
)

if len(selected_comparison_sites) < 2:
    st.info("Select at least two sites to run a comparison.")

else:
    st.caption(
        f"Mission period: {mission_start.date()} 00:00 UTC "
        f"to {mission_end.date()} 00:00 UTC"
    )

    st.caption(
        f"Interval: {mission_interval} minutes | "
        f"Panel power: {panel_power:.1f} W | "
        f"Antenna limit: {antenna_limit:.1f}° | "
        f"Panel tilt: {panel_tilt}° | "
        f"Panel azimuth: {panel_azimuth}°"
    )

    if st.button(
        "Run Multi-Site Mission Comparison",
        type="primary",
        use_container_width=True,
        key="run_multi_site_comparison",
    ):
        comparison_results = []
        comparison_errors = []

        progress = st.progress(0)
        status = st.empty()

        for index, site_name in enumerate(selected_comparison_sites):
            site_id = site_details[site_name]["Site ID"]
            site_latitude, site_longitude = sites[site_name]

            status.info(
                f"Calculating site {index + 1} of "
                f"{len(selected_comparison_sites)}: {site_name}"
            )

            try:
                # Each candidate gets its own terrain horizon.
                site_horizon, site_terrain = calculate_site_horizon(
                    site_id,
                    site_latitude,
                    site_longitude,
                )

                touchdown_elevation = None
                touchdown_slope = None

                if site_terrain["elevation"] is not None:
                    try:
                        _, _, _, elevation_meta = read_site_patch(
                            site_terrain["elevation"],
                            site_latitude,
                            site_longitude,
                        )
                        touchdown_elevation = elevation_meta["sample_value"]
                    except Exception:
                        touchdown_elevation = None

                if site_terrain["slope"] is not None:
                    try:
                        _, _, _, slope_meta = read_site_patch(
                            site_terrain["slope"],
                            site_latitude,
                            site_longitude,
                        )
                        touchdown_slope = slope_meta["sample_value"]
                    except Exception:
                        touchdown_slope = None

                result = calculate_mission_windows(
                    latitude_deg=site_latitude,
                    longitude_deg=site_longitude,
                    start_utc=mission_start,
                    end_utc=mission_end,
                    interval_minutes=mission_interval,
                    horizon_result=site_horizon,
                    antenna_min_elevation_deg=antenna_limit,
                    panel_rated_power_w=panel_power,
                    panel_tilt_deg=float(panel_tilt),
                    panel_azimuth_deg=float(panel_azimuth),
                )

                summary = result["summary"]

                comparison_results.append({
                    "Site": site_name,
                    "Site ID": site_id,
                    "Latitude (°)": site_latitude,
                    "Longitude (°)": site_longitude,
                    "Sunlight (h)": summary["sunlight_hours"],
                    "Sunlight (%)": summary["sunlight_percentage"],
                    "Communication (h)": summary["communication_hours"],
                    "Communication (%)": summary["communication_percentage"],
                    "Solar energy (Wh)": summary["estimated_energy_wh"],
                    "Touchdown elevation (m)": touchdown_elevation,
                    "Touchdown slope (°)": touchdown_slope,
                    "Terrain source": site_terrain["source"],
                    "Terrain resolution": site_terrain["resolution"],
                    "Horizon quality": site_terrain["horizon_quality"],
                    "Terrain horizon used": site_horizon is not None,
                })

            except Exception as comparison_error:
                comparison_errors.append({
                    "Site": site_name,
                    "Error": str(comparison_error),
                })

            progress.progress(
                (index + 1) / len(selected_comparison_sites)
            )

        status.empty()
        progress.empty()

        # Save the results and their matching settings.
        st.session_state["multi_site_results"] = comparison_results
        st.session_state["multi_site_errors"] = comparison_errors
        st.session_state["multi_site_signature"] = comparison_signature

        if comparison_results:
            st.success(
                f"Comparison completed for "
                f"{len(comparison_results)} site(s)."
            )

        if comparison_errors:
            st.warning(
                f"{len(comparison_errors)} site(s) could not be calculated."
            )


# ============================================================
# DISPLAY MULTI-SITE COMPARISON RESULTS
# ============================================================

multi_results = st.session_state.get("multi_site_results", [])
multi_errors = st.session_state.get("multi_site_errors", [])

# FIX:
# Read the signature directly from session state here.
# This ensures that newly saved results are recognized immediately.
multi_results_match = (
    bool(multi_results)
    and st.session_state.get("multi_site_signature")
    == comparison_signature
)

if multi_results and not multi_results_match:
    st.info(
        "The saved comparison belongs to different sites or mission "
        "settings. Run the comparison again to refresh the results."
    )

if multi_results_match:
    multi_df = pd.DataFrame(multi_results)

    st.divider()
    st.subheader("Multi-Site Comparison Results")

    st.dataframe(
        multi_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(" Sunlight Comparison")

    fig_compare_sun = go.Figure()

    fig_compare_sun.add_trace(
        go.Bar(
            x=multi_df["Site ID"],
            y=multi_df["Sunlight (h)"],
            name="Sunlight hours",
            text=multi_df["Sunlight (h)"].map(
                lambda value: f"{value:.2f} h"
            ),
            textposition="outside",
        )
    )

    fig_compare_sun.update_layout(
        title="Daily Sunlight Duration by Site",
        xaxis_title="Landing site",
        yaxis_title="Sunlight duration (hours)",
        template="plotly_dark",
        paper_bgcolor="#07111f",
        plot_bgcolor="#07111f",
        height=420,
    )

    st.plotly_chart(fig_compare_sun, use_container_width=True)

    st.subheader("Earth Communication Comparison")

    fig_compare_comms = go.Figure()

    fig_compare_comms.add_trace(
        go.Bar(
            x=multi_df["Site ID"],
            y=multi_df["Communication (h)"],
            name="Communication hours",
            text=multi_df["Communication (h)"].map(
                lambda value: f"{value:.2f} h"
            ),
            textposition="outside",
        )
    )

    fig_compare_comms.update_layout(
        title="Daily Earth Communication Opportunity by Site",
        xaxis_title="Landing site",
        yaxis_title="Communication duration (hours)",
        template="plotly_dark",
        paper_bgcolor="#07111f",
        plot_bgcolor="#07111f",
        height=420,
    )

    st.plotly_chart(fig_compare_comms, use_container_width=True)

    st.subheader(" Estimated Solar Energy Comparison")

    fig_compare_energy = go.Figure()

    fig_compare_energy.add_trace(
        go.Bar(
            x=multi_df["Site ID"],
            y=multi_df["Solar energy (Wh)"],
            name="Estimated energy",
            text=multi_df["Solar energy (Wh)"].map(
                lambda value: f"{value:.1f} Wh"
            ),
            textposition="outside",
        )
    )

    fig_compare_energy.update_layout(
        title="Estimated Daily Solar Energy by Site",
        xaxis_title="Landing site",
        yaxis_title="Estimated energy (Wh)",
        template="plotly_dark",
        paper_bgcolor="#07111f",
        plot_bgcolor="#07111f",
        height=420,
    )

    st.plotly_chart(fig_compare_energy, use_container_width=True)

    st.subheader("⛰️ Touchdown Terrain Comparison")

    terrain_compare_df = multi_df[
        [
            "Site ID",
            "Touchdown elevation (m)",
            "Touchdown slope (°)",
            "Terrain source",
            "Terrain resolution",
            "Horizon quality",
            "Terrain horizon used",
        ]
    ]

    st.dataframe(
        terrain_compare_df,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download Multi-Site Comparison CSV",
        data=multi_df.to_csv(index=False),
        file_name="lunaview_multi_site_mission_comparison.csv",
        mime="text/csv",
        key="download_multi_site_comparison",
    )

    st.caption(
        "Touchdown elevation and slope are center-pixel raster samples, "
        "not averages of the surrounding terrain patch. Terrain products "
        "have different spatial resolutions, so values are not equally "
        "precise. This comparison is not a landing-safety certification."
    )

if multi_errors:
    with st.expander("View sites with calculation errors"):
        st.dataframe(
            pd.DataFrame(multi_errors),
            use_container_width=True,
            hide_index=True,
        )

# ============================================================
# MULTI-DATE MISSION COMPARISON
# ============================================================

st.header("📅 Multi-Date Mission Comparison")

st.write(
    "Compare different landing dates at the currently selected "
    "site to understand how mission schedule changes affect "
    "solar power generation and direct-to-Earth communications."
)

st.info(
    "The selected site's terrain horizon is reused across all "
    "dates because the terrain does not change with the observation date. "
    "Sun and Earth positions are recalculated independently for each date."
)

st.subheader("Select Mission Dates")

date_col1, date_col2 = st.columns(2)

with date_col1:
    date_1 = st.date_input(
        "Mission date 1",
        value=observation_date,
        key="multi_date_1",
    )

    date_2 = st.date_input(
        "Mission date 2",
        value=observation_date + timedelta(days=7),
        key="multi_date_2",
    )

with date_col2:
    date_3 = st.date_input(
        "Mission date 3",
        value=observation_date + timedelta(days=14),
        key="multi_date_3",
    )

    date_4 = st.date_input(
        "Mission date 4",
        value=observation_date + timedelta(days=21),
        key="multi_date_4",
    )

use_date_3 = st.checkbox(
    "Include mission date 3",
    value=True,
    key="include_multi_date_3",
)

use_date_4 = st.checkbox(
    "Include mission date 4",
    value=True,
    key="include_multi_date_4",
)

selected_dates = [date_1, date_2]

if use_date_3:
    selected_dates.append(date_3)

if use_date_4:
    selected_dates.append(date_4)

st.caption(
    f"Selected landing site: {selected_site} | "
    f"Site ID: {selected_site_id}"
)

st.caption(
    f"Dates selected: {len(selected_dates)} | "
    f"Calculation interval: {mission_interval} minutes"
)

# ------------------------------------------------------------
# VALIDATE DATE SELECTION
# ------------------------------------------------------------

date_comparison_valid = True

if len(set(selected_dates)) != len(selected_dates):
    st.error(
        "Please select different dates. Duplicate dates "
        "cannot be compared."
    )
    date_comparison_valid = False

if len(selected_dates) < 2:
    st.warning(
        "At least two different mission dates are required."
    )
    date_comparison_valid = False


# ------------------------------------------------------------
# CREATE DATE COMPARISON SIGNATURE
# ------------------------------------------------------------

date_comparison_signature = (
    selected_site_id,
    float(latitude),
    float(longitude),
    tuple(sorted(str(date) for date in selected_dates)),
    int(mission_interval),
    float(panel_power),
    float(antenna_limit),
    float(panel_tilt),
    float(panel_azimuth),
    bool(horizon_result is not None),
)

# ------------------------------------------------------------
# RUN MULTI-DATE CALCULATIONS
# ------------------------------------------------------------

if st.button(
    "Compare Mission Dates",
    type="primary",
    use_container_width=True,
    key="run_multi_date_comparison",
    disabled=not date_comparison_valid,
):

    date_results = []
    date_errors = []

    progress = st.progress(0)
    status = st.empty()

    for index, mission_date in enumerate(selected_dates):

        status.info(
            f"Calculating date {index + 1} of "
            f"{len(selected_dates)}: {mission_date}"
        )

        try:

            date_start = datetime(
                mission_date.year,
                mission_date.month,
                mission_date.day,
                0,
                0,
                0,
                tzinfo=timezone.utc,
            )

            date_end = date_start + timedelta(days=1)

            result = calculate_mission_windows(
                latitude_deg=latitude,
                longitude_deg=longitude,
                start_utc=date_start,
                end_utc=date_end,
                interval_minutes=mission_interval,
                horizon_result=horizon_result,
                antenna_min_elevation_deg=antenna_limit,
                panel_rated_power_w=panel_power,
                panel_tilt_deg=float(panel_tilt),
                panel_azimuth_deg=float(panel_azimuth),
            )

            summary = result["summary"]

            date_results.append({
                "Mission date": str(mission_date),
                "Site ID": selected_site_id,
                "Site": selected_site,
                "Sunlight (h)": summary["sunlight_hours"],
                "Sunlight (%)": summary["sunlight_percentage"],
                "Communication (h)": summary["communication_hours"],
                "Communication (%)": summary["communication_percentage"],
                "Solar energy (Wh)": summary["estimated_energy_wh"],
                "Sample count": summary["sample_count"],
                "Terrain horizon used": horizon_result is not None,
            })

        except Exception as date_error:

            date_errors.append({
                "Mission date": str(mission_date),
                "Error": str(date_error),
            })

        progress.progress(
            (index + 1) / len(selected_dates)
        )

    progress.empty()
    status.empty()

    st.session_state["multi_date_results"] = date_results
    st.session_state["multi_date_errors"] = date_errors
    st.session_state["multi_date_signature"] = (
        date_comparison_signature
    )

    if date_results:
        st.success(
            f"Date comparison completed for "
            f"{len(date_results)} mission date(s)."
        )

    if date_errors:
        st.warning(
            f"{len(date_errors)} date(s) could not be calculated."
        )


# ============================================================
# DISPLAY MULTI-DATE RESULTS
# ============================================================

stored_date_results = st.session_state.get(
    "multi_date_results",
    [],
)

stored_date_errors = st.session_state.get(
    "multi_date_errors",
    [],
)

stored_date_signature = st.session_state.get(
    "multi_date_signature"
)

date_results_match = (
    bool(stored_date_results)
    and stored_date_signature == date_comparison_signature
)

if stored_date_results and not date_results_match:

    st.info(
        "Saved date-comparison results belong to different "
        "site coordinates, dates or mission settings. "
        "Run the comparison again to update them."
    )

if date_results_match:

    date_df = pd.DataFrame(stored_date_results)

    st.divider()

    st.subheader("📊 Mission Date Comparison Results")

    st.dataframe(
        date_df,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # SUNLIGHT COMPARISON
    # --------------------------------------------------------

    st.subheader("☀️ Sunlight Duration Comparison")

    fig_date_sun = go.Figure()

    fig_date_sun.add_trace(
        go.Bar(
            x=date_df["Mission date"],
            y=date_df["Sunlight (h)"],
            name="Sunlight duration",
            text=date_df["Sunlight (h)"].map(
                lambda value: f"{value:.2f} h"
            ),
            textposition="outside",
        )
    )

    fig_date_sun.update_layout(
        title="Daily Sunlight Duration by Landing Date",
        xaxis_title="Mission date (UTC)",
        yaxis_title="Sunlight duration (hours)",
        template="plotly_dark",
        paper_bgcolor="#07111f",
        plot_bgcolor="#07111f",
        font=dict(color="#e2e8f0"),
        height=420,
    )

    st.plotly_chart(
        fig_date_sun,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # COMMUNICATION COMPARISON
    # --------------------------------------------------------

    st.subheader("🌍 Direct-to-Earth Communication Comparison")

    fig_date_comms = go.Figure()

    fig_date_comms.add_trace(
        go.Bar(
            x=date_df["Mission date"],
            y=date_df["Communication (h)"],
            name="Communication duration",
            text=date_df["Communication (h)"].map(
                lambda value: f"{value:.2f} h"
            ),
            textposition="outside",
        )
    )

    fig_date_comms.update_layout(
        title="Daily Earth Communication Opportunities by Date",
        xaxis_title="Mission date (UTC)",
        yaxis_title="Communication duration (hours)",
        template="plotly_dark",
        paper_bgcolor="#07111f",
        plot_bgcolor="#07111f",
        font=dict(color="#e2e8f0"),
        height=420,
    )

    st.plotly_chart(
        fig_date_comms,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # SOLAR ENERGY COMPARISON
    # --------------------------------------------------------

    st.subheader("⚡ Solar Energy Comparison")

    fig_date_energy = go.Figure()

    fig_date_energy.add_trace(
        go.Bar(
            x=date_df["Mission date"],
            y=date_df["Solar energy (Wh)"],
            name="Estimated solar energy",
            text=date_df["Solar energy (Wh)"].map(
                lambda value: f"{value:.1f} Wh"
            ),
            textposition="outside",
        )
    )

    fig_date_energy.update_layout(
        title="Estimated Solar Energy by Landing Date",
        xaxis_title="Mission date (UTC)",
        yaxis_title="Estimated energy (Wh)",
        template="plotly_dark",
        paper_bgcolor="#07111f",
        plot_bgcolor="#07111f",
        font=dict(color="#e2e8f0"),
        height=420,
    )

    st.plotly_chart(
        fig_date_energy,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # BEST RESULTS
    # --------------------------------------------------------

    st.subheader("🏆 Mission Date Assessment")

    best_sunlight = date_df.loc[
        date_df["Sunlight (h)"].idxmax()
    ]

    best_communication = date_df.loc[
        date_df["Communication (h)"].idxmax()
    ]

    best_energy = date_df.loc[
        date_df["Solar energy (Wh)"].idxmax()
    ]

    best1, best2, best3 = st.columns(3)

    best1.metric(
        "Most sunlight",
        str(best_sunlight["Mission date"]),
        f"{best_sunlight['Sunlight (h)']:.2f} hours",
    )

    best2.metric(
        "Most communication",
        str(best_communication["Mission date"]),
        f"{best_communication['Communication (h)']:.2f} hours",
    )

    best3.metric(
        "Highest estimated energy",
        str(best_energy["Mission date"]),
        f"{best_energy['Solar energy (Wh)']:.2f} Wh",
    )

    st.caption(
        "These are separate single-metric comparisons. "
        "The date with the most sunlight may not be the date "
        "with the best communication availability."
    )

    # --------------------------------------------------------
    # EXPORT RESULTS
    # --------------------------------------------------------

    st.download_button(
        "Download Multi-Date Comparison CSV",
        data=date_df.to_csv(index=False),
        file_name="lunaview_multi_date_comparison.csv",
        mime="text/csv",
        key="download_multi_date_comparison",
    )

    st.caption(
        "Results are calculated independently for each UTC date "
        "using the selected site's coordinates and current mission "
        "settings. Solar energy is a simplified panel-output estimate. "
        "Communication estimates are geometric and do not include "
        "a radio link budget or data-rate analysis."
    )


# ------------------------------------------------------------
# DISPLAY DATE CALCULATION ERRORS
# ------------------------------------------------------------

if stored_date_errors:

    with st.expander("View mission dates with calculation errors"):

        st.dataframe(
            pd.DataFrame(stored_date_errors),
            use_container_width=True,
            hide_index=True,
        )

st.divider()



# ============================================================
# LUNAVIEW MISSION DECISION SUPPORT MODULES
# ============================================================
# These modules build on the existing LOLA rasters and the existing
# calculate_mission_windows() output. Missing scientific datasets are
# explicitly reported rather than replaced with geographic guesses.


def _luna_local_slope_from_dem(dem, x_coords, y_coords, latitude_deg):
    """Estimate slope in degrees from a local DEM patch."""
    if dem.shape[0] < 2 or dem.shape[1] < 2:
        raise ValueError("The DEM patch is too small to estimate slope.")
    dx_native = float(np.nanmedian(np.abs(np.diff(x_coords)))) if len(x_coords) > 1 else 1.0
    dy_native = float(np.nanmedian(np.abs(np.diff(y_coords)))) if len(y_coords) > 1 else 1.0
    # Geographic rasters use degrees. Convert local pixel spacing to metres.
    # Projected rasters are assumed to use metres, as with the supplied LOLA products.
    if dx_native < 2.0 and dy_native < 2.0:
        dx_m = max(0.01, dx_native * math.pi / 180.0 * MOON_RADIUS_M * max(0.01, math.cos(math.radians(latitude_deg))))
        dy_m = max(0.01, dy_native * math.pi / 180.0 * MOON_RADIUS_M)
    else:
        dx_m, dy_m = max(0.01, dx_native), max(0.01, dy_native)
    # np.gradient returns derivatives in array-axis order (row/y, column/x).
    dz_dy, dz_dx = np.gradient(dem.astype(float), dy_m, dx_m)
    return np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))


def _luna_astar_route(elevation_grid, slope_grid, start, goal, x_coords, y_coords, max_slope_deg):
    """8-connected slope-aware A* route over a local raster patch."""
    import heapq

    rows, cols = elevation_grid.shape
    sr, sc = start
    gr, gc = goal
    if not (0 <= sr < rows and 0 <= sc < cols and 0 <= gr < rows and 0 <= gc < cols):
        raise ValueError("Route endpoints fall outside the available terrain patch.")
    valid = np.isfinite(elevation_grid) & np.isfinite(slope_grid) & (slope_grid <= max_slope_deg)
    if not valid[sr, sc]:
        raise ValueError("The landing point is invalid or exceeds the selected slope limit.")
    if not valid[gr, gc]:
        raise ValueError("The destination is invalid or exceeds the selected slope limit.")

    dx = float(np.nanmedian(np.abs(np.diff(x_coords)))) if len(x_coords) > 1 else 1.0
    dy = float(np.nanmedian(np.abs(np.diff(y_coords)))) if len(y_coords) > 1 else 1.0
    # A* uses raster coordinate spacing for relative distance; output distance is
    # converted to metres below using the same local grid spacing.
    moves = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    def heuristic(r, c):
        return math.hypot((gr-r)*dy, (gc-c)*dx)
    queue = [(heuristic(sr, sc), 0.0, sr, sc)]
    came_from = {}
    costs = {(sr, sc): 0.0}
    closed = set()
    while queue:
        _, cost, r, c = heapq.heappop(queue)
        if (r, c) in closed:
            continue
        if (r, c) == (gr, gc):
            path = [(gr, gc)]
            while path[-1] != (sr, sc):
                path.append(came_from[path[-1]])
            path.reverse()
            return path
        closed.add((r, c))
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if nr < 0 or nc < 0 or nr >= rows or nc >= cols or not valid[nr, nc]:
                continue
            step = math.hypot(dy if dr else 0.0, dx if dc else 0.0)
            local_slope = max(float(slope_grid[r, c]), float(slope_grid[nr, nc]))
            # Prefer flatter ground while retaining a finite traversable cost.
            elevation_penalty = abs(float(elevation_grid[nr, nc] - elevation_grid[r, c]))
            new_cost = cost + step * (1.0 + local_slope / max(1.0, max_slope_deg)) + elevation_penalty * 0.02
            if new_cost < costs.get((nr, nc), float("inf")):
                costs[(nr, nc)] = new_cost
                came_from[(nr, nc)] = (r, c)
                heapq.heappush(queue, (new_cost + heuristic(nr, nc), new_cost, nr, nc))
    raise ValueError("No continuous route satisfies the selected slope and NoData constraints.")


st.header("🛬 Landing Safety & Terrain Suitability")
st.write(
    "Evaluate the local landing area using the selected LOLA slope raster. "
    "If a slope raster is unavailable, LunaView estimates slope from the elevation patch and labels it accordingly."
)

_safety_slope = None
_safety_source = None
_safety_meta = None
_safety_x = None
_safety_y = None
_safety_dem = None
_center_slope = None

if elevation_path is not None and Path(elevation_path).exists():
    try:
        _safety_x, _safety_y, _safety_dem, _safety_meta = read_site_patch(elevation_path, latitude, longitude)
        if slope_path is not None and Path(slope_path).exists():
            _sx, _sy, _candidate_slope, _slope_meta = read_site_patch(slope_path, latitude, longitude)
            if (_candidate_slope.shape == _safety_dem.shape and len(_sx) == len(_safety_x) and len(_sy) == len(_safety_y) and np.allclose(_sx, _safety_x) and np.allclose(_sy, _safety_y)):
                # Only combine rasters whose local pixel coordinates actually align.
                _safety_slope = _candidate_slope
                _safety_source = f"Provided slope raster: {_slope_meta['file']}"
        if _safety_slope is None:
            _safety_slope = _luna_local_slope_from_dem(_safety_dem, _safety_x, _safety_y, latitude)
            _safety_source = "Slope estimated from the local elevation raster (not a supplied slope product)"
        _valid_safety = np.isfinite(_safety_slope) & np.isfinite(_safety_dem)
        _valid_slopes = _safety_slope[_valid_safety]
        if _valid_slopes.size:
            _safe5 = float(np.mean(_valid_slopes <= 5.0) * 100.0)
            _safe10 = float(np.mean(_valid_slopes <= 10.0) * 100.0)
            _safe15 = float(np.mean(_valid_slopes <= 15.0) * 100.0)
            _center_row = int(np.argmin(np.abs(_safety_y - _safety_y[len(_safety_y)//2])))
            _center_col = int(np.argmin(np.abs(_safety_x - _safety_x[len(_safety_x)//2])))
            _center_slope = float(_safety_slope[_center_row, _center_col]) if np.isfinite(_safety_slope[_center_row, _center_col]) else None
            _sc1, _sc2, _sc3 = st.columns(3)
            _sc1.metric("Area at ≤5° slope", f"{_safe5:.1f}%")
            _sc2.metric("Area at ≤10° slope", f"{_safe10:.1f}%")
            _sc3.metric("Area at ≤15° slope", f"{_safe15:.1f}%")
            _safety_fig = go.Figure(go.Heatmap(x=_safety_x, y=_safety_y, z=np.where(_valid_safety, _safety_slope, np.nan), colorscale="Turbo", zmin=0, zmax=30, colorbar=dict(title="Slope (°)"), hovertemplate="X %{x:.1f}<br>Y %{y:.1f}<br>Slope %{z:.2f}°<extra></extra>"))
            _safety_fig.update_layout(title="Local Landing-Zone Slope Hazard Map", template="plotly_dark", paper_bgcolor="#07111f", plot_bgcolor="#07111f", xaxis_title="Raster X", yaxis_title="Raster Y", height=440)
            st.plotly_chart(_safety_fig, use_container_width=True)
            st.caption(f"Assessment source: {_safety_source}. Patch: {_safety_meta['file']} ({_safety_meta['crs']}).")
            if _center_slope is None:
                st.warning("Landing-point slope is NoData; no touchdown slope classification is available.")
            elif _center_slope <= 5:
                st.success(f"Landing-point slope estimate: {_center_slope:.2f}°. Favorable under the illustrative ≤5° screening threshold.")
            elif _center_slope <= 10:
                st.warning(f"Landing-point slope estimate: {_center_slope:.2f}°. Requires detailed hazard and lander-footprint assessment.")
            elif _center_slope <= 15:
                st.warning(f"Landing-point slope estimate: {_center_slope:.2f}°. High terrain concern under this screening model.")
            else:
                st.error(f"Landing-point slope estimate: {_center_slope:.2f}°. Exceeds the illustrative 15° screening threshold.")
            st.caption("Slope thresholds are configurable screening assumptions, not certified lander acceptance criteria. Raster resolution and local hazards limit confidence.")
        else:
            st.warning("The selected terrain patch contains no valid slope/elevation pixels.")
    except Exception as _safety_error:
        st.warning(f"Landing safety assessment could not be completed: {_safety_error}")
else:
    st.info("Landing safety requires a local LOLA elevation raster for this site.")

st.divider()


st.header("🏆 Mission Suitability Score & Explainable Ranking")
st.write("A transparent 0–100 screening score combines calculated sunlight, direct-to-Earth communication, simultaneous availability, and terrain slope. Missing inputs are excluded and the score is marked incomplete.")

if results_match and mission_result is not None:
    _sum = mission_result["summary"]
    _records_score = pd.DataFrame(mission_result["records"])
    _overlap_pct = float((_records_score["sun_visible"].astype(bool) & _records_score["communication_visible"].astype(bool)).mean() * 100.0) if not _records_score.empty else None
    _components = []
    _weights = []
    _labels = []
    for _label, _value, _weight in [
        ("Sunlight availability", float(_sum["sunlight_percentage"]), 0.35),
        ("Earth communication", float(_sum["communication_percentage"]), 0.25),
        ("Simultaneous Sun + Earth", _overlap_pct, 0.15),
        ("Terrain slope safety", (max(0.0, min(100.0, (15.0 - float(_center_slope)) / 15.0 * 100.0)) if '_center_slope' in globals() and _center_slope is not None else None), 0.25),
    ]:
        if _value is not None and np.isfinite(_value):
            _components.append(max(0.0, min(100.0, _value)) * _weight)
            _weights.append(_weight)
            _labels.append((_label, max(0.0, min(100.0, _value)), _weight))
    if _weights:
        _score = sum(_components) / sum(_weights)
        _coverage = sum(_weights) * 100.0
        _sc_a, _sc_b, _sc_c = st.columns(3)
        _sc_a.metric("Mission suitability", f"{_score:.1f} / 100")
        _sc_b.metric("Input coverage", f"{_coverage:.0f}%")
        _sc_c.metric("Screening category", "Strong" if _score >= 75 else "Moderate" if _score >= 50 else "Constrained")
        _score_df = pd.DataFrame([{"Factor": label, "Normalized result (%)": value, "Weight": f"{weight*100:.0f}%"} for label, value, weight in _labels])
        st.dataframe(_score_df, use_container_width=True, hide_index=True)
        st.progress(int(round(_coverage)))
        if _coverage < 100:
            st.warning("Score is provisional because one or more inputs are unavailable. The displayed score is reweighted across available factors; it is not directly comparable with a fully populated score.")
        st.caption("Weights: sunlight 35%, communications 25%, simultaneous availability 15%, terrain slope 25%. Terrain safety is a simple screening transform of touchdown slope, not a validated landing probability.")
    else:
        st.warning("No valid inputs are available to calculate a suitability score.")
else:
    st.info("Calculate the 24-hour mission windows above to generate a suitability score for the current site and settings.")

st.divider()


st.header("🧊 Permanently Shadowed Region (PSR) Data Explorer")
st.write("Load a georeferenced PSR mask raster to inspect actual mapped shadow regions around the selected coordinates. LunaView does not infer PSRs or water ice from latitude alone.")
_psr_upload = st.file_uploader("Upload PSR GeoTIFF mask (0 = not PSR, non-zero = PSR; preserve lunar georeferencing)", type=["tif", "tiff"], key="psr_geotiff_upload")
if _psr_upload is not None:
    try:
        from rasterio.io import MemoryFile
        with MemoryFile(_psr_upload.getvalue()) as _memfile:
            with _memfile.open() as _psr_src:
                if _psr_src.crs is None:
                    raise ValueError("The PSR raster has no CRS. A valid lunar CRS is required.")
                _psr_transformer = Transformer.from_crs(MOON_GEOGRAPHIC_CRS, _psr_src.crs, always_xy=True)
                if _psr_src.crs.is_geographic:
                    _psr_x0, _psr_y0 = float(longitude) % 360.0, float(latitude)
                else:
                    _psr_x0, _psr_y0 = _psr_transformer.transform(float(longitude), float(latitude))
                _pr, _pc = _psr_src.index(_psr_x0, _psr_y0)
                _half = 120
                _r0, _r1 = max(0, _pr-_half), min(_psr_src.height, _pr+_half)
                _c0, _c1 = max(0, _pc-_half), min(_psr_src.width, _pc+_half)
                if _r0 >= _r1 or _c0 >= _c1:
                    raise ValueError("Selected coordinates fall outside the uploaded PSR raster.")
                _psr_window = Window(_c0, _r0, _c1-_c0, _r1-_r0)
                _psr_arr = _psr_src.read(1, window=_psr_window, masked=True)
                _psr_data = np.ma.filled(_psr_arr, np.nan).astype(float)
                _psr_affine = _psr_src.window_transform(_psr_window)
                _psr_x = _psr_affine.c + (np.arange(_psr_data.shape[1]) + 0.5) * _psr_affine.a
                _psr_y = _psr_affine.f + (np.arange(_psr_data.shape[0]) + 0.5) * _psr_affine.e
                _psr_valid = np.isfinite(_psr_data)
                _psr_mask = np.where(_psr_valid & (_psr_data != 0), 1.0, np.where(_psr_valid, 0.0, np.nan))
                _psr_count = int(np.nansum(_psr_mask == 1.0))
                _psr_pct = float(np.nanmean(_psr_mask) * 100.0) if np.isfinite(_psr_mask).any() else 0.0
                _pc1, _pc2 = st.columns(2)
                _pc1.metric("PSR pixels in local patch", f"{_psr_count:,}")
                _pc2.metric("Mapped PSR area fraction", f"{_psr_pct:.1f}%")
                _psr_fig = go.Figure(go.Heatmap(x=_psr_x, y=_psr_y, z=_psr_mask, zmin=0, zmax=1, colorscale=[[0,"#334155"],[0.49,"#334155"],[0.5,"#38bdf8"],[1,"#38bdf8"]], colorbar=dict(title="PSR mask", tickvals=[0,1], ticktext=["Not PSR","PSR"]), connectgaps=False))
                _psr_fig.update_layout(title=f"Uploaded PSR Mask near {selected_site}", template="plotly_dark", paper_bgcolor="#07111f", plot_bgcolor="#07111f", xaxis_title="Raster X", yaxis_title="Raster Y", height=440)
                st.plotly_chart(_psr_fig, use_container_width=True)
                st.caption(f"Dataset: {_psr_upload.name} | CRS: {_psr_src.crs} | Raster resolution: {_psr_src.res}. Non-zero pixels are interpreted as PSR according to the upload instructions.")
    except Exception as _psr_error:
        st.error(f"PSR raster could not be read: {_psr_error}")
else:
    st.info("No PSR raster is loaded. Upload a scientifically sourced lunar PSR GeoTIFF to enable this map. A PSR indicates persistent lack of direct sunlight; it does not by itself confirm accessible water ice.")

st.divider()


st.header("🧭 Surface Exploration Route Planner")
st.write("Generate an experimental slope-aware rover route over the currently selected local LOLA elevation patch. The route is limited to the displayed raster window and is not a flight-qualified mobility plan.")
if _safety_dem is not None and _safety_x is not None and _safety_y is not None and _safety_slope is not None:
    _route_col1, _route_col2 = st.columns(2)
    with _route_col1:
        _route_dest_lat = st.number_input("Destination latitude (°)", min_value=-90.0, max_value=90.0, value=float(latitude), format="%.5f", key="route_dest_lat")
    with _route_col2:
        _route_dest_lon = st.number_input("Destination longitude (° east)", min_value=-180.0, max_value=180.0, value=float(longitude) + 2.0 if longitude <= 178.0 else float(longitude) - 2.0, format="%.5f", key="route_dest_lon")
    _route_slope_limit = st.slider("Maximum traversable slope (°)", min_value=5, max_value=35, value=15, key="route_slope_limit")
    if st.button("Calculate Slope-Aware Rover Route", type="primary", key="calculate_rover_route"):
        try:
            with rasterio.open(elevation_path) as _route_src:
                _route_transformer = Transformer.from_crs(MOON_GEOGRAPHIC_CRS, _route_src.crs, always_xy=True)
                if _route_src.crs.is_geographic:
                    _tx, _ty = float(_route_dest_lon) % 360.0, float(_route_dest_lat)
                else:
                    _tx, _ty = _route_transformer.transform(float(_route_dest_lon), float(_route_dest_lat))
            _goal_col = int(np.argmin(np.abs(_safety_x - _tx)))
            _goal_row = int(np.argmin(np.abs(_safety_y - _ty)))
            _start = (int(len(_safety_y)//2), int(len(_safety_x)//2))
            _goal = (_goal_row, _goal_col)
            _route = _luna_astar_route(_safety_dem, _safety_slope, _start, _goal, _safety_x, _safety_y, float(_route_slope_limit))
            _route_rows = np.array([point[0] for point in _route], dtype=int)
            _route_cols = np.array([point[1] for point in _route], dtype=int)
            _route_z = np.array([_safety_dem[r,c] for r,c in _route], dtype=float)
            # Convert native raster spacing to approximate metres.
            _rx = float(np.nanmedian(np.abs(np.diff(_safety_x)))) if len(_safety_x)>1 else 1.0
            _ry = float(np.nanmedian(np.abs(np.diff(_safety_y)))) if len(_safety_y)>1 else 1.0
            if _rx < 2 and _ry < 2:
                _route_dx_m = _rx * math.pi/180 * MOON_RADIUS_M * max(0.01, math.cos(math.radians(latitude)))
                _route_dy_m = _ry * math.pi/180 * MOON_RADIUS_M
            else:
                _route_dx_m, _route_dy_m = _rx, _ry
            _route_distance_m = sum(math.hypot((_route_rows[i]-_route_rows[i-1])*_route_dy_m, (_route_cols[i]-_route_cols[i-1])*_route_dx_m) for i in range(1,len(_route)))
            _route_gain = float(np.maximum(0, np.diff(_route_z)).sum()) if len(_route_z)>1 else 0.0
            _route_loss = float(np.maximum(0, -np.diff(_route_z)).sum()) if len(_route_z)>1 else 0.0
            _route_mean_slope = float(np.mean([_safety_slope[r,c] for r,c in _route]))
            _rc1, _rc2, _rc3 = st.columns(3)
            _rc1.metric("Route distance", f"{_route_distance_m/1000:.2f} km")
            _rc2.metric("Mean route slope", f"{_route_mean_slope:.1f}°")
            _rc3.metric("Cumulative ascent", f"{_route_gain:.1f} m")
            _route_fig = go.Figure(go.Heatmap(x=_safety_x, y=_safety_y, z=_safety_slope, colorscale="Turbo", zmin=0, zmax=35, colorbar=dict(title="Slope (°)"), connectgaps=False))
            _route_fig.add_trace(go.Scatter(x=_safety_x[_route_cols], y=_safety_y[_route_rows], mode="lines+markers", line=dict(color="#22c55e", width=4), marker=dict(size=4), name="Suggested rover route"))
            _route_fig.add_trace(go.Scatter(x=[_safety_x[_start[1]]], y=[_safety_y[_start[0]]], mode="markers+text", marker=dict(size=12,color="#facc15"), text=["START"], textposition="top center", name="Start"))
            _route_fig.add_trace(go.Scatter(x=[_safety_x[_goal[1]]], y=[_safety_y[_goal[0]]], mode="markers+text", marker=dict(size=12,color="#38bdf8"), text=["DESTINATION"], textposition="top center", name="Destination"))
            _route_fig.update_layout(title="Slope-Aware Route on Local LOLA Patch", template="plotly_dark", paper_bgcolor="#07111f", plot_bgcolor="#07111f", xaxis_title="Raster X", yaxis_title="Raster Y", height=500)
            st.plotly_chart(_route_fig, use_container_width=True)
            _route_df = pd.DataFrame({"Step":np.arange(len(_route)),"Raster row":_route_rows,"Raster column":_route_cols,"Elevation (m)":_route_z,"Slope (°)":[_safety_slope[r,c] for r,c in _route]})
            st.download_button("Download Rover Route CSV", data=_route_df.to_csv(index=False), file_name=f"{selected_site_id.lower()}_rover_route.csv", mime="text/csv", key="download_rover_route")
            st.caption(f"Estimated cumulative descent: {_route_loss:.1f} m. Route cost penalizes slope and elevation changes; it does not model rover traction, wheel slip, boulders, energy, or detailed hazard imagery.")
        except Exception as _route_error:
            st.error(f"Route calculation failed: {_route_error}")
else:
    st.info("A readable local elevation patch and a derived or supplied slope grid are required for route planning.")

st.divider()


st.header("🗓️ Mission Operations Scheduler")
st.write("Assign activities to consecutive time samples that satisfy their sunlight and direct-to-Earth communication requirements. Calculate the mission windows above first.")
if results_match and mission_result is not None:
    _ops_df = pd.DataFrame(mission_result["records"])
    _ops_df["_time"] = pd.to_datetime(_ops_df["utc_time"], utc=True)
    _ops_interval_h = float(mission_interval) / 60.0
    _default_tasks = pd.DataFrame([
        {"Activity":"Deploy instruments", "Duration (h)":1.0, "Needs sunlight":True, "Needs Earth link":False},
        {"Activity":"Science observation", "Duration (h)":2.0, "Needs sunlight":True, "Needs Earth link":False},
        {"Activity":"Transmit science data", "Duration (h)":1.0, "Needs sunlight":False, "Needs Earth link":True},
        {"Activity":"Rover traverse", "Duration (h)":2.0, "Needs sunlight":True, "Needs Earth link":False},
    ])
    _task_input = st.data_editor(_default_tasks, num_rows="dynamic", use_container_width=True, key="mission_operations_tasks", column_config={"Duration (h)":st.column_config.NumberColumn(min_value=0.25,max_value=24.0,step=0.25)})
    if st.button("Build Mission Operations Schedule", type="primary", key="build_mission_ops_schedule"):
        _busy = set()
        _scheduled = []
        _unscheduled = []
        _step_h = max(_ops_interval_h, 1e-6)
        for _, _task in _task_input.iterrows():
            _name = str(_task.get("Activity", "")).strip()
            _duration = float(_task.get("Duration (h)", 0) or 0)
            if not _name or _duration <= 0:
                continue
            _needed = max(1, int(math.ceil(_duration / _step_h)))
            _found = None
            for _i in range(0, len(_ops_df)-_needed+1):
                _indices = list(range(_i, _i+_needed))
                if any(_j in _busy for _j in _indices):
                    continue
                _window = _ops_df.iloc[_indices]
                if bool(_task.get("Needs sunlight", False)) and not bool(_window["sun_visible"].astype(bool).all()):
                    continue
                if bool(_task.get("Needs Earth link", False)) and not bool(_window["communication_visible"].astype(bool).all()):
                    continue
                _found = _indices
                break
            if _found is None:
                _unscheduled.append({"Activity":_name,"Duration (h)":_duration,"Reason":"No sufficiently long unoccupied window meets the selected constraints."})
            else:
                _busy.update(_found)
                _scheduled.append({"Activity":_name,"Start (UTC)":_ops_df.iloc[_found[0]]["_time"].strftime("%Y-%m-%d %H:%M"),"End (UTC)":min(_ops_df.iloc[_found[-1]]["_time"]+pd.Timedelta(hours=_step_h), pd.Timestamp(mission_end)).strftime("%Y-%m-%d %H:%M"),"Requested duration (h)":_duration,"Allocated samples":len(_found),"Sunlight required":bool(_task.get("Needs sunlight",False)),"Earth link required":bool(_task.get("Needs Earth link",False))})
        st.session_state["lunaview_ops_schedule"] = _scheduled
        st.session_state["lunaview_ops_unscheduled"] = _unscheduled
    _saved_ops = st.session_state.get("lunaview_ops_schedule")
    if _saved_ops is not None:
        if _saved_ops:
            _scheduled_df = pd.DataFrame(_saved_ops)
            st.subheader("Proposed Activity Schedule")
            st.dataframe(_scheduled_df, use_container_width=True, hide_index=True)
            st.download_button("Download Mission Operations Schedule", data=_scheduled_df.to_csv(index=False), file_name="lunaview_mission_operations_schedule.csv", mime="text/csv", key="download_ops_schedule")
        else:
            st.info("No activities were scheduled. Check the task durations and available windows.")
        _saved_unscheduled = st.session_state.get("lunaview_ops_unscheduled", [])
        if _saved_unscheduled:
            st.warning("Some activities could not be placed within the available mission windows.")
            st.dataframe(pd.DataFrame(_saved_unscheduled), use_container_width=True, hide_index=True)
else:
    st.info("The operations scheduler is enabled after a current 24-hour mission-window calculation is available.")

st.divider()


st.header("🔋 Power & Battery Survival Simulator")
st.write("Estimate battery state-of-charge through the calculated mission period using the modelled solar-panel output and a constant spacecraft load.")
if results_match and mission_result is not None:
    _battery_load = st.number_input("Average spacecraft load (W)", min_value=0.0, max_value=5000.0, value=60.0, step=5.0, key="battery_load_w")
    _battery_capacity = st.number_input("Usable battery capacity (Wh)", min_value=1.0, max_value=100000.0, value=1500.0, step=50.0, key="battery_capacity_wh")
    _battery_initial_pct = st.slider("Initial battery charge (%)", min_value=0, max_value=100, value=100, key="battery_initial_pct")
    _battery_reserve_pct = st.slider("Minimum reserve (%)", min_value=0, max_value=90, value=20, key="battery_reserve_pct")
    _battery_df = pd.DataFrame(mission_result["records"])
    _battery_df["time"] = pd.to_datetime(_battery_df["utc_time"], utc=True)
    _battery_dt = float(mission_interval) / 60.0
    _soc = float(_battery_capacity) * float(_battery_initial_pct) / 100.0
    _reserve_wh = float(_battery_capacity) * float(_battery_reserve_pct) / 100.0
    _soc_values = []
    _energy_values = []
    for _idx, _rec in _battery_df.iterrows():
        _solar_w = max(0.0, float(_rec.get("panel_power_w", 0.0) or 0.0))
        _net_wh = (_solar_w - float(_battery_load)) * _battery_dt
        _soc = min(float(_battery_capacity), max(0.0, _soc + _net_wh))
        _soc_values.append(_soc)
        _energy_values.append(_net_wh)
    _battery_df["Battery energy (Wh)"] = _soc_values
    _battery_df["Battery charge (%)"] = _battery_df["Battery energy (Wh)"] / float(_battery_capacity) * 100.0
    _battery_min = float(_battery_df["Battery energy (Wh)"].min()) if not _battery_df.empty else 0.0
    _battery_end = float(_battery_df["Battery energy (Wh)"].iloc[-1]) if not _battery_df.empty else 0.0
    _battery_below = bool((_battery_df["Battery energy (Wh)"] < _reserve_wh).any()) if not _battery_df.empty else True
    _bc1, _bc2, _bc3 = st.columns(3)
    _bc1.metric("Ending battery energy", f"{_battery_end:.1f} Wh")
    _bc2.metric("Minimum simulated energy", f"{_battery_min:.1f} Wh")
    _bc3.metric("Reserve status", "BREACHED" if _battery_below else "Maintained")
    _battery_fig = go.Figure(go.Scatter(x=_battery_df["time"], y=_battery_df["Battery charge (%)"], mode="lines", line=dict(color="#22c55e",width=3), fill="tozeroy", name="Battery state of charge"))
    _battery_fig.add_hline(y=float(_battery_reserve_pct), line_dash="dash", line_color="#f97316", annotation_text="Reserve threshold")
    _battery_fig.update_layout(title="Estimated Battery State of Charge", xaxis_title="Time (UTC)", yaxis_title="Charge (%)", yaxis=dict(range=[0,100]), template="plotly_dark", paper_bgcolor="#07111f", plot_bgcolor="#07111f", height=400)
    st.plotly_chart(_battery_fig, use_container_width=True)
    if _battery_below:
        st.error("The simulated battery falls below the selected reserve. Increase storage, reduce loads, or revise the mission schedule.")
    else:
        st.success("The simulated battery remains above the selected reserve for this calculation period.")
    st.caption("Simplified energy-balance model: constant load, sampled solar output, no battery temperature/efficiency/degradation model, no transient peaks, and no independent validation of panel performance.")
else:
    st.info("Calculate the current 24-hour mission windows to enable the battery simulator.")

st.divider()


st.divider()

# ============================================================
# RAPID MISSION DECISION BRIEF
# ============================================================

st.header("🧠 Rapid Mission Decision Brief")
st.write(
    "Translate the current 24-hour site/date analysis into an explainable screening decision. "
    "The thresholds below are user-selected planning gates, not NASA certification limits."
)
_decision_c1, _decision_c2, _decision_c3 = st.columns(3)
with _decision_c1:
    _decision_min_sun = st.slider(
        "Minimum sunlight availability (%)", 0, 100, 40, 5,
        key="decision_min_sunlight_pct",
    )
with _decision_c2:
    _decision_min_earth = st.slider(
        "Minimum Earth communication availability (%)", 0, 100, 25, 5,
        key="decision_min_earth_pct",
    )
with _decision_c3:
    _decision_min_overlap = st.slider(
        "Minimum simultaneous Sun + Earth (%)", 0, 100, 15, 5,
        key="decision_min_overlap_pct",
    )
_decision_max_slope = st.slider(
    "Maximum touchdown slope for this screening gate (degrees)",
    1, 30, 15, 1, key="decision_max_slope_deg",
)

if results_match and mission_result is not None:
    _decision_summary = mission_result["summary"]
    _decision_records = pd.DataFrame(mission_result["records"])
    _decision_overlap_pct = (
        float((
            _decision_records["sun_visible"].fillna(False).astype(bool)
            & _decision_records["communication_visible"].fillna(False).astype(bool)
        ).mean() * 100.0)
        if not _decision_records.empty else None
    )
    _decision_sun_pct = float(_decision_summary.get("sunlight_percentage", 0.0))
    _decision_earth_pct = float(_decision_summary.get("communication_percentage", 0.0))
    _decision_sun_elev = pd.to_numeric(
        _decision_records.get("sun_elevation_deg", pd.Series(dtype=float)), errors="coerce"
    )
    _decision_panel = pd.to_numeric(
        _decision_records.get("panel_power_w", pd.Series(dtype=float)), errors="coerce"
    )
    _decision_checks = [
        {"Requirement": "Sunlight availability", "Result": _decision_sun_pct,
         "Threshold": _decision_min_sun, "Unit": "%",
         "Status": "MEETS GATE" if _decision_sun_pct >= _decision_min_sun else "BELOW GATE"},
        {"Requirement": "Earth communication availability", "Result": _decision_earth_pct,
         "Threshold": _decision_min_earth, "Unit": "%",
         "Status": "MEETS GATE" if _decision_earth_pct >= _decision_min_earth else "BELOW GATE"},
        {"Requirement": "Simultaneous Sun + Earth availability", "Result": _decision_overlap_pct,
         "Threshold": _decision_min_overlap, "Unit": "%",
         "Status": ("NOT AVAILABLE" if _decision_overlap_pct is None else
                    "MEETS GATE" if _decision_overlap_pct >= _decision_min_overlap else "BELOW GATE")},
    ]
    if _center_slope is None:
        _decision_checks.append({
            "Requirement": "Touchdown slope", "Result": None,
            "Threshold": _decision_max_slope, "Unit": "°", "Status": "NOT ASSESSED — terrain data unavailable"
        })
    else:
        _decision_checks.append({
            "Requirement": "Touchdown slope", "Result": float(_center_slope),
            "Threshold": _decision_max_slope, "Unit": "°",
            "Status": "MEETS GATE" if float(_center_slope) <= _decision_max_slope else "BELOW GATE"
        })
    _decision_df = pd.DataFrame(_decision_checks)
    _decision_failures = int((_decision_df["Status"] == "BELOW GATE").sum())
    _decision_unknowns = int(_decision_df["Status"].str.startswith(("NOT AVAILABLE", "NOT ASSESSED")).sum())
    if _decision_failures == 0 and _decision_unknowns == 0:
        _decision_status = "SCREENING GATES MET"
        _decision_explanation = "All selected screening thresholds are met by the current model outputs. This is not a certification of mission feasibility."
    elif _decision_failures == 0:
        _decision_status = "CONDITIONALLY SCREENED"
        _decision_explanation = "No evaluated threshold failed, but one or more required inputs are unavailable. Resolve the missing assessment before comparing this case with fully evaluated cases."
    elif _decision_failures == 1:
        _decision_status = "REVIEW REQUIRED"
        _decision_explanation = "One screening threshold is not met. Consider changing the landing epoch, site, operating schedule, or system assumptions."
    else:
        _decision_status = "CONSTRAINED — REPLAN"
        _decision_explanation = "Multiple screening thresholds are not met. Revisit the site/date combination or mission architecture before treating this scenario as a candidate."

    _dc1, _dc2, _dc3, _dc4 = st.columns(4)
    _dc1.metric("Decision status", _decision_status)
    _dc2.metric("Mean solar elevation", f"{_decision_sun_elev.mean():.2f}°" if _decision_sun_elev.notna().any() else "—")
    _dc3.metric("Peak solar elevation", f"{_decision_sun_elev.max():.2f}°" if _decision_sun_elev.notna().any() else "—")
    _dc4.metric("Peak estimated panel output", f"{_decision_panel.max():.1f} W" if _decision_panel.notna().any() else "—")
    st.write(_decision_explanation)
    st.dataframe(_decision_df, use_container_width=True, hide_index=True)
    _decision_actions = []
    for _check in _decision_checks:
        if _check["Status"] == "BELOW GATE":
            if _check["Requirement"] == "Sunlight availability":
                _decision_actions.append("Test a different landing date/time or a nearby ridge with a more favorable terrain horizon; verify the local illumination model.")
            elif _check["Requirement"] == "Earth communication availability":
                _decision_actions.append("Test a lower antenna-elevation requirement only if hardware permits, or evaluate relay coverage and a full RF link budget.")
            elif _check["Requirement"] == "Simultaneous Sun + Earth availability":
                _decision_actions.append("Reschedule high-power/high-data-rate activities into overlapping sunlight and Earth-visible windows, or evaluate a relay architecture.")
            elif _check["Requirement"] == "Touchdown slope":
                _decision_actions.append("Inspect a flatter touchdown patch and higher-resolution terrain; include lander footprint, roughness, boulders and navigation uncertainty.")
    if _decision_unknowns:
        _decision_actions.append("Resolve missing terrain or geometry inputs; unavailable data must not be interpreted as a pass.")
    if _decision_actions:
        st.markdown("**Recommended review actions**")
        for _action in dict.fromkeys(_decision_actions):
            st.markdown(f"- {_action}")
    _decision_export = _decision_df.copy()
    _decision_export.insert(0, "Site", selected_site)
    _decision_export.insert(1, "Site ID", selected_site_id)
    _decision_export.insert(2, "Observation date UTC", str(observation_date))
    _decision_export.insert(3, "Screening decision", _decision_status)
    st.download_button(
        "Download rapid decision brief (CSV)",
        data=_decision_export.to_csv(index=False),
        file_name=f"{selected_site_id.lower()}_rapid_decision_brief.csv",
        mime="text/csv", key="download_rapid_decision_brief",
    )
    with st.expander("Decision method and interpretation"):
        st.markdown(
            "- Availability percentages are calculated from the current sampled 24-hour mission window.\n"
            "- Simultaneous availability means both the Sun and the direct-to-Earth geometric communication condition are true at the same sample.\n"
            "- Touchdown slope is a local raster estimate at the selected coordinate; it is not a landing-footprint hazard analysis.\n"
            "- Solar elevation and panel output are sampled model outputs. The panel estimate is not a validated power-system prediction.\n"
            "- The decision is a transparent screening gate based on the thresholds you choose; it is not a probability of success, flight rule, or NASA-approved decision."
        )
else:
    st.info("Calculate the current 24-hour mission windows to generate a decision brief for the selected site and date.")

st.divider()

st.header("📄 Mission Assessment Report")
st.write("Create a portable summary of the current site, selected date, data availability, mission windows, and screening assessments.")
_report_lines = [
    "LUNAVIEW — PRELIMINARY LUNAR MISSION ASSESSMENT",
    f"Generated UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
    f"Site: {selected_site} ({selected_site_id})",
    f"Coordinates: latitude {latitude:.5f}°, longitude {longitude:.5f}° east",
    f"Observation date: {observation_date} UTC",
    f"Terrain source: {terrain['source']}",
    f"Terrain resolution: {terrain['resolution']}",
    f"Terrain horizon: {'available' if horizon_result is not None else 'unavailable'}",
]
if results_match and mission_result is not None:
    _rs = mission_result["summary"]
    _report_lines.extend([
        f"Sunlight: {_rs['sunlight_hours']:.2f} h ({_rs['sunlight_percentage']:.1f}%)",
        f"Earth communications: {_rs['communication_hours']:.2f} h ({_rs['communication_percentage']:.1f}%)",
        f"Estimated solar energy: {_rs['estimated_energy_wh']:.2f} Wh",
    ])
else:
    _report_lines.append("Mission windows: not calculated for the current settings")
_report_lines.extend([
    "",
    "LIMITATIONS",
    "This is a preliminary screening tool, not a certified landing-site safety or mission-design system.",
    "Solar output is a simplified modelled panel-power estimate, not measured irradiance or a validated energy-yield prediction. Communications are geometric and do not include a full RF link budget.",
    "Terrain results depend on the selected LOLA raster resolution and coverage.",
    "PSR mapping requires an externally supplied, georeferenced PSR dataset; PSR does not prove accessible water ice.",
    "Rover routing is an experimental slope-aware path over the local raster patch only.",
    "All thresholds and operational schedules require mission-team validation.",
])
_report_text = "\n".join(_report_lines)
st.download_button("Download Mission Assessment Report (TXT)", data=_report_text, file_name=f"{selected_site_id.lower()}_mission_assessment.txt", mime="text/plain", key="download_mission_assessment_report")

with st.expander("Data sources, assumptions & model limitations"):
    st.markdown("""
    - **Terrain:** LOLA elevation/slope rasters configured in `data/terrain/`; the app displays the source and nominal resolution when available.
    - **Sun/Earth geometry:** LunaView's installed calculation modules; validate ephemeris settings and epoch coverage for operational use.
    - **Terrain horizon:** derived from the selected elevation raster; coarse global products may miss small local obstacles.
    - **Landing safety:** local slope screening only; does not assess boulders, roughness, lander footprint, plume effects, navigation uncertainty, or certified vehicle constraints.
    - **PSR:** requires a georeferenced PSR mask upload. Shadowing is not equivalent to confirmed water-ice concentration or accessibility.
    - **Power:** estimated panel output and a constant-load battery model; no thermal, degradation, charge-efficiency, or peak-load model.
    - **Communications:** geometric visibility only; no antenna pattern, occultation by spacecraft structures, RF link budget, or data-rate model.
    - **Rover route:** A* over the local DEM patch with a slope threshold; no wheel-soil interaction, obstacle imagery, localization uncertainty, or detailed rover dynamics.
    - **Mission scheduler:** greedy time-window allocation using sampled sunlight/communications flags; not a full resource-constrained project scheduler.
    """)



# ============================================================
# NASA-ALIGNMENT UPGRADE: COMBINED SITE + DATE MATRIX
# ============================================================

st.divider()
st.header("🧭 Integrated Landing Site & Date Decision Matrix")
st.write(
    "Compare multiple candidate sites across multiple landing dates in "
    "one calculation. Select 2–4 sites and 2–4 landing dates (up to 16 "
    "site/date cases). Each site uses its own available terrain horizon; "
    "each date is evaluated independently."
)
st.info(
    "This matrix is designed for rapid mission screening. It reports "
    "Sun visibility, direct-to-Earth geometric communication availability, "
    "simultaneous Sun + Earth availability, and estimated panel energy. "
    "It does not replace a validated trajectory, RF link budget, or lander safety review."
)

_matrix_site_defaults = [
    name for name in sites
    if site_details[name]["Site ID"] in {"SITE01", "SITE04"}
]
if len(_matrix_site_defaults) < 2:
    _matrix_site_defaults = list(sites.keys())[:2]

_matrix_sites = st.multiselect(
    "Select candidate sites (2–4)",
    options=list(sites.keys()),
    default=_matrix_site_defaults[:2],
    max_selections=4,
    key="integrated_matrix_sites",
)
_matrix_date_cols = st.columns(4)
with _matrix_date_cols[0]:
    _matrix_date_a = st.date_input(
        "Landing date A (UTC)", value=observation_date,
        key="integrated_matrix_date_a",
    )
with _matrix_date_cols[1]:
    _matrix_date_b = st.date_input(
        "Landing date B (UTC)", value=observation_date + timedelta(days=7),
        key="integrated_matrix_date_b",
    )
with _matrix_date_cols[2]:
    _matrix_date_c = st.date_input(
        "Landing date C (UTC)", value=observation_date + timedelta(days=14),
        key="integrated_matrix_date_c",
    )
with _matrix_date_cols[3]:
    _matrix_date_d = st.date_input(
        "Landing date D (UTC)", value=observation_date + timedelta(days=21),
        key="integrated_matrix_date_d",
    )
_matrix_use_c, _matrix_use_d = st.columns(2)
with _matrix_use_c:
    _matrix_include_c = st.checkbox(
        "Include landing date C", value=True, key="integrated_matrix_include_c"
    )
with _matrix_use_d:
    _matrix_include_d = st.checkbox(
        "Include landing date D", value=True, key="integrated_matrix_include_d"
    )
_matrix_dates = [_matrix_date_a, _matrix_date_b]
if _matrix_include_c:
    _matrix_dates.append(_matrix_date_c)
if _matrix_include_d:
    _matrix_dates.append(_matrix_date_d)
_matrix_interval = st.selectbox(
    "Matrix sampling interval",
    options=[30, 60],
    index=1,
    format_func=lambda value: f"Every {value} minutes",
    key="integrated_matrix_interval",
)
_matrix_valid = len(_matrix_sites) >= 2 and len(set(_matrix_dates)) == len(_matrix_dates)
if len(_matrix_sites) < 2:
    st.caption("Choose at least two sites to enable the matrix.")
elif len(set(_matrix_dates)) != len(_matrix_dates):
    st.caption("Choose different dates; duplicate dates cannot be compared.")

def _matrix_transition_hours(records_frame, state_column, start_time):
    """Return first sampled state transition in hours from a scenario start."""
    if records_frame.empty or state_column not in records_frame.columns:
        return None, "Not available"
    _states = records_frame[state_column].fillna(False).astype(bool).tolist()
    if not _states:
        return None, "Not available"
    _event_label = (
        "Sunset" if state_column == "sun_visible" and _states[0]
        else "Next sunrise" if state_column == "sun_visible"
        else "Earthset" if state_column == "earth_disk_visible" and _states[0]
        else "Next Earthrise" if state_column == "earth_disk_visible"
        else "DTE window loss" if state_column == "communication_visible" and _states[0]
        else "Next DTE acquisition"
    )
    if "utc_time" not in records_frame.columns:
        return None, _event_label
    _times = pd.to_datetime(records_frame["utc_time"], utc=True, errors="coerce")
    for _idx in range(1, len(_states)):
        if _states[_idx] != _states[0] and pd.notna(_times.iloc[_idx]):
            _hours = (_times.iloc[_idx] - pd.Timestamp(start_time)).total_seconds() / 3600.0
            return max(0.0, float(_hours)), _event_label
    return None, _event_label


_matrix_signature = (
    tuple((site_details[n]["Site ID"], float(sites[n][0]), float(sites[n][1])) for n in _matrix_sites),
    tuple(str(d) for d in _matrix_dates), int(_matrix_interval),
    float(panel_power), float(antenna_limit), float(panel_tilt), float(panel_azimuth),
)
if st.button(
    "Build Integrated Site–Date Matrix",
    type="primary",
    use_container_width=True,
    disabled=not _matrix_valid,
    key="run_integrated_site_date_matrix",
):
    _matrix_rows = []
    _matrix_errors = []
    _matrix_progress = st.progress(0)
    _matrix_status = st.empty()
    _matrix_total = len(_matrix_sites) * len(_matrix_dates)
    _matrix_done = 0
    _matrix_horizons = {}
    for _site_name in _matrix_sites:
        _site_id = site_details[_site_name]["Site ID"]
        _site_lat, _site_lon = sites[_site_name]
        try:
            _site_horizon, _site_terrain = calculate_site_horizon(_site_id, _site_lat, _site_lon)
            _matrix_horizons[_site_name] = (_site_horizon, _site_terrain)
        except Exception as _horizon_exc:
            _matrix_horizons[_site_name] = (None, {"source": "Terrain calculation failed", "resolution": "Unavailable", "horizon_quality": "Unavailable"})
            _matrix_errors.append({"Site": _site_name, "Date": "—", "Error": f"Terrain horizon: {_horizon_exc}"})

        for _landing_date in _matrix_dates:
            _matrix_status.info(f"Calculating {_site_name} for {_landing_date}…")
            try:
                _start = datetime.combine(_landing_date, datetime.min.time(), tzinfo=timezone.utc)
                _end = _start + timedelta(days=1)
                _horizon_for_site, _terrain_for_site = _matrix_horizons[_site_name]
                _result = calculate_mission_windows(
                    latitude_deg=float(_site_lat), longitude_deg=float(_site_lon),
                    start_utc=_start, end_utc=_end,
                    interval_minutes=int(_matrix_interval),
                    horizon_result=_horizon_for_site,
                    antenna_min_elevation_deg=float(antenna_limit),
                    panel_rated_power_w=float(panel_power),
                    panel_tilt_deg=float(panel_tilt),
                    panel_azimuth_deg=float(panel_azimuth),
                )
                _records = pd.DataFrame(_result["records"])
                if _records.empty:
                    raise ValueError("No time samples were returned by the mission-window calculation.")
                _sun_flags = _records["sun_visible"].fillna(False).astype(bool)
                _earth_flags = _records["communication_visible"].fillna(False).astype(bool)
                _both_flags = _sun_flags & _earth_flags
                _hours_per_sample = float(_matrix_interval) / 60.0
                _summary = _result["summary"]
                _solar_elevation = pd.to_numeric(
                    _records.get("sun_elevation_deg", pd.Series(dtype=float)),
                    errors="coerce",
                )
                _panel_output = pd.to_numeric(
                    _records.get("panel_power_w", pd.Series(dtype=float)),
                    errors="coerce",
                )
                _sun_event_h, _sun_event_label = _matrix_transition_hours(
                    _records, "sun_visible", _start
                )
                _earth_event_h, _earth_event_label = _matrix_transition_hours(
                    _records, "earth_disk_visible", _start
                )
                _dte_event_h, _dte_event_label = _matrix_transition_hours(
                    _records, "communication_visible", _start
                )
                _matrix_rows.append({
                    "Site": _site_name,
                    "Site ID": _site_id,
                    "Landing date (UTC)": str(_landing_date),
                    "Sunlight (h/24h)": float(_summary.get("sunlight_hours", 0.0)),
                    "Sunlight (%)": float(_summary.get("sunlight_percentage", 0.0)),
                    "Earth comms (h/24h)": float(_summary.get("communication_hours", 0.0)),
                    "Earth comms (%)": float(_summary.get("communication_percentage", 0.0)),
                    "Sun + Earth overlap (h/24h)": float(_both_flags.sum() * _hours_per_sample),
                    "Sun + Earth overlap (%)": float(_both_flags.mean() * 100.0),
                    "Estimated energy (Wh)": float(_summary.get("estimated_energy_wh", 0.0)),
                    "Mean solar elevation (°)": float(_solar_elevation.mean()) if _solar_elevation.notna().any() else np.nan,
                    "Peak solar elevation (°)": float(_solar_elevation.max()) if _solar_elevation.notna().any() else np.nan,
                    "Minimum solar elevation (°)": float(_solar_elevation.min()) if _solar_elevation.notna().any() else np.nan,
                    "Peak panel output (W)": float(_panel_output.max()) if _panel_output.notna().any() else np.nan,
                    "Next Sun event": _sun_event_label,
                    "Sun event from start (h)": _sun_event_h,
                    "Next Earth event": _earth_event_label,
                    "Earth event from start (h)": _earth_event_h,
                    "Next DTE event": _dte_event_label,
                    "DTE event from start (h)": _dte_event_h,
                    "Terrain source": _terrain_for_site.get("source", "Unknown"),
                    "Terrain resolution": _terrain_for_site.get("resolution", "Unknown"),
                    "Terrain horizon used": _horizon_for_site is not None,
                })
            except Exception as _matrix_exc:
                _matrix_errors.append({"Site": _site_name, "Date": str(_landing_date), "Error": str(_matrix_exc)})
            _matrix_done += 1
            _matrix_progress.progress(_matrix_done / _matrix_total)
    _matrix_status.empty()
    _matrix_progress.empty()
    st.session_state["integrated_matrix_rows"] = _matrix_rows
    st.session_state["integrated_matrix_errors"] = _matrix_errors
    st.session_state["integrated_matrix_signature"] = _matrix_signature
    if _matrix_rows:
        st.success(f"Integrated comparison completed: {len(_matrix_rows)} site/date combinations.")
    else:
        st.error("No site/date combinations could be calculated. Review the errors below.")

_saved_matrix_rows = st.session_state.get("integrated_matrix_rows", [])
_saved_matrix_errors = st.session_state.get("integrated_matrix_errors", [])
if _saved_matrix_rows and st.session_state.get("integrated_matrix_signature") == _matrix_signature:
    _matrix_df = pd.DataFrame(_saved_matrix_rows)
    st.subheader("Integrated mission comparison results")
    st.dataframe(_matrix_df, use_container_width=True, hide_index=True)
    _heatmap_metric = st.selectbox(
        "Heatmap metric",
        options=["Sunlight (%)", "Earth comms (%)", "Sun + Earth overlap (%)"],
        index=2,
        key="integrated_matrix_heatmap_metric",
    )
    _heatmap_pivot = _matrix_df.pivot(index="Site", columns="Landing date (UTC)", values=_heatmap_metric)
    _heatmap = go.Figure(data=go.Heatmap(
        z=_heatmap_pivot.values,
        x=[str(v) for v in _heatmap_pivot.columns],
        y=list(_heatmap_pivot.index),
        colorscale="Viridis",
        colorbar=dict(title="Availability (%)"),
        zmin=0, zmax=100,
        hovertemplate="Site: %{y}<br>Date: %{x}<br>Availability: %{z:.1f}%<extra></extra>",
    ))
    _heatmap.update_layout(
        title=f"{_heatmap_metric} by site and landing date",
        xaxis_title="Landing date (UTC)", yaxis_title="Candidate site",
        template="plotly_dark", paper_bgcolor="#07111f", plot_bgcolor="#07111f",
        font=dict(color="#E2E8F0"), height=max(360, 90 * len(_heatmap_pivot.index)),
    )
    st.plotly_chart(_heatmap, use_container_width=True)
    st.download_button(
        "Download integrated site–date matrix CSV",
        data=_matrix_df.to_csv(index=False),
        file_name="lunaview_integrated_site_date_matrix.csv",
        mime="text/csv",
        key="download_integrated_site_date_matrix",
    )
    st.caption(
        "Availability and event transitions are sampled at the selected interval; event times are first-transition estimates "
        "from the start of each 24-hour case, not high-precision ephemeris predictions. Solar elevation statistics are "
        "sampled centre-Sun elevations. Rows with no terrain horizon use the calculation module's fallback behavior."
    )
elif _saved_matrix_rows:
    st.info("Matrix settings changed. Run the integrated comparison again to refresh these results.")
if _saved_matrix_errors and st.session_state.get("integrated_matrix_signature") == _matrix_signature:
    with st.expander("View integrated matrix calculation errors"):
        st.dataframe(pd.DataFrame(_saved_matrix_errors), use_container_width=True, hide_index=True)


# ============================================================
# NASA-ALIGNMENT UPGRADE: SCENARIO SENSITIVITY ANALYSIS
# ============================================================

st.divider()
st.header("🧪 Mission Scenario Sensitivity & Robustness")
st.write(
    "Test whether the selected mission remains viable when planning assumptions change. "
    "The baseline uses the selected coordinates and time, then compares a later landing date, "
    "a shifted start time, a stricter antenna elevation threshold, and a 1 km northward site shift."
)
st.warning(
    "Sensitivity results are comparative screening outputs, not probabilities or certified tolerances. "
    "The 1 km location case recalculates the local terrain horizon when suitable terrain data are available."
)
_sens_col1, _sens_col2 = st.columns(2)
with _sens_col1:
    _sens_interval = st.selectbox(
        "Sensitivity sampling interval",
        options=[30, 60],
        index=1,
        format_func=lambda value: f"Every {value} minutes",
        key="sensitivity_interval",
    )
with _sens_col2:
    _sens_antenna_delta = st.slider(
        "Antenna threshold increase for stress test (degrees)",
        min_value=1, max_value=10, value=2,
        key="sensitivity_antenna_delta",
    )
_sens_start = utc_datetime
_sens_end = _sens_start + timedelta(days=1)
_sens_signature = (
    selected_site_id, float(latitude), float(longitude), _sens_start.isoformat(),
    int(_sens_interval), float(panel_power), float(antenna_limit),
    float(panel_tilt), float(panel_azimuth), int(_sens_antenna_delta),
)
if st.button(
    "Run Mission Sensitivity Analysis",
    type="primary",
    use_container_width=True,
    key="run_mission_sensitivity",
):
    _sens_rows = []
    _sens_errors = []
    _sens_progress = st.progress(0)
    _sens_status = st.empty()
    def _spherical_offset_km(lat_deg, lon_deg, distance_km, bearing_deg):
        """Move a point along a great-circle bearing on the spherical Moon."""
        angular = float(distance_km) * 1000.0 / MOON_RADIUS_M
        bearing = math.radians(float(bearing_deg))
        lat1 = math.radians(float(lat_deg))
        lon1 = math.radians(float(lon_deg))
        lat2 = math.asin(
            math.sin(lat1) * math.cos(angular)
            + math.cos(lat1) * math.sin(angular) * math.cos(bearing)
        )
        lon2 = lon1 + math.atan2(
            math.sin(bearing) * math.sin(angular) * math.cos(lat1),
            math.cos(angular) - math.sin(lat1) * math.sin(lat2),
        )
        lon_out = (math.degrees(lon2) + 180.0) % 360.0 - 180.0
        return math.degrees(lat2), lon_out

    _sens_scenarios = [
        {"name": "Baseline", "lat": float(latitude), "lon": float(longitude), "start": _sens_start, "antenna": float(antenna_limit), "use_current_horizon": True},
        {"name": "+1 day landing date", "lat": float(latitude), "lon": float(longitude), "start": _sens_start + timedelta(days=1), "antenna": float(antenna_limit), "use_current_horizon": True},
        {"name": "+14.75 days (half synodic month)", "lat": float(latitude), "lon": float(longitude), "start": _sens_start + timedelta(days=14.75), "antenna": float(antenna_limit), "use_current_horizon": True},
        {"name": "+29.5 days (one synodic month)", "lat": float(latitude), "lon": float(longitude), "start": _sens_start + timedelta(days=29.5), "antenna": float(antenna_limit), "use_current_horizon": True},
        {"name": "+1 hour start-time shift", "lat": float(latitude), "lon": float(longitude), "start": _sens_start + timedelta(hours=1), "antenna": float(antenna_limit), "use_current_horizon": True},
        {"name": f"Antenna threshold +{_sens_antenna_delta}°", "lat": float(latitude), "lon": float(longitude), "start": _sens_start, "antenna": min(90.0, float(antenna_limit) + float(_sens_antenna_delta)), "use_current_horizon": True},
    ]
    for _bearing, _direction in [(0, "north"), (90, "east"), (180, "south"), (270, "west")]:
        _offset_lat, _offset_lon = _spherical_offset_km(latitude, longitude, 1.0, _bearing)
        _sens_scenarios.append({
            "name": f"+1 km {_direction}ward location shift",
            "lat": _offset_lat, "lon": _offset_lon, "start": _sens_start,
            "antenna": float(antenna_limit), "use_current_horizon": False,
        })
    for _sens_index, _scenario in enumerate(_sens_scenarios):
        _sens_status.info(f"Evaluating scenario {_sens_index + 1} of {len(_sens_scenarios)}: {_scenario['name']}")
        try:
            if _scenario["use_current_horizon"]:
                _scenario_horizon = horizon_result
                _scenario_terrain = terrain
            else:
                _scenario_horizon, _scenario_terrain = calculate_site_horizon(
                    selected_site_id, _scenario["lat"], _scenario["lon"]
                )
            _scenario_end = _scenario["start"] + timedelta(days=1)
            _scenario_result = calculate_mission_windows(
                latitude_deg=_scenario["lat"], longitude_deg=_scenario["lon"],
                start_utc=_scenario["start"], end_utc=_scenario_end,
                interval_minutes=int(_sens_interval),
                horizon_result=_scenario_horizon,
                antenna_min_elevation_deg=float(_scenario["antenna"]),
                panel_rated_power_w=float(panel_power),
                panel_tilt_deg=float(panel_tilt), panel_azimuth_deg=float(panel_azimuth),
            )
            _scenario_summary = _scenario_result["summary"]
            _scenario_records = pd.DataFrame(_scenario_result["records"])
            _scenario_solar_el = pd.to_numeric(
                _scenario_records.get("sun_elevation_deg", pd.Series(dtype=float)),
                errors="coerce",
            )
            _scenario_panel_w = pd.to_numeric(
                _scenario_records.get("panel_power_w", pd.Series(dtype=float)),
                errors="coerce",
            )
            _scenario_both = (
                _scenario_records["sun_visible"].fillna(False).astype(bool)
                & _scenario_records["communication_visible"].fillna(False).astype(bool)
            )
            _sens_rows.append({
                "Scenario": _scenario["name"],
                "Start (UTC)": _scenario["start"].strftime("%Y-%m-%d %H:%M"),
                "Latitude (°)": _scenario["lat"],
                "Longitude (°E)": _scenario["lon"],
                "Antenna threshold (°)": _scenario["antenna"],
                "Sunlight (h)": float(_scenario_summary.get("sunlight_hours", 0.0)),
                "Earth comms (h)": float(_scenario_summary.get("communication_hours", 0.0)),
                "Sun + Earth overlap (h)": float(_scenario_both.sum() * float(_sens_interval) / 60.0),
                "Estimated energy (Wh)": float(_scenario_summary.get("estimated_energy_wh", 0.0)),
                "Minimum solar elevation (°)": float(_scenario_solar_el.min()) if _scenario_solar_el.notna().any() else np.nan,
                "Mean solar elevation (°)": float(_scenario_solar_el.mean()) if _scenario_solar_el.notna().any() else np.nan,
                "Peak solar elevation (°)": float(_scenario_solar_el.max()) if _scenario_solar_el.notna().any() else np.nan,
                "Peak panel output (W)": float(_scenario_panel_w.max()) if _scenario_panel_w.notna().any() else np.nan,
                "Terrain horizon used": _scenario_horizon is not None,
                "Terrain source": _scenario_terrain.get("source", "Unknown"),
            })
        except Exception as _sens_exc:
            _sens_errors.append({"Scenario": _scenario["name"], "Error": str(_sens_exc)})
        _sens_progress.progress((_sens_index + 1) / len(_sens_scenarios))
    _sens_status.empty()
    _sens_progress.empty()
    st.session_state["mission_sensitivity_rows"] = _sens_rows
    st.session_state["mission_sensitivity_errors"] = _sens_errors
    st.session_state["mission_sensitivity_signature"] = _sens_signature
    if _sens_rows:
        st.success(f"Sensitivity analysis completed for {len(_sens_rows)} scenario(s).")

_saved_sens_rows = st.session_state.get("mission_sensitivity_rows", [])
_saved_sens_errors = st.session_state.get("mission_sensitivity_errors", [])
if _saved_sens_rows and st.session_state.get("mission_sensitivity_signature") == _sens_signature:
    _sens_df = pd.DataFrame(_saved_sens_rows)
    _sens_baseline = _sens_df[_sens_df["Scenario"] == "Baseline"]
    if not _sens_baseline.empty:
        _sens_base = _sens_baseline.iloc[0]
        for _metric_name in ["Sunlight (h)", "Earth comms (h)", "Sun + Earth overlap (h)", "Estimated energy (Wh)"]:
            _sens_df[f"Change in {_metric_name}"] = _sens_df[_metric_name] - float(_sens_base[_metric_name])
    st.subheader("Scenario comparison and change from baseline")
    st.dataframe(_sens_df, use_container_width=True, hide_index=True)
    _sens_chart = go.Figure()
    for _sens_metric, _sens_label in [
        ("Sunlight (h)", "Sunlight"),
        ("Earth comms (h)", "Earth communications"),
        ("Sun + Earth overlap (h)", "Simultaneous availability"),
    ]:
        _sens_chart.add_trace(go.Bar(name=_sens_label, x=_sens_df["Scenario"], y=_sens_df[_sens_metric]))
    _sens_chart.update_layout(
        title="How planning changes affect available operating time",
        barmode="group", xaxis_title="Scenario", yaxis_title="Hours in 24-hour window",
        template="plotly_dark", paper_bgcolor="#07111f", plot_bgcolor="#07111f",
        font=dict(color="#E2E8F0"), height=440,
    )
    st.plotly_chart(_sens_chart, use_container_width=True)
    _sens_solar_chart = go.Figure()
    _sens_solar_chart.add_trace(go.Bar(
        x=_sens_df["Scenario"],
        y=_sens_df["Mean solar elevation (°)"],
        name="Mean solar elevation",
        text=_sens_df["Mean solar elevation (°)"].map(
            lambda value: f"{value:.1f}°" if pd.notna(value) else "—"
        ),
        textposition="outside",
    ))
    _sens_solar_chart.update_layout(
        title="Solar elevation response to small mission changes",
        xaxis_title="Scenario",
        yaxis_title="Mean centre-Sun elevation (°)",
        template="plotly_dark", paper_bgcolor="#07111f", plot_bgcolor="#07111f",
        font=dict(color="#E2E8F0"), height=390,
    )
    st.plotly_chart(_sens_solar_chart, use_container_width=True)
    st.download_button(
        "Download mission sensitivity CSV",
        data=_sens_df.to_csv(index=False),
        file_name=f"{selected_site_id.lower()}_mission_sensitivity.csv",
        mime="text/csv",
        key="download_mission_sensitivity",
    )
    st.caption(
        "The four location perturbations are 1 km great-circle offsets north, east, south and west on a spherical Moon; "
        "they are not surveyed landing ellipses or full horizontal position uncertainty. Date cases include +1 day, "
        "half a synodic month and one synodic month. Each scenario is a separate 24-hour window; the +1 hour case "
        "shifts both window endpoints. The 29.5-day comparison is a scheduling sensitivity case, not a complete "
        "18.6-year lunar libration or seasonal analysis."
    )
if _saved_sens_errors and st.session_state.get("mission_sensitivity_signature") == _sens_signature:
    with st.expander("View scenario calculation errors"):
        st.dataframe(pd.DataFrame(_saved_sens_errors), use_container_width=True, hide_index=True)


# ============================================================
# NASA-ALIGNMENT UPGRADE: MODEL TRACEABILITY & DATA QUALITY
# ============================================================

st.divider()
st.header("🔬 Scientific Data Quality, Traceability & Limitations")
st.write(
    "Mission planners need to know not only the predicted conditions, but also "
    "which data and assumptions produced them. This panel summarizes the current "
    "run's data coverage and the limits that affect confidence."
)
_quality_rows = [
    {"Analysis element": "Landing-site coordinates", "Current source/status": selected_details.get("Coordinate Reference", "Not specified"), "What to verify": "Confirm the selenographic coordinate frame, longitude convention, epoch, and source for each candidate."},
    {"Analysis element": "Terrain elevation and slope", "Current source/status": terrain.get("source", "Unknown") + " | " + terrain.get("resolution", "Unknown"), "What to verify": "Check raster provenance, horizontal/vertical reference systems, pixel scale, coverage, and NoData gaps."},
    {"Analysis element": "Local terrain horizon", "Current source/status": "Available" if horizon_result is not None else "Unavailable; geometric fallback may apply", "What to verify": "Higher-resolution local topography is needed to resolve small ridges, boulders, and nearby obstructions."},
    {"Analysis element": "Sun and Earth geometry", "Current source/status": "LunaView calculation modules", "What to verify": "Validate against a documented JPL Horizons or NAIF SPICE configuration, including ephemeris/kernel version and time scale."},
    {"Analysis element": "Earth communications", "Current source/status": "Geometric line-of-sight threshold", "What to verify": "Add antenna pattern, RF link budget, relay availability, data rate, pointing and spacecraft/terrain masking for operational use."},
    {"Analysis element": "Solar power", "Current source/status": "Sampled panel-output estimate using selected panel geometry and rated power", "What to verify": "Validate incidence-angle response, solar irradiance, temperature, degradation, dust, conversion efficiency, wiring/storage losses and realistic load profiles."},
    {"Analysis element": "PSR and water ice", "Current source/status": "Requires a user-supplied georeferenced PSR mask", "What to verify": "Use authoritative PSR products and distinguish persistent shadow from confirmed water-ice abundance and accessibility."},
]
_quality_df = pd.DataFrame(_quality_rows)
st.dataframe(_quality_df, use_container_width=True, hide_index=True)
st.markdown(
    "**Important validation target:** compare Sun illumination and Earth visibility against NASA polar illumination products "
    "and/or independently configured SPICE/Horizons calculations before interpreting the output as operationally predictive. "
    "NASA research shows that polar illumination and Earth visibility vary with terrain, date, and lunar libration; "
    "a single static site score cannot capture all of these effects."
)
