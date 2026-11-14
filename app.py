
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
        display = (
            f"{record['site_name']} — {record['mission']}"
        )

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
    """Prefer individual terrain, then polar terrain, then global LOLA."""

    site_id = str(site_id).upper()

    if site_id in HIGH_RES_SITE_IDS:
        file_site_id = site_id[0].upper() + site_id[1:].lower()

        elevation = (
            TERRAIN_DIR / f"{file_site_id}_elevation.tif"
        )

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

        transformer = Transformer.from_crs(
            MOON_GEOGRAPHIC_CRS,
            src.crs,
            always_xy=True,
        )

        if src.crs.is_geographic:
            x_site = float(longitude) % 360.0
            y_site = float(latitude)
        else:
            x_site, y_site = transformer.transform(
                float(longitude),
                float(latitude),
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

        return float(
            max(abs(src.res[0]), abs(src.res[1]))
        )


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
.stApp {
    background: #07111f;
    color: #f1f5f9;
}

.stApp, .stApp p, .stApp label, .stApp li, .stApp span {
    color: #e2e8f0;
}

.stMarkdown h1 {
    color: #7dd3fc !important;
}

.stMarkdown h2, .stMarkdown h3 {
    color: #38bdf8 !important;
}

[data-testid="stMetric"] {
    background: #13243a;
    border: 1px solid #38516e;
    padding: 18px;
    border-radius: 10px;
}

[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

.stCaption {
    color: #cbd5e1 !important;
}

hr {
    border-color: #38516e;
}

.stApp input,
.stApp textarea,
.stApp select,
.stApp [data-baseweb="input"] input {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    background-color: #13243a !important;
    border: 1px solid #526b88 !important;
}

.stApp [data-baseweb="select"] > div,
[data-baseweb="menu"] {
    background-color: #13243a !important;
}

.stApp [data-baseweb="select"] *,
[data-baseweb="menu"] * {
    color: #ffffff !important;
}

[data-testid="stSidebar"] {
    background-color: #101f32 !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #f1f5f9 !important;
}

[data-testid="stSidebar"] [data-testid="stDateInput"],
[data-testid="stSidebar"] [data-testid="stTimeInput"] {
    color-scheme: light !important;
}

[data-testid="stSidebar"] [data-testid="stDateInput"] *,
[data-testid="stSidebar"] [data-testid="stTimeInput"] * {
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
}

[data-testid="stSidebar"] [data-testid="stDateInput"] input,
[data-testid="stSidebar"] [data-testid="stTimeInput"] input {
    background: #ffffff !important;
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
}

.stApp [data-testid="stAlert"],
.stApp [data-testid="stNotification"] {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #cbd5e1 !important;
}

.stApp [data-testid="stAlert"] *,
.stApp [data-testid="stNotification"] * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

.stApp [data-testid="stDownloadButton"] button,
.stApp [data-testid="stDownloadButton"] button * {
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
    background-color: #ffffff !important;
}

.stApp [data-testid="stExpander"] {
    background-color: #13243a !important;
    border-color: #526b88 !important;
}

.stApp [data-testid="stExpander"] * {
    color: #e2e8f0;
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
st.sidebar.caption("LunaView preliminary mission analysis")
st.sidebar.caption("Time standard: UTC")


# ============================================================
# MISSION OVERVIEW
# ============================================================

st.header("Mission Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Selected Site",
        selected_site.split(" — ")[0],
    )

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

    col1.metric(
        "Sun Azimuth",
        f"{sun['sun_azimuth_deg']:.2f}°",
    )

    col2.metric(
        "Sun Elevation",
        f"{sun['sun_elevation_deg']:.2f}°",
    )

    col3.metric(
        "Sun Distance",
        f"{sun['sun_range_km']:,.0f} km",
    )

    if sun["sun_elevation_deg"] > 0:
        st.success(
            "Sun is above the idealized lunar horizon."
        )
    else:
        st.warning(
            "Sun is below the idealized lunar horizon."
        )

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

    col1.metric(
        "Earth Azimuth",
        f"{earth['earth_azimuth_deg']:.2f}°",
    )

    col2.metric(
        "Earth Elevation",
        f"{earth['earth_elevation_deg']:.2f}°",
    )

    col3.metric(
        "Earth Distance",
        f"{earth['earth_distance_km']:,.0f} km",
    )

    st.metric(
        "Earth Angular Radius",
        f"{earth['earth_angular_radius_deg']:.3f}°",
    )

    if earth["earth_disk_visible"]:
        st.success(
            "Earth disk is above the idealized lunar horizon."
        )
    else:
        st.warning(
            "Earth disk is below the idealized lunar horizon."
        )

    if earth["earth_full_disk_visible"]:
        st.info(
            "The full Earth disk is geometrically visible."
        )
    else:
        st.info(
            "The full Earth disk is not geometrically visible."
        )

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
        r=[
            90 + item["Latitude (°)"]
            for item in comparison_data
        ],
        theta=[
            item["Longitude (°)"]
            for item in comparison_data
        ],
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

st.plotly_chart(
    fig_sites,
    use_container_width=True,
)

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

status_cols[0].metric(
    "Terrain source",
    terrain["source"],
)

status_cols[1].metric(
    "Nominal resolution",
    terrain["resolution"],
)

status_cols[2].metric(
    "Site ID",
    selected_site_id,
)

# Shared terrain horizon result for the mission planner.
# It is populated by the 360-degree horizon calculation below.
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
        f"Elevation dataset found: "
        f"{elevation_path.relative_to(ROOT)}"
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

        st.plotly_chart(
            fig_elev,
            use_container_width=True,
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Patch minimum elevation",
            f"{np.nanmin(valid_elev):.1f} m",
        )

        c2.metric(
            "Patch mean elevation",
            f"{np.nanmean(valid_elev):.1f} m",
        )

        c3.metric(
            "Patch maximum elevation",
            f"{np.nanmax(valid_elev):.1f} m",
        )

        # ----------------------------------------------------
        # SLOPE MAP
        # ----------------------------------------------------

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

                    st.plotly_chart(
                        fig_slope,
                        use_container_width=True,
                    )

                    s1, s2, s3 = st.columns(3)

                    s1.metric(
                        "Patch minimum slope",
                        f"{np.nanmin(valid_slope):.2f}°",
                    )

                    s2.metric(
                        "Patch mean slope",
                        f"{np.nanmean(valid_slope):.2f}°",
                    )

                    s3.metric(
                        "Patch maximum slope",
                        f"{np.nanmax(valid_slope):.2f}°",
                    )

                    st.caption(
                        f"Slope raster: {slope_meta['file']} | "
                        f"Valid pixels: "
                        f"{slope_meta['valid_pixels_in_patch']:,}"
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

        # ----------------------------------------------------
        # INTERACTIVE 3D TERRAIN
        # ----------------------------------------------------

        st.subheader("⛰️ Interactive 3D Lunar Terrain")

        step = max(
            1,
            int(max(elevation.shape) / 140),
        )

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

        st.plotly_chart(
            fig_3d,
            use_container_width=True,
        )

    except Exception as terrain_error:
        st.error(
            f"Terrain data could not be sampled for "
            f"{selected_site}: {terrain_error}"
        )

st.divider()


# ============================================================
# 360-DEGREE TERRAIN HORIZON
# ============================================================

st.header("🏔️ 360° Terrain Horizon Analysis")

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

            distance_step = max(
                400.0,
                pixel_spacing,
            )

            st.warning(
                "This site uses the coarse global LOLA DEM. "
                "The horizon is a regional estimate and may "
                "miss small local obstacles."
            )

        elif terrain["horizon_quality"].startswith("Regional"):

            distance_step = max(
                80.0,
                pixel_spacing,
            )

            st.info(
                "Horizon is based on the shared 80 m south-polar "
                "terrain mosaic. Small local obstacles may be missed."
            )

        else:

            distance_step = max(
                10.0,
                pixel_spacing,
            )

            st.info(
                "Horizon uses the site-specific high-resolution terrain tile."
            )

        with st.spinner(
            "Calculating 360° terrain horizon for the selected site..."
        ):

            terrain_longitude = longitude

            if (
                Path(elevation_path).resolve()
                == GLOBAL_ELEVATION.resolve()
            ):
                terrain_longitude = longitude % 360.0

            horizon_result = calculate_terrain_horizon(
                terrain_path=str(elevation_path),
                latitude=latitude,
                longitude=terrain_longitude,
                max_distance_m=5000,
                distance_step_m=float(distance_step),
                azimuth_step_deg=1,
            )

        st.success(
            "360° terrain horizon calculation completed."
        )

        st.metric(
            "Observer elevation",
            f"{horizon_result['observer_elevation_m']:.2f} m",
        )

        horizon_df = pd.DataFrame(
            horizon_result["horizon_profile"]
        )

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
            xaxis=dict(
                range=[0, 359],
                dtick=30,
            ),
            hovermode="x unified",
        )

        st.plotly_chart(
            fig_horizon,
            use_container_width=True,
        )

        if comparison is not None:

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Sun azimuth",
                f"{sun['sun_azimuth_deg']:.2f}°",
            )

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
# MISSION WINDOW PLANNER — ALL 32 LANDING SITES
# ============================================================

st.header("🚀 Mission Window Planner")

st.write(
    "Calculate sunlight duration, solar-panel energy and "
    "direct-to-Earth communication opportunities for "
    "any of the 32 landing sites."
)

st.caption(
    "The planner uses the selected site's coordinates "
    "and its available terrain horizon."
)


# ------------------------------------------------------------
# MISSION CONFIGURATION
# ------------------------------------------------------------

st.subheader("Mission Configuration")

mc1, mc2, mc3 = st.columns(3)

with mc1:
    mission_interval = st.selectbox(
        "Calculation interval",
        [5, 10, 15, 30, 60],
        index=3,
        format_func=lambda value: f"{value} minutes",
        key="mission_interval",
    )

with mc2:
    panel_power = st.number_input(
        "Solar panel rated power (W)",
        min_value=1.0,
        max_value=5000.0,
        value=100.0,
        step=10.0,
        key="mission_panel_power",
    )

with mc3:
    antenna_limit = st.number_input(
        "Minimum antenna elevation (°)",
        min_value=0.0,
        max_value=90.0,
        value=5.0,
        step=1.0,
        key="mission_antenna_limit",
    )

panel_tilt = st.slider(
    "Solar panel tilt from horizontal (°)",
    min_value=0,
    max_value=90,
    value=0,
    key="mission_panel_tilt",
)

panel_azimuth = st.slider(
    "Solar panel facing direction (° clockwise from north)",
    min_value=0,
    max_value=359,
    value=0,
    key="mission_panel_azimuth",
)


# ------------------------------------------------------------
# MISSION PERIOD
# ------------------------------------------------------------

mission_start = datetime(
    observation_date.year,
    observation_date.month,
    observation_date.day,
    0,
    0,
    0,
    tzinfo=timezone.utc,
)

mission_end = mission_start + timedelta(days=1)

st.info(
    f"Selected site: {selected_site} | "
    f"Latitude: {latitude:.5f}° | "
    f"Longitude: {longitude:.5f}° | "
    f"Mission period: {mission_start.date()} UTC "
    f"to {mission_end.date()} UTC"
)


# ------------------------------------------------------------
# TERRAIN QUALITY INFORMATION
# ------------------------------------------------------------

if horizon_result is not None:

    st.success(
        f"Terrain-aware mission calculation available. "
        f"Terrain source: {terrain['source']}."
    )

    if terrain["horizon_quality"].startswith("Coarse global"):
        st.warning(
            "Caution: global LOLA terrain is coarse. "
            "Small ridges and local obstacles may not be represented."
        )

    elif terrain["horizon_quality"].startswith("Regional"):
        st.info(
            "The regional 80 m terrain model may miss obstacles "
            "smaller than its pixel spacing."
        )

else:

    st.warning(
        "No valid terrain horizon is available for this site. "
        "The mission calculation will use a geometric horizon."
    )


# ------------------------------------------------------------
# RUN MISSION CALCULATION
# ------------------------------------------------------------

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

        st.session_state["mission_terrain_used"] = (
            horizon_result is not None
        )

        st.session_state["mission_date"] = str(
            observation_date
        )

        st.session_state["mission_interval_used"] = mission_interval

        st.session_state["mission_panel_power_used"] = panel_power

        st.session_state["mission_antenna_limit_used"] = antenna_limit

        st.success(
            f"Mission calculation completed for {selected_site}."
        )

    except Exception as mission_error:

        st.error(
            f"Mission-window calculation failed: {mission_error}"
        )


# ------------------------------------------------------------
# DISPLAY STORED MISSION RESULTS
# ------------------------------------------------------------

mission_result = st.session_state.get("mission_result")

stored_site_id = st.session_state.get("mission_site_id")

stored_date = st.session_state.get("mission_date")

results_match = (
    mission_result is not None
    and stored_site_id == selected_site_id
    and stored_date == str(observation_date)
)

if mission_result is not None and not results_match:

    st.info(
        "The displayed mission results belong to a different "
        "site or date. Press Calculate 24-hour Mission Windows "
        "to generate results for the current selection."
    )

if results_match:

    summary = mission_result["summary"]

    records = mission_result["records"]

    mission_df = pd.DataFrame(records)

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
        f"Date: {stored_date} UTC | "
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

    # --------------------------------------------------------
    # SUMMARY METRICS
    # --------------------------------------------------------

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

    d.metric(
        "Time samples",
        str(summary["sample_count"]),
    )

    # --------------------------------------------------------
    # SUNLIGHT TIMELINE
    # --------------------------------------------------------

    st.subheader("☀️ Sunlight and Terrain Horizon")

    fig_sun = go.Figure()

    fig_sun.add_trace(
        go.Scatter(
            x=mission_df["time"],
            y=mission_df["sun_elevation_deg"],
            mode="lines",
            name="Sun elevation",
            line=dict(
                color="#facc15",
                width=2,
            ),
        )
    )

    fig_sun.add_trace(
        go.Scatter(
            x=mission_df["time"],
            y=mission_df["sun_terrain_horizon_deg"],
            mode="lines",
            name="Terrain horizon",
            line=dict(
                color="#f97316",
                width=2,
                dash="dash",
            ),
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

    st.plotly_chart(
        fig_sun,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # SOLAR POWER TIMELINE
    # --------------------------------------------------------

    st.subheader("⚡ Solar Panel Power")

    fig_power = go.Figure()

    fig_power.add_trace(
        go.Scatter(
            x=mission_df["time"],
            y=mission_df["panel_power_w"],
            mode="lines",
            name="Solar panel output",
            line=dict(
                color="#22c55e",
                width=2,
            ),
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

    st.plotly_chart(
        fig_power,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # EARTH COMMUNICATION TIMELINE
    # --------------------------------------------------------

    st.subheader("🌍 Earth Communication Visibility")

    fig_comms = go.Figure()

    fig_comms.add_trace(
        go.Scatter(
            x=mission_df["time"],
            y=mission_df["earth_clearance_deg"],
            mode="lines",
            name="Earth clearance",
            line=dict(
                color="#38bdf8",
                width=2,
            ),
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

    st.plotly_chart(
        fig_comms,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # VISIBILITY STATUS TIMELINE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DETAILED MISSION RECORDS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DOWNLOAD RESULTS
    # --------------------------------------------------------

    st.download_button(
        "Download Mission Window Results CSV",
        data=mission_df.to_csv(index=False),
        file_name=(
            f"{selected_site_id.lower()}_mission_windows.csv"
        ),
        mime="text/csv",
        key="download_mission_windows",
    )

    st.caption(
        "Solar energy is a simplified panel-output estimate. "
        "Communication visibility is geometric and does not "
        "include a radio link budget, antenna pointing model, "
        "or data-rate analysis."
    )


# ============================================================
# FINAL DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "LunaView is a preliminary lunar mission planning and "
    "comparison tool. DEM resolution, coordinate reference "
    "frame, data provenance and local coverage must be verified "
    "before operational landing decisions. Global terrain "
    "models may not resolve small local obstacles."
)
