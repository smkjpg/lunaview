import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timezone
from pathlib import Path

from calculations.sun_position import calculate_sun_position
from calculations.earth_position import calculate_earth_position
from calculations.terrain import load_terrain
from calculations.terrain_horizon import (
    calculate_terrain_horizon,
    compare_sun_with_terrain,
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="LunaView | Lunar Mission Planner",
    page_icon="🌕",
    layout="wide",
)
st.markdown("""
<style>
/* Main background */
.stApp {
    background: #07111f;
    color: #f1f5f9;
}

/* General text */
.stApp, .stApp p, .stApp label,
.stApp li, .stApp span {
    color: #e2e8f0;
}

/* Headings */
.stMarkdown h1 {
    color: #7dd3fc !important;
}

.stMarkdown h2,
.stMarkdown h3 {
    color: #38bdf8 !important;
}

/* Metric cards */
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

/* Captions and dividers */
.stCaption {
    color: #cbd5e1 !important;
}

hr {
    border-color: #38516e;
}

/* Input controls */
.stApp input,
.stApp textarea,
.stApp select,
.stApp [data-baseweb="input"] input {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    background-color: #13243a !important;
    border: 1px solid #526b88 !important;
}

.stApp input::placeholder,
.stApp textarea::placeholder {
    color: #bdc9d8 !important;
}

.stApp [data-baseweb="select"] > div,
[data-baseweb="menu"] {
    background-color: #13243a !important;
}

.stApp [data-baseweb="select"] *,
[data-baseweb="menu"] * {
    color: #ffffff !important;
}

/* Date and time controls are styled separately in the sidebar below. */

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #101f32 !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #f1f5f9 !important;
}

/* Date (UTC) and Time (UTC): force dark text in the actual input fields */
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
[data-testid="stSidebar"] [data-testid="stTimeInput"] input,
[data-testid="stSidebar"] [data-testid="stDateInput"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-testid="stTimeInput"] [data-baseweb="input"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
    border: 1px solid #8aa2bd !important;
    opacity: 1 !important;
    caret-color: #172033 !important;
}

[data-testid="stSidebar"] [data-testid="stDateInput"] button,
[data-testid="stSidebar"] [data-testid="stTimeInput"] button {
    color: #172033 !important;
}

[data-testid="stSidebar"] [data-testid="stDateInput"] svg,
[data-testid="stSidebar"] [data-testid="stTimeInput"] svg {
    color: #172033 !important;
    fill: #172033 !important;
}

/* White information bars: every word is black */
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

/* Download Terrain Horizon CSV button: dark readable text */
.stApp [data-testid="stDownloadButton"] button,
.stApp [data-testid="stDownloadButton"] button *,
.stApp [data-testid="stDownloadButton"] a,
.stApp [data-testid="stDownloadButton"] a * {
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
}

.stApp [data-testid="stDownloadButton"] button,
.stApp [data-testid="stDownloadButton"] a {
    background-color: #ffffff !important;
    border: 1px solid #8aa2bd !important;
}

/* Keep expanders in the dark mission-control style */
.stApp [data-testid="stExpander"] {
    background-color: #13243a !important;
    border-color: #526b88 !important;
}

.stApp [data-testid="stExpander"] * {
    color: #e2e8f0;
}

</style>
""", unsafe_allow_html=True)

st.title("🌕 LunaView")
st.subheader("Lunar South Pole Mission Planning System")

st.markdown(
    """
    **MISSION CONTROL / OBSERVATION PLANNER**

    Evaluate solar geometry, Earth visibility and local terrain
    conditions for candidate lunar south-pole landing sites.
    """
)


# ==================================================
# CANDIDATE AND HISTORICAL LANDING SITES
# ==================================================

# The original eight south-pole candidate sites are always available.
sites = {
    "Connecting Ridge (Site01)": (-89.4632, -137.49),
    "Shackleton Rim (Site04)": (-89.7668, -171.87),
    "Nobile Rim 1 (Site06)": (-85.4381, 37.36667),
    "Peak Near Shackleton (Site07)": (-88.811, 123.6901),
    "de Gerlache Rim 1 (Site11)": (-88.6834, -67.9321),
    "Leibnitz Beta Plateau (Site20)": (-85.4266, 31.74276),
    "Malapert Massif (Site23)": (-85.9948, -0.23578),
    "de Gerlache-Kocher Massif (Site42)": (-85.8296, -116.322),
}

# Extra site information is retained for the comparison table.
site_details = {
    name: {
        "Site ID": name.split("(")[-1].replace(")", "") if "(" in name else "",
        "Mission": "LunaView candidate",
        "Agency": "LunaView",
        "Site Type": "Candidate",
        "Status": "Proposed",
        "Coordinate Reference": "LunaView candidate coordinates; verify frame before precision use",
        "Source": "",
    }
    for name in sites
}

# Load the additional landing sites from the project CSV database.
# Put lunar_sites.csv in data/sites/ in the repository.
SITE_DATABASE_PATH = Path(__file__).resolve().parent / "data" / "sites" / "lunar_sites.csv"

if SITE_DATABASE_PATH.exists():
    try:
        site_database = pd.read_csv(SITE_DATABASE_PATH)
        required_columns = {
            "site_id", "site_name", "mission", "agency", "site_type",
            "status", "latitude_deg", "longitude_deg_east",
        }
        missing_columns = required_columns.difference(site_database.columns)
        if missing_columns:
            st.warning(
                "The lunar site database is missing these columns: "
                + ", ".join(sorted(missing_columns))
            )
        else:
            for _, row in site_database.iterrows():
                site_id = str(row["site_id"]).strip()

                # Connecting Ridge is already in the original candidate list.
                if site_id.upper() == "SITE01":
                    continue

                try:
                    site_lat = float(row["latitude_deg"])
                    site_lon = float(row["longitude_deg_east"])
                except (TypeError, ValueError):
                    continue

                if not np.isfinite(site_lat) or not np.isfinite(site_lon):
                    continue
                if not -90 <= site_lat <= 90 or not -180 <= site_lon <= 180:
                    continue

                site_name = str(row["site_name"]).strip()
                mission = str(row["mission"]).strip()
                display_name = f"{site_name} — {mission}"

                # Avoid duplicate labels if the CSV contains repeated records.
                if display_name in sites:
                    continue

                sites[display_name] = (site_lat, site_lon)
                site_details[display_name] = {
                    "Site ID": site_id,
                    "Mission": mission,
                    "Agency": str(row.get("agency", "")),
                    "Site Type": str(row.get("site_type", "")),
                    "Status": str(row.get("status", "")),
                    "Coordinate Reference": str(row.get("coordinate_reference", "")),
                    "Source": str(row.get("source_url", "")),
                }

    except Exception as error:
        st.warning(f"Additional lunar site database could not be loaded: {error}")
else:
    st.warning(
        "Additional landing sites were not loaded. Make sure "
        "data/sites/lunar_sites.csv is included in the project."
    )


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("🚀 Mission Configuration")

selected_site = st.sidebar.selectbox(
    "Select landing site",
    list(sites.keys()),
)

default_lat, default_lon = sites[selected_site]

latitude = st.sidebar.number_input(
    "Latitude (degrees)",
    value=float(default_lat),
    min_value=-90.0,
    max_value=90.0,
    format="%.5f",
)

longitude = st.sidebar.number_input(
    "Longitude (degrees)",
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


# ==================================================
# MISSION OVERVIEW
# ==================================================

st.header("Mission Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Selected Site", selected_site)

with col2:
    st.metric("Latitude", f"{latitude:.5f}°")

with col3:
    st.metric("Longitude", f"{longitude:.5f}°")

st.caption(f"OBSERVATION EPOCH (UTC): {utc_string}")

st.divider()


# ==================================================
# SUN POSITION
# ==================================================

st.header("☀️ Solar Position Analysis")

sun = None

try:
    sun = calculate_sun_position(
        latitude,
        longitude,
        utc_string,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Sun Azimuth",
            f"{sun['sun_azimuth_deg']:.2f}°",
        )

    with col2:
        st.metric(
            "Sun Elevation",
            f"{sun['sun_elevation_deg']:.2f}°",
        )

    with col3:
        st.metric(
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


# ==================================================
# EARTH POSITION
# ==================================================

st.header("🌍 Earth Position Analysis")

try:
    earth = calculate_earth_position(
        latitude,
        longitude,
        utc_string,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Earth Azimuth",
            f"{earth['earth_azimuth_deg']:.2f}°",
        )

    with col2:
        st.metric(
            "Earth Elevation",
            f"{earth['earth_elevation_deg']:.2f}°",
        )

    with col3:
        st.metric(
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
        st.info("The full Earth disk is geometrically visible.")
    else:
        st.info(
            "The full Earth disk is not geometrically visible."
        )

except Exception as error:
    st.error(f"Earth calculation failed: {error}")


st.divider()


# ==================================================
# CANDIDATE SITE COMPARISON
# ==================================================

st.header("📍 Candidate Site Comparison")

comparison_data = []

for site_name, coordinates in sites.items():
    site_lat, site_lon = coordinates
    details = site_details.get(site_name, {})

    comparison_data.append({
        "Site": site_name,
        "Site ID": details.get("Site ID", ""),
        "Mission": details.get("Mission", ""),
        "Agency": details.get("Agency", ""),
        "Site Type": details.get("Site Type", ""),
        "Status": details.get("Status", ""),
        "Latitude": site_lat,
        "Longitude": site_lon,
        "Coordinate Reference": details.get("Coordinate Reference", ""),
    })

comparison_df = pd.DataFrame(comparison_data)

st.caption(
    f"{len(sites)} locations loaded: 8 original south-pole candidates "
    "plus additional sites from the lunar site database, when available. "
    "This is a location catalogue, not a landing-site safety ranking."
)

st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Coordinate references may differ between source datasets. "
    "The plotted positions are schematic and should not be treated as "
    "a precision geodetic comparison until all coordinates are normalized "
    "to the same lunar reference frame."
)

fig_sites = go.Figure()

fig_sites.add_trace(
    go.Scatterpolar(
        r=[90 + item["Latitude"] for item in comparison_data],
        theta=[item["Longitude"] for item in comparison_data],
        mode="markers+text",
        text=[
            item["Site"].split("(")[0].strip()
            for item in comparison_data
        ],
        textposition="top center",
        marker=dict(size=10),
        name="Candidate sites",
    )
)

fig_sites.update_layout(
    title=dict(
        text="Schematic Lunar South Pole Site Comparison",
        font=dict(color="#7dd3fc", size=22),
    ),
    template="plotly_dark",
    paper_bgcolor="#07111f",
    plot_bgcolor="#07111f",
    font=dict(color="#e2e8f0"),
    polar=dict(
        angularaxis=dict(
            direction="clockwise",
            rotation=90,
        )
    ),
    height=550,
)

st.plotly_chart(
    fig_sites,
    use_container_width=True,
)


st.divider()


# ==================================================
# TERRAIN ELEVATION MAP
# ==================================================

st.header("🗺️ LOLA Terrain Elevation")

TERRAIN_PATH = "data/terrain/Site01_elevation.tif"


def extract_terrain_arrays(terrain_data):
    if isinstance(terrain_data, dict):
        x = terrain_data["x"]
        y = terrain_data["y"]
        elevation = terrain_data["elevation"]

    elif isinstance(terrain_data, tuple):
        if len(terrain_data) < 3:
            raise ValueError("Terrain loader returned fewer than three items.")
        x, y, elevation = terrain_data[:3]

    else:
        raise TypeError(
            f"Unexpected terrain data type: {type(terrain_data)}"
        )

    elevation = np.ma.filled(
        np.ma.asarray(elevation, dtype=float),
        np.nan,
    )

    return np.asarray(x), np.asarray(y), elevation


if selected_site == "Connecting Ridge (Site01)":

    try:
        terrain_data = load_terrain(TERRAIN_PATH)

        x, y, elevation = extract_terrain_arrays(terrain_data)

        # 2D TERRAIN MAP
        fig_terrain = go.Figure(
            data=go.Heatmap(
                x=x,
                y=y,
                z=elevation,
                colorscale="Viridis",
                colorbar=dict(title="Elevation (m)"),
            )
        )

        fig_terrain.update_layout(
            title="Connecting Ridge LOLA Elevation Model",
            template="plotly_dark",
            xaxis_title="Projected X (m)",
            yaxis_title="Projected Y (m)",
            height=600,
        )

        st.plotly_chart(fig_terrain, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Terrain Minimum",
                f"{np.nanmin(elevation):.2f} m",
            )

        with col2:
            st.metric(
                "Terrain Maximum",
                f"{np.nanmax(elevation):.2f} m",
            )

        # INTERACTIVE 3D TERRAIN
        st.subheader("Interactive 3D Lunar Terrain")

        # Reduce display resolution for smoother interaction
        step = max(1, int(max(elevation.shape) / 150))

        x_3d = x[::step]
        y_3d = y[::step]
        z_3d = elevation[::step, ::step]

        fig_3d = go.Figure(
            data=[
                go.Surface(
                    x=x_3d,
                    y=y_3d,
                    z=z_3d,
                    colorscale="Viridis",
                    colorbar=dict(title="Elevation (m)"),
                    connectgaps=False,
                )
            ]
        )

        fig_3d.update_layout(
            template="plotly_dark",
            height=650,
            scene=dict(
                xaxis_title="Projected X (m)",
                yaxis_title="Projected Y (m)",
                zaxis_title="Elevation (m)",
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=0.25),
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        )

        st.plotly_chart(fig_3d, use_container_width=True)

    except Exception as error:
        st.error(f"Terrain loading failed: {error}")


st.header("🏔️ 360° Terrain Horizon Analysis")

st.write(
    "Calculate the surrounding terrain horizon and compare "
    "it with the Sun's position."
)

if selected_site == "Connecting Ridge (Site01)":

    try:

        with st.spinner(
            "Calculating terrain horizon in 360 directions..."
        ):

            horizon_result = calculate_terrain_horizon(
                terrain_path=TERRAIN_PATH,
                latitude=latitude,
                longitude=longitude,
                max_distance_m=5000,
                distance_step_m=10,
                azimuth_step_deg=1,
            )

        st.success(
            "360° terrain horizon calculation completed."
        )

        st.metric(
            "Observer Elevation",
            f"{horizon_result['observer_elevation_m']:.2f} m",
        )

        horizon_df = pd.DataFrame(
            horizon_result["horizon_profile"]
        )

        # --------------------------------------------------
        # TERRAIN HORIZON GRAPH
        # --------------------------------------------------

        fig_horizon = go.Figure()

        fig_horizon.add_trace(
            go.Scatter(
                x=horizon_df["azimuth_deg"],
                y=horizon_df["horizon_elevation_deg"],
                mode="lines",
                name="Terrain horizon",
                line=dict(
                    color="orange",
                    width=2,
                ),
                fill="tozeroy",
            )
        )

        # --------------------------------------------------
        # SUN VISIBILITY COMPARISON
        # --------------------------------------------------

        if sun is not None:

            sun_azimuth = sun["sun_azimuth_deg"]
            sun_elevation = sun["sun_elevation_deg"]

            comparison = compare_sun_with_terrain(
                horizon_result,
                sun_azimuth,
                sun_elevation,
            )

            fig_horizon.add_vline(
                x=sun_azimuth,
                line_dash="dash",
                line_color="yellow",
                annotation_text="Sun azimuth",
            )

            fig_horizon.add_hline(
                y=sun_elevation,
                line_dash="dot",
                line_color="cyan",
                annotation_text="Sun elevation",
            )

        fig_horizon.update_layout(
            title="Terrain Horizon Across 360°",
            template="plotly_dark",
            paper_bgcolor="#07111f",
            plot_bgcolor="#07111f",
            font=dict(color="#e2e8f0"),
            xaxis_title="Azimuth (degrees)",
            yaxis_title="Horizon Elevation (degrees)",
            height=550,
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

        if sun is not None:

            st.subheader("☀️ Sun Visibility Result")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Sun Azimuth",
                    f"{sun_azimuth:.2f}°",
                )

            with col2:
                st.metric(
                    "Terrain Horizon",
                    f"{comparison['terrain_horizon_deg']:.2f}°",
                )

            with col3:
                st.metric(
                    "Angular Clearance",
                    f"{comparison['angular_clearance_deg']:.2f}°",
                )

            comparison_status = comparison.get("status")
            if comparison_status is None:
                comparison_status = (
                    "ABOVE TERRAIN HORIZON"
                    if comparison["angular_clearance_deg"] > 0
                    else "BELOW TERRAIN HORIZON"
                )

            if comparison_status == "ABOVE TERRAIN HORIZON":
                st.success(
                    "RESULT: SUN IS ABOVE THE SAMPLED TERRAIN HORIZON"
                )
            else:
                st.warning(
                    "RESULT: SUN IS BELOW THE SAMPLED TERRAIN HORIZON"
                )

            horizon_azimuth = comparison.get(
                "horizon_azimuth_deg",
                sun_azimuth,
            )

            st.caption(
                "Horizon comparison azimuth: "
                f"{horizon_azimuth:.1f}°"
            )

        else:

            st.warning(
                "Sun calculation unavailable. "
                "Showing terrain horizon without Sun comparison."
            )

        # --------------------------------------------------
        # COMPLETE HORIZON DATA
        # --------------------------------------------------

        with st.expander("View complete 360° horizon data"):

            st.dataframe(
                horizon_df,
                use_container_width=True,
                hide_index=True,
            )

        st.download_button(
            "Download terrain horizon CSV",
            data=horizon_df.to_csv(index=False),
            file_name="lunaview_terrain_horizon.csv",
            mime="text/csv",
        )

    except Exception as error:

        st.error(
            f"Terrain horizon analysis failed: {error}"
        )

else:

    st.info(
        "Terrain horizon analysis currently supports "
        "Connecting Ridge (Site01) only."
    )


st.divider()