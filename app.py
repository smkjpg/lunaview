
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timezone

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

st.title("🌕 LunaView")
st.subheader("Lunar South Pole Mission Planning System")

st.write(
    "Explore lunar observation conditions, solar geometry, "
    "Earth visibility and local terrain horizons."
)


# ==================================================
# CANDIDATE LANDING SITES
# ==================================================

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

st.write("Observation time:", utc_string)

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

    comparison_data.append({
        "Site": site_name,
        "Latitude": site_lat,
        "Longitude": site_lon,
    })

comparison_df = pd.DataFrame(comparison_data)

st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True,
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
    title="Schematic Lunar South Pole Site Comparison",
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

st.caption(
    "Schematic comparison only. Candidate coordinates "
    "require independent scientific verification."
)


st.divider()


# ==================================================
# TERRAIN ELEVATION MAP
# ==================================================

st.header("🗺️ LOLA Terrain Elevation")

TERRAIN_PATH = "data/terrain/Site01_elevation.tif"


def extract_terrain_arrays(terrain_data):
    """
    Accept the existing terrain loader's dictionary or
    tuple return format and extract x, y and elevation.
    """

    if isinstance(terrain_data, dict):

        x = terrain_data["x"]
        y = terrain_data["y"]
        elevation = terrain_data["elevation"]

    elif isinstance(terrain_data, tuple):

        if len(terrain_data) < 3:
            raise ValueError(
                "Terrain loader returned fewer than three items."
            )

        # The existing loader may return:
        # x, y, elevation, metadata
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

        x, y, elevation = extract_terrain_arrays(
            terrain_data
        )

        fig_terrain = go.Figure(
            data=go.Heatmap(
                x=x,
                y=y,
                z=elevation,
                colorscale="Viridis",
                colorbar=dict(
                    title="Elevation (m)"
                ),
            )
        )

        fig_terrain.update_layout(
            title="Connecting Ridge LOLA Elevation Model",
            xaxis_title="Projected X (m)",
            yaxis_title="Projected Y (m)",
            height=650,
        )

        st.plotly_chart(
            fig_terrain,
            use_container_width=True,
        )

        st.metric(
            "Terrain Minimum",
            f"{np.nanmin(elevation):.2f} m",
        )

        st.metric(
            "Terrain Maximum",
            f"{np.nanmax(elevation):.2f} m",
        )

    except Exception as error:
        st.error(f"Terrain loading failed: {error}")

else:

    st.info(
        "Terrain elevation data is currently available only "
        "for Connecting Ridge (Site01)."
    )


st.divider()


# ==================================================
# TERRAIN HORIZON ANALYSIS
# ==================================================

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

            if comparison["status"] == "ABOVE TERRAIN HORIZON":

                st.success(
                    "RESULT: SUN IS ABOVE THE SAMPLED TERRAIN HORIZON"
                )

            else:

                st.warning(
                    "RESULT: SUN IS BELOW THE SAMPLED TERRAIN HORIZON"
                )

            st.caption(
                "Nearest sampled horizon azimuth: "
                f"{comparison['horizon_azimuth_deg']:.1f}°"
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

