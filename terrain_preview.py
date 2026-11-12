
import streamlit as st
import rasterio
import numpy as np
import plotly.graph_objects as go
from rasterio.enums import Resampling
from rasterio.transform import xy, Affine

st.set_page_config(
    page_title="LunaView Terrain Preview",
    layout="wide"
)

st.title("LunaView — Lunar Terrain Preview")
st.write("NASA LOLA elevation data: Connecting Ridge (Site01)")

terrain_path = "data/terrain/Site01_elevation.tif"

try:
    with rasterio.open(terrain_path) as src:

        # Reduce resolution for faster interactive display
        output_height = 500
        output_width = 500

        elevation = src.read(
            1,
            out_shape=(output_height, output_width),
            resampling=Resampling.average
        )

        # Calculate map coordinates for the reduced image
        small_transform = src.transform * Affine.scale(
            src.width / output_width,
            src.height / output_height
        )

        rows = np.arange(output_height)
        cols = np.arange(output_width)

        x = np.array([
            xy(small_transform, 0, col)[0]
            for col in cols
        ])

        y = np.array([
            xy(small_transform, row, 0)[1]
            for row in rows
        ])

        elevation = np.where(
            np.isfinite(elevation),
            elevation,
            np.nan
        )

        minimum = float(np.nanmin(elevation))
        maximum = float(np.nanmax(elevation))
        mean = float(np.nanmean(elevation))

        col1, col2, col3 = st.columns(3)

        col1.metric("Minimum elevation", f"{minimum:.1f} m")
        col2.metric("Maximum elevation", f"{maximum:.1f} m")
        col3.metric("Mean elevation", f"{mean:.1f} m")

        figure = go.Figure(
            data=go.Heatmap(
                z=elevation,
                x=x,
                y=y,
                colorscale="Earth",
                colorbar=dict(
                    title="Elevation (m)"
                ),
                hovertemplate=(
                    "Easting: %{x:.1f} m<br>"
                    "Northing: %{y:.1f} m<br>"
                    "Elevation: %{z:.1f} m"
                    "<extra></extra>"
                )
            )
        )

        figure.update_layout(
            title="Connecting Ridge — LOLA Elevation Map",
            xaxis_title="Lunar polar stereographic X (m)",
            yaxis_title="Lunar polar stereographic Y (m)",
            height=750
        )

        figure.update_yaxes(autorange="reversed")
        figure.update_yaxes(scaleanchor="x", scaleratio=1)

        st.plotly_chart(figure, use_container_width=True)

        st.info(
            "This is a terrain elevation preview. "
            "It does not yet calculate local shadows or "
            "terrain-obstructed Sun visibility."
        )

except FileNotFoundError:
    st.error(f"Terrain file not found: {terrain_path}")

except Exception as error:
    st.error(f"Could not display terrain: {error}")
