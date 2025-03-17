import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="UFO Sightings Dashboard",
    page_icon="🛸",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-top: 1rem;
    }
    .stMetric {
        background-color: black;
        padding: 10px;
        border-radius: 5px;
    }
    .footer {
        font-size: 0.8rem;
        color: #888;
        text-align: center;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("scrubbed_updated.csv")

    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    
    df['year'] = df['datetime'].dt.year
    df['decade'] = (df['year'] // 10) * 10
    df["month"] = df["datetime"].dt.month  
    df['duration_seconds'] = pd.to_numeric(df['duration (seconds)'], errors='coerce')
    
    df = df[
        (df['latitude'].notna()) &
        (df['longitude'].notna()) &
        (df['latitude'] != 0) &
        (df['longitude'] != 0)
    ]
    
    duration_bins = [0, 60, 300, 900, 1800, 3600, float('inf')]
    duration_labels = ['< 1 min', '1-5 mins', '5-15 mins', '15-30 mins', '30-60 mins', '> 1 hour']
    df['duration_category'] = pd.cut(df['duration_seconds'], bins=duration_bins, labels=duration_labels)
    
    for col in ['country', 'state', 'shape']:
        if col in df.columns:
            df[col] = df[col].str.lower() if df[col].dtype == 'object' else df[col]
    
    return df

seasons = {
    "Winter": [12, 1, 2],
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Fall": [9, 10, 11]}

def filter_data(df, year_range, shapes, duration_cats, season=None, countries=None):
    filtered = df.copy()
    
    filtered = filtered[
        (filtered['year'] >= year_range[0]) &
        (filtered['year'] <= year_range[1])
    ]
    
    if shapes:
        filtered = filtered[filtered['shape'].isin(shapes)]
        
    if duration_cats:
        filtered = filtered[filtered['duration_category'].isin(duration_cats)]
    
    if season:
        filtered = filtered[filtered['month'].isin(seasons[season])]
    
    if countries:
        filtered = filtered[filtered['country'].isin(countries)]
    
    return filtered

def create_map(df, map_style="mapbox://styles/mapbox/dark-v10"):
    map_df = df[['latitude', 'longitude', 'shape', 'year', 'duration_seconds', 'city', 'state', 'country']].copy()
    map_df['latitude'] = pd.to_numeric(map_df['latitude'], errors='coerce')

    shape_colors = {
        'light': [255, 255, 0],      # Yellow
        'triangle': [0, 255, 0],     # Green
        'circle': [0, 0, 255],       # Blue
        'unknown': [128, 128, 128],  # Gray
        'fireball': [255, 0, 0],     # Red
        'other': [255, 165, 0],      # Orange
        'sphere': [128, 0, 128],     # Purple
        'disk': [0, 255, 255],       # Cyan
        'oval': [255, 192, 203],     # Pink
        'formation': [165, 42, 42],  # Brown
        'cigar': [255, 255, 255],    # White
        'chevron': [255, 0, 255],    # Magenta
        'rectangle': [0, 128, 0],    # Dark Green
        'diamond': [30, 144, 255],   # Dodger Blue
        'teardrop': [219, 112, 147], # Pale Violet Red
        'cylinder': [0, 191, 255],   # Deep Sky Blue
        'changing': [148, 0, 211],   # Dark Violet
        'flash': [255, 215, 0],      # Gold
        'cone': [139, 69, 19],       # Saddle Brown
        'cross': [220, 20, 60]       # Crimson
    }
    
    default_color = [128, 128, 128]  # Gray

    map_df['color'] = map_df['shape'].apply(lambda x: shape_colors.get(x, default_color))
    map_df['size'] = np.clip(np.log1p(map_df['duration_seconds']) / 2, 1, 5)

    map_df['tooltip'] = map_df.apply(
        lambda row: f"Shape: {row['shape']}\n " +
                    f"Year: {row['year']}\n " +
                    f"Location: {row['city']}, {row['state']}, {row['country']}\n " +
                    f"Duration: {row['duration_seconds']} seconds",
        axis=1
    )
    
    view_state = pdk.ViewState(
        latitude=np.mean(map_df['latitude']),
        longitude=np.mean(map_df['longitude']),
        zoom=3,
        pitch=0
    )
    
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=['longitude', 'latitude'],
        get_color='color',
        get_radius='size * 20000', 
        pickable=True,
        opacity=0.5,
        stroked=True,
        filled=True,
        radius_scale=3,
        radius_min_pixels=1,
        radius_max_pixels=3
    )

    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=map_df,
        get_position=['longitude', 'latitude'],
        opacity=0.9,
        aggregation='"SUM"',
        visible=False
    )
    
    deck = pdk.Deck(
        map_style=map_style,
        initial_view_state=view_state,
        layers=[scatter_layer, heatmap_layer],
        tooltip={"text": "{tooltip}"}
    )
    
    return deck

# Main app
def main():
    st.markdown("<h1 class='main-header'>🛸Worldwide UFO Sightings 1906 - 2014🛸</h1>", unsafe_allow_html=True)

    with st.spinner("Loading data..."):
        df = load_data()
    
    total_sightings = len(df)
    year_range = (int(df['year'].min()), int(df['year'].max()))
    all_shapes = sorted(df['shape'].dropna().unique())
    all_duration_cats = sorted(df['duration_category'].dropna().unique())
    all_seasons = list(seasons.keys()) 

    st.sidebar.markdown("## Filters")

    st.sidebar.markdown("### Time Period")
    selected_year_range = st.sidebar.slider(
        "Select Years",
        min_value=year_range[0],
        max_value=year_range[1],
        value=year_range,
        step=1
    )
    
    st.sidebar.markdown("### UFO Shape")
    selected_shapes = st.sidebar.multiselect(
        "Select Shapes",
        options=all_shapes,
        default=None
    )

    st.sidebar.markdown("### Sighting Duration")
    selected_durations = st.sidebar.multiselect(
        "Select Duration",
        options=all_duration_cats,
        default=None
    )

    st.sidebar.markdown("### Season")
    selected_season = st.sidebar.selectbox(
        "Select Season",
        options=["All"] + all_seasons,  
        index=0  
    )
    
    st.sidebar.markdown("### Map Style")
    map_style_options = {
        "Dark": "mapbox://styles/mapbox/dark-v10",
        "Light": "mapbox://styles/mapbox/light-v10",
        "Satellite": "mapbox://styles/mapbox/satellite-v9",
        "Outdoors": "mapbox://styles/mapbox/outdoors-v11"
    }
        
    display_style = st.sidebar.selectbox(
        "Select Map Style",
        options=list(map_style_options.keys()),
        index=0
    )

    map_style = map_style_options[display_style]

    st.sidebar.markdown("### Map Layers")
    show_scatter = st.sidebar.checkbox("Show Scatter Plot", value=True)
    show_heatmap = st.sidebar.checkbox("Show Heatmap", value=False)
    
    filtered_df = filter_data(df, selected_year_range, selected_shapes, selected_durations, selected_season if selected_season != "All" else None)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sightings", f"{len(filtered_df):,}")
    
    with col2:
        if not selected_shapes:
            top_shape = filtered_df['shape'].value_counts().index[0] if len(filtered_df) > 0 else "N/A"
        else:
            top_shape = selected_shapes[0] if selected_shapes else "N/A"
        st.metric("Most Common Shape", top_shape.title() if isinstance(top_shape, str) else "N/A")
    
    with col3:
        avg_duration = filtered_df['duration_seconds'].median()
        st.metric("Median Duration", f"{avg_duration:.0f} sec" if not np.isnan(avg_duration) else "N/A")
    
    with col4:
        top_year = filtered_df['year'].value_counts().index[0] if len(filtered_df) > 0 else "N/A"
        st.metric("Year with Most Sightings", top_year)
    
    tab1 = st.tabs(["Map"])
    
    with tab1[0]:
        st.markdown("<h3 class='sub-header'>UFO Sightings Map</h3>", unsafe_allow_html=True)
        
        if len(filtered_df) > 0:
            deck = create_map(filtered_df, map_style)
            
            deck.layers[0].visible = show_scatter
            deck.layers[1].visible = show_heatmap
            st.pydeck_chart(deck)
        else:
            st.warning("No data available for the selected filters. Please adjust your selection.")
    
    st.markdown("<div class='footer'>Data source: https://www.kaggle.com/datasets/NUFORC/ufo-sightings/data</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()