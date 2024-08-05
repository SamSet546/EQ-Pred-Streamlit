import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import requests
import time

url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTJdjJGMh3LE80MLupe_JRd9u7b6Wv96ZCKyF9P7UBzeMKdgdZpYRicZA9_f33VK0ar9Pnn09wVi-wV/pub?output=csv'
df = pd.read_csv(url)

chart_data = df.loc[:72116, ['latitude', 'longitude', 'magnitude']]  #72116 = Max Num of data points
chart_data_cut = df.loc[:100, ['latitude', 'longitude', 'magnitude']]

# Convert to DataFrame
df_1 = pd.DataFrame(chart_data)
df_2 = pd.DataFrame(chart_data_cut)

# Shift longitude and latitude values by 1 so that the target values are always the pair after the source 
df_2['source_latitude'] = df_2['latitude']
df_2['source_longitude'] = df_2['longitude']
df_2['target_latitude'] = df_2['latitude'].shift(-1)
df_2['target_longitude'] = df_2['longitude'].shift(-1)

# Remove the last row since it has no target
df_2 = df_2[:-1]

# Add source_color and target_color 
df_2['source_color'] = df_2['source_latitude'].apply(lambda lat: [int((lat + 90) % 255), 140, 0])
df_2['target_color'] = df_2['target_latitude'].apply(lambda lat: [int((lat + 90) % 255), 140, 0])

# Add inbound and outbound with some example values
df_2['inbound'] = np.random.rand(len(df_2)) * 100
df_2['outbound'] = np.random.rand(len(df_2)) * 100

# Add 'icon' column with the value 'marker' for all rows
df_1['icon'] = 'marker'

#Creating a 'color' column for df_1
df_1['color'] = df_1['magnitude'].apply(lambda lat: [int((lat + 90) % 255), 20, 0])

# Add a predefined list of location names for demonstration
predefined_names = {
    (row['latitude'], row['longitude']): f"Location {i+1}"
    for i, row in df_1.iterrows()
}

#Location names to display on the markers
df_1['name'] = df_1.apply(lambda row: predefined_names.get((row['latitude'], row['longitude']), 'Unknown'), axis=1)

st.pydeck_chart(pdk.Deck(
    #mapbox_key='<mapbox-access-token>',
    map_style='mapbox://styles/mapbox/satellite-v9',  # 'None' will return a blank map
    initial_view_state=pdk.ViewState(
        latitude=-17.9408,
        longitude=-178.4997,
        zoom=7.5,
        pitch=50,
    ),
    layers=[
        pdk.Layer(
            'ScatterplotLayer',
            data=df_1,  # Use df_new instead of chart_data
            get_position=['longitude', 'latitude'],
            get_fill_color='source_color',
            get_radius='magnitude',
            radius_scale=10,
            pickable=True
        ),
        pdk.Layer(
            'ArcLayer',
            data=df_2,  # Use df_new instead of chart_data
            get_source_position=['source_longitude', 'source_latitude'],
            get_target_position=['target_longitude', 'target_latitude'],
            get_source_color='source_color',
            get_target_color='target_color',
            width_scale=5,
            get_width='magnitude',
            pickable=True,
            opacity=0.1
        ),
        pdk.Layer(
            'IconLayer',
            data=df_1,
            get_color='color',
            get_icon='icon',
            get_size=35,
            get_position=['longitude', 'latitude'],
            icon_atlas='https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas.png',
            icon_mapping='https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas.json',
            pickable=True,
            opacity=0.5
        ),
        pdk.Layer(
            'HexagonLayer',
            data=df_1,  # Use df_new instead of chart_data
            get_position=['longitude', 'latitude'],
            radius=1000,  # works only with specific value of radius
            get_elevation='magnitude',  # get_elevation should be a single field
            elevation_scale=10,
            elevation_range=[0, 10000],
            pickable=True,
            extruded=True,
            visible=True
        ),
        pdk.Layer(
            'HeatmapLayer',
            data=df_1,
            get_position=['longitude', 'latitude'],
            get_weight=['magnitude'],
            radius_pixels=25,
            aggregation='SUM'
        )
        ],
    tooltip={"html": "<b>Name:</b> {name} <br/> <b>Latitude:</b> {latitude} <br/> <b>Longitude:</b> {longitude} <br/> <b>Magnitude:</b> {magnitude}",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white"}} # Display the name of the location where the markers are 
))

