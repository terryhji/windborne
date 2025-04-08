import streamlit as st
import requests
import json
import plotly.graph_objects as go
import math
from groq import Groq
import textwrap
from streamlit_autorefresh import st_autorefresh
import time
import os

st.set_page_config(layout="wide")

st_autorefresh(interval=1800000, key="thirty_minute_refresh")

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    km = 6371 * c
    return km

def fetch_and_plot():
    lat_lon = []
    errors = []

    for hr in range(0, 24):
        hr_str = str(hr).zfill(2)
        url = f"https://a.windbornesystems.com/treasure/{hr_str}.json"
        try:
            data = requests.get(url).json()
            for i, coord in enumerate(data):
                if i < len(lat_lon):
                    lat_lon[i].append(coord[1])
                    lat_lon[i].append(coord[0])
                else:
                    lat_lon.append([coord[1], coord[0]])
        except json.decoder.JSONDecodeError as e:
            if "Expecting value" in str(e):
                errors.append(f"Hour {hr_str}: 404 Not Found")
            if "Extra data" in str(e):
                errors.append(f"Hour {hr_str}: Incorrect JSON Format")
            continue

    balloon_paths = {}
    for i, coords in enumerate(lat_lon):
        lons, lats = [], []
        for j in range(len(coords)):
            if j % 2 == 0:
                lons.append(coords[j])
            else:
                lats.append(coords[j])
        balloon_paths[i] = {"lon": lons, "lat": lats}

    fig = go.Figure()
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    threshold_km = 5000  
    large_movements = {}

    for balloon_id, path in balloon_paths.items():
        fig.add_trace(go.Scattergeo(
            lon=path["lon"],
            lat=path["lat"],
            mode="lines+markers",
            marker=dict(size=4),
            line=dict(width=2, color=colors[balloon_id % len(colors)]),
            name="Balloon " + str(balloon_id)
        ))

        lons = path["lon"]
        lats = path["lat"]

        balloon_movements = []
        for idx in range(1, len(lons)):
            lon1, lat1 = lons[idx-1], lats[idx-1]
            lon2, lat2 = lons[idx], lats[idx]
            distance = haversine(lon1, lat1, lon2, lat2)
            if distance > threshold_km:
                balloon_movements.append({
                    "start_coord": (lon1, lat1),
                    "end_coord": (lon2, lat2),
                    "distance_km": distance
                })
        
        if balloon_movements:
            large_movements[balloon_id] = balloon_movements

    fig.update_geos(
        projection_type="orthographic",
        showcountries=True,
        countrycolor="black"
    )

    fig.update_layout(
        title="Windborne Balloon Trajectories (Past 23 Hours)",
        margin={"r":0, "t":40, "l":0, "b":0}
    )

    if errors:
        error_text = "<br>".join(errors)
        fig.add_annotation(
            text=f"<b>Errors:</b><br>{error_text}",
            align='left',
            showarrow=False,
            xref="paper", yref="paper",
            x=0, y=0,  
            bordercolor="red",
            borderwidth=1,
            bgcolor="white",
            font=dict(size=10, color="red")
        )

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"),)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": """Based on the dataset of balloon movements at the end of the prompt, where some legs record extreme distances 
                (ranging from approximately 6000 km to over 17000 km), what severe or unusual weather conditions might be causing such 
                dramatic shifts in trajectories? Also, could these anomalies indicate a combination of extreme meteorological 
                events (like intense storms or wind shear)? Add location specific causes related to the start or end coordinates. 
                Format this into a paragraph, no bullet points. Max 1000 characters.""" + str(large_movements)
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    prompt_result = chat_completion.choices[0].message.content

    def add_line_breaks(text, width=40):
        return "<br>".join(textwrap.wrap(text, width=width))

    wrapped_prompt = add_line_breaks(prompt_result, width=60)

    # Add as annotation
    fig.add_annotation(
        text=f"<b>Llama 3.3 70b Versatile (Groq):</b><br>{wrapped_prompt}",
        align='left',
        showarrow=False,
        xref="paper", yref="paper",
        x=0, y=1,
        xanchor='left', yanchor='top',
        bordercolor="green",
        borderwidth=1,
        bgcolor="white",
        font=dict(size=10, color="green"),
    )

    return fig

strf = time.strftime("%T", time.gmtime(time.time()))
st.title(f"Windborne Balloon Monitoring Dashboard - Last Updated {strf} UTC")

fig = fetch_and_plot()

st.plotly_chart(fig, use_container_width=True)

