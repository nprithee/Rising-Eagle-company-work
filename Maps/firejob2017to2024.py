
#%%
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time

# Replace the link with the raw GitHub link to your CSV file
github_link = "https://raw.githubusercontent.com/nprithee/Rising-Eagle-company-work/main/Map_data/Firejob2017to2024.csv"

# Read the data into a DataFrame
try:
    df = pd.read_csv(github_link)
except pd.errors.ParserError as e:
    print("Error parsing CSV file:", e)

# Initialize the map
m = folium.Map(location=[38.9072, -77.0369], zoom_start=8)  # Washington DC coordinates

# Create marker cluster
marker_cluster = MarkerCluster().add_to(m)

# Function to geocode with retries and delay
def geocode_with_retry(geolocator, address, max_retries=2):
    retries = 0
    while retries < max_retries:
        try:
            location = geolocator.geocode(address, timeout=10)  # Increased timeout
            return location
        except GeocoderTimedOut:
            retries += 1
            time.sleep(1)  # Adding a delay between retries
    return None

# Define marker colors for each year
year_colors = {
    2017: 'green',
    2018: 'blue',
    2019: 'red',
    2020: 'orange',
    2021: 'purple',
    2022: 'black',
    2023: 'gray',
    2024: 'darkred'
}

# Add markers for each job
for _, row in df.iterrows():
    popup_html = f"<b>Job Number:</b> {row['Job Number']}<br>"
    popup_html += f"<b>Property Type:</b> {row['Property Type']}<br>"
    popup_html += f"<b>Sale Amount:</b> {row['Sale Amount']}<br>"
    popup_html += f"<b>Call Date:</b> {row['Call Date']}"

    # Use geopy to get coordinates for the loss city
    geolocator = Nominatim(user_agent="job_map")
    
    # Try including state name along with the city name
    location = geocode_with_retry(geolocator, f"{row['Loss City']}, Virginia, USA")
    
    # If not found in Virginia, try Maryland
    if not location:
        location = geocode_with_retry(geolocator, f"{row['Loss City']}, Maryland, USA")
    
    # If not found in Maryland, try Washington DC
    if not location:
        location = geocode_with_retry(geolocator, f"{row['Loss City']}, Washington DC, USA")
    
    if location:
        lat, lon = location.latitude, location.longitude
        year = int(row['Call Date'])  # Extract year directly from the 'Call Date' column
        color = year_colors.get(year, 'gray')  # Default to gray if year is not defined
        folium.Marker(location=[lat, lon], popup=popup_html, icon=folium.Icon(color=color)).add_to(marker_cluster)

# Save map to HTML
m.save("Firejob_map2017to2024.html")

# %%
