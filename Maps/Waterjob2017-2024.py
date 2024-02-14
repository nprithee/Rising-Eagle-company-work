
#%%
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time

# Replace the link with the raw GitHub link to your CSV file
github_link = "https://raw.githubusercontent.com/nprithee/Rising-Eagle-company-work/main/Map_data/Waterjob2017-2024.csv"

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
        folium.Marker(location=[lat, lon], popup=popup_html).add_to(marker_cluster)

# Save map to HTML
m.save("waterjob_map.html")



# %%
