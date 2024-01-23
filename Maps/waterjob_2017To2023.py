#%%
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

# Replace the link with your GitHub link
github_link = "https://raw.githubusercontent.com/Rising-Eagle-Construction-LLC/Data/edd385b9f63f99fe3c1a70077d3a178161484a5c/WATERjob_report2017-2023.csv"
df = pd.read_csv(github_link)

# Initialize geocoder
geolocator = Nominatim(user_agent="job_map")

# Add latitude and longitude columns to the DataFrame
df['Latitude'] = None
df['Longitude'] = None

# Geocode using both 'Loss City' and 'County'
for index, row in df.iterrows():
    try:
        location = geolocator.geocode(f"{row['Loss City']}, {row['County']}", timeout=10)  # Increase the timeout value as needed
        if location:
            df.at[index, 'Latitude'] = location.latitude
            df.at[index, 'Longitude'] = location.longitude
        else:
            print(f"Geocoding failed for {row['Loss City']}, {row['County']}")
    except (GeocoderTimedOut, GeocoderUnavailable) as e:
        print(f"Geocoding timeout/unavailable for {row['Loss City']}, {row['County']}: {e}")

# Create a folium map centered at a specific location
map_center = [df['Latitude'].mean(), df['Longitude'].mean()]  # Centered at the mean of latitudes and longitudes
mymap = folium.Map(location=map_center, zoom_start=10)

# Create a marker cluster group
marker_cluster = MarkerCluster().add_to(mymap)

# Add markers to the map
for index, row in df.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"Job Name: {row['Job Name']}<br>Call Date: {row['Call Date']}<br>Property Type: {row['Property Type']}<br>Sale Amount: {row['Sale Amount']}",
        icon=None  # You can customize the icon if needed
    ).add_to(marker_cluster)

# Save the map as an HTML file
mymap.save("job_map.html")



# %%
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim

# Replace the link with your GitHub link
github_link = "https://raw.githubusercontent.com/Rising-Eagle-Construction-LLC/Data/edd385b9f63f99fe3c1a70077d3a178161484a5c/WATERjob_report2017-2023.csv"
df = pd.read_csv(github_link)

# Initialize geocoder
geolocator = Nominatim(user_agent="job_map")

# Add latitude and longitude columns to the DataFrame
df['Latitude'] = None
df['Longitude'] = None

# Geocode using both 'Loss City' and 'County'
for index, row in df.iterrows():
    try:
        location = geolocator.geocode(f"{row['Loss City']}, {row['County']}")
        if location:
            df.at[index, 'Latitude'] = location.latitude
            df.at[index, 'Longitude'] = location.longitude
        else:
            print(f"Geocoding failed for {row['Loss City']}, {row['County']}")
    except Exception as e:
        print(f"Error geocoding for {row['Loss City']}, {row['County']}: {e}")

# Create a folium map centered at a specific location
map_center = [df['Latitude'].mean(), df['Longitude'].mean()]  # Centered at the mean of latitudes and longitudes
mymap = folium.Map(location=map_center, zoom_start=10)

# Create a marker cluster group
marker_cluster = MarkerCluster().add_to(mymap)

# Add markers to the map
for index, row in df.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"Job Name: {row['Job Name']}<br>Call Date: {row['Call Date']}<br>Property Type: {row['Property Type']}<br>Sale Amount: {row['Sale Amount']}",
        icon=None  # You can customize the icon if needed
    ).add_to(marker_cluster)

# Save the map as an HTML file
mymap.save("job_map.html")

# %%
