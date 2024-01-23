import pandas as pd
import folium
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim

# Replace the link with the raw GitHub link to your CSV file
github_link = "https://raw.githubusercontent.com/nprithee/Dataset-storage/b673419484717ad94f72edeba2e0b7317f3365d2/Report_job.csv"

# Read the data into a DataFrame
try:
    df = pd.read_csv(github_link)
    print(df.head())  # Check the first few rows to verify if the data is loaded correctly
except pd.errors.ParserError as e:
    print("Error parsing CSV file:", e)


# Initialize geocoder
geolocator = Nominatim(user_agent="job_map")

# Add latitude and longitude columns to the DataFrame
df['Latitude'] = None
df['Longitude'] = None

# Geocode using both 'Loss City' and 'County'
for index, row in df.iterrows():
    location = geolocator.geocode(f"{row['Loss City']}, {row['County']}")
    if location:
        df.at[index, 'Latitude'] = location.latitude
        df.at[index, 'Longitude'] = location.longitude
    else:
        print(f"Geocoding failed for {row['Loss City']}, {row['County']}")

# Create a folium map centered at a specific location
map_center = [df['Latitude'].mean(), df['Longitude'].mean()]  # Centered at the mean of latitudes and longitudes
mymap = folium.Map(location=map_center, zoom_start=10)

# Create a marker cluster group
marker_cluster = MarkerCluster().add_to(mymap)

# Add markers to the map
for index, row in df.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"{row['Job Name']} ({row['Loss City']}, {row['County']})",
        icon=None  # You can customize the icon if needed
    ).add_to(marker_cluster)

# Save the map as an HTML file
mymap.save("job_map.html")

#%%


# Now you can use the DataFrame as usual

# %%
