from pystac_client import Client
from odc.stac import load
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
import plotly.express as px
import os

'''
source: https://towardsdatascience.com/create-an-interactive-map-to-display-time-series-of-satellite-imagery-e9346e165e27
'''


def search_satellite_images(collection="sentinel-2-l2a",
                            bbox=[-120.15, 38.93, -119.88, 39.25],
                            date="2023-01-01/2023-03-12",
                            cloud_cover=(0, 10)):
    """
    Search for satellite images based on collection, bounding box, date range, and cloud cover.

    :param collection: Collection name (default: "sentinel-2-l2a").
    :param bbox: Bounding box [min_lon, min_lat, max_lon, max_lat] (default: Lake Tahoe Region).
    :param date: Date range "YYYY-MM-DD/YYYY-MM-DD" (default: "2023-01-01/2023-12-30").
                    Max range: 2015-17-10/present
    :param cloud_cover: Tuple representing cloud cover range (min, max) (default: (0, 10)).
    :return: Data loaded based on search criteria.
    """
    # Define the search client
    client = Client.open("https://earth-search.aws.element84.com/v1")
    search = client.search(collections=[collection],
                           bbox=bbox,
                           datetime=date,
                           query=[f"eo:cloud_cover<{cloud_cover[1]}", f"eo:cloud_cover>{cloud_cover[0]}"])

    # Print the number of matched items
    print(f"Number of images found: {search.matched()}")

    data = load(search.items(), bbox=bbox, groupby="solar_day", chunks={})

    print(f"Number of days in data: {len(data.time)}")

    return data


def count_water_pixels(data, lake_id):
    """
    Counts water pixels from Sentinel-2 SCL data for each time step.

    :param data: xarray Dataset with Sentinel-2 SCL data.
    :return: DataFrame with dates, water counts, and snow counts.
    """
    water_counts = []
    date_labels = []
    water_area = []
    coverage_ratio = []
    # Determine the number of time steps
    numb_days = len(data.time)

    # Iterate through each time step
    for t in range(numb_days):
        scl_image = data[["scl"]].isel(time=t).to_array()
        dt = pd.to_datetime(scl_image.time.values)
        year = dt.year
        month = dt.month
        day = dt.day

        date_string = f"{year}-{month:02d}-{day:02d}"
        print(date_string)

        '''
        Count the number of pixels corresponding to water
        To change this, use the following classifications:
        0: No data
        1: Saturated or defective
        2: Dark area pixels
        3: Cloud shadows
        4: Vegetation
        5: Bare soils
        6: Water
        7: Unclassified
        8: Cloud medium probability
        9: Cloud high probability
        10: Thin cirrus
        11: Snow or ice
        source: https://docs.digitalearthafrica.org/en/latest/data_specs/Sentinel-2_Level-2A_specs.html
        '''
        count_water = np.count_nonzero(scl_image == 6)  # Water

        surface_area = count_water*10*10/(10**6)

        count_pixels = np.count_nonzero((scl_image == 1) | (scl_image == 2) |
                                        (scl_image == 3) | (scl_image == 4) |
                                        (scl_image == 5) | (scl_image == 6) |
                                        (scl_image == 7) | (scl_image == 8) |
                                        (scl_image == 9) | (scl_image == 10) |
                                        (scl_image == 11))
        total_pixels = data.sizes['y']*data.sizes['x']

        coverage = count_pixels*10*10/1e6
        lake_area = total_pixels*10*10/1e6

        ratio = coverage/lake_area

        if ratio < 0.8:
            continue

        water_counts.append(count_water)
        date_labels.append(date_string)
        water_area.append(surface_area)
        coverage_ratio.append(ratio)

    # Convert date labels to pandas datetime format
    datetime_index = pd.to_datetime(date_labels)

    # Create a dictionary for constructing the DataFrame
    data_dict = {
        'Date': datetime_index,
        'ID': lake_id,
        'Water Counts': water_counts,
        'Pixel Counts': count_pixels,
        'Total Pixels': total_pixels,
        'Coverage Ratio': coverage_ratio,
        'Water Surface Area': water_area
    }

    # Create the DataFrame
    df = pd.DataFrame(data_dict)

    return df


def get_centroids_and_bboxes(shapefile_path):
    """
    Processes a shapefile to return a DataFrame containing the ID, centroid,
    and bounding box (bbox) of each polygon.

    :param shapefile_path: Path to the shapefile.
    :return: A pandas DataFrame with the ID, centroid, and bbox of each polygon.
    """

    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Reproject to EPSG:4326
    gdf_proj = gdf.to_crs("EPSG:4326")

    centroids = []
    bboxes = []

    # Process each polygon to get its centroid and bbox
    for index, row in gdf_proj.iterrows():
        # Centroid
        centroid_lat = row.geometry.centroid.y
        centroid_lon = row.geometry.centroid.x
        centroids.append((centroid_lat, centroid_lon))

        # Bounding Box
        minx, miny, maxx, maxy = row.geometry.bounds
        bbox = (minx, miny, maxx, maxy)
        bboxes.append(bbox)

    # Create the DataFrame
    df = pd.DataFrame({
        'ID': gdf_proj.index,
        'Centroid_Lat': [lat for lat, lon in centroids],
        'Centroid_Lon': [lon for lat, lon in centroids],
        'BBox_Min_Lon': [bbox[0] for bbox in bboxes],
        'BBox_Min_Lat': [bbox[1] for bbox in bboxes],
        'BBox_Max_Lon': [bbox[2] for bbox in bboxes],
        'BBox_Max_Lat': [bbox[3] for bbox in bboxes]
    })

    return df


def plot_timeseries_for_spot(spot_id, ts_df):
    df_spot = ts_df[ts_df['ID'] == spot_id]
    print(df_spot)
    fig = px.line(df_spot, x='Date', y='Water Surface Area', title=f'Time Series for Lake {spot_id}')

    # Add X and Y axis labels
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Water Surface Area (sq km)"
    )

    filepath = f'tmp_{int(spot_id)}.html'
    fig.write_html(filepath, include_plotlyjs='cdn')
    return filepath


dir_path = os.path.dirname(os.path.realpath(__file__))
shapefile_path = os.path.join(dir_path, "shp", "lake_boundingboxes.shp")
lakes_df = get_centroids_and_bboxes(shapefile_path)
print(lakes_df)

all_water_pixels_dfs = []

centroid_lats = []
centroid_longs = []

for lake_id in lakes_df.ID:
    print(lake_id)
    lake_df = lakes_df[lakes_df['ID'] == lake_id]
    centroid_lats.append(lake_df.iloc[0].Centroid_Lat)
    centroid_longs.append(lake_df.iloc[0].Centroid_Lon)

    if not lake_df.empty:
        bbox = [lake_df.iloc[0].BBox_Min_Lon, lake_df.iloc[0].BBox_Min_Lat,
                lake_df.iloc[0].BBox_Max_Lon, lake_df.iloc[0].BBox_Max_Lat]

        data = search_satellite_images(collection="sentinel-2-l2a",
                                       date="2024-01-01/2024-05-14",
                                       cloud_cover=(0, 5),
                                       bbox=bbox)
        # Pass the lake_id
        water_pixels_df = count_water_pixels(data, lake_id)

        # Append
        all_water_pixels_dfs.append(water_pixels_df)

zoom_lat = (max(centroid_lats) + min(centroid_lats)) / 2
zoom_long = (max(centroid_longs) + min(centroid_longs)) / 2

# Concatenate all DataFrames into a single DataFrame
final_df = pd.concat(all_water_pixels_dfs, ignore_index=True)

# Create a map TODO: set location as parameter
m = folium.Map(location=[zoom_lat, zoom_long], zoom_start=7)

# Add markers with Plotly time series popups
for index, row in lakes_df.iterrows():
    html_path = plot_timeseries_for_spot(row['ID'], final_df)
    iframe = folium.IFrame(html=open(html_path).read(), width=500, height=300)
    popup = folium.Popup(iframe, max_width=2650)
    folium.Marker([row['Centroid_Lat'], row['Centroid_Lon']], popup=popup).add_to(m)

m.save(os.path.join(dir_path, 'output', 'map_with_timeseries.html'))

# Clean up temporary HTML files
for spot_id in lakes_df['ID']:
    os.remove(f'tmp_{spot_id}.html')
