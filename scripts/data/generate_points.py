import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import random
from dotenv import load_dotenv
from pathlib import Path
from shapely.geometry import box
import mercantile


def generate_tile_centroids(counties_gdf, zoom=16):
    """
    Generate centroids of Web Mercator tiles at a given zoom level that 
    intersect with the bounding box of the input GeoDataFrame.

    Returns a GeoDataFrame with centroid points and associated quadkeys.
    """
    counties_wgs = counties_gdf.to_crs("EPSG:4326")
    min_lon, min_lat, max_lon, max_lat = counties_wgs.total_bounds

    tiles = list(mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zoom))
    
    records = []
    for tile in tiles:
        bounds = mercantile.bounds(tile)
        centroid_lon = (bounds.west + bounds.east) / 2
        centroid_lat = (bounds.south + bounds.north) / 2
        quadkey = mercantile.quadkey(tile)
        records.append({"geometry": Point(centroid_lon, centroid_lat), "quadkey": quadkey})

    centroid_gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

    # Clip to actual city boundary
    centroid_gdf = gpd.clip(centroid_gdf, counties_wgs)

    return centroid_gdf


def filter_points_within_boundary(grid_gdf, boundary_gdf):
    """
    Filter out points that are not within the geometry of the specified boundary.
    Ensures CRS consistency before applying spatial filter.
    """
    # Reproject grid to boundary CRS if needed
    if grid_gdf.crs != boundary_gdf.crs:
        grid_gdf = grid_gdf.to_crs(boundary_gdf.crs)

    # Use unary_union to handle multipolygons or multiple features robustly
    boundary_geom = boundary_gdf.union_all()
    grid_within = grid_gdf[grid_gdf.geometry.within(boundary_geom)].copy()
    return grid_within


def split_train_val_test(points_within, test_size=0.2, val_size=0.2):
    """
    Split the points into train, validation, and test sets and return them as separate GeoDataFrames.
    
    Parameters:
        points_within (GeoDataFrame): The full dataset.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the *remaining* train set to use for validation.
        
    Returns:
        train_gdf, val_gdf, test_gdf: GeoDataFrames for train, val, and test splits.
    """
    # First, split off the test set
    train_val_gdf, test_gdf = train_test_split(points_within, test_size=test_size, random_state=42, shuffle=True)

    # Then, split the remaining into train and val
    relative_val_size = val_size / (1 - test_size)
    train_gdf, val_gdf = train_test_split(train_val_gdf, test_size=relative_val_size, random_state=42, shuffle=True)

    return train_gdf, val_gdf, test_gdf


def save_csv(gdf, output_file):
    """
    Save the GeoDataFrame to a CSV file.
    """
    gdf[['point_id', 'latitude', 'longitude', 'NAME_2', 'NAME_1']].to_csv(output_file, index=False)

def plot_and_save(counties_gdf, train_gdf, val_gdf, test_gdf, output_image):
    """
    Plot the counties and the split points (train and val) and save the plot as an image file.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    counties_gdf.to_crs(epsg=4326).plot(ax=ax, facecolor='none', edgecolor='black')
    train_gdf.plot(ax=ax, color='blue', markersize=3, label='Train')
    val_gdf.plot(ax=ax, color='red', markersize=3, label='Val')
    test_gdf.plot(ax=ax, color='green', markersize=3, label='Test')
    plt.legend()
    plt.title("NYC Sampled Points Split into Train/Val with County and State Info")
    plt.savefig(output_image, dpi=300)  # Save plot to file
    plt.close()

def main():

    load_dotenv("config.env")

    BOUNDARY_GEOJSON = Path(os.getenv("BOUNDARY_GEOJSON"))
    CITY= os.getenv("CITY")
    COUNTRY = os.getenv("COUNTRY")
    ZOOM = int(os.getenv("ZOOM"))
    POINTS_CSV = Path(os.getenv("POINTS_CSV"))

    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    # File paths
    output_image = f'./data/points/{CITY}_sampled_points_plot.png'

    # EPSG codes per city (for local UTM projection)
    city_epsg = {
        'new york city': 2263,     # NAD83 / New York Long Island
        'paris': 32631,       # WGS84 / UTM zone 31N
        'los angeles': 26911, # NAD83 / UTM zone 11N
        'berlin': 32633,      # WGS84 / UTM zone 33N
        'chicago': 26916,     # NAD83 / UTM zone 16N
        'dallas': 26914,      # NAD83 / UTM zone 14N
        'amsterdam': 32631,   # WGS84 / UTM zone 31N
        'mumbai': 32643,   # WGS84 / UTM zone 31N
        'sydney': 32756,   # WGS84 / UTM zone 31N
        'rome': 32633         # WGS84 / UTM zone 33N (Rome, Italy)
    }

    if CITY not in city_epsg:
        raise ValueError(f"No EPSG projection defined for city '{CITY}'.")

    # Read boundary and reproject to appropriate UTM
    boundary_gdf = gpd.read_file(BOUNDARY_GEOJSON)
    boundary_gdf = boundary_gdf.to_crs(epsg=city_epsg[CITY])

    # Generate grid points within counties
    grid_gdf = generate_tile_centroids(boundary_gdf, zoom=ZOOM)

    # Filter points within counties
    points_within = filter_points_within_boundary(grid_gdf, boundary_gdf)

    # Get centroids and add lat/lon
    points_within["centroid"] = points_within.centroid
    points_within = points_within.to_crs(epsg=4326)
    points_within["longitude"] = points_within.geometry.x
    points_within["latitude"] = points_within.geometry.y

    # Use quadkey as point_id
    points_within.rename(columns = {"quadkey": "point_id"}, inplace=True)

    points_within['NAME_2'] = CITY.replace('_', ' ').title()
    points_within['NAME_1'] = COUNTRY

    # Ensure point_id is unique
    points_within.sort_values(by = ['point_id'], inplace = True)
    points_within['rand_point'] = points_within['point_id']
    points_within.set_index('rand_point', inplace = True)

    # Split into train/val sets
    train_gdf, val_gdf, test_gdf = split_train_val_test(points_within)

    # Save train and validation CSVs
    save_csv(points_within, POINTS_CSV)
    save_csv(train_gdf, f"./data/points/{CITY}_train_points.csv")
    save_csv(val_gdf, f"./data/points/{CITY}_val_points.csv")
    save_csv(test_gdf, f"./data/points/{CITY}_test_points.csv")

    # Plot and save the image
    plot_and_save(boundary_gdf, train_gdf, val_gdf, test_gdf, output_image)

    print(f"Plot saved as: {output_image}")

if __name__ == "__main__":
    main()
