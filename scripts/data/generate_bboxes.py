import os
from pathlib import Path
from typing import Optional

import pandas as pd
import geopandas as gpd
import mercantile
from shapely.geometry import box
from dotenv import load_dotenv


def generate_bboxes(
    input_csv: str, 
    output_dir: str, 
    combined_output_path: str, 
    zoom: int = 18
) -> None:
    """
    Generate bounding boxes from CSV of points and save as GeoParquet files.
    
    Parameters:
    input_csv: Path to CSV file with 'latitude', 'longitude', 'point_id' columns
    output_dir: Directory to save individual bbox GeoParquet files
    combined_output_path: Path for combined GeoParquet file
    zoom: Zoom level for tile-based bounding boxes (default: 18)
    """
    # Check if combined output already exists
    if os.path.exists(combined_output_path):
        print(f"Combined output already exists at {combined_output_path}. Skipping generation.")
        return
    
    # Load points from CSV
    points = pd.read_csv(input_csv)
    
    # Validate required columns
    required_cols = ['latitude', 'longitude', 'point_id']
    missing_cols = [col for col in required_cols if col not in points.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each point
    bbox_data = []
    skipped_count = 0
    
    for _, row in points.iterrows():
        latitude = float(row['latitude'])
        longitude = float(row['longitude'])
        point_id = row['point_id']
        
        # Generate bounding box from tile
        try:
            tile = mercantile.tile(longitude, latitude, zoom)
            bbox = mercantile.bounds(tile)
            bbox_geom = box(bbox.west, bbox.south, bbox.east, bbox.north)
            
            # Add to combined data
            bbox_data.append({
                'point_id': point_id,
                'geometry': bbox_geom,
                'latitude': latitude,
                'longitude': longitude,
                'zoom': zoom,
                'tile_x': tile.x,
                'tile_y': tile.y
            })
            
        except Exception as e:
            print(f"Error processing point {point_id} at ({latitude}, {longitude}): {e}")
            continue
    
    # Create and save combined GeoParquet
    if bbox_data:
        combined_gdf = gpd.GeoDataFrame(bbox_data, crs='EPSG:4326')
        combined_gdf.to_parquet(combined_output_path)
        
        print(f"Generated {len(bbox_data)} bounding boxes:")
        print(f"  - Individual files: {len(bbox_data) - skipped_count} new, {skipped_count} skipped")
        print(f"  - Combined file: {combined_output_path}")
    else:
        print("No bounding boxes were generated.")

if __name__ == "__main__":
    load_dotenv("config.env")

    input_csv = os.getenv("POINTS_CSV")
    output_dir = os.getenv("BOUNDING_BOXES_DIR")
    combined_output_path = os.getenv("COMBINED_BBOX_OUTPUT_PATH")
    zoom = int(os.getenv("ZOOM"))

    generate_bboxes(input_csv, output_dir, combined_output_path, zoom)
