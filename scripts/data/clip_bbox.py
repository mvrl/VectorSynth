import os
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Set, Tuple

import geopandas as gpd
from dotenv import load_dotenv
from tqdm import tqdm


def setup_logging(log_file: str = "clip_batch.log") -> logging.Logger:
    """Configure logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def filter_tag_list(tag_list: List[Tuple[str, str]], parent_tags: Set[str]) -> List[Tuple[str, str]]:
    """Filter a list of tag tuples to keep only those with keys in parent_tags."""
    return [tag for tag in tag_list if tag[0] in parent_tags]


def filter_tags_within_rows(gdf: gpd.GeoDataFrame, parent_tags: Set[str]) -> gpd.GeoDataFrame:
    """Filter each row's tags list to keep only tag tuples whose keys are in parent_tags."""
    filtered_gdf = gdf.copy()
    filtered_gdf['tags'] = filtered_gdf['tags'].apply(
        lambda x: filter_tag_list(x, parent_tags)
    )
    return filtered_gdf


def load_config() -> dict:
    """Load configuration from environment variables."""
    load_dotenv("config.env")
    
    config = {
        'osm_pbf_path': Path(os.getenv("OSM_PBF")),
        'combined_bbox_path': Path(os.getenv("COMBINED_BBOX_OUTPUT_PATH")),
        'output_dir': Path(os.getenv("CLIPPED_BBOX_GEOPARQUET_DIR"), os.getenv("CITY")),
        'tags': set(os.getenv("TAGS", "").split(",")),
        'max_workers': int(os.getenv("MAX_WORKERS", "4")),
        'chunk_size': int(os.getenv("CHUNK_SIZE", "25")),
        'boundary_geojson': os.getenv("BOUNDARY_GEOJSON"),
    }
    
    # Derive geoparquet path
    geoparquet_dir = config['osm_pbf_path'].parent.parent / "converted"
    geoparquet_file = config['osm_pbf_path'].stem.replace(".osm", "") + ".geoparquet"
    config['osm_geoparquet_path'] = geoparquet_dir / geoparquet_file
    
    return config


def load_and_prepare_city_gdf(
    geoparquet_path: Path, 
    tags: Set[str], 
    logger: logging.Logger,
    boundary_path: str = None
) -> gpd.GeoDataFrame:
    """Load and filter the main city GeoDataFrame, with optional boundary filtering."""
    logger.info(f"Loading city GeoDataFrame from {geoparquet_path}")
    city_gdf = gpd.read_parquet(geoparquet_path)

    # Filter tags and remove rows with no valid tags
    city_gdf = filter_tags_within_rows(city_gdf, tags)
    city_gdf = city_gdf[city_gdf['tags'].str.len() > 0]

    if boundary_path:
        logger.info(f"Applying boundary filtering using {boundary_path}")
        boundary_gdf = gpd.read_file(boundary_path)
        if not boundary_gdf.crs == city_gdf.crs:
            boundary_gdf = boundary_gdf.to_crs(city_gdf.crs)

        # Bounding box filter first, then clip for precision
        bounds = boundary_gdf.total_bounds
        city_gdf = city_gdf.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
        city_gdf = gpd.clip(city_gdf, boundary_gdf)

        logger.info(f"Filtered to {len(city_gdf)} features inside boundary")

    # Pre-build spatial index
    _ = city_gdf.sindex

    return city_gdf

def get_bbox_data(combined_bbox_path: Path, logger: logging.Logger) -> gpd.GeoDataFrame:
    """Load bounding boxes from combined GeoParquet file."""
    logger.info(f"Loading bounding boxes from {combined_bbox_path}")
    
    bbox_gdf = gpd.read_parquet(combined_bbox_path)
    logger.info(f"Loaded {len(bbox_gdf)} bounding boxes")
    
    return bbox_gdf


def chunk_dataframe(gdf: gpd.GeoDataFrame, size: int) -> List[gpd.GeoDataFrame]:
    """Split a GeoDataFrame into chunks of specified size."""
    return [gdf.iloc[i:i + size] for i in range(0, len(gdf), size)]


def clip_batch(bbox_chunk: gpd.GeoDataFrame, city_gdf: gpd.GeoDataFrame, 
               output_dir: Path, logger: logging.Logger) -> None:
    """Process a batch of bounding boxes."""
    for idx, bbox_row in bbox_chunk.iterrows():
        bbox_id = bbox_row['point_id']
        bbox_geom = bbox_row.geometry
        output_path = output_dir / f"bbox_{bbox_id}.parquet"
        
        # Skip if output already exists and is non-empty
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(f"✅ Skipping bbox {bbox_id}: output exists")
            continue
        
        try:
            clipped = gpd.clip(city_gdf, bbox_geom)
            
            clipped.to_parquet(output_path)
            logger.info(f"🗂️ Clipped {len(clipped)} features to bbox {bbox_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed bbox {bbox_id}: {e}")


def main():
    """Main execution function."""
    logger = setup_logging()
    config = load_config()
    
    # Create output directory
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    city_gdf = load_and_prepare_city_gdf(
        config['osm_geoparquet_path'], 
        config['tags'], 
        logger,
        config.get('boundary_geojson')
    )
    
    # Load bounding boxes and create chunks
    bbox_gdf = get_bbox_data(config['combined_bbox_path'], logger)
    bbox_chunks = chunk_dataframe(bbox_gdf, config['chunk_size'])
    
    logger.info(
        f"📦 Starting clip: {len(bbox_gdf)} bounding boxes in "
        f"{len(bbox_chunks)} chunks (chunk size: {config['chunk_size']})"
    )
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=config['max_workers']) as executor:
        futures = [
            executor.submit(
                clip_batch, 
                chunk, 
                city_gdf, 
                config['output_dir'], 
                logger
            ) 
            for chunk in bbox_chunks
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
            try:
                future.result()
            except Exception as e:
                logger.error(f"❗ Uncaught error in chunk: {e}")
    
    logger.info("✅ All clipping finished.")


if __name__ == "__main__":
    main()