import os
import logging
from pathlib import Path
from typing import List

import geopandas as gpd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from shapely.validation import make_valid

def setup_logging(log_file: str = "clip_buildings.log") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def get_all_building_files(buildings_dir: Path) -> List[Path]:
    return list(buildings_dir.glob("*.parquet"))


def clip_tile_to_bboxes(tile_path: Path, bbox_gdf: gpd.GeoDataFrame, output_dir: Path, logger: logging.Logger):
    try:
        tile_gdf = gpd.read_parquet(tile_path)

        # Fix invalid geometries using make_valid
        invalid_mask = ~tile_gdf.geometry.is_valid
        if invalid_mask.any():
            logger.info(f"🧹 Fixing {invalid_mask.sum()} invalid geometries in {tile_path.name} using make_valid")
            tile_gdf.loc[invalid_mask, "geometry"] = tile_gdf.loc[invalid_mask, "geometry"].apply(make_valid)


        # Reproject tile_gdf to match bbox CRS if necessary
        if tile_gdf.crs != bbox_gdf.crs:
            tile_gdf = tile_gdf.to_crs(bbox_gdf.crs)
            logger.info(f"🌀 Reprojected {tile_path.name} to match bbox CRS")

        # Spatial filter to limit to intersecting bboxes
        intersecting_bboxes = bbox_gdf[bbox_gdf.intersects(tile_gdf.geometry.union_all())]

        if intersecting_bboxes.empty:
            logger.info(f"❎ No intersecting bboxes for {tile_path.name}")
            return

        for _, bbox_row in intersecting_bboxes.iterrows():
            bbox_id = bbox_row.get("point_id") or bbox_row.name
            clipped = gpd.clip(tile_gdf, bbox_row.geometry)

            if clipped.empty:
                continue

            out_path = output_dir / f"{tile_path.stem}_bbox_{bbox_id}.parquet"
            clipped.to_parquet(out_path)
            logger.info(f"✅ Clipped {len(clipped)} features from {tile_path.name} to bbox {bbox_id}")

    except Exception as e:
        logger.error(f"❌ Error processing {tile_path.name}: {e}")



def main():
    parser = argparse.ArgumentParser(description="Clip building tiles to bounding boxes.")
    parser.add_argument("--buildings_dir", type=str, required=True, help="Directory of building GeoParquet files")
    parser.add_argument("--bbox_file", type=str, required=True, help="GeoParquet file with bounding boxes")
    parser.add_argument("--clipped_dir", type=str, required=True, help="Output directory for clipped results")
    parser.add_argument("--log_file", type=str, default="clip_buildings.log", help="Log file path")
    parser.add_argument("--max_workers", type=int, default=4, help="Parallel workers")

    args = parser.parse_args()
    logger = setup_logging(args.log_file)

    buildings_dir = Path(args.buildings_dir)
    clipped_dir = Path(args.clipped_dir)
    clipped_dir.mkdir(parents=True, exist_ok=True)

    logger.info("📦 Loading bounding boxes...")
    bbox_gdf = gpd.read_parquet(args.bbox_file)
    logger.info(f"✅ Loaded {len(bbox_gdf)} bounding boxes")

    building_files = get_all_building_files(buildings_dir)
    logger.info(f"🔍 Found {len(building_files)} building files to clip")

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                clip_tile_to_bboxes,
                tile_path,
                bbox_gdf,
                clipped_dir,
                logger
            ) for tile_path in building_files
        ]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Clipping tiles"):
            pass

    logger.info("✅ All clipping completed.")


if __name__ == "__main__":
    main()