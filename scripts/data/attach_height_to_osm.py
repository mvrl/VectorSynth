import os
import re
import logging
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from tqdm import tqdm

def load_config() -> dict:
    load_dotenv("config.env")
    config = {
        'bbox_dir': Path(os.getenv("CLIPPED_BBOX_GEOPARQUET_DIR")) / os.getenv("CITY"),
        'bh_dir': Path(os.getenv("CLIPPED_BUILDING_BBOX_GEOPARQUET_DIR")) / os.getenv("CITY"),
        'output_dir': Path(os.getenv("CLIPPED_BBOX_GEOPARQUET_DIR")) / "with_heights" / os.getenv("CITY") ,
        'max_workers': int(os.getenv("MAX_WORKERS", "8")),
    }
    return config

def setup_logging(log_file: str = "attach_height.log") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_building_height_path_from_bbox(bbox_path: Path, bh_dir: Path) -> Path | None:
    bbox_fname = bbox_path.name
    quadkey_match = re.search(r"\d{9,}", bbox_fname)
    if not quadkey_match:
        raise ValueError(f"No quadkey found in bbox filename: {bbox_fname}")
    full_quadkey = quadkey_match.group(0)
    for fname in bh_dir.iterdir():
        if full_quadkey in fname.name and fname.suffix == ".parquet":
            return fname
    return None

def is_probably_building(tags):
    if not isinstance(tags, (list, np.ndarray)):
        return False
    building_keys = {"building", "amenity", "shop", "office", "public_building", "industrial", "man_made"}
    for tag in tags:
        if isinstance(tag, (list, np.ndarray)) and tag[0] in building_keys:
            return True
    return False

def append_height_tag(tags, height):
    import pandas as pd
    if pd.notnull(height):
        new_tags = [list(t) for t in tags] if isinstance(tags, (list, np.ndarray)) else []
        new_tags.append(["height", f"{height}m"])
        return new_tags
    return tags

def process_single_bbox(bbox_path: Path, bh_dir: Path, output_dir: Path, logger: logging.Logger):
    try:
        logger.info(f"Processing {bbox_path.name}")
        gdf_bbox = gpd.read_parquet(bbox_path)

        bh_path = get_building_height_path_from_bbox(bbox_path, bh_dir)
        if bh_path is None:
            logger.warning(f"No matching building height file for {bbox_path.name}. Saving original.")
            output_path = output_dir / bbox_path.name
            gdf_bbox.to_parquet(output_path)
            return f"Wrote original {bbox_path.name} (no BH match)"

        gdf_bh = gpd.read_parquet(bh_path)

        if gdf_bbox.crs != gdf_bh.crs:
            gdf_bh = gdf_bh.to_crs(gdf_bbox.crs)

        gdf_bh = gdf_bh[pd.notnull(gdf_bh["height"])]

        gdf_buildings = gdf_bbox[gdf_bbox["tags"].apply(is_probably_building)].copy()

        if gdf_buildings.empty:
            logger.warning(f"No building features in {bbox_path.name}. Saving original.")
            output_path = output_dir / bbox_path.name
            gdf_bbox.to_parquet(output_path)
            return f"Wrote original {bbox_path.name} (no buildings)"

        matched = gpd.sjoin(gdf_buildings, gdf_bh[["geometry", "height"]], how="left", predicate="intersects")
        matched = matched.drop(columns=["index_right"])

        matched["tags"] = [append_height_tag(tags, height) for tags, height in zip(matched["tags"], matched["height"])]

        non_buildings = gdf_bbox[~gdf_bbox.index.isin(gdf_buildings.index)]
        gdf_final = pd.concat([non_buildings, matched], ignore_index=True).reset_index(drop=True)
        gdf_final = gpd.GeoDataFrame(gdf_final, crs=gdf_bbox.crs)

        output_path = output_dir / bbox_path.name.replace(".parquet", "_with_heights.parquet")
        gdf_final.drop(columns=["height"], inplace=True)
        gdf_final.to_parquet(output_path)
        logger.info(f"Saved augmented bbox to {output_path.name}")
        return f"Processed {bbox_path.name}"

    except Exception as e:
        logger.error(f"Error processing {bbox_path.name}: {e}")
        return f"Error {bbox_path.name}: {e}"

    except Exception as e:
        logger.error(f"Error processing {bbox_path.name}: {e}")
        return f"Error {bbox_path.name}: {e}"

def process_all_bboxes_parallel(bbox_dir: Path, bh_dir: Path, output_dir: Path, logger: logging.Logger, max_workers: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    bbox_files = list(bbox_dir.glob("*.parquet"))
    logger.info(f"Found {len(bbox_files)} bbox files in {bbox_dir}")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_bbox, bbox, bh_dir, output_dir, logger): bbox for bbox in bbox_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing BBoxes"):
            res = future.result()
            results.append(res)
            logger.info(res)
    return results

def main():
    config = load_config()
    logger = setup_logging()

    logger.info("Starting parallel bbox processing with building heights...")

    process_all_bboxes_parallel(
        bbox_dir=config['bbox_dir'],
        bh_dir=config['bh_dir'],
        output_dir=config['output_dir'],
        logger=logger,
        max_workers=config['max_workers']
    )

    logger.info("Finished all bbox files.")

if __name__ == "__main__":
    main()
