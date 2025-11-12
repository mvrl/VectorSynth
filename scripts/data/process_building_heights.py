import geopandas as gpd
import pandas as pd
import mercantile
from shapely.geometry import shape
import requests
from requests.exceptions import Timeout
from tqdm import tqdm
import gzip
import json
import time
import os
import argparse
import logging


def setup_logging(log_file=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file) if log_file else logging.StreamHandler()
        ]
    )

def size_to_bytes(size_str):
    size_str = size_str.strip()
    if size_str.endswith("KB"):
        return float(size_str[:-2]) * 1024
    elif size_str.endswith("MB"):
        return float(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith("B"):
        return float(size_str[:-1])
    return 0

def load_boundary(boundary_path):
    start = time.time()
    gdf = gpd.read_file(boundary_path)
    logging.info(f"Loaded boundary from {boundary_path} in {time.time() - start:.2f} seconds")
    return gdf.unary_union

def compute_quadkeys(geom, zoom=9):
    start = time.time()
    minx, miny, maxx, maxy = geom.bounds
    tiles = list(mercantile.tiles(minx, miny, maxx, maxy, zooms=zoom))
    quadkeys = list({mercantile.quadkey(t) for t in tiles})
    logging.info(f"Computed {len(quadkeys)} quadkeys at zoom {zoom} in {time.time() - start:.2f} seconds")
    return quadkeys

def load_dataset_index(csv_url):
    start = time.time()
    df = pd.read_csv(csv_url, dtype=str)
    df["SizeBytes"] = df["Size"].apply(size_to_bytes)
    logging.info(f"Loaded dataset index from {csv_url} in {time.time() - start:.2f} seconds")
    return df

def process_quadkey(qk, dataset_df, boundary_geom, output_dir):
    rows = dataset_df[dataset_df["QuadKey"] == qk]
    if rows.empty:
        logging.warning(f"No entries found for QuadKey {qk}")
        return

    best_row = rows.loc[rows["SizeBytes"].idxmax()]
    tile_url = best_row["Url"]
    logging.info(f"Fetching tile for QuadKey {qk} from {tile_url}")

    try:
        start = time.time()
        features = []
        with requests.get(tile_url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with gzip.open(r.raw, mode='rt') as f:
                for line in tqdm(f, desc=f"Reading {qk}", leave=False):
                    features.append(json.loads(line))
        logging.info(f"Read {len(features)} features for {qk} in {time.time() - start:.2f} sec")

        df = pd.DataFrame(features)

        # Extract and convert
        df["height"] = df["properties"].apply(lambda d: d.get("height", None)).round().astype("Int32")
        # df["confidence"] = df["properties"].apply(lambda d: d.get("confidence", None))
        df["geometry"] = [shape(geom) for geom in tqdm(df.geometry, desc="Converting geometry", leave=False)]
        df.drop(columns=["properties"], inplace=True)

        gdf = gpd.GeoDataFrame(df, crs="EPSG:4326")

        clip_start = time.time()
        gdf = gdf[gdf.geometry.intersects(boundary_geom)]
        logging.info(f"Clipped {qk} to boundary in {time.time() - clip_start:.2f} sec; {len(gdf)} buildings remain")

        save_path = os.path.join(output_dir, f"buildings_{qk}.parquet")
        gdf.to_parquet(save_path)
        logging.info(f"Saved {len(gdf)} buildings to {save_path}")

    except Timeout:
        logging.warning(f"Timeout occurred for QuadKey {qk} at {tile_url}")
    except Exception as e:
        logging.error(f"Error processing QuadKey {qk}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download and process Microsoft building data.")
    parser.add_argument("--boundary", type=str, required=True, help="Path to city boundary GeoJSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for GeoParquet files")
    parser.add_argument("--zoom", type=int, default=9, help="Zoom level for tiling (default: 9)")
    parser.add_argument("--csv_url", type=str,
                        default="https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv",
                        help="URL to dataset index CSV")
    parser.add_argument("--log_file", type=str, help="Optional log file path")

    args = parser.parse_args()
    setup_logging(args.log_file)

    os.makedirs(args.output_dir, exist_ok=True)

    overall_start = time.time()
    boundary_geom = load_boundary(args.boundary)
    quadkeys = compute_quadkeys(boundary_geom, zoom=args.zoom)
    dataset_df = load_dataset_index(args.csv_url)

    for qk in tqdm(quadkeys, desc="Processing QuadKeys"):
        process_quadkey(qk, dataset_df, boundary_geom, args.output_dir)

    logging.info(f"✅ Done. Total time: {time.time() - overall_start:.2f} seconds")

if __name__ == "__main__":
    main()
