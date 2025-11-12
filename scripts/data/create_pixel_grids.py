import argparse
import pandas as pd
import torch
import mercantile
from shapely.geometry import LineString, Polygon, MultiPolygon
import skimage.draw
from shapely import wkt
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm
import numpy as np
from collections import Counter
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed
from utils import process_osm_tags

GEOM_PRIORITY = {
    "LineString": 1,
    "MultiLineString": 1,
    "Polygon": 2,
    "MultiPolygon": 2
}

def tag_key(tag):
    if isinstance(tag, tuple):
        return tag[0]
    elif isinstance(tag, str):
        return tag.split()[0]
    return None


def to_pixel_coords(coords, bounds, img_size):
    west, south, east, north = bounds
    coords = torch.tensor(coords)
    lons, lats = coords[:, 0], coords[:, 1]
    norm_x = torch.clamp((lons - west) / (east - west), 0, 1)
    norm_y = 1 - torch.clamp((lats - south) / (north - south), 0, 1)
    px = (norm_x * (img_size - 1)).long()
    py = (norm_y * (img_size - 1)).long()
    return px, py

def rasterize_polygon(grid, polygon, tag_indices, img_size):
    # Rasterize polygon exterior
    exterior_coords = list(polygon.exterior.coords)
    rr, cc = skimage.draw.polygon(
        [c[1] for c in exterior_coords],  # y = row
        [c[0] for c in exterior_coords],  # x = col
        shape=(img_size, img_size)
    )
    for r, c in zip(rr, cc):
        grid[r][c].extend(tag_indices)

    # Rasterize holes (interiors) as empty space
    for interior in polygon.interiors:
        interior_coords = list(interior.coords)
        rr, cc = skimage.draw.polygon(
            [c[1] for c in interior_coords],
            [c[0] for c in interior_coords],
            shape=(img_size, img_size)
        )
        for r, c in zip(rr, cc):
            for tag in tag_indices:
                if tag in grid[r][c]:
                    grid[r][c].remove(tag)

def rasterize_line(grid, coords_px, tag_indices):
    for i in range(len(coords_px[0]) - 1):
        rr, cc = skimage.draw.line(
            coords_px[1][i].item(), coords_px[0][i].item(),
            coords_px[1][i+1].item(), coords_px[0][i+1].item()
        )
        for r, c in zip(rr, cc):
            grid[r][c].extend(tag_indices)

def should_buffer(tags):
    return any(tag_key(t) in ("highway", "waterway") for t in tags)

def rasterize_geometry(grid, geom, geom_type, tag_indices, bounds, img_size, tags):
    west, south, east, north = bounds

    # Convert all geometry coords to pixel coords once
    def get_all_coords(geom, geom_type):
        if geom_type == "LineString":
            return list(geom.coords)
        elif geom_type == "MultiLineString":
            coords = []
            for ls in geom.geoms:
                coords.extend(ls.coords)
            return coords
        elif geom_type == "Polygon":
            return list(geom.exterior.coords)
        elif geom_type == "MultiPolygon":
            coords = []
            for poly in geom.geoms:
                coords.extend(poly.exterior.coords)
            return coords
        else:
            return []

    coords = get_all_coords(geom, geom_type)
    if not coords:
        return

    px, py = to_pixel_coords(coords, bounds, img_size)

    # Handle buffering and rasterization for line-like geometries
    if should_buffer(tags) and geom_type in ("LineString", "MultiLineString"):
        geom_pixels = LineString(zip(px.numpy(), py.numpy()))
        buffer_px = 5
        buffered_geom = geom_pixels.buffer(buffer_px)

        polygons = []
        if isinstance(buffered_geom, Polygon):
            polygons = [buffered_geom]
        elif isinstance(buffered_geom, MultiPolygon):
            polygons = list(buffered_geom.geoms)

        for poly in polygons:
            rasterize_polygon(grid, poly, tag_indices, img_size)
    else:
        # No buffer case: rasterize normally
        if geom_type == "LineString":
            rasterize_line(grid, (px, py), tag_indices)
        elif geom_type == "MultiLineString":
            for ls in geom.geoms:
                ls_px, ls_py = to_pixel_coords(list(ls.coords), bounds, img_size)
                rasterize_line(grid, (ls_px, ls_py), tag_indices)
        elif geom_type == "Polygon":
            px_poly, py_poly = to_pixel_coords(list(geom.exterior.coords), bounds, img_size)
            rasterize_polygon(grid, Polygon(zip(px_poly.numpy(), py_poly.numpy())), tag_indices, img_size)
        elif geom_type == "MultiPolygon":
            for poly in geom.geoms:
                px_poly, py_poly = to_pixel_coords(list(poly.exterior.coords), bounds, img_size)
                rasterize_polygon(grid, Polygon(zip(px_poly.numpy(), py_poly.numpy())), tag_indices, img_size)


def process_point(args):
    row, geoparquet_dir, tag_to_index, output_npy_dir, img_size = args
    point_id = row['point_id']
    parquet_path = geoparquet_dir / f"bbox_{point_id}.parquet"
    if not parquet_path.exists():
        return f"Missing: bbox_{point_id}.parquet"

    lon, lat = row["longitude"], row["latitude"]
    tile = mercantile.tile(lon, lat, 16)
    bounds = mercantile.bounds(tile)
    df = gpd.read_parquet(parquet_path)

    if df.empty:
        return f"Skipping {point_id}: parquet file is empty"

    if isinstance(df["geometry"].iloc[0], str):
        df["geometry"] = df["geometry"].apply(wkt.loads)
        df = gpd.GeoDataFrame(df, geometry="geometry")

    df = df[~df.geometry.geom_type.isin(["Point", "MultiPoint"])]
    if df.empty:
        return f"Skipping {point_id}: no non-point geometries left after filtering"

    max_index = len(tag_to_index) - 1
    grid = [[[] for _ in range(img_size)] for _ in range(img_size)]

    for _, feat in df.iterrows():
        raw_tags = feat["tags"]
        cleaned_tags = process_osm_tags(raw_tags)

        tag_indices = [tag_to_index[str(t)] for t in cleaned_tags if str(t) in tag_to_index and tag_to_index[str(t)] <= max_index]

        rasterize_geometry(grid, feat["geometry"], feat["geometry"].geom_type, tag_indices, bounds, img_size, cleaned_tags)

    total_tag_pixels = sum(1 for r in range(img_size) for c in range(img_size) if grid[r][c])
    if total_tag_pixels == 0:
        return f"Warning: Empty grid for bbox_{point_id}"

    np_grid = np.empty((img_size, img_size), dtype=object)
    for r in range(img_size):
        for c in range(img_size):
            np_grid[r, c] = tuple(sorted(set(grid[r][c])))

    np.save(output_npy_dir / f"bbox_{point_id}.npy", np_grid)
    return f"Processed bbox_{point_id} with {total_tag_pixels} tagged pixels"

def rasterize_npy_parallel(point_df, geoparquet_dir, tag_to_index, output_npy_dir, img_size, n_workers=None):
    output_npy_dir.mkdir(parents=True, exist_ok=True)
    n_workers = n_workers or max(1, cpu_count() - 1)
    args_list = [(row, geoparquet_dir, tag_to_index, output_npy_dir, img_size) for _, row in point_df.iterrows()]

    with Pool(n_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_point, args_list), total=len(args_list), desc="Stage 1: Saving .npy grids"))
    for res in results:
        print(res)

def process_npy_for_surviving_taglists(npy_path, img_size, min_pixels_per_image):
    grid = np.load(npy_path, allow_pickle=True)
    taglist_pixel_counts = Counter()

    for r in range(img_size):
        for c in range(img_size):
            if grid[r, c]:
                taglist = tuple(sorted(set(grid[r, c])))
                taglist_pixel_counts[taglist] += 1

    return {tl for tl, count in taglist_pixel_counts.items() if count >= min_pixels_per_image}

def build_taglist_vocab_from_npy(npy_dir, img_size, min_pixels_per_image=100, n_jobs=8):
    npy_files = list(npy_dir.glob("*.npy"))

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_npy_for_surviving_taglists)(npy_file, img_size, min_pixels_per_image)
        for npy_file in tqdm(npy_files, desc="Collecting surviving taglists")
    )

    all_surviving = set()
    for r in results:
        all_surviving.update(r)

    print(f"Found {len(all_surviving)} taglists surviving pixel threshold")
    return {tl: i for i, tl in enumerate(sorted(all_surviving))}

def process_npy_to_tensor_final_pass(npy_path, tensor_out_dir, taglist_to_index, img_size, min_pixels_per_image):
    grid = np.load(npy_path, allow_pickle=True)
    tensor = torch.full((img_size, img_size), -1, dtype=torch.long)

    taglist_pixel_counts = Counter()
    for r in range(img_size):
        for c in range(img_size):
            if grid[r, c]:
                taglist = tuple(sorted(set(grid[r, c])))
                taglist_pixel_counts[taglist] += 1

    valid_taglists = {tl for tl, c in taglist_pixel_counts.items() if c >= min_pixels_per_image}

    for r in range(img_size):
        for c in range(img_size):
            if grid[r, c]:
                taglist = tuple(sorted(set(grid[r, c])))
                if taglist in valid_taglists and taglist in taglist_to_index:
                    tensor[r, c] = taglist_to_index[taglist]

    out_path = tensor_out_dir / (npy_path.stem + ".pt")
    torch.save(tensor, out_path)
    return f"Saved {out_path.name}"

def convert_to_tensor_with_final_vocab(npy_dir, tensor_out_dir, taglist_to_index, img_size, min_pixels_per_image=100, n_jobs=8):
    tensor_out_dir.mkdir(parents=True, exist_ok=True)
    npy_files = list(npy_dir.glob("*.npy"))

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_npy_to_tensor_final_pass)(
            npy_file, tensor_out_dir, taglist_to_index, img_size, min_pixels_per_image
        ) for npy_file in tqdm(npy_files, desc="Final tensor creation")
    )

    for res in results:
        print(res)

def main(args):
    point_df = pd.read_csv(args.point_csv)
    geoparquet_dir = Path(args.geoparquet_dir)
    output_npy_dir = Path(args.output_npy_dir)
    tensor_out_dir = Path(args.tensor_out_dir)
    save_dir = Path(args.save_dir)

    output_npy_dir.mkdir(parents=True, exist_ok=True)
    tensor_out_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    tag_to_index = torch.load(args.tag_vocab)
    rasterize_npy_parallel(point_df, geoparquet_dir, tag_to_index, output_npy_dir, args.img_size, args.n_workers)

    taglist_to_index = build_taglist_vocab_from_npy(output_npy_dir, args.img_size, args.min_pixels_per_image, args.n_jobs)
    torch.save(sorted(taglist_to_index.keys()), Path(args.taglist_vocab))

    convert_to_tensor_with_final_vocab(
        npy_dir=output_npy_dir,
        tensor_out_dir=tensor_out_dir,
        taglist_to_index=taglist_to_index,
        img_size=args.img_size,
        min_pixels_per_image=args.min_pixels_per_image,
        n_jobs=args.n_jobs
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--point_csv", default="./data/metadata/combined_points.csv")
    parser.add_argument("--geoparquet_dir", default="./data/clipped_bbox_geoparquet_combined")
    parser.add_argument("--tag_vocab", default="./data/metadata/tag_vocab.pt")
    parser.add_argument("--taglist_vocab", default="./data/metadata/taglist_vocab.pt")
    parser.add_argument("--output_npy_dir", default="./data/pixel_grid")
    parser.add_argument("--tensor_out_dir", default="./data/pixel_tags")
    parser.add_argument("--save_dir", default="./data/taglist_counts")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--n_workers", type=int, default=None)
    parser.add_argument("--min_pixels_per_image", type=int, default=100)
    parser.add_argument("--n_jobs", type=int, default=8)
    args = parser.parse_args()

    main(args)
