import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import shutil
import logging
import numpy as np

# === Configuration ===
TENSOR_DIR = Path("./data/pixel_tags")
COMBINED_POINTS_CSV = Path("./data/metadata/combined_points.csv")
FINAL_POINTS_CSV = COMBINED_POINTS_CSV.parent / "final_points.csv"
COVERAGE_CSV = COMBINED_POINTS_CSV.parent / "bbox_coverage.csv"  # New coverage CSV
COVERAGE_THRESHOLD = 0.0
IMG_SIZE = 512

# Directories to delete
DIRS_TO_DELETE = [
    "./data/processed_buildings",
    "./data/bounding_boxes",
    "./data/clipped_bbox_geoparquet",
    "./data/clipped_bbox_geoparquet_combined",
    "./data/clipped_building_bbox_geoparquet",
    "./data/pixel_grid"
]

def setup_logger(log_file="filter_cleanup.log"):
    logger = logging.getLogger("FilterCleanup")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

def calculate_coverage(tensor_path, img_size):
    tensor = torch.load(tensor_path)
    valid = (tensor != -1).sum().item()
    total = tensor.numel()
    return valid / total

def main():
    logger = setup_logger()
    logger.info(f"Scanning tensors in {TENSOR_DIR}")
    
    tensor_files = list(TENSOR_DIR.glob("bbox_*.pt"))
    logger.info(f"Found {len(tensor_files)} tensor files.")
    
    valid_ids = []
    removed_ids = []
    coverage_data = []  # Store coverage data for all bbox_ids
    
    for tensor_path in tqdm(tensor_files, desc="Checking coverage"):
        coverage = calculate_coverage(tensor_path, IMG_SIZE)
        bbox_id = tensor_path.stem.replace("bbox_", "")
        
        # Store coverage data for CSV
        coverage_data.append({
            'bbox_id': bbox_id,
            'coverage': coverage,
            'above_threshold': coverage >= COVERAGE_THRESHOLD
        })
        
        if coverage >= COVERAGE_THRESHOLD:
            valid_ids.append(bbox_id)
        else:
            removed_ids.append(bbox_id)
            tensor_path.unlink()
    
    # Create and save coverage CSV
    coverage_df = pd.DataFrame(coverage_data)
    coverage_df = coverage_df.sort_values('coverage', ascending=False)  # Sort by coverage descending
    coverage_df.to_csv(COVERAGE_CSV, index=False)
    logger.info(f"Saved coverage data to {COVERAGE_CSV} with {len(coverage_df)} entries.")
    
    logger.info(f"Kept {len(valid_ids)} tensors.")
    logger.info(f"Removed {len(removed_ids)} tensors (coverage < {COVERAGE_THRESHOLD:.0%}).")
    
    # Filter the CSV
    df = pd.read_csv(COMBINED_POINTS_CSV)
    df_filtered = df[df["point_id"].astype(str).isin(valid_ids)].copy()
    
    # For chicago, make the split all test
    df_filtered.loc[df_filtered['NAME_2'].str.startswith('Chicago'), 'split'] = 'test'
    df_filtered.to_csv(FINAL_POINTS_CSV, index=False)
    logger.info(f"Saved filtered DataFrame to {FINAL_POINTS_CSV} with {len(df_filtered)} points.")
    
    # Count per city
    logger.info("\n--- Remaining points per city ---")
    city_counts = df_filtered["city"].value_counts()
    for city, count in city_counts.items():
        logger.info(f"{city}: {count} points")
    
    # Count per split
    logger.info("\n--- Remaining points per split ---")
    split_counts = df_filtered["split"].value_counts()
    for split, count in split_counts.items():
        logger.info(f"{split}: {count} points")
    
    # Coverage statistics
    logger.info(f"\n--- Coverage Statistics ---")
    logger.info(f"Average coverage: {coverage_df['coverage'].mean():.3f}")
    logger.info(f"Median coverage: {coverage_df['coverage'].median():.3f}")
    logger.info(f"Min coverage: {coverage_df['coverage'].min():.3f}")
    logger.info(f"Max coverage: {coverage_df['coverage'].max():.3f}")
    logger.info(f"Coverage above threshold ({COVERAGE_THRESHOLD:.0%}): {coverage_df['above_threshold'].sum()} / {len(coverage_df)}")
    
    # Cleanup intermediate directories
    logger.info("\nCleaning up intermediate directories:")
    for dir_path in DIRS_TO_DELETE:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
            logger.info(f"Deleted: {path}")
        else:
            logger.info(f"Not found or already removed: {path}")
    
    logger.info("Done.")

if __name__ == "__main__":
    main()