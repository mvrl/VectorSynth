import pandas as pd
from pathlib import Path
import shutil

def load_and_combine_point_csvs(base_dir: Path) -> pd.DataFrame:
    """Load all *_points.csv files and combine into a single DataFrame with new point IDs."""
    combined_data = []
    point_id_counter = 0

    for city_file in base_dir.glob("*_points.csv"):
        city_name = city_file.stem.replace("_points", "")
        
        # Determine split from filename
        if 'test' in city_name:
            split = 'test'
            city_name = city_name.replace('_test', '')
        elif 'val' in city_name:
            split = 'val'
            city_name = city_name.replace('_val', '')
        else:
            split = 'train'
            city_name = city_name.replace('_train', '')
        
        df = pd.read_csv(city_file)

        # Add split and city columns
        df['split'] = split
        if 'city' not in df.columns:
            df['city'] = city_name

        combined_data.append(df)

    final_df = pd.concat(combined_data, ignore_index=True)
    return final_df


def save_combined_csv(df: pd.DataFrame, output_path: Path):
    """Save the combined DataFrame to a CSV file."""
    df.to_csv(output_path, index=False)
    print(f"✅ Combined {len(df)} points saved to: {output_path}")


import shutil
from pathlib import Path
import pandas as pd

def move_and_rename_parquets(df: pd.DataFrame, old_base: Path, new_base: Path):
    """
    Move .parquet files starting with bbox_{id} from old location to new one.
    Handles files with varying suffixes like bbox_{id}_*.parquet.
    """
    new_base.mkdir(parents=True, exist_ok=True)
    missing_files = 0
    multiple_matches = 0

    for _, row in df.iterrows():
        city = row["city"]
        point_id = row["point_id"]

        city_dir = old_base / city
        matching_files = list(city_dir.glob(f"bbox_{point_id}*.parquet"))

        if len(matching_files) == 1:
            src_path = matching_files[0]
            dst_path = new_base / f"bbox_{point_id}.parquet"
            shutil.move(src_path, dst_path)
        elif len(matching_files) == 0:
            missing_files += 1
        else:
            print(f"⚠️ Multiple matches for bbox_{point_id} in {city_dir}. Skipping.")
            multiple_matches += 1

    print(f"✅ Finished moving parquet files to: {new_base}")
    if missing_files:
        print(f"⚠️ {missing_files} .parquet files were missing.")
    if multiple_matches:
        print(f"⚠️ {multiple_matches} entries had multiple matches and were skipped.")



def main():
    # --- Config paths ---
    base_dir = Path("./data/points/")
    geoparquet_base = Path("./data/clipped_bbox_geoparquet/with_heights")
    new_parquet_dir = Path("./data/clipped_bbox_geoparquet_combined")
    output_csv_path = "./data/metadata/combined_points.csv"

    # Ensure output directories exist
    new_parquet_dir.mkdir(parents=True, exist_ok=True)
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)

    # --- Combine points ---
    combined_df = load_and_combine_point_csvs(base_dir)
    save_combined_csv(combined_df, output_csv_path)

    # --- Move geoparquet files ---
    move_and_rename_parquets(combined_df, geoparquet_base, new_parquet_dir)

if __name__ == "__main__":
    main()