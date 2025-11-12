import torch
import geopandas as gpd
from collections import Counter
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from utils import process_osm_tags


def setup_logger(name, log_file):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def main():
    config = {
        "CLIPPED_PARQUET_PATH": "./data/clipped_bbox_geoparquet_combined",
        "LOGGING_FOLDER": "./data/logs",
        "OUTPUT_DIR": "./data/metadata",
        "MIN_TAG_COUNT": 20
    }

    logger = setup_logger(
        "tags_logger",
        Path(config['LOGGING_FOLDER']) / "process_tags.log"
    )

    Path(config['OUTPUT_DIR']).mkdir(parents=True, exist_ok=True)

    logger.info("Reading clipped bbox GeoParquet...")
    df = gpd.read_parquet(config['CLIPPED_PARQUET_PATH'])
    logger.info(f"Loaded {len(df)} features")

    all_cleaned_tags = []
    skipped_empty = 0

    for raw_tags in tqdm(df["tags"], desc="Processing tags"):
        if raw_tags is None or (isinstance(raw_tags, float) and np.isnan(raw_tags)):
            skipped_empty += 1
            continue

        cleaned = process_osm_tags(raw_tags)
        if not cleaned:
            skipped_empty += 1
            continue

        all_cleaned_tags.extend(cleaned)

    logger.info(f"Skipped {skipped_empty} empty or fully dropped tag lists.")
    logger.info("Building raw tag vocabulary...")

    tag_counter = Counter(all_cleaned_tags)
    torch.save(tag_counter, Path(config["OUTPUT_DIR"]) / "tag_counter.pt")
    logger.info("Saved tag counter.")

    tag_vocab_raw = {tag: i for i, tag in enumerate(sorted(tag_counter.keys()))}
    torch.save(tag_vocab_raw, Path(config["OUTPUT_DIR"]) / "tag_vocab_raw.pt")
    logger.info(f"Saved raw vocab with {len(tag_vocab_raw)} tags.")

    # === Filter rare tags ===
    min_tag_count = config["MIN_TAG_COUNT"]
    valid_tags = {tag for tag, count in tag_counter.items() if count >= min_tag_count}
    tag_vocab = {tag: i for i, tag in enumerate(sorted(valid_tags))}
    torch.save(tag_vocab, Path(config["OUTPUT_DIR"]) / "tag_vocab.pt")
    logger.info(f"Saved filtered vocab with {len(tag_vocab)} tags (threshold = {min_tag_count}).")


if __name__ == "__main__":
    main()
