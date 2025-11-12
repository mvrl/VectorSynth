import quackosm as qosm
from pathlib import Path
import os
from dotenv import load_dotenv
import logging


def setup_logger(name, log_file, level=logging.INFO):
    """Function to set up a logger with file and stream handlers."""
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def main():
    load_dotenv("config.env")

    OSM_PBF_PATH = Path(os.getenv("OSM_PBF"))
    LOGGING_FOLDER = Path(os.getenv("LOGGING_FOLDER"))

    # Derive output path from input PBF path
    output_dir = OSM_PBF_PATH.parent.parent / "converted"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = OSM_PBF_PATH.stem.replace(".osm", "") + ".geoparquet"
    OSM_GEOPARQUET_PATH = output_dir / output_file

    # Setup logger
    LOGGING_FOLDER.mkdir(parents=True, exist_ok=True)
    log_file = LOGGING_FOLDER / "convert_pbf_to_parquet.log"
    main_logger = setup_logger("convert_logger", log_file)

    main_logger.info("Starting conversion from PBF to GeoParquet.")
    main_logger.info(f"Input PBF: {OSM_PBF_PATH}")
    main_logger.info(f"Output GeoParquet path: {OSM_GEOPARQUET_PATH}")

    try:
        gpq_path = qosm.convert_pbf_to_parquet(
            OSM_PBF_PATH.as_posix(),
            result_file_path=OSM_GEOPARQUET_PATH.as_posix()
        )
        main_logger.info(f"GeoParquet file written to: {gpq_path}")
    except Exception as e:
        main_logger.error("Error during conversion", exc_info=True)

if __name__ == "__main__":
    main()