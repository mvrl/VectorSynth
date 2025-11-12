# Data Generation Pipeline

A fast pipeline for processing dense urban OSM data, avoiding the slowness of Overpass API in areas with thousands of overlapping polygons.

> **Alternative**: For less dense regions, consider [overpy](https://python-overpy.readthedocs.io/en/latest/) instead.

## Prerequisites

- Download region-level OSM PBF files from [Geofabrik](https://download.geofabrik.de/)
- Obtain boundary shapefiles (see Zenodo)
- Configure `config.env` with paths and parameters

### 🧑‍💻 Setting up environment

Create a conda environment:

```bash
conda env create -f environment.yaml
conda activate vectorsynth_download
```

## Pipeline Execution Order

### Vector Data Processing (per city, then combine)

1. **Convert PBF to GeoParquet**
   ```bash
   python osm2geoparquet.py
   ```

2. **Generate sample points** within boundaries
   ```bash
   python generate_points.py
   ```

3. **Generate bounding boxes** from points
   ```bash
   python generate_bboxes.py
   ```

4. **Clip OSM tags** to bounding boxes
   ```bash
   python clip_bbox.py
   ```

5. **Process building heights** (Microsoft Buildings dataset)
   ```bash
   python process_building_heights.py
   python clip_building_heights.py
   python attach_height_to_osm.py
   ```

6. **Combine cities** and merge metadata
   ```bash
   python combine_points.py
   ```

7. **Clean and filter tags**
   ```bash
   python clean_tags.py
   ```

8. **Rasterize tags to pixel grids**
   ```bash
   python create_pixel_grids.py
   ```

9. **Compute text embeddings** for tags
   ```bash
   python compute_embeddings.py
   ```

10. **Cleanup intermediate files** and filter by coverage
    ```bash
    python filter_cleanup.py
    ```

### Satellite Imagery

11. **Download satellite tiles**
    ```bash
    python download_sat.py
    ```

12. **Generate captions** using LLaVA
    ```bash
    python get_llava_captions.py
    ```

## Output Structure

- `data/metadata/`: Tag vocabularies, point metadata, coverage statistics
- `data/pixel_tags/`: Rasterized tag grids as PyTorch tensors
- `data/tag_embeddings/`: Precomputed text embeddings
- `data/sat_images/`: Satellite imagery tiles
- `data/logs/`: Processing logs