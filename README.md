# Prefect EO Data Pipeline: Orchestration, Embeddings, Classification and STAC for Evacuation Routes

This portfolio project showcases how to combine workflow orchestration with Prefect and Earth observation problem solving, using evacuation route analysis as the driving use case. It brings together machine learning and embeddings, data preprocessing and quality checks, STAC catalog creation, and Dockerized data pipelines, with production-minded features such as task retries, explicit flow dependencies, and a human-in-the-loop step that simulates an operator of a public protection ministry providing input mid-pipeline. The goal is to demonstrate end-to-end capabilities in building reliable, reproducible, and scalable geospatial data workflows.

## Orchestration logic

The pipeline is orchestrated with Prefect as a main flow that coordinates two subflows. The second subflow runs only if the first has completed successfully, so no downstream step ever works on missing or partial data.

### Design principles

Each flow is decomposed into its own subflows and tasks instead of being written as one long function. This follows Prefect best practices and gives the pipeline several operational benefits:

- **Isolated logs and state:** every subflow and task run has its own state and log stream in the Prefect UI, so a failure can be traced to the exact step without searching through one monolithic log.
- **Targeted retries:** retries are configured at task level, so a transient failure reruns only the task that failed, not the whole pipeline or the work already completed.
- **Explicit dependencies:** failures propagate from task to subflow to flow, which is what allows the main flow to gate Flow 2 on the success of Flow 1.
- **Testability and reuse:** small single-purpose tasks can be tested in isolation, and steps such as the training sample can be rerun on their own.

### Flow 1: Road surface classification

**Data preparation.** This flow downloads OpenStreetMap (OSM) road data and standardizes it into a usable schema by assigning codes based on each feature's categories and fields. For roads where OSM has no surface information, it downloads Tessera embeddings so the model can infer whether they are paved or unpaved.

**Training sample.** This step can be rerun by the user on their own data. Embeddings are gathered and preprocessed to keep only road pixels whose center falls within the road. Pixels near a road's edge are discarded because, at this spatial resolution, their spectral signal is dominated by the surroundings rather than the road itself. The remaining pixels are split into train and test sets, with all pixels of the same road assigned to the same set to prevent data leakage. A spatial buffer is also applied to select geographically diverse road samples from each area of interest (AOI). Class imbalance can be handled in two modes: undersampling, which samples down to the size of the smallest class, or class weight balancing.

**Inference.** The model predicts every pixel along a road and produces two outputs: a classification raster with confidence channels, and a shapefile with the complete paved/unpaved surface inference per road. A road is assigned a class only if more than 70% of its pixels agree on it; otherwise it is labeled "uncertain" rather than forced into a class.

### Flow 2: Evacuation routing

This flow uses the road surface classification from Flow 1 to compute evacuation routes with the OSMnx API. Only roads confirmed as paved are kept in the road graph. Roads classified as unpaved or labeled "uncertain" are removed before routing, so the resulting routes never rely on roads whose surface could not be verified. This is a deliberately conservative choice for an evacuation context, where routing over a road that turns out to be unpassable is worse than excluding a road that was actually fine.

**Human-in-the-loop step.** The flow creates an `evacuation_routes_start_point` point shapefile and uploads it, then suspends. This simulates the human-machine interaction of an operator at a public protection ministry, who defines or updates the evacuation start points before the analysis continues. Once the operator uploads the file, the flow resumes.

**Route calculation.** After resuming, the flow calculates the evacuation routes from the start points over the filtered road graph.

**Quality checks.** The flow then runs quality checks on the output, covering topology errors and topological handling of the road network. Road geometries must be split at intersections so the network stays connected and routable, but not where roads cross without connecting, such as bridges and tunnels, where splitting would create false connections.

## STAC catalog and result reuse

Results are published as a [STAC](https://stacspec.org/) (SpatioTemporal Asset Catalog) catalog. The catalog also acts as a lookup, so an area that has already been processed is never inferred twice.

**Collection check first.** Before inference starts, the flow checks `collection.json` for the requested AOI. An AOI counts as a match only if it falls within the footprint of an existing product. If the AOI only partly overlaps an existing product, the inference runs for the whole AOI. If a matching product already exists, Flow 1 is skipped and Flow 2 uses the stored output directly. If not, Flow 1 runs the full inference, and when it ends the pipeline creates two STAC items: one for the road surface inference and one for the evacuation routes.

### Collection

The collection (`embedding-based-surface-classification`) is the entry point of the catalog. It describes the product and links to every item produced so far:

- **Extent:** the spatial bounding box and temporal interval covered by the items.
- **License and providers:** the output license (CC-BY-4.0), the producer, and credit to the licensors of the input data (TESSERA embeddings and OpenStreetMap contributors).
- **Item links:** one `item` link per processed tile.

```json
{
  "type": "Collection",
  "id": "embedding-based-surface-classification",
  "stac_version": "1.1.0",
  "title": "Embedding Based Surface Classification (EBSC)",
  "description": "Surface material predictions (per surface and per pixel) derived from satellite embeddings and applied on OpenStreetMap surfaces for prediction.",
  "license": "CC-BY-4.0",
  "links": [
    { "rel": "root", "href": "./collection.json", "type": "application/json" },
    { "rel": "item", "href": "./EBSC_39.77000_21.18300_EPSG4326_pixel_classifier_vote_v0.2.0/EBSC_39.77000_21.18300_EPSG4326_pixel_classifier_vote_v0.2.0.json", "type": "application/geo+json" }
  ]
}
```

*(Excerpt. Extent and providers omitted for brevity.)*

### Road surface inference item

Each item describes one processed tile and is built to make the result traceable and reproducible:

- **Identifier:** encodes the tile coordinates, the CRS, the classification method and the software version, for example `EBSC_39.77000_21.18300_EPSG4326_pixel_classifier_vote_v0.2.0`.
- **Provenance properties:** the software version (`processing` extension), the classification method, the embedding model, the embeddings source date and the OSM snapshot date. The item `datetime` is the embeddings source date, while `created` records when the processing ran.
- **Assets:** the per-road inference shapefile with its sidecar files, the per-pixel inference GeoTIFF, and the inputs used (the embeddings GeoTIFF and the clipped OSM roads). Inputs carry the `source` role so the lineage is explicit.
- **Integrity:** every asset has a checksum and a file size (`file` extension), so a result can be verified.
- **Data lake layout:** asset paths point into the data lake zones, with inputs under `raw_zone` and results under `gold_zone`.

```json
{
  "type": "Feature",
  "stac_version": "1.1.0",
  "stac_extensions": [
    "https://stac-extensions.github.io/processing/v1.2.0/schema.json",
    "https://stac-extensions.github.io/file/v2.1.0/schema.json",
    "https://stac-extensions.github.io/projection/v2.0.0/schema.json"
  ],
  "id": "EBSC_39.77000_21.18300_EPSG4326_pixel_classifier_vote_v0.2.0",
  "properties": {
    "datetime": "2026-03-12T00:00:00Z",
    "processing:software": { "embedding-based-surface-classification": "0.2.0" },
    "EBSC:method": "pixel_classifier_vote",
    "EBSC:embedding_model": "tessera-foundation-model-v1",
    "EBSC:embeddings_source_date": "2026-03-12",
    "EBSC:osm_snapshot": "2026-10-19",
    "proj:code": "EPSG:4326"
  },
  "assets": {
    "per_road_inference_shp": {
      "href": "../../datalake/gold_zone/per_road_inference_39.77000_21.18300_EPSG4326.shp",
      "title": "Per-road inference",
      "roles": ["data"]
    },
    "per_road_pixel_inference_tif": {
      "href": "../../datalake/gold_zone/per_road_pixel_inference_39.77000_21.18300_EPSG4326.tif",
      "title": "Per-pixel inference",
      "roles": ["data"]
    },
    "embeddings_tif": {
      "href": "../../datalake/raw_zone/retrieved_embeddings/tessera_embeddings_39.77000_21.18300_EPSG4326.tif",
      "title": "Embeddings used as input",
      "roles": ["data", "source"]
    }
  },
  "collection": "embedding-based-surface-classification"
}
```

*(Excerpt. Geometry, links, checksums, sizes and the remaining assets omitted for brevity.)*

### Evacuation routes item

The evacuation routes produced by Flow 2 are registered as a second item, so the routes are discoverable and traceable like the classification they build on. It follows the same pattern as the inference item:

- **Footprint and CRS:** the geometry and bounding box are those of the AOI, and the `projection` extension records its coordinate reference system.
- **Assets:** the route outputs as shapefiles, namely the route lines and the start points. Each file of a shapefile is its own asset, and shapefiles carry both the `data` and `visual` roles.
- **Styling:** an SLD style file is included as an asset for each layer, so the layers render consistently in GIS clients.
- **Thumbnail:** a preview image is generated by rendering the output layers over the AOI with those SLD styles. The start points layer is left out of the preview.
- **Lineage:** a `processing:lineage` property describes the inputs the routes were derived from.
- **Time:** the item `datetime` is the processing time.

### Data lake

STAC assets point into a data lake organized in zones: `raw_zone` for inputs such as the retrieved embeddings and OSM roads, and `gold_zone` for the final outputs. The data lake is stored in RustFS. The pipeline also creates the same `datalake` folder structure in the repository, which is then uploaded to RustFS, so the relative asset paths in the STAC items resolve against the same layout in both places.

## How to run

The project runs locally with Docker Compose, which starts a Prefect server and configures RustFS as the object storage.

### Prerequisites

- Docker and Docker Compose

### 1. Configure credentials

Inside the `local_environment` folder, create a `.env` file based on `.env.example` and set the RustFS credentials:

```bash
cd local_environment
cp .env.example .env
```

Then edit `.env` with your own RustFS credentials.

### 2. Start the environment

```bash
docker compose up -d
```

This starts a local Prefect server and configures RustFS. Wait until the command has finished and the containers are running.

### 3. Run the flow

1. Open the Prefect UI at [http://localhost:4200](http://localhost:4200).
2. Go to **Deployments** and select the main flow.
3. Choose **Run → Custom run**.
4. Set the area of interest as a bounding box in WGS84 (EPSG:4326), in the format `min_lon, min_lat, max_lon, max_lat`.
5. Start the run.

### 4. Provide the evacuation start points

Once Flow 1 completes, Flow 2 suspends and waits for the `evacuation_routes_start_point` shapefile. Upload your file in the Prefect UI to resume the flow.

### Stopping the environment

```bash
docker compose down
```
