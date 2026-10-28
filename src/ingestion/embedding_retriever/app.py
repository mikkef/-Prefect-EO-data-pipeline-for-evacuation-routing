from src.constants import raw_zone_path
from pathlib import Path
from src.ingestion.embedding_retriever.ingest_embeddings import generate_tessera_embeddings


def ingest_embeddings(aoi_shp, logger):
    output_dir = Path(raw_zone_path) / "retrieved_embeddings"

    out = generate_tessera_embeddings(
        aoi_shapefile=str(aoi_shp),
        output_dir=str(output_dir),
        year=2025,
        logger=logger
    )

    logger.info(f"Fetched {out['n_tiles']} tiles")
    logger.info(f"Mosaic (clipped to AOI): {out['mosaic_file']}")

    return out['mosaic_file']