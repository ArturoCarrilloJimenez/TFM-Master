"""ETL independiente de Categorización Semántica (Paso 1 de Silver: Bronze -> Relevantes + Caché)."""

import argparse
import asyncio
from pathlib import Path
from typing import Any

from articulos.models.categorizador import CategorizadorArticulos
from dependencies.dependencies import get_compactor, get_storage_repository
from shared.ids import generar_categorizer_hash, generar_record_id
from shared.logging import Logger
from shared.parquet_cache import (
    cargar_parquet_cache,
    cargar_parquet_local_o_r2,
    guardar_parquet_cache,
    guardar_parquet_local_y_r2,
)

logger = Logger("ETLCategorizadorArticulos")

CATEGORIZER_VERSION: str = "v1"

DEFAULT_INPUT_S3_KEY: str = "raw/huggingface/multisource_bancos_2010_2023.parquet"
DEFAULT_CAT_CACHE_S3_KEY: str = "processed/huggingface/categorizer_cache.parquet"
DEFAULT_CATEGORIZED_S3_KEY: str = "processed/huggingface/categorized_checkpoint.parquet"

DEFAULT_LOCAL_CAT_CACHE_PATH: Path = Path("data/silver_categorizer_cache.parquet")
DEFAULT_CHECKPOINT_PATH: Path = Path("data/silver_categorized_checkpoint.parquet")


def asegurar_record_ids(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Garantiza que todo registro posea un `record_id` determinista antes de categorizar."""
    out: list[dict[str, Any]] = []
    for row in records:
        if row.get("record_id"):
            out.append(row)
        else:
            r = dict(row)
            r["record_id"] = generar_record_id(
                str(r.get("ticker", "")),
                str(r.get("date", "")),
                str(r.get("title", "")),
            )
            out.append(r)
    return out


async def cargar_cache_categorizador(
    local_cache_path: Path = DEFAULT_LOCAL_CAT_CACHE_PATH,
    cache_s3_key: str = DEFAULT_CAT_CACHE_S3_KEY,
) -> dict[str, dict[str, Any]]:
    """Carga la caché inmutable del categorizador por `cat_hash` desde disco local o R2."""
    return await cargar_parquet_cache("cat_hash", local_cache_path, cache_s3_key)


async def guardar_cache_categorizador(
    cache_map: dict[str, dict[str, Any]],
    local_cache_path: Path = DEFAULT_LOCAL_CAT_CACHE_PATH,
    cache_s3_key: str = DEFAULT_CAT_CACHE_S3_KEY,
) -> None:
    """Persiste la caché inmutable del categorizador en Parquet ZSTD local y en R2."""
    await guardar_parquet_cache(cache_map, local_cache_path, cache_s3_key, "categorizador")


async def cargar_relevantes_checkpoint(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    categorized_s3_key: str = DEFAULT_CATEGORIZED_S3_KEY,
) -> list[dict[str, Any]] | None:
    """Recupera el dataset categorizado relevante desde disco local o desde el rollback en R2."""
    return await cargar_parquet_local_o_r2(
        checkpoint_path,
        categorized_s3_key,
        nombre="checkpoint categorizado",
    )


def categorizar_relevantes_realtime(
    records: list[dict[str, Any]],
    categorizador: CategorizadorArticulos | None = None,
    batch_size: int = 128,
) -> list[dict[str, Any]]:
    """Categoriza un lote en memoria (AWS Lambda / tiempo real) y filtra los registros de ruido."""
    con_id = asegurar_record_ids(records)
    model = categorizador or CategorizadorArticulos()
    categorizados = model.clasificar_dataset(con_id, batch_size=batch_size)
    return [row for row in categorizados if not bool(row.get("is_ruido", False))]


async def ejecutar_categorizacion(
    input_s3_key: str = DEFAULT_INPUT_S3_KEY,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    categorized_s3_key: str = DEFAULT_CATEGORIZED_S3_KEY,
    cat_cache_path: str | Path = DEFAULT_LOCAL_CAT_CACHE_PATH,
    cat_cache_s3_key: str = DEFAULT_CAT_CACHE_S3_KEY,
    categorizador_batch_size: int = 128,
    usar_checkpoint: bool = False,
    force_recategorize: bool = False,
) -> list[dict[str, Any]]:
    """Ejecuta la categorización incremental de Bronze, filtra ruido y guarda checkpoint local + R2."""
    ckpt_path = Path(checkpoint_path)
    cache_path = Path(cat_cache_path)

    if usar_checkpoint and not force_recategorize:
        cached_relevantes = await cargar_relevantes_checkpoint(ckpt_path, categorized_s3_key)
        if cached_relevantes is not None:
            return cached_relevantes

    cat_cache = {} if force_recategorize else await cargar_cache_categorizador(cache_path, cat_cache_s3_key)

    compact = get_compactor()
    repo = get_storage_repository()
    logger.info(f"Descargando Bronze del Data Lake: [{repo.bucket_name}/{input_s3_key}]...")
    try:
        raw_bytes = await repo.get(input_s3_key)
    finally:
        await repo.close()

    records = asegurar_record_ids(compact.decompact(raw_bytes))
    row_cat_hashes: list[str] = []
    pendientes_por_hash: dict[str, dict[str, Any]] = {}
    for row in records:
        ch = generar_categorizer_hash(
            ticker=str(row.get("ticker", "")),
            title=str(row.get("title", "")),
            article=str(row.get("article") or ""),
            ticker_count=int(row.get("ticker_count") or 1),
            is_multisubject=bool(row.get("is_multisubject", False)),
            version=CATEGORIZER_VERSION,
        )
        row_cat_hashes.append(ch)
        if ch not in cat_cache and ch not in pendientes_por_hash:
            pendientes_por_hash[ch] = row

    logger.info(
        f"Categorizador Bronze ({len(records):,} filas) -> En caché ($0 CPU): {len(records) - len(pendientes_por_hash):,} | "
        f"Pendientes únicos CPU: {len(pendientes_por_hash):,}"
    )

    if pendientes_por_hash:
        model = CategorizadorArticulos()
        pend_hashes = list(pendientes_por_hash.keys())
        pend_rows = list(pendientes_por_hash.values())
        clasificados = model.clasificar_dataset(pend_rows, batch_size=categorizador_batch_size)
        for ch, res in zip(pend_hashes, clasificados):
            cat_cache[ch] = {
                "cat_hash": ch,
                "process_text": res.get("process_text"),
                "category": res.get("category"),
                "confianza": res.get("confianza"),
                "is_ruido": bool(res.get("is_ruido", False)),
            }
        await guardar_cache_categorizador(cat_cache, cache_path, cat_cache_s3_key)

    relevantes: list[dict[str, Any]] = []
    for row, ch in zip(records, row_cat_hashes):
        cat_info = cat_cache[ch]
        if not bool(cat_info.get("is_ruido", False)):
            merged = dict(row)
            merged["process_text"] = cat_info["process_text"]
            merged["category"] = cat_info["category"]
            merged["confianza"] = cat_info["confianza"]
            merged["is_ruido"] = False
            relevantes.append(merged)

    logger.info(
        f"Poda de ruido completada: {len(relevantes):,} relevantes conservados "
        f"({len(records) - len(relevantes):,} descartados por is_ruido=True)."
    )

    await guardar_parquet_local_y_r2(
        relevantes,
        ckpt_path,
        categorized_s3_key,
        nombre="checkpoint categorizado",
    )
    return relevantes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETL Categorizador Semántico (Bronze -> Relevantes + R2).")
    parser.add_argument("--input-s3-key", type=str, default=DEFAULT_INPUT_S3_KEY)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--use-checkpoint", action="store_true", help="Reutiliza checkpoint local o de R2.")
    parser.add_argument("--force-recategorize", action="store_true", help="Ignora caché del categorizador.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(
        ejecutar_categorizacion(
            input_s3_key=args.input_s3_key,
            categorizador_batch_size=args.batch_size,
            usar_checkpoint=args.use_checkpoint,
            force_recategorize=args.force_recategorize,
        )
    )
