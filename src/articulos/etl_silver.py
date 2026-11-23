"""Orquestador ETL Silver: encadena Categorización (`etl_categorizador`) + Sentimiento (`etl_sentimientos`)."""

import argparse
import asyncio
from pathlib import Path
import time
from typing import Any

from articulos.etl_categorizador import (
    DEFAULT_CAT_CACHE_S3_KEY,
    DEFAULT_CATEGORIZED_S3_KEY,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_INPUT_S3_KEY,
    DEFAULT_LOCAL_CAT_CACHE_PATH,
    categorizar_relevantes_realtime,
    ejecutar_categorizacion,
)
from articulos.etl_sentimientos import (
    DEFAULT_BATCH_STATE_PATH,
    DEFAULT_CACHE_S3_KEY,
    DEFAULT_LOCAL_CACHE_PATH,
    analizar_sentimiento_realtime,
    cargar_cache_sentimiento,
    ejecutar_sentimiento_batch,
    guardar_cache_sentimiento,
)
from articulos.models.categorizador import CategorizadorArticulos, is_running_in_lambda
from articulos.models.sentiment import AnthropicSentimentEngine
from config.settings import settings
from dependencies.dependencies import get_compactor, get_storage_repository
from shared.logging import Logger
from shared.parquet_cache import guardar_parquet_local_y_r2

logger = Logger("ETLSilverArticulos")

# Poda Silver: elimina textos crudos redundantes y `article` extenso tras extraer `process_text` e `input_hash`
COLUMNAS_DESCARTAR_SILVER: set[str] = {"text", "publisher", "url", "source_dataset", "article"}

DEFAULT_OUTPUT_S3_KEY: str = "processed/huggingface/multisource_bancos_2010_2023.parquet"
DEFAULT_LOCAL_OUTPUT_PATH: Path = Path("data/processed_multisource_bancos_2010_2023.parquet")


def resolver_modo_ejecucion(mode: str = "auto") -> str:
    """Determina automáticamente si ejecutar en modo 'realtime' (AWS Lambda) o 'batch' (histórico)."""
    if mode in {"realtime", "batch"}:
        return mode
    return "realtime" if is_running_in_lambda() else "batch"


def podar_y_unir_silver(
    relevantes: list[dict[str, Any]],
    row_hashes: list[str],
    cache_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Adjunta `input_hash` y columnas de sentimiento Claude y poda campos crudos para Silver."""
    silver_rows: list[dict[str, Any]] = []
    for row, h in zip(relevantes, row_hashes):
        sent = cache_map.get(h, {})
        base = {k: v for k, v in row.items() if k not in COLUMNAS_DESCARTAR_SILVER}
        base["input_hash"] = h
        base["sentiment_label"] = sent.get("sentiment_label")
        base["sentiment_score"] = sent.get("sentiment_score")
        base["p_pos"] = sent.get("p_pos")
        base["p_neg"] = sent.get("p_neg")
        base["p_neu"] = sent.get("p_neu")
        base["sentiment_motivo"] = sent.get("sentiment_motivo")
        base["sentiment_model"] = sent.get("sentiment_model")
        base["sentiment_prompt_ver"] = sent.get("sentiment_prompt_ver")
        silver_rows.append(base)
    return silver_rows


def transformar_a_silver_realtime(
    records: list[dict[str, Any]],
    categorizador: CategorizadorArticulos | None = None,
    engine: AnthropicSentimentEngine | None = None,
    cache_map: dict[str, dict[str, Any]] | None = None,
    categorizador_batch_size: int = 128,
    sentiment_batch_size: int = settings.anthropic_batch_size,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Ejecuta el pipeline Silver completo en tiempo real (AWS Lambda / lotes cortos)."""
    cache_actual = dict(cache_map or {})
    relevantes = categorizar_relevantes_realtime(
        records,
        categorizador=categorizador,
        batch_size=categorizador_batch_size,
    )
    if not relevantes:
        return [], cache_actual

    row_hashes, cache_actual = analizar_sentimiento_realtime(
        relevantes=relevantes,
        engine=engine,
        cache_map=cache_actual,
        sentiment_batch_size=sentiment_batch_size,
    )
    return podar_y_unir_silver(relevantes, row_hashes, cache_actual), cache_actual


async def persistir_silver_final(
    silver_rows: list[dict[str, Any]],
    output_s3_key: str,
    local_backup_path: Path,
) -> None:
    """Compacta la capa Silver final y la persiste en disco local y Cloudflare R2."""
    await guardar_parquet_local_y_r2(silver_rows, local_backup_path, output_s3_key, nombre="Silver")


async def process_data(
    input_s3_key: str = DEFAULT_INPUT_S3_KEY,
    output_s3_key: str = DEFAULT_OUTPUT_S3_KEY,
    local_backup_path: str | Path = DEFAULT_LOCAL_OUTPUT_PATH,
    cache_s3_key: str = DEFAULT_CACHE_S3_KEY,
    local_cache_path: str | Path = DEFAULT_LOCAL_CACHE_PATH,
    cat_cache_s3_key: str = DEFAULT_CAT_CACHE_S3_KEY,
    local_cat_cache_path: str | Path = DEFAULT_LOCAL_CAT_CACHE_PATH,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    categorized_s3_key: str = DEFAULT_CATEGORIZED_S3_KEY,
    batch_state_path: str | Path = DEFAULT_BATCH_STATE_PATH,
    batch_size: int = 128,
    sentiment_batch_size: int = settings.anthropic_batch_size,
    mode: str = "auto",
    step: str = "all",
    wait: bool = True,
    dry_run: bool = False,
    use_checkpoint: bool = False,
    force_recategorize: bool = False,
    poll_interval_s: int = 30,
    batch_id: str | None = None,
) -> dict[str, Any]:
    """Orquesta el pipeline Silver completo (Paso 1 Categorización -> Paso 2 Sentimiento -> Silver R2)."""
    t_start = time.time()
    modo_real = resolver_modo_ejecucion(mode)
    backup_path = Path(local_backup_path)
    cache_path = Path(local_cache_path)

    cache_map = await cargar_cache_sentimiento(cache_path, cache_s3_key)
    logger.info(f"Modo detectado: [{modo_real}] | Step: [{step}] | Hashes sentimiento en caché: {len(cache_map):,}")

    if modo_real == "realtime":
        repo = get_storage_repository()
        try:
            raw_bytes = await repo.get(input_s3_key)
        finally:
            await repo.close()
        records = get_compactor().decompact(raw_bytes)
        silver_rows, cache_map = transformar_a_silver_realtime(
            records=records,
            cache_map=cache_map,
            categorizador_batch_size=batch_size,
            sentiment_batch_size=sentiment_batch_size,
        )
        await guardar_cache_sentimiento(cache_map, cache_path, cache_s3_key)
        await persistir_silver_final(silver_rows, output_s3_key, backup_path)
        return {"mode": modo_real, "rows": len(silver_rows), "elapsed_s": round(time.time() - t_start, 2)}

    # --- MODO BATCH HISTÓRICO: Paso 1 (Categorizador) + Paso 2 (Sentimiento) ---
    usar_ckpt = use_checkpoint or step == "collect"
    relevantes = await ejecutar_categorizacion(
        input_s3_key=input_s3_key,
        checkpoint_path=checkpoint_path,
        categorized_s3_key=categorized_s3_key,
        cat_cache_path=local_cat_cache_path,
        cat_cache_s3_key=cat_cache_s3_key,
        categorizador_batch_size=batch_size,
        usar_checkpoint=usar_ckpt,
        force_recategorize=force_recategorize,
    )

    sent_res = await ejecutar_sentimiento_batch(
        relevantes=relevantes,
        cache_map=cache_map,
        sentiment_batch_size=sentiment_batch_size,
        step=step,
        dry_run=dry_run,
        wait=wait,
        poll_interval_s=poll_interval_s,
        batch_id=batch_id,
        cache_s3_key=cache_s3_key,
        local_cache_path=cache_path,
        batch_state_path=batch_state_path,
    )

    if sent_res["status"] != "completed":
        return {k: v for k, v in sent_res.items() if k not in {"row_hashes", "cache_map", "status"}}

    silver_rows = podar_y_unir_silver(relevantes, sent_res["row_hashes"], sent_res["cache_map"])
    await persistir_silver_final(silver_rows, output_s3_key, backup_path)
    logger.info(f"Pipeline Silver finalizado en {time.time() - t_start:.1f} s.")
    return {"mode": "batch", "step": step, "rows": len(silver_rows), "batch_id": sent_res.get("batch_id")}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETL Silver Unificado: Categorización + Sentimiento Claude.")
    parser.add_argument("--step", choices=["all", "prepare", "submit", "collect"], default="all")
    parser.add_argument("--mode", choices=["auto", "batch", "realtime"], default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Calcula hashes y coste sin llamar a Anthropic.")
    parser.add_argument("--no-wait", action="store_true", help="Envía el Batch y sale sin esperar a collect.")
    parser.add_argument("--use-checkpoint", action="store_true", help="Reutiliza checkpoint categorizado local/R2.")
    parser.add_argument("--force-recategorize", action="store_true", help="Ignora caché del categorizador.")
    parser.add_argument("--poll-interval", type=int, default=30, help="Segundos entre consultas de estado Batch.")
    parser.add_argument("--batch-id", type=str, default=None, help="batch_id manual para --step collect.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(
        process_data(
            step=args.step,
            mode=args.mode,
            dry_run=args.dry_run,
            wait=not args.no_wait,
            use_checkpoint=args.use_checkpoint,
            force_recategorize=args.force_recategorize,
            poll_interval_s=args.poll_interval,
            batch_id=args.batch_id,
        )
    )
