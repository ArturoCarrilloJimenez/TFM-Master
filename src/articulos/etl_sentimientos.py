"""ETL independiente de Sentimiento LLM (Paso 2 de Silver: Relevantes -> Caché Sentimiento)."""

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any

from articulos.etl_categorizador import (
    DEFAULT_CATEGORIZED_S3_KEY,
    DEFAULT_CHECKPOINT_PATH,
    cargar_relevantes_checkpoint,
)
from articulos.models.sentiment import AnthropicSentimentEngine, preparar_entradas_sentimiento
from config.settings import settings
from shared.logging import Logger
from shared.parquet_cache import cargar_parquet_cache, guardar_parquet_cache

logger = Logger("ETLSentimientosArticulos")

DEFAULT_CACHE_S3_KEY: str = "processed/huggingface/sentiment_cache.parquet"
DEFAULT_LOCAL_CACHE_PATH: Path = Path("data/silver_sentiment_cache.parquet")
DEFAULT_BATCH_STATE_PATH: Path = Path("data/anthropic_batch_state.json")

# Tarifas oficiales Claude Haiku 5.5 con descuento Batch API -50% (USD por 1M tokens)
BATCH_IN_USD_1M: float = 0.05
BATCH_OUT_USD_1M: float = 0.25


def estimar_coste_batch_usd(n_items: int, avg_in_tok: float = 310.0, avg_out_tok: float = 165.0) -> float:
    """Estima el coste USD en modo Anthropic Batch (-50%) según longitud media de prompt/respuesta."""
    return ((n_items * avg_in_tok) / 1e6) * BATCH_IN_USD_1M + ((n_items * avg_out_tok) / 1e6) * BATCH_OUT_USD_1M


async def cargar_cache_sentimiento(
    local_cache_path: Path = DEFAULT_LOCAL_CACHE_PATH,
    cache_s3_key: str = DEFAULT_CACHE_S3_KEY,
) -> dict[str, dict[str, Any]]:
    """Carga la caché inmutable de sentimiento por `input_hash` desde disco local o R2."""
    return await cargar_parquet_cache("input_hash", local_cache_path, cache_s3_key)


async def guardar_cache_sentimiento(
    cache_map: dict[str, dict[str, Any]],
    local_cache_path: Path = DEFAULT_LOCAL_CACHE_PATH,
    cache_s3_key: str = DEFAULT_CACHE_S3_KEY,
) -> None:
    """Persiste la caché inmutable de sentimientos en Parquet ZSTD local y en R2."""
    await guardar_parquet_cache(cache_map, local_cache_path, cache_s3_key, "sentimiento")


def analizar_sentimiento_realtime(
    relevantes: list[dict[str, Any]],
    engine: AnthropicSentimentEngine | None = None,
    cache_map: dict[str, dict[str, Any]] | None = None,
    sentiment_batch_size: int = settings.anthropic_batch_size,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Ejecuta inferencia de sentimiento síncrona (AWS Lambda) con caché incremental por `input_hash`."""
    cache_actual = dict(cache_map or {})
    llm = engine or AnthropicSentimentEngine()
    row_hashes, unicas = preparar_entradas_sentimiento(
        relevantes, model=llm.model, prompt_version=llm.prompt_version
    )
    pendientes = [e for h, e in unicas.items() if h not in cache_actual]
    if pendientes:
        nuevos = llm.clasificar_realtime(pendientes, batch_size=sentiment_batch_size)
        cache_actual.update(nuevos)
    return row_hashes, cache_actual


async def ejecutar_sentimiento_batch(
    relevantes: list[dict[str, Any]],
    cache_map: dict[str, dict[str, Any]] | None = None,
    engine: AnthropicSentimentEngine | None = None,
    sentiment_batch_size: int = settings.anthropic_batch_size,
    step: str = "all",
    dry_run: bool = False,
    wait: bool = True,
    poll_interval_s: int = 30,
    batch_id: str | None = None,
    cache_s3_key: str = DEFAULT_CACHE_S3_KEY,
    local_cache_path: str | Path = DEFAULT_LOCAL_CACHE_PATH,
    batch_state_path: str | Path = DEFAULT_BATCH_STATE_PATH,
) -> dict[str, Any]:
    """Ejecuta el flujo Batch API (-50%) de sentimiento (`prepare`, `submit`, `collect`) sobre relevantes."""
    cache_path = Path(local_cache_path)
    state_path = Path(batch_state_path)
    cache_actual = (
        dict(cache_map) if cache_map is not None else await cargar_cache_sentimiento(cache_path, cache_s3_key)
    )

    llm = engine or AnthropicSentimentEngine()
    row_hashes, unicas = preparar_entradas_sentimiento(
        relevantes, model=llm.model, prompt_version=llm.prompt_version
    )
    pendientes = [e for h, e in unicas.items() if h not in cache_actual]
    requests_list, chunk_map = llm.construir_batch_requests(pendientes, batch_size=sentiment_batch_size)
    est_usd = estimar_coste_batch_usd(len(pendientes))

    logger.info(
        f"Resumen Sentimiento LLM -> Relevantes: {len(relevantes):,} | Únicos por texto: {len(unicas):,} | "
        f"En caché ($0): {len(unicas) - len(pendientes):,} | Pendientes API: {len(pendientes):,} "
        f"({len(requests_list):,} bloques de {sentiment_batch_size}) | Coste Batch est.: ~${est_usd:.2f}"
    )

    if dry_run or step == "prepare":
        return {
            "status": "prepared",
            "mode": "batch",
            "step": "prepare",
            "relevantes": len(relevantes),
            "unicos": len(unicas),
            "pendientes": len(pendientes),
            "lotes_batch": len(requests_list),
            "coste_estimado_usd": round(est_usd, 4),
            "row_hashes": row_hashes,
            "cache_map": cache_actual,
            "batch_id": batch_id,
        }

    active_batch_id = batch_id
    if step in {"all", "submit"} and pendientes:
        resp_batch = llm.enviar_batch(requests_list)
        active_batch_id = str(resp_batch["id"])
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(
                {
                    "batch_id": active_batch_id,
                    "model": llm.model,
                    "prompt_version": llm.prompt_version,
                    "total_requests": len(requests_list),
                    "chunk_map": chunk_map,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info(f"Batch enviado a Anthropic -> batch_id=[{active_batch_id}] ({len(requests_list):,} chunks).")

        if step == "submit" or not wait:
            return {
                "status": "submitted",
                "mode": "batch",
                "step": "submit",
                "batch_id": active_batch_id,
                "lotes": len(requests_list),
                "row_hashes": row_hashes,
                "cache_map": cache_actual,
            }

    chunks_ok = 0
    chunks_err = 0
    if step in {"all", "collect"} and pendientes:
        if not state_path.exists():
            raise FileNotFoundError(f"No existe estado de Batch en {state_path}. Ejecuta primero --step submit.")
        state_data = json.loads(state_path.read_text(encoding="utf-8"))
        raw_batch_id = active_batch_id or state_data["batch_id"]
        active_batch_id = re.sub(
            r"[\x00-\x1f\x7f]",
            "",
            str(raw_batch_id).replace("\r", "").replace("\n", ""),
        )
        saved_chunk_map: dict[str, list[str]] = state_data["chunk_map"]

        status_obj = llm.consultar_batch(active_batch_id)
        while status_obj.get("processing_status") != "ended":
            counts = status_obj.get("request_counts", {})
            logger.info(
                f"Batch [{active_batch_id}] estado={status_obj.get('processing_status')} | "
                f"succeeded={counts.get('succeeded', 0)} processing={counts.get('processing', 0)} "
                f"errored={counts.get('errored', 0)}"
            )
            if not wait:
                return {
                    "status": "in_progress",
                    "mode": "batch",
                    "step": "collect",
                    "batch_id": active_batch_id,
                    "row_hashes": row_hashes,
                    "cache_map": cache_actual,
                }
            await asyncio.sleep(poll_interval_s)
            status_obj = llm.consultar_batch(active_batch_id)

        results_url = status_obj.get("results_url")
        if not results_url:
            raise RuntimeError(f"Batch [{active_batch_id}] finalizado sin results_url: {status_obj}")

        nuevos, chunks_ok, chunks_err = llm.descargar_resultados_batch(results_url, saved_chunk_map)
        cache_actual.update(nuevos)
        await guardar_cache_sentimiento(cache_actual, cache_path, cache_s3_key)
        logger.info(
            f"Batch [{active_batch_id}] recolectado -> chunks OK={chunks_ok}, err={chunks_err}, "
            f"nuevos hashes guardados={len(nuevos):,}."
        )
        if chunks_err > 0:
            logger.warning(
                f"Quedaron {chunks_err} chunks sin completar. Vuelve a ejecutar `etl_sentimientos.py --step all` "
                "para reintentar únicamente los hashes faltantes."
            )

    return {
        "status": "completed",
        "mode": "batch",
        "step": step,
        "batch_id": active_batch_id,
        "row_hashes": row_hashes,
        "cache_map": cache_actual,
        "chunks_ok": chunks_ok,
        "chunks_err": chunks_err,
    }


async def _run_standalone(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    categorized_s3_key: str = DEFAULT_CATEGORIZED_S3_KEY,
    step: str = "all",
    dry_run: bool = False,
    wait: bool = True,
    poll_interval_s: int = 30,
    batch_id: str | None = None,
) -> dict[str, Any]:
    relevantes = await cargar_relevantes_checkpoint(checkpoint_path, categorized_s3_key)
    if relevantes is None:
        raise FileNotFoundError(
            f"No se encontró checkpoint categorizado ni en {checkpoint_path} ni en R2 [{categorized_s3_key}]. "
            "Ejecuta primero `etl_categorizador.py`."
        )
    return await ejecutar_sentimiento_batch(
        relevantes=relevantes,
        step=step,
        dry_run=dry_run,
        wait=wait,
        poll_interval_s=poll_interval_s,
        batch_id=batch_id,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETL Sentimiento Claude sobre checkpoint categorizado.")
    parser.add_argument("--step", choices=["all", "prepare", "submit", "collect"], default="all")
    parser.add_argument("--dry-run", action="store_true", help="Calcula hashes y coste sin llamar a Anthropic.")
    parser.add_argument("--no-wait", action="store_true", help="Envía el Batch y sale sin esperar a collect.")
    parser.add_argument("--poll-interval", type=int, default=30, help="Segundos entre consultas de estado Batch.")
    parser.add_argument("--batch-id", type=str, default=None, help="batch_id manual para --step collect.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(
        _run_standalone(
            step=args.step,
            dry_run=args.dry_run,
            wait=not args.no_wait,
            poll_interval_s=args.poll_interval,
            batch_id=args.batch_id,
        )
    )
