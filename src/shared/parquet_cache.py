"""Utilidades compartidas de caché y rollback en Parquet ZSTD (disco local + Cloudflare R2)."""

from pathlib import Path
from typing import Any

from dependencies.dependencies import get_compactor, get_storage_repository
from shared.logging import Logger

logger = Logger("ParquetCache")


async def cargar_parquet_local_o_r2(
    local_path: Path,
    s3_key: str,
    nombre: str = "dataset",
) -> list[dict[str, Any]] | None:
    """Carga un Parquet ZSTD desde disco local o hace fallback a Cloudflare R2.

    Si el archivo existe en disco local se lee directamente. En caso contrario, intenta
    descargarlo de R2, lo replica en `local_path` para reintentos futuros y lo descompacta.
    """
    compact = get_compactor()
    if local_path.exists():
        logger.info(f"Reutilizando {nombre} local: [{local_path}]")
        return compact.decompact(local_path.read_bytes())

    repo = get_storage_repository()
    try:
        raw_bytes = await repo.get(s3_key)
        rows = compact.decompact(raw_bytes)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(raw_bytes)
        logger.info(f"{nombre.capitalize()} recuperado de R2 [{s3_key}] y guardado localmente en [{local_path}].")
        return rows
    except Exception:
        return None
    finally:
        await repo.close()


async def guardar_parquet_local_y_r2(
    rows: list[dict[str, Any]],
    local_path: Path,
    s3_key: str,
    nombre: str = "dataset",
) -> bytes:
    """Compacta registros en Parquet ZSTD, guarda copia local y sincroniza con Cloudflare R2."""
    compact = get_compactor()
    parquet_bytes = compact.compact(rows)

    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(parquet_bytes)
    logger.info(f"{nombre.capitalize()} guardado en disco: [{local_path}] ({len(parquet_bytes):,} bytes).")

    repo = get_storage_repository()
    try:
        await repo.save(s3_key, parquet_bytes)
        logger.info(f"{nombre.capitalize()} sincronizado en R2 [{s3_key}] ({len(rows):,} filas).")
    finally:
        await repo.close()

    return parquet_bytes


async def cargar_parquet_cache(
    key_col: str,
    local_path: Path,
    s3_key: str,
) -> dict[str, dict[str, Any]]:
    """Carga una tabla de caché indexada por `key_col` desde disco local o R2."""
    rows = await cargar_parquet_local_o_r2(local_path, s3_key, nombre=f"caché {key_col}")
    if not rows:
        return {}
    return {str(r[key_col]): r for r in rows if r.get(key_col)}


async def guardar_parquet_cache(
    cache_map: dict[str, dict[str, Any]],
    local_path: Path,
    s3_key: str,
    nombre: str = "caché",
) -> None:
    """Persiste un diccionario de caché en Parquet ZSTD local y en Cloudflare R2."""
    if not cache_map:
        return
    await guardar_parquet_local_y_r2(list(cache_map.values()), local_path, s3_key, nombre=f"caché {nombre}")
