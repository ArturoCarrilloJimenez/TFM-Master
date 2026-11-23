"""Tests unitarios para el helper transversal de caché Parquet/R2 (src/shared/parquet_cache.py)."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from infrastructure.compaction.parquet import ParquetCompactor
from shared.parquet_cache import (
    cargar_parquet_cache,
    cargar_parquet_local_o_r2,
    guardar_parquet_cache,
    guardar_parquet_local_y_r2,
)


@pytest.mark.asyncio
async def test_cargar_parquet_local_o_r2_fallback_y_copia_local(tmp_path):
    """Si falta en disco local, descarga de R2 y deja copia local para ejecuciones posteriores."""
    compactor = ParquetCompactor()
    sample_rows = [{"cat_hash": "h1", "category": "earnings", "is_ruido": False}]
    parquet_bytes = compactor.compact(sample_rows)

    local_path = tmp_path / "subdir" / "ckpt.parquet"
    mock_repo = MagicMock()
    mock_repo.get = AsyncMock(return_value=parquet_bytes)
    mock_repo.close = AsyncMock()

    with patch("shared.parquet_cache.get_compactor", return_value=compactor), \
         patch("shared.parquet_cache.get_storage_repository", return_value=mock_repo):
        res_r2 = await cargar_parquet_local_o_r2(local_path, "processed/ckpt.parquet", nombre="checkpoint")
        assert res_r2 == sample_rows
        assert local_path.exists()
        mock_repo.get.assert_awaited_once_with("processed/ckpt.parquet")
        mock_repo.close.assert_awaited_once()

        # Segunda llamada: debe leer directamente de disco sin consultar R2
        mock_repo.get.reset_mock()
        res_local = await cargar_parquet_local_o_r2(local_path, "processed/ckpt.parquet", nombre="checkpoint")
        assert res_local == sample_rows
        mock_repo.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_cargar_y_guardar_parquet_cache(tmp_path):
    """Verifica guardado ZSTD dual (local + R2) y recarga indexada por columna hash."""
    compactor = ParquetCompactor()
    local_cache = tmp_path / "cache.parquet"
    mock_repo = MagicMock()
    mock_repo.save = AsyncMock()
    mock_repo.close = AsyncMock()

    cache_in = {"hash_a": {"input_hash": "hash_a", "sentiment_label": 1}}
    with patch("shared.parquet_cache.get_compactor", return_value=compactor), \
         patch("shared.parquet_cache.get_storage_repository", return_value=mock_repo):
        await guardar_parquet_cache(cache_in, local_cache, "processed/sent_cache.parquet", nombre="sentimiento")
        assert local_cache.exists()
        mock_repo.save.assert_awaited_once()

        loaded = await cargar_parquet_cache("input_hash", local_cache, "processed/sent_cache.parquet")
        assert loaded["hash_a"]["sentiment_label"] == 1


@pytest.mark.asyncio
async def test_guardar_parquet_local_y_r2_cierra_repo_ante_error(tmp_path):
    """Asegura cierre limpio del repositorio aunque falle la subida a R2."""
    compactor = ParquetCompactor()
    mock_repo = MagicMock()
    mock_repo.save = AsyncMock(side_effect=RuntimeError("R2 timeout"))
    mock_repo.close = AsyncMock()

    with patch("shared.parquet_cache.get_compactor", return_value=compactor), \
         patch("shared.parquet_cache.get_storage_repository", return_value=mock_repo):
        with pytest.raises(RuntimeError, match="R2 timeout"):
            await guardar_parquet_local_y_r2([{"a": 1}], tmp_path / "out.parquet", "k.parquet")

    mock_repo.close.assert_awaited_once()
