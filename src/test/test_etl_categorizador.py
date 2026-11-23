"""Tests unitarios para el ETL independiente de Categorización (src/articulos/etl_categorizador.py)."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from articulos.etl_categorizador import (
    asegurar_record_ids,
    categorizar_relevantes_realtime,
    ejecutar_categorizacion,
)
from infrastructure.compaction.parquet import ParquetCompactor


def test_asegurar_record_ids_y_realtime_poda_ruido():
    """Asigna record_id determinista y descarta filas con is_ruido=True."""
    raw = [
        {"ticker": "JPM", "date": "2023-04-14", "title": "JPM Q1 beat", "article": "Strong quarter"},
        {"ticker": "BAC", "date": "2023-04-14", "title": "Noise", "article": "Cooking recipe"},
    ]
    con_id = asegurar_record_ids(raw)
    assert len(con_id[0]["record_id"]) == 24
    assert len(con_id[1]["record_id"]) == 24

    mock_cat = MagicMock()
    mock_cat.clasificar_dataset.side_effect = lambda recs, **kw: [
        {**recs[0], "process_text": "Strong quarter", "category": "earnings", "confianza": 85.0, "is_ruido": False},
        {**recs[1], "process_text": "Cooking recipe", "category": "ruido", "confianza": 12.0, "is_ruido": True},
    ]

    relevantes = categorizar_relevantes_realtime(raw, categorizador=mock_cat)
    assert len(relevantes) == 1
    assert relevantes[0]["ticker"] == "JPM"


@pytest.mark.asyncio
async def test_ejecutar_categorizacion_cache_y_checkpoint_r2(tmp_path):
    """Verifica deduplicación por cat_hash, checkpoint dual local+R2 y reutilización con usar_checkpoint."""
    compactor = ParquetCompactor()
    bronze_rows = [
        {
            "ticker": "C",
            "date": "2023-05-01",
            "title": "Citigroup expands wealth",
            "article": "Citigroup announced advisors.",
        },
        {
            "ticker": "MS",
            "date": "2023-05-01",
            "title": "Unrelated horoscope",
            "article": "Astrology weekly",
        },
    ]
    raw_bytes = compactor.compact(bronze_rows)

    mock_repo = MagicMock()
    mock_repo.bucket_name = "test-bucket"
    mock_repo.get = AsyncMock(side_effect=[RuntimeError("no cat cache"), raw_bytes])
    mock_repo.save = AsyncMock()
    mock_repo.close = AsyncMock()

    mock_cat = MagicMock()
    mock_cat.clasificar_dataset.side_effect = lambda recs, **kw: [
        {**recs[0], "process_text": "Citigroup announced advisors.", "category": "m&a", "confianza": 74.0, "is_ruido": False},
        {**recs[1], "process_text": "Astrology weekly", "category": "ruido", "confianza": 10.0, "is_ruido": True},
    ]

    cat_cache_path = tmp_path / "cat_cache.parquet"
    ckpt_path = tmp_path / "ckpt.parquet"

    with patch("articulos.etl_categorizador.get_compactor", return_value=compactor), \
         patch("articulos.etl_categorizador.get_storage_repository", return_value=mock_repo), \
         patch("shared.parquet_cache.get_compactor", return_value=compactor), \
         patch("shared.parquet_cache.get_storage_repository", return_value=mock_repo), \
         patch("articulos.etl_categorizador.CategorizadorArticulos", return_value=mock_cat):
        relevantes = await ejecutar_categorizacion(
            checkpoint_path=ckpt_path,
            cat_cache_path=cat_cache_path,
            usar_checkpoint=False,
        )

    assert len(relevantes) == 1
    assert relevantes[0]["ticker"] == "C"
    assert cat_cache_path.exists()
    assert ckpt_path.exists()
    # Se guarda cat_cache en R2 + checkpoint categorizado en R2
    assert mock_repo.save.await_count == 2

    # Borrar checkpoint local para verificar recuperación desde fallback R2 cuando usar_checkpoint=True
    saved_ckpt_bytes = ckpt_path.read_bytes()
    ckpt_path.unlink()
    mock_repo.get = AsyncMock(return_value=saved_ckpt_bytes)
    mock_cat.clasificar_dataset.reset_mock()

    with patch("shared.parquet_cache.get_compactor", return_value=compactor), \
         patch("shared.parquet_cache.get_storage_repository", return_value=mock_repo), \
         patch("articulos.etl_categorizador.CategorizadorArticulos", return_value=mock_cat) as mock_cls:
        relevantes_r2 = await ejecutar_categorizacion(
            checkpoint_path=ckpt_path,
            cat_cache_path=cat_cache_path,
            usar_checkpoint=True,
        )

    assert len(relevantes_r2) == 1
    assert ckpt_path.exists()
    mock_cls.assert_not_called()
