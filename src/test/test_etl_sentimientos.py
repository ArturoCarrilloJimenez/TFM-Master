"""Tests unitarios para el ETL independiente de Sentimiento (src/articulos/etl_sentimientos.py)."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from articulos.etl_sentimientos import (
    _run_standalone,
    analizar_sentimiento_realtime,
    ejecutar_sentimiento_batch,
    estimar_coste_batch_usd,
)
from articulos.models.sentiment import preparar_entradas_sentimiento
from infrastructure.compaction.parquet import ParquetCompactor


def test_estimar_coste_y_realtime_con_cache():
    """Verifica cálculo de coste estimado Batch y reutilización de caché en modo realtime."""
    assert estimar_coste_batch_usd(1000) > 0
    relevantes = [
        {
            "ticker": "JPM",
            "title": "JPMorgan Q1 profit jumps",
            "article": "JPMorgan Chase beat consensus.",
            "process_text": "JPMorgan Chase beat consensus.",
        }
    ]
    _, unicas = preparar_entradas_sentimiento(relevantes)
    h = next(iter(unicas.keys()))

    mock_engine = MagicMock()
    mock_engine.model = "claude-haiku-5-5"
    mock_engine.prompt_version = "v1_en"
    mock_engine.clasificar_realtime.return_value = {
        h: {"input_hash": h, "sentiment_label": 1, "sentiment_score": 0.9}
    }

    hashes, cache = analizar_sentimiento_realtime(relevantes, engine=mock_engine, cache_map={})
    assert hashes == [h]
    assert cache[h]["sentiment_label"] == 1
    mock_engine.clasificar_realtime.assert_called_once()

    # Segunda pasada con caché llena: 0 llamadas LLM
    mock_engine.clasificar_realtime.reset_mock()
    _, cache2 = analizar_sentimiento_realtime(relevantes, engine=mock_engine, cache_map=cache)
    mock_engine.clasificar_realtime.assert_not_called()
    assert cache2[h]["sentiment_score"] == 0.9


@pytest.mark.asyncio
async def test_ejecutar_sentimiento_batch_dry_run_y_submit_collect(tmp_path):
    """Valida dry-run sin gasto y flujo submit -> collect guardando sentiment_cache."""
    compactor = ParquetCompactor()
    relevantes = [
        {
            "record_id": "abc123456789012345678901",
            "ticker": "GS",
            "title": "Goldman Sachs profit jumps",
            "article": "Goldman Sachs beat estimates.",
            "process_text": "Goldman Sachs beat estimates.",
        }
    ]
    _, unicas = preparar_entradas_sentimiento(relevantes)
    h = next(iter(unicas.keys()))

    mock_repo = MagicMock()
    mock_repo.get = AsyncMock(side_effect=RuntimeError("no sent cache"))
    mock_repo.save = AsyncMock()
    mock_repo.close = AsyncMock()

    mock_engine = MagicMock()
    mock_engine.model = "claude-haiku-5-5"
    mock_engine.prompt_version = "v1_en"
    mock_engine.construir_batch_requests.return_value = ([{"custom_id": "chunk_00000", "params": {}}], {"chunk_00000": [h]})
    mock_engine.enviar_batch.return_value = {"id": "msgbatch_sent_1"}
    mock_engine.consultar_batch.return_value = {"processing_status": "ended", "results_url": "https://mock/res"}
    mock_engine.descargar_resultados_batch.return_value = (
        {h: {"input_hash": h, "sentiment_label": 1, "sentiment_score": 0.8}},
        1,
        0,
    )

    cache_path = tmp_path / "sent_cache.parquet"
    state_path = tmp_path / "batch_state.json"

    with patch("shared.parquet_cache.get_compactor", return_value=compactor), \
         patch("shared.parquet_cache.get_storage_repository", return_value=mock_repo):
        res_dry = await ejecutar_sentimiento_batch(
            relevantes=relevantes,
            engine=mock_engine,
            dry_run=True,
            local_cache_path=cache_path,
            batch_state_path=state_path,
        )
        assert res_dry["status"] == "prepared"
        assert res_dry["pendientes"] == 1
        mock_engine.enviar_batch.assert_not_called()

        res_ok = await ejecutar_sentimiento_batch(
            relevantes=relevantes,
            engine=mock_engine,
            step="all",
            wait=True,
            poll_interval_s=0,
            local_cache_path=cache_path,
            batch_state_path=state_path,
        )

    assert res_ok["status"] == "completed"
    assert res_ok["batch_id"] == "msgbatch_sent_1"
    assert cache_path.exists()
    assert state_path.exists()
    mock_repo.save.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_standalone_falla_si_no_hay_checkpoint(tmp_path):
    """Exige checkpoint categorizado previo cuando se ejecuta etl_sentimientos de forma aislada."""
    with patch("articulos.etl_sentimientos.cargar_relevantes_checkpoint", AsyncMock(return_value=None)):
        with pytest.raises(FileNotFoundError, match="etl_categorizador.py"):
            await _run_standalone(checkpoint_path=tmp_path / "missing.parquet", dry_run=True)
