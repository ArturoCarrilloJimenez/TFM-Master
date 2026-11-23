"""Tests de integración/orquestación para el pipeline ETL Silver (src/articulos/etl_silver.py)."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from articulos.etl_silver import (
    COLUMNAS_DESCARTAR_SILVER,
    process_data,
    resolver_modo_ejecucion,
    transformar_a_silver_realtime,
)
from articulos.models.sentiment import preparar_entradas_sentimiento
from infrastructure.compaction.parquet import ParquetCompactor


def test_resolver_modo_ejecucion_lambda_vs_local(monkeypatch):
    """Verifica cambio automático a realtime en AWS Lambda y batch en local."""
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
    monkeypatch.delenv("LAMBDA_TASK_ROOT", raising=False)
    assert resolver_modo_ejecucion("auto") == "batch"
    assert resolver_modo_ejecucion("realtime") == "realtime"

    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "tfm-daily-inference")
    assert resolver_modo_ejecucion("auto") == "realtime"


def test_transformar_a_silver_realtime_poda_ruido_y_enriquece():
    """Verifica poda de is_ruido=True, poda de columnas crudas, trazabilidad y sentimiento realtime."""
    raw_records = [
        {
            "ticker": "JPM",
            "date": "2023-04-14 08:00:00",
            "title": "JPMorgan Q1 profit surges on net interest income",
            "article": "JPMorgan Chase beat analyst estimates easily.",
            "text": "full raw text",
            "publisher": "Reuters",
            "url": "https://example.com/1",
            "source_dataset": "fnspid",
        },
        {
            "ticker": "BAC",
            "date": "2023-04-14 09:00:00",
            "title": "Random unrelated lifestyle piece",
            "article": "Nothing about banking here.",
            "text": "full noise text",
            "publisher": "Other",
            "url": "https://example.com/2",
            "source_dataset": "fnspid",
        },
    ]

    mock_cat = MagicMock()
    mock_cat.clasificar_dataset.side_effect = lambda recs, **kw: [
        {**recs[0], "process_text": "JPMorgan Chase beat analyst estimates easily.", "category": "earnings", "confianza": 82.0, "is_ruido": False},
        {**recs[1], "process_text": "Nothing about banking here.", "category": "ruido", "confianza": 15.0, "is_ruido": True},
    ]

    mock_engine = MagicMock()
    mock_engine.model = "claude-haiku-5-5"
    mock_engine.prompt_version = "v1_en"

    def fake_realtime(entradas, batch_size=10):
        return {
            e.input_hash: {
                "input_hash": e.input_hash,
                "sentiment_label": 1,
                "sentiment_score": 0.9,
                "p_pos": 0.92,
                "p_neg": 0.02,
                "p_neu": 0.06,
                "sentiment_motivo": "Strong NII beat",
                "sentiment_model": "claude-haiku-5-5",
                "sentiment_prompt_ver": "v1_en",
            }
            for e in entradas
        }

    mock_engine.clasificar_realtime.side_effect = fake_realtime

    silver_rows, cache = transformar_a_silver_realtime(
        raw_records, categorizador=mock_cat, engine=mock_engine
    )

    assert len(silver_rows) == 1
    row = silver_rows[0]
    for col in COLUMNAS_DESCARTAR_SILVER:
        assert col not in row
    assert len(row["record_id"]) == 24
    assert len(row["input_hash"]) == 64
    assert row["sentiment_label"] == 1
    assert row["sentiment_score"] == 0.9
    assert len(cache) == 1


@pytest.mark.asyncio
async def test_process_data_batch_dry_run_y_end_to_end(tmp_path):
    """Verifica dry-run sin llamadas API y orquestación completa categorizador -> sentimientos -> R2."""
    raw_records = [
        {
            "record_id": "abc123456789012345678901",
            "ticker": "C",
            "company_name": "Citigroup Inc.",
            "date": "2023-05-01 10:00:00",
            "title": "Citigroup expands wealth management",
            "article": "Citigroup announced new wealth advisors.",
            "text": "dropped",
            "publisher": "Reuters",
            "url": "https://example.com/c",
            "source_dataset": "fnspid",
        }
    ]
    classified = [
        {**raw_records[0], "process_text": "Citigroup announced new wealth advisors.", "category": "m&a", "confianza": 71.0, "is_ruido": False}
    ]

    real_compactor = ParquetCompactor()
    raw_bytes = real_compactor.compact(raw_records)

    mock_repo = MagicMock()
    mock_repo.bucket_name = "test-bucket"
    mock_repo.save = AsyncMock()
    mock_repo.close = AsyncMock()

    mock_cat = MagicMock()
    mock_cat.clasificar_dataset.side_effect = lambda recs, **kw: [
        {**recs[0], "process_text": "Citigroup announced new wealth advisors.", "category": "m&a", "confianza": 71.0, "is_ruido": False}
    ]

    mock_engine = MagicMock()
    mock_engine.model = "claude-haiku-5-5"
    mock_engine.prompt_version = "v1_en"
    mock_engine.construir_batch_requests.return_value = ([{"custom_id": "chunk_00000", "params": {}}], {"chunk_00000": ["h1"]})
    mock_engine.enviar_batch.return_value = {"id": "msgbatch_test_1"}
    mock_engine.consultar_batch.return_value = {"processing_status": "ended", "results_url": "https://mock/res"}

    def fake_download(url, cmap):
        _, unicas = preparar_entradas_sentimiento(classified)
        h = next(iter(unicas.keys()))
        return ({h: {"input_hash": h, "sentiment_label": 1, "sentiment_score": 0.6, "p_pos": 0.7, "p_neg": 0.1, "p_neu": 0.2, "sentiment_motivo": "Expansion", "sentiment_model": "claude-haiku-5-5", "sentiment_prompt_ver": "v1_en"}}, 1, 0)

    mock_engine.descargar_resultados_batch.side_effect = fake_download

    # 1. Dry-run: categoriza, siembra cat_cache + checkpoint en R2 y no llama a enviar_batch
    mock_repo.get = AsyncMock(side_effect=[RuntimeError("no sent cache"), RuntimeError("no cat cache"), raw_bytes])
    with patch("shared.parquet_cache.get_compactor", return_value=real_compactor), \
         patch("shared.parquet_cache.get_storage_repository", return_value=mock_repo), \
         patch("articulos.etl_categorizador.get_compactor", return_value=real_compactor), \
         patch("articulos.etl_categorizador.get_storage_repository", return_value=mock_repo), \
         patch("articulos.etl_categorizador.CategorizadorArticulos", return_value=mock_cat), \
         patch("articulos.etl_sentimientos.AnthropicSentimentEngine", return_value=mock_engine):
        summary_dry = await process_data(
            mode="batch",
            dry_run=True,
            local_backup_path=tmp_path / "out.parquet",
            local_cache_path=tmp_path / "cache.parquet",
            local_cat_cache_path=tmp_path / "cat_cache.parquet",
            checkpoint_path=tmp_path / "ckpt.parquet",
            batch_state_path=tmp_path / "state.json",
        )
    assert summary_dry["pendientes"] == 1
    assert (tmp_path / "cat_cache.parquet").exists()
    assert (tmp_path / "ckpt.parquet").exists()
    mock_engine.enviar_batch.assert_not_called()

    # 2. Ejecución all: reutiliza cat_cache sin re-ejecutar CategorizadorArticulos
    mock_repo.get = AsyncMock(side_effect=[RuntimeError("no sent cache"), raw_bytes])
    with patch("shared.parquet_cache.get_compactor", return_value=real_compactor), \
         patch("shared.parquet_cache.get_storage_repository", return_value=mock_repo), \
         patch("articulos.etl_categorizador.get_compactor", return_value=real_compactor), \
         patch("articulos.etl_categorizador.get_storage_repository", return_value=mock_repo), \
         patch("articulos.etl_categorizador.CategorizadorArticulos", return_value=mock_cat) as mock_cat_cls, \
         patch("articulos.etl_sentimientos.AnthropicSentimentEngine", return_value=mock_engine):
        summary_all = await process_data(
            mode="batch",
            step="all",
            wait=True,
            poll_interval_s=0,
            local_backup_path=tmp_path / "out.parquet",
            local_cache_path=tmp_path / "cache.parquet",
            local_cat_cache_path=tmp_path / "cat_cache.parquet",
            checkpoint_path=tmp_path / "ckpt.parquet",
            batch_state_path=tmp_path / "state.json",
        )
    assert summary_all["rows"] == 1
    assert summary_all["batch_id"] == "msgbatch_test_1"
    mock_cat_cls.assert_not_called()
    assert (tmp_path / "out.parquet").exists()
    assert (tmp_path / "cache.parquet").exists()
    assert mock_repo.save.await_count == 5  # (cat_cache + ckpt_r2) + (ckpt_r2 + sent_cache + silver final)
