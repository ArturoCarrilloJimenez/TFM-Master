"""Tests unitarios para el pipeline ETL de categorización semántica (src/articulos/etl_categorizacion.py)."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from articulos.etl_categorizacion import COLUMNAS_DESCARTAR_SILVER, process_data


@pytest.mark.asyncio
async def test_process_data_pipeline_completo(tmp_path):
    """Verifica la orquestación completa de process_data: descarga R2, inferencia, poda Silver y guardado dual."""
    mock_raw_records = [
        {
            "ticker": "JPM",
            "company_name": "JPMorgan Chase & Co.",
            "title": "JPMorgan Profit Beats Forecast",
            "article": "Full details on quarterly earnings.",
            "text": "Full text to be dropped",
            "publisher": "SentARL",
            "url": "https://example.com/jpm",
            "source_dataset": "fnspid_news",
            "ticker_count": 1,
            "is_multisubject": False,
        }
    ]

    mock_classified = [
        {
            **mock_raw_records[0],
            "process_text": "Full details on quarterly earnings.",
            "category": "earnings",
            "confianza": 75.0,
            "is_ruido": False,
        }
    ]

    mock_compactor = MagicMock()
    mock_compactor.decompact.return_value = mock_raw_records
    mock_compactor.compact.return_value = b"PARQUET_ZSTD_MOCK_BYTES"

    mock_repo_in = MagicMock()
    mock_repo_in.bucket_name = "test-lake"
    mock_repo_in.get = AsyncMock(return_value=b"RAW_IN_BYTES")
    mock_repo_in.close = AsyncMock()

    mock_repo_out = MagicMock()
    mock_repo_out.bucket_name = "test-lake"
    mock_repo_out.save = AsyncMock()
    mock_repo_out.close = AsyncMock()

    mock_model = MagicMock()
    mock_model.clasificar_dataset.return_value = mock_classified

    backup_target = tmp_path / "processed_test.parquet"

    with patch("articulos.etl_categorizacion.get_compactor", return_value=mock_compactor), \
         patch("articulos.etl_categorizacion.get_storage_repository", side_effect=[mock_repo_in, mock_repo_out]), \
         patch("articulos.etl_categorizacion.CategorizadorArticulos", return_value=mock_model):

        await process_data(
            input_s3_key="raw/test.parquet",
            output_s3_key="processed/test.parquet",
            local_backup_path=str(backup_target),
            batch_size=64,
        )

        # 1. Verificación de lectura de entrada
        mock_repo_in.get.assert_awaited_once_with("raw/test.parquet")
        mock_repo_in.close.assert_awaited_once()
        mock_compactor.decompact.assert_called_once_with(b"RAW_IN_BYTES")

        # 2. Verificación de llamada al modelo
        mock_model.clasificar_dataset.assert_called_once_with(mock_raw_records, batch_size=64)

        # 3. Verificación de poda Silver
        compact_arg = mock_compactor.compact.call_args[0][0]
        assert len(compact_arg) == 1
        clean_row = compact_arg[0]
        for col in COLUMNAS_DESCARTAR_SILVER:
            assert col not in clean_row
        assert clean_row["ticker"] == "JPM"
        assert clean_row["category"] == "earnings"

        # 4. Verificación de backup en disco
        assert backup_target.exists()
        assert backup_target.read_bytes() == b"PARQUET_ZSTD_MOCK_BYTES"

        # 5. Verificación de subida a R2
        mock_repo_out.save.assert_awaited_once_with("processed/test.parquet", b"PARQUET_ZSTD_MOCK_BYTES")
        mock_repo_out.close.assert_awaited_once()
