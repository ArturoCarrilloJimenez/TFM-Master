import pytest
from unittest.mock import MagicMock, AsyncMock

from sec.adaptador import SECFiling
from sec.etl_historico import SECHistoricalETL, run_sec_historical_etl


@pytest.fixture
def mock_adapter():
    adapter = MagicMock()
    adapter.resolve_cik.side_effect = lambda ticker: (12345, f"{ticker} Corp")

    def fake_get_filings(ticker, forms, start_date, end_date):
        return [
            SECFiling(
                ticker=ticker,
                company_name=f"{ticker} Corp",
                cik=12345,
                form="8-K",
                filing_date="2022-05-10",
                report_date="2022-05-10",
                acceptance_date_time="",
                accession_number="0000012345-22-000001",
                items="2.02",
                primary_document="doc8k.htm",
                primary_doc_description="",
                document_url="https://www.sec.gov/Archives/edgar/data/12345/000001234522000001/doc8k.htm",
                index_url="https://www.sec.gov/Archives/edgar/data/12345/000001234522000001/0000012345-22-000001-index.htm",
            ),
            SECFiling(
                ticker=ticker,
                company_name=f"{ticker} Corp",
                cik=12345,
                form="10-Q",
                filing_date="2022-08-10",
                report_date="2022-08-10",
                acceptance_date_time="",
                accession_number="0000012345-22-000002",
                items="",
                primary_document="doc10q.htm",
                primary_doc_description="",
                document_url="https://www.sec.gov/Archives/edgar/data/12345/000001234522000002/doc10q.htm",
                index_url="https://www.sec.gov/Archives/edgar/data/12345/000001234522000002/0000012345-22-000002-index.htm",
            ),
            SECFiling(
                ticker=ticker,
                company_name=f"{ticker} Corp",
                cik=12345,
                form="10-K",
                filing_date="2023-02-15",
                report_date="2023-02-15",
                acceptance_date_time="",
                accession_number="0000012345-23-000003",
                items="",
                primary_document="doc10k.htm",
                primary_doc_description="",
                document_url="https://www.sec.gov/Archives/edgar/data/12345/000001234523000003/doc10k.htm",
                index_url="https://www.sec.gov/Archives/edgar/data/12345/000001234523000003/0000012345-23-000003-index.htm",
            ),
        ]

    adapter.get_filings.side_effect = fake_get_filings
    return adapter


@pytest.fixture
def mock_compactor():
    compactor = MagicMock()
    compactor.compact.return_value = b"PAR1_MOCK_PARQUET_BYTES"
    return compactor


@pytest.fixture
def mock_storage():
    storage = MagicMock()
    storage.save = AsyncMock()
    return storage


@pytest.fixture
def mock_scraper_client():
    client = MagicMock()
    client.dispatch_tasks = AsyncMock(
        return_value={
            "job_id": "mock-job",
            "status": "SUCCESS",
            "total_tasks": 6,
            "total_batches": 1,
            "accepted_tasks": 6,
        }
    )
    return client


@pytest.mark.asyncio
async def test_sec_historical_etl_run(mock_adapter, mock_compactor, mock_storage, mock_scraper_client):
    etl = SECHistoricalETL(
        adapter=mock_adapter,
        compactor=mock_compactor,
        storage=mock_storage,
        scraper_client=mock_scraper_client,
    )

    result = await etl.run(
        start_date="2022-01-01",
        end_date="2023-12-31",
        target_s3_key="raw/sec/10k-10q-2010-2023.parquet",
        dispatch_to_scraper=True,
    )

    assert result["status"] == "SUCCESS"
    assert result["s3_key"] == "raw/sec/10k-10q-2010-2023.parquet"
    assert result["total_universe_discovered"] == 18
    # 6 banks * 2 (10-Q y 10-K) = 12 compactados
    assert result["total_10k_10q_compacted"] == 12
    # 6 banks * 1 (8-K) = 6 listos para el scraper
    assert result["total_8k_for_scraper"] == 6
    assert len(result["filings_8k"]) == 6
    assert result["scraper_result"]["status"] == "SUCCESS"

    # Verificar que el compactador fue llamado con los 12 registros de 10-Q/10-K
    mock_compactor.compact.assert_called_once()
    args, kwargs = mock_compactor.compact.call_args
    assert len(args[0]) == 12
    assert kwargs.get("compression") == "ZSTD"

    # Verificar que storage.save fue invocado con la key y los bytes correctos
    mock_storage.save.assert_awaited_once_with(
        key="raw/sec/10k-10q-2010-2023.parquet",
        body=b"PAR1_MOCK_PARQUET_BYTES",
    )

    # Verificar que el scraper_client recibió las 6 tareas 8-K
    mock_scraper_client.dispatch_tasks.assert_awaited_once()
    s_args, s_kwargs = mock_scraper_client.dispatch_tasks.call_args
    assert len(s_kwargs["tasks"]) == 6
    assert s_kwargs["job_id"] == "sec-8k-2022-01-01-to-2023-12-31"


def test_run_sec_historical_etl_wrapper(mock_adapter, mock_compactor, mock_storage, mock_scraper_client):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("sec.etl_historico.SECEdgarAdapter", lambda: mock_adapter)
        mp.setattr("sec.etl_historico.get_compactor", lambda: mock_compactor)
        mp.setattr("sec.etl_historico.get_storage_repository", lambda: mock_storage)
        mp.setattr("sec.etl_historico.get_scraper_client", lambda: mock_scraper_client)

        result = run_sec_historical_etl(
            start_date="2022-01-01",
            end_date="2023-12-31",
            target_s3_key="raw/sec/10k-10q-2010-2023.parquet",
            dispatch_to_scraper=False,
        )
        assert result["status"] == "SUCCESS"
        assert result["total_10k_10q_compacted"] == 12
        assert result["scraper_result"] is None


@pytest.mark.asyncio
async def test_sec_historical_etl_daily_production_window(
    mock_adapter, mock_compactor, mock_storage, mock_scraper_client
):
    """Verifica que el ETL funcione en producción con ventanas diarias (día t y t-1)."""
    etl = SECHistoricalETL(
        adapter=mock_adapter,
        compactor=mock_compactor,
        storage=mock_storage,
        scraper_client=mock_scraper_client,
    )

    result = await etl.run(
        start_date="2023-10-18",
        end_date="2023-10-19",
        target_s3_key="raw/sec/daily/2023-10-19.parquet",
        dispatch_to_scraper=True,
    )

    assert result["status"] == "SUCCESS"
    assert result["s3_key"] == "raw/sec/daily/2023-10-19.parquet"
    mock_adapter.get_filings.assert_called()
    # Verificar que el adaptador recibió exactamente las fechas diarias solicitadas
    for call in mock_adapter.get_filings.call_args_list:
        assert call.kwargs["start_date"] == "2023-10-18"
        assert call.kwargs["end_date"] == "2023-10-19"


def test_tikets_shared_consistency():
    from shared.tikets import TIKETS
    assert len(TIKETS) == 6
    for ticker in ["JPM", "BAC", "WFC", "C", "GS", "MS"]:
        assert ticker in TIKETS
        cik_str, name = TIKETS[ticker]
        assert cik_str.startswith("CIK")
        assert len(name) > 0
    assert TIKETS["MS"][0] == "CIK0000895421"
