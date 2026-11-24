import json
from unittest.mock import AsyncMock, MagicMock
import pytest

from sec.etl_proces_raw_data_8k import (
    clean_text,
    extract_items_from_text,
    ETLProcesRawData8K,
)


def test_clean_text():
    dirty = "Line 1 \x00 with  spaces\t and \ufffd artifact  "
    assert clean_text(dirty) == "Line 1 with spaces and artifact"


def test_extract_items_from_text():
    sample_text = """
    UNITED STATES SECURITIES AND EXCHANGE COMMISSION
    Item 2.02 Results of Operations and Financial Condition.
    On October 13, 2023, the Company announced third-quarter financial results.
    
    Item 8.01 Other Events.
    The Board of Directors declared a quarterly dividend of $1.05 per share.
    
    SIGNATURE
    Pursuant to the requirements of the Securities Exchange Act of 1934...
    """
    items = extract_items_from_text(sample_text)
    assert len(items) == 2
    assert items[0]["item"] == "Item 2.02"
    assert "Results of Operations" in items[0]["item_description"]
    assert "third-quarter financial results" in items[0]["content"]

    assert items[1]["item"] == "Item 8.01"
    assert "Other Events" in items[1]["item_description"]
    assert "quarterly dividend" in items[1]["content"]


def test_etl_initialization_with_injected_dependencies():
    mock_compactor = MagicMock()
    mock_storage = MagicMock()
    etl = ETLProcesRawData8K(compactor=mock_compactor, storage=mock_storage)
    assert etl.days_window == 7
    assert etl.compactor is mock_compactor
    assert etl.storage is mock_storage


@pytest.mark.asyncio
async def test_etl_proces_raw_data_8k_run_flow():
    """Valida el flujo run() con compactor centralizado (decompact y compact) y storage."""
    mock_compactor = MagicMock()
    mock_storage = MagicMock()

    mock_storage.list_keys = AsyncMock(
        return_value=["compacted-data/job_id=test-job/year=2023/month=10/day=15/part-0.parquet"]
    )
    mock_storage.get = AsyncMock(return_value=b"fake_raw_bytes")
    mock_storage.save = AsyncMock()

    raw_mock_records = [
        {
            "url": "https://sec.gov/doc8k.htm",
            "date": "2023-10-15 10:00:00",
            "context": json.dumps({
                "ticker": "JPM",
                "company_name": "JPMorgan Chase & Co.",
                "cik": 19617,
                "form": "8-K",
            }),
            "data": json.dumps({
                "body_text": [
                    "Item 2.02 Results of Operations.",
                    "Third quarter net income rose significantly.",
                    "SIGNATURE",
                ]
            }),
        }
    ]

    mock_compactor.decompact.return_value = raw_mock_records
    mock_compactor.compact.return_value = b"fake_compacted_parquet"

    etl = ETLProcesRawData8K(
        job_id="test-job",
        compact_dates=["2023-10-15"],
        compactor=mock_compactor,
        storage=mock_storage,
    )

    df_out = await etl.run()

    mock_storage.get.assert_called_once()
    mock_compactor.decompact.assert_called_once_with(b"fake_raw_bytes")
    mock_compactor.compact.assert_called_once()
    mock_storage.save.assert_called_once()

    assert len(df_out) == 1
    assert df_out.iloc[0]["ticker"] == "JPM"
    assert df_out.iloc[0]["item"] == "Item 2.02"


def test_extract_items_preserves_exhibit_99_1_decimal():
    sample_text = (
        "Item 2.02 Results of Operations and Financial Condition. "
        "Including Exhibit 99.1 and Exhibit 99.2, net income was $12.9 billion. "
        "SIGNATURE"
    )
    items = extract_items_from_text(sample_text)
    assert len(items) == 1
    assert items[0]["item"] == "Item 2.02"
    assert items[0]["item_description"] == "Results of Operations and Financial Condition"
    assert "Exhibit 99.1" in items[0]["content"]


@pytest.mark.asyncio
async def test_etl_proces_raw_data_8k_persists_separate_exhibits_linked_by_accession():
    mock_compactor = MagicMock()
    mock_storage = MagicMock()

    mock_storage.list_keys = AsyncMock(
        return_value=["compacted-data/job_id=test-job/year=2024/month=10/day=11/part-0.parquet"]
    )
    mock_storage.get = AsyncMock(return_value=b"fake_raw_bytes")
    mock_storage.save = AsyncMock()

    raw_mock_records = [
        {
            "url": "https://www.sec.gov/Archives/edgar/data/19617/000001961724000555/jpm-20241011.htm",
            "date": "2024-10-11 10:00:00",
            "context": json.dumps({
                "ticker": "JPM",
                "company_name": "JPMorgan Chase & Co.",
                "cik": 19617,
                "form": "8-K",
                "filing_date": "2024-10-11",
                "acceptance_date_time": "2024-10-11T10:45:00.000Z",
                "accession_number": "0000019617-24-000555",
                "items": "2.02,9.01",
                "index_url": "https://www.sec.gov/Archives/edgar/data/19617/000001961724000555/0000019617-24-000555-index.htm",
            }),
            "data": json.dumps({
                "body_text": [
                    "Item 2.02 Results of Operations. On October 11, 2024, JPMorgan reported net income. SIGNATURE"
                ],
                "exhibits": [
                    {
                        "url": "https://www.sec.gov/Archives/edgar/data/19617/000001961724000555/ex991.htm",
                        "label": "EX-99.1",
                        "metadata": {
                            "description": "EARNINGS RELEASE - 3Q24",
                            "size": "205642",
                        },
                        "data": {"body_text": ["JPMorgan Chase reports 3Q24 net income of $12.9 billion."]},
                    },
                    {
                        "url": "https://www.sec.gov/Archives/edgar/data/19617/000001961724000555/ex992.pdf",
                        "label": "EX-99.2",
                        "metadata": {"description": "SUPPLEMENT", "size": "100000"},
                        "error": "Contenido no HTML: application/pdf",
                    },
                ],
            }),
        }
    ]

    mock_compactor.decompact.return_value = raw_mock_records
    mock_compactor.compact.side_effect = [b"items_parquet", b"exhibits_parquet"]

    etl = ETLProcesRawData8K(
        job_id="test-job",
        target_key="raw/sec/8k-2010-2023.parquet",
        exhibits_target_key="raw/sec/8k-exhibits-2010-2023.parquet",
        compact_dates=["2024-10-11"],
        compactor=mock_compactor,
        storage=mock_storage,
    )

    df_items = await etl.run()

    assert len(df_items) == 1
    assert df_items.iloc[0]["accession_number"] == "0000019617-24-000555"

    df_ex = etl.last_exhibits_df
    assert df_ex is not None
    assert len(df_ex) == 2
    assert list(df_ex["exhibit_type"]) == ["EX-99.1", "EX-99.2"]
    assert df_ex.iloc[0]["accession_number"] == "0000019617-24-000555"
    assert df_ex.iloc[0]["filing_items"] == "2.02,9.01"
    assert df_ex.iloc[0]["exhibit_description"] == "EARNINGS RELEASE - 3Q24"
    assert "12.9 billion" in df_ex.iloc[0]["content"]
    assert df_ex.iloc[1]["error"] == "Contenido no HTML: application/pdf"

    assert mock_storage.save.await_count == 2
    saved_keys = [c.kwargs["key"] for c in mock_storage.save.await_args_list]
    assert saved_keys == ["raw/sec/8k-2010-2023.parquet", "raw/sec/8k-exhibits-2010-2023.parquet"]
