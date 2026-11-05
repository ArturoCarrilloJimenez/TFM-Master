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
