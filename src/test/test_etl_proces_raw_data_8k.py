import pytest
from unittest.mock import MagicMock
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
