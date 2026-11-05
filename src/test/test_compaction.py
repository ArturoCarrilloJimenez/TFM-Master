import io
import json
import pytest
from unittest.mock import patch

from infrastructure.compaction.parquet import ParquetCompactor
from sec.adaptador import SECFiling


@pytest.fixture
def compactor():
    return ParquetCompactor()


@pytest.fixture
def sample_records():
    return [
        {
            "ticker": "MS",
            "cik": 895421,
            "form": "10-Q",
            "filing_date": "2023-09-30",
            "document_url": "https://www.sec.gov/Archives/edgar/data/895421/000089542123000441/ms-20230930.htm",
        },
        {
            "ticker": "JPM",
            "cik": 19617,
            "form": "10-K",
            "filing_date": "2023-12-31",
            "document_url": "https://www.sec.gov/Archives/edgar/data/19617/000001961723000524/jpm-20231231.htm",
        },
    ]


def test_compact_and_decompact_list_of_dicts(compactor, sample_records):
    parquet_bytes = compactor.compact(sample_records, compression="ZSTD")
    assert isinstance(parquet_bytes, bytes)
    assert len(parquet_bytes) > 0

    # Header de Apache Parquet es siempre 'PAR1'
    assert parquet_bytes.startswith(b"PAR1")

    recovered = compactor.decompact(parquet_bytes)
    assert len(recovered) == 2
    assert recovered[0]["ticker"] == "MS"
    assert recovered[1]["ticker"] == "JPM"
    assert recovered[0]["form"] == "10-Q"


def test_decompact_with_bytesio_buffer(compactor, sample_records):
    """Valida soporte polimórfico al recibir un buffer io.BytesIO en decompact()."""
    parquet_bytes = compactor.compact(sample_records, compression="ZSTD")
    buffer = io.BytesIO(parquet_bytes)
    recovered = compactor.decompact(buffer)
    assert len(recovered) == 2
    assert recovered[0]["ticker"] == "MS"


def test_compact_single_dict(compactor):
    single_record = {"ticker": "BAC", "form": "8-K", "value": 42}
    parquet_bytes = compactor.compact(single_record)
    recovered = compactor.decompact(parquet_bytes)
    assert len(recovered) == 1
    assert recovered[0]["ticker"] == "BAC"
    assert recovered[0]["value"] == 42


def test_compact_json_string(compactor):
    json_data = json.dumps([
        {"ticker": "C", "revenue": 1000},
        {"ticker": "WFC", "revenue": 2000},
    ])
    parquet_bytes = compactor.compact(json_data)
    recovered = compactor.decompact(parquet_bytes)
    assert len(recovered) == 2
    assert recovered[0]["ticker"] == "C"


def test_compact_pydantic_v2_models(compactor):
    filings = [
        SECFiling(
            ticker="GS",
            company_name="Goldman Sachs Group Inc.",
            cik=886982,
            form="8-K",
            filing_date="2023-01-15",
            report_date="2023-01-15",
            acceptance_date_time="2023-01-15T10:00:00.000Z",
            accession_number="0000886982-23-000001",
            items="2.02",
            primary_document="gs-20230115.htm",
            primary_doc_description="",
            document_url="https://sec.gov/doc.htm",
            index_url="https://sec.gov/index.htm",
        )
    ]
    parquet_bytes = compactor.compact(filings)
    recovered = compactor.decompact(parquet_bytes)
    assert len(recovered) == 1
    assert recovered[0]["ticker"] == "GS"
    assert recovered[0]["cik"] == 886982


def test_compact_empty_list_raises(compactor):
    with pytest.raises(ValueError, match="No se puede compactar una lista vacía"):
        compactor.compact([])


def test_compact_invalid_json_string_raises(compactor):
    with pytest.raises(ValueError, match="Cadena JSON inválida"):
        compactor.compact("{invalid_json: true")


def test_compact_scalar_json_raises(compactor):
    with pytest.raises(ValueError, match="El JSON provisto debe ser un objeto"):
        compactor.compact("42")


def test_compact_invalid_type_raises(compactor):
    with pytest.raises(TypeError, match="Tipo de entrada no soportado"):
        compactor.compact(12345)


def test_compact_list_with_invalid_element_type_raises(compactor):
    with pytest.raises(TypeError, match="Tipo de elemento no soportado"):
        compactor.compact([123, 456])


def test_decompact_empty_payload_raises(compactor):
    with pytest.raises(ValueError, match="El payload binario Parquet está vacío"):
        compactor.decompact(b"")


def test_compact_runtime_error_on_pyarrow_failure(compactor, sample_records):
    with patch("pyarrow.parquet.write_table", side_effect=Exception("PyArrow write failure")):
        with pytest.raises(RuntimeError, match="Error al generar binario Parquet"):
            compactor.compact(sample_records)


def test_decompact_runtime_error_on_invalid_bytes(compactor):
    with pytest.raises(RuntimeError, match="Error al leer binario Parquet"):
        compactor.decompact(b"CORRUPTED_BYTES_NOT_PARQUET")
