import json
import pytest
from unittest.mock import patch, MagicMock
import urllib.error

from sec.adaptador import SECEdgarAdapter, SECFiling


@pytest.fixture
def adapter():
    return SECEdgarAdapter(min_request_interval=0.0)


def test_sec_filing_model_validation():
    filing = SECFiling(
        ticker="MS",
        company_name="Morgan Stanley",
        cik=895421,
        form="8-K",
        filing_date="2023-10-18",
        report_date="2023-10-18",
        acceptance_date_time="2023-10-18T07:15:00.000Z",
        accession_number="0000895421-23-000555",
        items="2.02,8.01",
        primary_document="ms-20231018.htm",
        primary_doc_description="CURRENT REPORT",
        document_url="https://www.sec.gov/Archives/edgar/data/895421/000089542123000555/ms-20231018.htm",
        index_url="https://www.sec.gov/Archives/edgar/data/895421/000089542123000555/0000895421-23-000555-index.htm",
    )
    assert filing.ticker == "MS"
    assert filing.cik == 895421
    assert "ms-20231018.htm" in filing.document_url
    assert "0000895421-23-000555-index.htm" in filing.index_url


def test_resolve_cik_known_ticker(adapter):
    cik, name = adapter.resolve_cik("ms")
    assert cik == 895421
    assert "Morgan Stanley" in name

    cik_jpm, name_jpm = adapter.resolve_cik("JPM")
    assert cik_jpm == 19617


def test_resolve_cik_dynamic_lookup(adapter):
    mock_tickers_data = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
    }

    with patch.object(adapter, "_fetch_json", return_value=mock_tickers_data):
        cik, name = adapter.resolve_cik("aapl")
        assert cik == 320193
        assert name == "Apple Inc."

        # Debe responder desde la caché interna sin volver a llamar a _fetch_json
        with patch.object(adapter, "_fetch_json", side_effect=AssertionError("No debe llamarse")):
            cik_cached, name_cached = adapter.resolve_cik("AAPL")
            assert cik_cached == 320193


def test_resolve_cik_not_found(adapter):
    with patch.object(adapter, "_fetch_json", return_value={}):
        with pytest.raises(ValueError, match="no fue localizado en el registro oficial"):
            adapter.resolve_cik("NONEXISTENT_TICKER_XYZ")


def test_fetch_json_http_error(adapter):
    with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError(
        url="http://test.url", code=403, msg="Forbidden", hdrs={}, fp=None
    )):
        with pytest.raises(urllib.error.HTTPError):
            adapter._fetch_json("http://test.url")


def test_fetch_json_runtime_error(adapter):
    with patch("urllib.request.urlopen", side_effect=ConnectionResetError("Socket reset")):
        with pytest.raises(RuntimeError, match="Fallo al conectar con la SEC"):
            adapter._fetch_json("http://test.url")


def test_get_filings_recent_and_paginated(adapter):
    main_submission = {
        "cik": "0000895421",
        "entityType": "operating",
        "name": "MORGAN STANLEY",
        "filings": {
            "recent": {
                "accessionNumber": ["0000895421-23-000002", "0000895421-23-000001", "0000895421-20-000099"],
                "filingDate": ["2023-10-18", "2023-07-15", "2020-01-15"],
                "reportDate": ["2023-10-18", "2023-07-15", "2020-01-15"],
                "acceptanceDateTime": ["2023-10-18T07:00:00.000Z", "", ""],
                "act": ["34", "34", "34"],
                "form": ["8-K", "10-Q", "SC 13G"],
                "primaryDocument": ["ms-20231018.htm", "ms-20230715.htm", "doc.htm"],
                "primaryDocDescription": ["Current Report", "Quarterly Report", "Other"],
                "items": [["2.02", "8.01"], "", ""],
            },
            "files": [
                {
                    "name": "CIK0000895421-submissions-001.json",
                    "filingCount": 2000,
                    "filingFrom": "2015-01-01",
                    "filingTo": "2019-12-31",
                },
                {
                    "name": "CIK0000895421-submissions-002.json",
                    "filingCount": 2000,
                    "filingFrom": "2000-01-01",
                    "filingTo": "2005-12-31",  # Fuera del rango buscado (2015-2023)
                },
            ],
        },
    }

    paginated_submission = {
        "accessionNumber": ["0000895421-17-000010"],
        "filingDate": ["2017-04-19"],
        "reportDate": ["2017-04-19"],
        "acceptanceDateTime": [""],
        "act": ["34"],
        "form": ["8-K"],
        "primaryDocument": ["ms-20170419.htm"],
        "primaryDocDescription": ["Current Report"],
        "items": ["2.02"],
    }

    def fake_fetch_json(url: str):
        if "CIK0000895421.json" in url:
            return main_submission
        if "CIK0000895421-submissions-001.json" in url:
            return paginated_submission
        raise ValueError(f"URL no esperada en mock: {url}")

    with patch.object(adapter, "_fetch_json", side_effect=fake_fetch_json):
        # 1. Filtro 2015 a 2023 solo para 8-K y 10-Q
        filings = adapter.get_filings(
            ticker="MS",
            forms=["8-K", "10-Q"],
            start_date="2015",
            end_date="2023",
        )

        assert len(filings) == 3
        # Orden descendente
        assert filings[0].filing_date == "2023-10-18"
        assert filings[0].form == "8-K"
        assert filings[0].items == "2.02,8.01"
        assert filings[0].document_url == "https://www.sec.gov/Archives/edgar/data/895421/000089542123000002/ms-20231018.htm"
        assert filings[0].index_url == "https://www.sec.gov/Archives/edgar/data/895421/000089542123000002/0000895421-23-000002-index.htm"

        assert filings[1].filing_date == "2023-07-15"
        assert filings[1].form == "10-Q"

        # Procedente del paginado
        assert filings[2].filing_date == "2017-04-19"
        assert filings[2].form == "8-K"

        # 2. Comprobar to_dict_list
        dicts = adapter.to_dict_list(filings)
        assert len(dicts) == 3
        assert isinstance(dicts[0], dict)
        assert dicts[0]["ticker"] == "MS"
        assert "document_url" in dicts[0]


def test_get_filings_date_exclusion(adapter):
    main_submission = {
        "cik": "0000895421",
        "entityType": "operating",
        "name": "MORGAN STANLEY",
        "filings": {
            "recent": {
                "accessionNumber": ["0000895421-23-000001", "0000895421-22-000001"],
                "filingDate": ["2023-05-01", "2022-05-01"],
                "reportDate": ["", ""],
                "acceptanceDateTime": ["", ""],
                "act": ["", ""],
                "form": ["8-K", "8-K"],
                "primaryDocument": ["doc1.htm", "doc2.htm"],
                "primaryDocDescription": ["", ""],
                "items": ["", ""],
            },
            "files": [],
        },
    }

    with patch.object(adapter, "_fetch_json", return_value=main_submission):
        filings = adapter.get_filings(
            ticker="MS",
            forms=["8-K"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )
        assert len(filings) == 1
        assert filings[0].filing_date == "2023-05-01"
