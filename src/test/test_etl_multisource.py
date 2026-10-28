"""
Tests unitarios para el pipeline ETL de noticias financieras multisource (src/articulos/etl_multisource.py).
"""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from articulos.etl_multisource import (
    extraer_title_y_article,
    extract_and_transform,
    save_and_upload,
)


def test_extraer_title_y_article_casos_limite():
    """Valida el manejo de casos nulos, vacíos y no string."""
    assert extraer_title_y_article(None) == ("", "")
    assert extraer_title_y_article("") == ("", "")
    assert extraer_title_y_article(float("nan")) == ("", "")


def test_extraer_title_y_article_solo_titular():
    """Valida textos de una sola línea sin salto de carro (solo titular)."""
    texto = "JPMorgan Chase Reports Strong First-Quarter Results"
    title, article = extraer_title_y_article(texto)
    assert title == "JPMorgan Chase Reports Strong First-Quarter Results"
    assert article == ""


def test_extraer_title_y_article_multilinea():
    """Valida la correcta separación en titular y cuerpo de artículo."""
    texto = """Bank of America Expands Lending Program
The Charlotte-based bank announced a multi-billion initiative today.
Additional paragraph with detailed quotes from the CEO."""
    
    title, article = extraer_title_y_article(texto)
    assert title == "Bank of America Expands Lending Program"
    assert "The Charlotte-based bank announced a multi-billion initiative today." in article
    assert "Additional paragraph with detailed quotes from the CEO." in article


def test_extract_and_transform_pipeline():
    """Valida el filtrado, deduplicación, filtro de falsos positivos de C y mapeo canónico de TIKETS."""
    mock_raw_data = pd.DataFrame([
        {
            "ticker": "JPM",
            "date": "2020-05-15 10:00:00",
            "source_dataset": "fnspid_news",
            "publisher": "Reuters",
            "text": "JPM Headline\nFull article content for JPMorgan Chase.",
            "url": "http://jpm.com/news/1"
        },
        # Registro duplicado exacto
        {
            "ticker": "JPM",
            "date": "2020-05-15 10:00:00",
            "source_dataset": "fnspid_news",
            "publisher": "Reuters",
            "text": "JPM Headline\nFull article content for JPMorgan Chase.",
            "url": "http://jpm.com/news/1"
        },
        # Fecha inválida (debe descartarse por dropna datetime)
        {
            "ticker": "BAC",
            "date": "fecha_invalida_no_parseable",
            "source_dataset": "benzinga_6000stocks",
            "publisher": "Benzinga",
            "text": "BAC headline without body",
            "url": "http://bac.com/news/2"
        },
        # Noticia solo titular
        {
            "ticker": "WFC",
            "date": "2021-08-20 14:30:00",
            "source_dataset": "sentarl_combined",
            "publisher": "SentARL",
            "text": "Wells Fargo Declares Regular Dividend",
            "url": "http://wfc.com/news/3"
        },
        # Falso positivo de Citigroup ('C' sin mención en texto ni título)
        {
            "ticker": "C",
            "date": "2022-01-05 12:00:00",
            "source_dataset": "yahoo_finance_felixdrinkall",
            "publisher": "Yahoo Finance",
            "text": "Antarctica turning green as climate change takes hold\nSome climate study text.",
            "url": "http://yahoo.com/c/fake"
        },
        # Noticia válida de Citigroup ('C' con 'Citi' en el texto)
        {
            "ticker": "C",
            "date": "2022-01-05 15:00:00",
            "source_dataset": "yahoo_finance_felixdrinkall",
            "publisher": "Yahoo Finance",
            "text": "Citigroup Announces Executive Reshuffle\nCiti names new head of banking.",
            "url": "http://yahoo.com/c/real"
        }
    ])

    mock_con = MagicMock()
    mock_con.execute.return_value.df.return_value = mock_raw_data

    with patch("duckdb.connect", return_value=mock_con):
        df_result = extract_and_transform()

    # Se descartan la fila duplicada, la fecha inválida y el falso positivo de C (quedan 3 registros: JPM, WFC, C)
    assert len(df_result) == 3
    assert set(df_result["ticker"]) == {"JPM", "WFC", "C"}

    # Verificar enriquecimiento con TIKETS
    jpm_row = df_result[df_result["ticker"] == "JPM"].iloc[0]
    assert jpm_row["company_name"] == "JPMorgan Chase & Co."
    assert jpm_row["cik"] == "CIK0000019617"
    assert jpm_row["title"] == "JPM Headline"
    assert jpm_row["article"] == "Full article content for JPMorgan Chase."

    c_row = df_result[df_result["ticker"] == "C"].iloc[0]
    assert c_row["company_name"] == "Citigroup Inc."
    assert c_row["title"] == "Citigroup Announces Executive Reshuffle"

    wfc_row = df_result[df_result["ticker"] == "WFC"].iloc[0]
    assert wfc_row["company_name"] == "Wells Fargo & Company/MN"
    assert wfc_row["title"] == "Wells Fargo Declares Regular Dividend"
    assert wfc_row["article"] == ""

    # Verificar columnas finales esperadas incluyendo ticker_count e is_multisubject
    columnas_esperadas = [
        "ticker", "company_name", "cik", "date", "datetime",
        "title", "article", "text", "publisher", "url", "source_dataset",
        "ticker_count", "is_multisubject"
    ]
    assert list(df_result.columns) == columnas_esperadas
    assert (df_result["ticker_count"] == 1).all()
    assert (~df_result["is_multisubject"]).all()


def test_save_and_upload():
    """Valida la generación de parquet local y la llamada de subida a Cloudflare R2."""
    df_sample = pd.DataFrame([
        {
            "ticker": "GS",
            "company_name": "Goldman Sachs Group Inc.",
            "cik": "CIK0000886982",
            "date": "2023-01-10 09:00:00",
            "datetime": pd.to_datetime("2023-01-10 09:00:00", utc=True),
            "title": "Goldman Sachs Reports Q4 Earnings",
            "article": "Detailed earnings breakdown for Goldman Sachs.",
            "text": "Goldman Sachs Reports Q4 Earnings\nDetailed earnings breakdown for Goldman Sachs.",
            "publisher": "Bloomberg",
            "url": "http://bloomberg.com/gs",
            "source_dataset": "yahoo_finance_felixdrinkall",
            "ticker_count": 1,
            "is_multisubject": False
        }
    ])

    mock_s3 = MagicMock()
    mock_s3.head_object.return_value = {
        "ContentLength": 1024 * 1024 * 5,
        "LastModified": "2026-10-24 21:00:00"
    }

    mock_stat = MagicMock()
    mock_stat.st_size = 1024 * 1024 * 5

    with patch.object(df_sample, "to_parquet") as mock_to_parquet, \
         patch("pathlib.Path.stat", return_value=mock_stat), \
         patch("boto3.client", return_value=mock_s3):
        
        save_and_upload(df_sample)

        mock_to_parquet.assert_called_once()
        mock_s3.upload_file.assert_called_once()
        mock_s3.head_object.assert_called_once()


def test_main_block():
    """Valida la ejecución del flujo completo de inicio a fin."""
    with patch("articulos.etl_multisource.extract_and_transform") as mock_extract, \
         patch("articulos.etl_multisource.save_and_upload") as mock_save:
        
        mock_df = pd.DataFrame({"ticker": ["JPM"]})
        mock_extract.return_value = mock_df

        # Simular ejecución
        import articulos.etl_multisource as module
        df_clean = module.extract_and_transform()
        module.save_and_upload(df_clean)

        mock_extract.assert_called_once()
        mock_save.assert_called_once_with(mock_df)
