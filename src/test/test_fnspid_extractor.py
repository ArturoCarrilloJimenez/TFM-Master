"""
Tests unitarios para la lógica de filtrado y extracción de FNSPID.
"""
import io
from src.shared.fnspid_extractor import (
    normalizar_ticker,
    procesar_stream_noticias,
    ajustar_limite_csv
)


def test_ajustar_limite_csv():
    """Verifica que el límite de campo CSV se ejecute sin excepciones."""
    ajustar_limite_csv()


def test_normalizar_ticker_variantes():
    """Comprueba la limpieza de cadenas y tratamiento de nulos."""
    assert normalizar_ticker(" jpm ") == "JPM"
    assert normalizar_ticker("bac") == "BAC"
    assert normalizar_ticker("Wfc") == "WFC"
    assert normalizar_ticker(None) == ""
    assert normalizar_ticker("") == ""


def test_procesar_stream_noticias_filtrado_correcto():
    """Evalúa que solo se capturen los bancos seleccionados."""
    csv_mock = (
        "Date,Stock_symbol,Article_title,Publisher\n"
        "2020-01-01,JPM,Ganancias récord,Bloomberg\n"
        "2020-01-02,AAPL,Presentación iPhone,Reuters\n"
        "2020-01-03, bac ,Dividendo aprobado,CNBC\n"
        "2020-01-04,TSLA,Entregas trimestrales,Yahoo\n"
    )
    stream = io.StringIO(csv_mock)
    bancos = {'JPM', 'BAC'}

    resultado = procesar_stream_noticias(stream, bancos)

    assert len(resultado) == 2
    tickers = [item["Stock_symbol"] for item in resultado]
    assert "JPM" in tickers
    assert "BAC" in tickers
    assert "AAPL" not in tickers


def test_procesar_stream_noticias_sin_coincidencias():
    """Evalúa que retorne lista vacía si ningún registro coincide."""
    csv_mock = (
        "Date,Stock_symbol,Article_title\n"
        "2021-05-10,MSFT,Azure Cloud\n"
        "2021-05-11,GOOG,Alphabet AI\n"
    )
    stream = io.StringIO(csv_mock)
    resultado = procesar_stream_noticias(stream, {'GS', 'C'})
    assert len(resultado) == 0