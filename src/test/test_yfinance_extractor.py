"""
Tests unitarios para la lógica de formateo y extracción de cotizaciones de yfinance.
"""
import pandas as pd
from shared.yfinance_extractor import aplanar_columnas_multiindex


def test_aplanar_columnas_multiindex_basico():
    """Valida la unión correcta de nombres de columnas multinivel."""
    columnas_multi = pd.MultiIndex.from_tuples([
        ('Close', 'JPM'),
        ('Close', 'BAC'),
        ('Volume', 'JPM')
    ])
    datos = [
        [100.0, 30.0, 500000],
        [101.5, 30.8, 620000]
    ]
    fechas = pd.date_range("2020-01-01", periods=2)
    df_mock = pd.DataFrame(datos, index=fechas, columns=columnas_multi)

    df_resultado = aplanar_columnas_multiindex(df_mock)

    columnas_esperadas = ['Close_JPM', 'Close_BAC', 'Volume_JPM']
    assert list(df_resultado.columns) == columnas_esperadas
    assert df_resultado.index.name == "Date"
    assert df_resultado.shape == (2, 3)


def test_aplanar_columnas_index_simple_no_altera():
    """Valida que si las columnas ya son un Index simple, no se rompa la estructura."""
    df_simple = pd.DataFrame({'Close_JPM': [100.0, 102.0]})
    df_resultado = aplanar_columnas_multiindex(df_simple)

    assert list(df_resultado.columns) == ['Close_JPM']
    assert df_resultado.index.name == "Date"
