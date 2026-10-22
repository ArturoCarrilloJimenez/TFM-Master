"""
Módulo para la extracción y formateo de cotizaciones bursátiles vía yfinance.
"""
from typing import List, Optional
import pandas as pd
import yfinance as yf

BANCOS_TICKERS_DEFAULT: List[str] = ["JPM", "BAC", "WFC", "GS", "MS", "C"]


def aplanar_columnas_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte el MultiIndex de columnas devuelto por yfinance en un Index simple
    uniendo los niveles mediante guiones bajos (ej. ('Close', 'JPM') -> 'Close_JPM').
    """
    df_procesado = df.copy()
    if isinstance(df_procesado.columns, pd.MultiIndex):
        df_procesado.columns = [
            '_'.join(str(col).strip() for col in c if str(col).strip())
            for c in df_procesado.columns
        ]
    df_procesado.index.name = "Date"
    return df_procesado


def descargar_cotizaciones_raw(
    tickers: Optional[List[str]] = None,
    start_date: str = "2010-01-01",
    end_date: str = "2024-01-01"
) -> pd.DataFrame:
    """
    Descarga cotizaciones históricas sin ajuste automático y aplana sus columnas.
    """
    simbolos = tickers or BANCOS_TICKERS_DEFAULT
    df_raw = yf.download(
        tickers=simbolos,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        threads=True
    )
    return aplanar_columnas_multiindex(df_raw)