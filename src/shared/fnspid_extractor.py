"""
Módulo para la normalización y filtrado en streaming del dataset FNSPID.
"""
import sys
import csv
import io
from typing import Set, List, Dict, Any, Optional

BANCOS_TARGET_DEFAULT: Set[str] = {'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS'}


def ajustar_limite_csv() -> None:
    """Eleva el límite de tamaño de celda CSV para admitir artículos largos."""
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)


def normalizar_ticker(ticker_raw: Optional[Any]) -> str:
    """Limpia y pasa a mayúsculas el símbolo bursátil."""
    if not ticker_raw:
        return ""
    return str(ticker_raw).strip().upper()


def procesar_stream_noticias(
    stream_io: io.TextIOBase,
    bancos_permitidos: Set[str]
) -> List[Dict[str, str]]:
    """
    Lee un stream CSV y captura los registros que coinciden con los bancos objetivo.
    Conserva todos los campos y valores textuales originales sin descarte por fecha.
    """
    reader = csv.DictReader(stream_io)
    registros: List[Dict[str, str]] = []

    for row in reader:
        ticker = normalizar_ticker(row.get("Stock_symbol"))
        if ticker in bancos_permitidos:
            row["Stock_symbol"] = ticker
            registros.append(row)

    return registros