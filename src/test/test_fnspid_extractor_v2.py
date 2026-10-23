import os
import pytest
from unittest.mock import patch, MagicMock
import io

# 1. Inyectamos variables de entorno FALSAS antes de importar el script.
# Esto evita que el entorno de GitHub Actions (que no tiene el .env) colapse.
os.environ["AWS_ACCESS_KEY_ID"] = "mock_key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "mock_secret"
os.environ["S3_ENDPOINT_URL"] = "https://mock.eu.r2.cloudflarestorage.com"
os.environ["S3_BUCKET_NAME"] = "mock_bucket"

from src.shared.fnspid_extractor_v2 import (
    _limpiar_ticker,
    clasificar_fila,
    _construir_filas_capturadas,
    procesar_fuente_streaming
)

def test_limpiar_ticker():
    assert _limpiar_ticker(" jpm ") == "JPM"
    assert _limpiar_ticker("NAN") == ""
    assert _limpiar_ticker(None) == ""

def test_clasificar_fila_exacto():
    resultado = clasificar_fila("MS", "Titular irrelevante", "http://url.com")
    assert resultado == [("MS", "EXACT_TICKER", "MS")]

def test_clasificar_fila_sucio():
    resultado = clasificar_fila("NYSE:JPM", "Titular irrelevante", "http://url.com")
    assert resultado[0] == ("JPM", "DIRTY_TICKER", "NYSE:JPM")

def test_clasificar_fila_rescate_titular():
    resultado = clasificar_fila("", "Morgan Stanley anuncia resultados", "http://url.com")
    assert len(resultado) == 1
    assert resultado[0][0] == "MS"
    assert resultado[0][1] == "RESCUED_NO_TICKER"
    assert "Title:Morgan Stanley" in resultado[0][2]

def test_clasificar_fila_rescate_url():
    resultado = clasificar_fila("", "Noticia genérica", "http://news.com/goldman-sachs-rally")
    assert len(resultado) == 1
    assert resultado[0][0] == "GS"
    assert resultado[0][1] == "RESCUED_NO_TICKER"
    assert "URL:goldman-sachs" in resultado[0][2]

def test_construir_filas():
    row = {"Stock_symbol": "basura", "Article_title": "Morgan Stanley wins"}
    matches = [("MS", "RESCUED_NO_TICKER", "Title:Morgan Stanley")]
    nuevas = _construir_filas_capturadas(row, matches, "test_fuente")
    
    assert len(nuevas) == 1
    assert nuevas[0]["Raw_Stock_symbol"] == "basura"
    assert nuevas[0]["Stock_symbol"] == "MS"
    assert nuevas[0]["Fuente_Archivo"] == "test_fuente"

@patch("src.shared.fnspid_extractor.urllib.request.urlopen")
def test_procesar_fuente_streaming(mock_urlopen):
    # Simulamos un CSV en memoria para no descargar nada de internet
    csv_content = b"Stock_symbol,Article_title,Url\nMS,Fake news,http://\n,Goldman Sachs sube,http://\n"
    
    mock_response = MagicMock()
    mock_response.__enter__.return_value = io.BytesIO(csv_content)
    mock_urlopen.return_value = mock_response

    registros = []
    procesar_fuente_streaming("mock_csv", "http://fake-url.com", registros)

    # Debería capturar el ticker MS exacto y rescatar GS del titular
    assert len(registros) == 2
    assert registros[0]["Stock_symbol"] == "MS"
    assert registros[1]["Stock_symbol"] == "GS"