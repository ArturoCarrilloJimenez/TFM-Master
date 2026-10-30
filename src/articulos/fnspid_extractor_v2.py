"""
Pipeline Ingesta Raw FNSPID v2: Universo 6 Grandes Bancos USA.
Recorre nasdaq_exteral_data.csv y All_external.csv en streaming.
Captura por Ticker exacto, Ticker sucio, Titular y URL, clasificando el motivo de captura.
Conserva copia en LOCAL y sube a Cloudflare R2 en raw/huggingface/fnspid_bancos_v2.parquet.
"""

import asyncio
import sys
import csv
import io
import re
import urllib.request
from pathlib import Path
import pandas as pd
from dependencies.dependencies import get_storage_repository
from shared.logging import Logger
from shared.tikets import TIKETS

logger = Logger("FNSIPDExtractorV2")

# 1. Ampliar límite de celda CSV para admitir artículos largos
max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)

# 2. Configuración y rutas de destino
URLS_HF_CSV = {
    "nasdaq_external": "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv",
    "all_external": "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/All_external.csv",
}

LOCAL_PARQUET = "fnspid_bancos_v2.parquet"
KEY_DESTINO = "raw/huggingface/fnspid_bancos_v2.parquet"

# Universo canónico de los 6 bancos derivado de shared.tikets
BANCOS: set[str] = set(TIKETS.keys())
VALORES_VACIOS = {"", "NAN", "NONE", "NULL"}

# 3. Expresiones regulares compiladas para máxima velocidad en el bucle
REGEX_DIRTY_TICKER = {
    'JPM': re.compile(r'(?:^|[\s,:$;/])JPM(?:$|[\s,.;/])'),
    'BAC': re.compile(r'(?:^|[\s,:$;/])BAC(?:$|[\s,.;/])'),
    'WFC': re.compile(r'(?:^|[\s,:$;/])WFC(?:$|[\s,.;/])'),
    'GS':  re.compile(r'(?:^|[\s,:$;/])GS(?:$|[\s,.;/])'),
    'MS':  re.compile(r'(?:^|[\s,:$;/])MS(?:$|[\s,.;/])'),
    'C':   re.compile(r'(?:^|[\s,:$;/])C(?:$|[\s,.;/])'),
}

REGEX_TITLE = {
    'JPM': re.compile(r'\b(jpmorgan|jp\s+morgan|j\.p\.\s*morgan|jamie\s+dimon|\(jpm\)|nyse:\s*jpm)\b', re.IGNORECASE),
    'BAC': re.compile(r'\b(bank\s+of\s+america|bofa|merrill\s+lynch|\(bac\)|nyse:\s*bac)\b', re.IGNORECASE),
    'WFC': re.compile(r'\b(wells\s+fargo|\(wfc\)|nyse:\s*wfc)\b', re.IGNORECASE),
    'C':   re.compile(r'\b(citigroup|citibank|citi\s+group|\bciti\b|\(nyse:\s*c\))\b', re.IGNORECASE),
    'GS':  re.compile(r'\b(goldman\s+sachs|\(gs\)|nyse:\s*gs)\b', re.IGNORECASE),
    'MS':  re.compile(r'\b(morgan\s+stanley|\(ms\)|nyse:\s*ms)\b', re.IGNORECASE),
}

REGEX_URL = {
    'JPM': re.compile(r'(jpmorgan|jp-morgan|[-/]jpm[-/.])', re.IGNORECASE),
    'BAC': re.compile(r'(bank-of-america|bofa|merrill-lynch|[-/]bac[-/.])', re.IGNORECASE),
    'WFC': re.compile(r'(wells-fargo|[-/]wfc[-/.])', re.IGNORECASE),
    'C':   re.compile(r'(citigroup|citibank|[-/]citi[-/.])', re.IGNORECASE),
    'GS':  re.compile(r'(goldman-sachs|[-/]gs[-/.])', re.IGNORECASE),
    'MS':  re.compile(r'(morgan-stanley|[-/]ms[-/.])', re.IGNORECASE),
}


def _limpiar_ticker(raw_ticker: str) -> str:
    """Normaliza el símbolo bursátil y descarta nulos textuales."""
    limpio = str(raw_ticker or "").strip().upper()
    return "" if limpio in VALORES_VACIOS else limpio


def _obtener_detalle_match(banco: str, title: str, url: str) -> str:
    """Devuelve el detalle si hay coincidencia en título o URL, o cadena vacía."""
    match_t = REGEX_TITLE[banco].search(title)
    if match_t:
        return f"Title:{match_t.group(0)}"
    match_u = REGEX_URL[banco].search(url)
    if match_u:
        return f"URL:{match_u.group(0)}"
    return ""


def _buscar_en_texto(ticker_clean: str, title: str, url: str) -> list[tuple[str, str, str]]:
    """Evalúa menciones en titular o URL cuando no hay match directo de ticker."""
    hallazgos = []
    metodo = "CROSS_MENTION_OTHER_TICKER" if ticker_clean else "RESCUED_NO_TICKER"
    prefijo = f"Orig:{ticker_clean}|" if ticker_clean else ""

    for banco in BANCOS:
        detalle = _obtener_detalle_match(banco, title, url)
        if detalle:
            hallazgos.append((banco, metodo, f"{prefijo}{detalle}"))

    return hallazgos


def clasificar_fila(raw_ticker: str, title: str, url: str) -> list[tuple[str, str, str]]:
    """
    Evalúa si una fila pertenece a alguno de los 6 bancos.
    Devuelve una lista de tuplas: [(ticker_asignado, metodo_captura, detalle_match), ...]
    """
    ticker_clean = _limpiar_ticker(raw_ticker)

    # 1. Match Exacto en Stock_symbol
    if ticker_clean in BANCOS:
        return [(ticker_clean, "EXACT_TICKER", ticker_clean)]

    # 2. Match en Stock_symbol sucio (ej. "NYSE:JPM" o "JPM, BAC")
    if ticker_clean:
        sucios = [
            (banco, "DIRTY_TICKER", ticker_clean)
            for banco, pattern in REGEX_DIRTY_TICKER.items()
            if pattern.search(ticker_clean)
        ]
        if sucios:
            return sucios

    # 3. Búsqueda en Article_title y Url
    return _buscar_en_texto(ticker_clean, str(title or ""), str(url or ""))


def _construir_filas_capturadas(row: dict[str, str], matches: list[tuple[str, str, str]], fuente_nombre: str) -> list[dict[str, str]]:
    """Genera los diccionarios enriquecidos para cada match de una fila."""
    raw_ticker = str(row.get("Stock_symbol") or "").strip()
    nuevas = []
    for ticker_inferido, metodo, detalle in matches:
        nueva_fila = dict(row)
        nueva_fila["Raw_Stock_symbol"] = raw_ticker
        nueva_fila["Stock_symbol"] = ticker_inferido
        nueva_fila["Metodo_Captura"] = metodo
        nueva_fila["Detalle_Match"] = detalle
        nueva_fila["Fuente_Archivo"] = fuente_nombre
        nuevas.append(nueva_fila)
    return nuevas


def procesar_fuente_streaming(fuente_nombre: str, url_csv: str, registros: list[dict[str, str]]) -> None:
    """Descarga en streaming un CSV de Hugging Face y acumula las filas filtradas."""
    logger.info(f"Iniciando lectura en streaming de: {fuente_nombre} ({url_csv})")
    

    req = urllib.request.Request(url_csv, headers={'User-Agent': 'Mozilla/5.0'})
    total_leidos = 0
    capturados_fuente = 0

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            stream = io.TextIOWrapper(response, encoding='utf-8', errors='replace')
            reader = csv.DictReader(stream)

            for row in reader:
                total_leidos += 1
                matches = clasificar_fila(
                    row.get("Stock_symbol"),
                    row.get("Article_title"),
                    row.get("Url")
                )
                if matches:
                    nuevas_filas = _construir_filas_capturadas(row, matches, fuente_nombre)
                    registros.extend(nuevas_filas)
                    capturados_fuente += len(nuevas_filas)

                if total_leidos % 500000 == 0:
                    logger.info(
                        f"[{fuente_nombre}] Filas leídas: {total_leidos:,} | "
                        f"Capturadas aquí: {capturados_fuente:,} | "
                        f"Total acumulado: {len(registros):,}"
                    )
    except Exception as e:
        logger.warning(f"Interrupción o fin de stream en {fuente_nombre} tras {total_leidos:,} filas: {e}")

if __name__ == '__main__':
    # 4. Ejecución del pipeline de lectura
    registros_totales = []
    for nombre_fuente, url_archivo in URLS_HF_CSV.items():
        procesar_fuente_streaming(nombre_fuente, url_archivo, registros_totales)

    logger.info(f"Lectura global finalizada. Total registros capturados: {len(registros_totales):,}")

    # 5. Generación de Parquet LOCAL (se conserva en disco para análisis)
    logger.info(f"Generando archivo Parquet local: {LOCAL_PARQUET}...")
    df = pd.DataFrame(registros_totales).astype(str)
    df.to_parquet(LOCAL_PARQUET, index=False, compression='zstd')
    size_mb = Path(LOCAL_PARQUET).stat().st_size / (1024 * 1024)
    logger.info(f"Archivo local guardado con éxito: {LOCAL_PARQUET} ({size_mb:.2f} MB)")

    # Resumen rápido por consola antes de subir
    
    logger.info("Resumen por método de captura y banco:\n" + str(pd.crosstab(df["Stock_symbol"], df["Metodo_Captura"], margins=True)))

    # 6. Subida a Cloudflare R2 vía Storage Repository
    logger.info(f"Subiendo a Cloudflare R2 vía Storage Repository: {KEY_DESTINO}...")
    storage = get_storage_repository()
    asyncio.run(storage.save(key=KEY_DESTINO, body=Path(LOCAL_PARQUET).read_bytes()))
    logger.info("Subida completada con éxito.")
    
    logger.info(f"Copia persistida en local: {Path(LOCAL_PARQUET).resolve()}")

