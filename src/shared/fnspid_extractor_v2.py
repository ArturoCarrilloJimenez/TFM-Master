"""
Pipeline Ingesta Raw FNSPID v2: Universo 6 Grandes Bancos USA.
Recorre nasdaq_exteral_data.csv y All_external.csv en streaming.
Captura por Ticker exacto, Ticker sucio, Titular y URL, clasificando el motivo de captura.
Conserva copia en LOCAL y sube a Cloudflare R2 en raw/huggingface/fnspid_bancos_v2.parquet.
"""

import os
import sys
import csv
import io
import re
import urllib.request
from pathlib import Path
import boto3
from botocore.client import Config
import pandas as pd
from dotenv import load_dotenv

# 1. Ampliar límite de celda CSV para admitir artículos largos
max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)

# 2. Carga de credenciales desde .env
env_path = Path(__file__).resolve().parent / ".env"
for parent in Path(__file__).resolve().parents:
    if (parent / ".env").exists():
        env_path = parent / ".env"
        break
load_dotenv(dotenv_path=env_path)

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "").rstrip("/")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "").strip()
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "auto")
S3_PREFIX_RAW = os.getenv("S3_PREFIX_RAW", "raw").strip("/")

if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_ENDPOINT_URL, S3_BUCKET_NAME]):
    raise ValueError("Faltan variables críticas de R2 en el archivo .env")

if S3_BUCKET_NAME in S3_ENDPOINT_URL:
    S3_ENDPOINT_URL = S3_ENDPOINT_URL.replace(f"/{S3_BUCKET_NAME}", "")

if "r2.cloudflarestorage.com" in S3_ENDPOINT_URL and ".eu.r2." not in S3_ENDPOINT_URL:
    S3_ENDPOINT_URL = S3_ENDPOINT_URL.replace(".r2.cloudflarestorage.com", ".eu.r2.cloudflarestorage.com")

# Ambas fuentes de datos de FNSPID en Hugging Face
URLS_HF_CSV = {
    "nasdaq_external": "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv",
    "all_external": "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/All_external.csv"
}

LOCAL_PARQUET = "fnspid_bancos_v2.parquet"
KEY_DESTINO = f"{S3_PREFIX_RAW}/huggingface/fnspid_bancos_v2.parquet"

BANCOS = {'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS'}
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


def _buscar_en_texto(ticker_clean: str, title: str, url: str) -> list:
    """Evalúa menciones en titular o URL cuando no hay match directo de ticker."""
    hallazgos = []
    metodo = "CROSS_MENTION_OTHER_TICKER" if ticker_clean else "RESCUED_NO_TICKER"
    prefijo = f"Orig:{ticker_clean}|" if ticker_clean else ""

    for banco in BANCOS:
        detalle = _obtener_detalle_match(banco, title, url)
        if detalle:
            hallazgos.append((banco, metodo, f"{prefijo}{detalle}"))

    return hallazgos


def clasificar_fila(raw_ticker: str, title: str, url: str) -> list:
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


def _construir_filas_capturadas(row: dict, matches: list, fuente_nombre: str) -> list:
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


def procesar_fuente_streaming(fuente_nombre: str, url_csv: str, registros: list) -> None:
    """Descarga en streaming un CSV de Hugging Face y acumula las filas filtradas."""
    separador = "=" * 80
    print(f"\n{separador}")
    print(f"Iniciando lectura en streaming de: {fuente_nombre} ({url_csv})")
    print(separador)

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
                    print(
                        f"[{fuente_nombre}] Filas leídas: {total_leidos:,} | "
                        f"Capturadas aquí: {capturados_fuente:,} | "
                        f"Total acumulado: {len(registros):,}"
                    )
    except Exception as e:
        print(f"\n[AVISO] Interrupción o fin de stream en {fuente_nombre} tras {total_leidos:,} filas: {e}")

if __name__ == '__main__': 
    
# 4. Ejecución del pipeline de lectura
registros_totales = []
for nombre_fuente, url_archivo in URLS_HF_CSV.items():
    procesar_fuente_streaming(nombre_fuente, url_archivo, registros_totales)

print(f"\nLectura global finalizada. Total registros capturados: {len(registros_totales):,}")

# 5. Generación de Parquet LOCAL (se conserva en disco para análisis)
print(f"\nGenerando archivo Parquet local: {LOCAL_PARQUET}...")
df = pd.DataFrame(registros_totales).astype(str)
df.to_parquet(LOCAL_PARQUET, index=False, compression='snappy')
size_mb = os.path.getsize(LOCAL_PARQUET) / (1024 * 1024)
print(f"Archivo local guardado con éxito: {LOCAL_PARQUET} ({size_mb:.2f} MB)")

# Resumen rápido por consola antes de subir
print("\n--- RESUMEN POR MÉTODO DE CAPTURA Y BANCO ---")
print(pd.crosstab(df["Stock_symbol"], df["Metodo_Captura"], margins=True))

# 6. Subida a Cloudflare R2
s3_client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION,
    config=Config(
        s3={"addressing_style": "path"},
        signature_version="s3v4"
    )
)

print(f"\nSubiendo a Cloudflare R2: s3://{S3_BUCKET_NAME}/{KEY_DESTINO}...")
# Cloudflare R2 es S3-compatible y no utiliza ExpectedBucketOwner de AWS IAM
s3_client.upload_file(LOCAL_PARQUET, S3_BUCKET_NAME, KEY_DESTINO)  # NOSONAR
print("¡SUBIDA COMPLETADA CON ÉXITO!")
print(f"1. Copia en la nube: s3://{S3_BUCKET_NAME}/{KEY_DESTINO}")
print(f"2. Copia en tu disco local: {os.path.abspath(LOCAL_PARQUET)}")

