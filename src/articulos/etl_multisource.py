"""
ETL Job: Procesamiento y carga del dataset multisource para los 6 bancos (2010-2023).
Extrae de los parquets locales, separa explícitamente en 'title' (titular) y 'article' (cuerpo),
filtra falsos positivos léxicos en Citigroup ('C'), calcula el conteo de tickers asociados ('ticker_count'),
normaliza, enriquece, deduplica y carga a Cloudflare R2 bajo 'raw/huggingface/multisource_bancos_2010_2023.parquet'.
"""

import asyncio
import sys
from pathlib import Path
import duckdb
import pandas as pd

# Asegurar path para módulos del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dependencies.dependencies import get_storage_repository
from shared.logging import Logger

logger = Logger("ETLMultisource")
from shared.tikets import TIKETS

DATA_DIR = PROJECT_ROOT / "src" / "data" / "financial_news_multisource" / "data"
LOCAL_OUTPUT = PROJECT_ROOT / "src" / "data" / "multisource_bancos_2010_2023.parquet"
R2_OBJECT_KEY = "raw/huggingface/multisource_bancos_2010_2023.parquet"


def extraer_title_y_article(raw_text: str) -> tuple[str, str]:
    """Separa el texto en titular (primera línea) y cuerpo del artículo (resto del texto)."""
    if not raw_text or pd.isna(raw_text):
        return "", ""
    raw = str(raw_text).strip()
    if "\n" in raw:
        parts = raw.split("\n", 1)
        title = parts[0].strip()
        article = parts[1].strip()
    else:
        title = raw
        article = ""
    return title, article


def extract_and_transform() -> pd.DataFrame:
    """Extrae datos con DuckDB aplicando filtros de fecha y tickers."""
    
    logger.info("1. Iniciando extracción y transformación con DuckDB...")
    

    con = duckdb.connect()
    tickers = list(TIKETS.keys())
    tickers_sql = ", ".join([f"'{t}'" for t in tickers])

    query = f"""
    WITH fnspid AS (
        SELECT 
            'fnspid_news' AS source_dataset,
            date,
            text,
            extra_fields->>'publisher' AS publisher,
            extra_fields->>'url' AS url,
            unnest(from_json(extra_fields->>'stocks', '["VARCHAR"]')) AS ticker
        FROM read_parquet('{DATA_DIR.as_posix()}/fnspid_news/*.parquet')
        WHERE date >= '2010-01-01' AND date <= '2023-12-31T23:59:59'
          AND (
              extra_fields LIKE '%"JPM"%' OR extra_fields LIKE '%"BAC"%' OR
              extra_fields LIKE '%"WFC"%' OR extra_fields LIKE '%"C"%' OR
              extra_fields LIKE '%"GS"%' OR extra_fields LIKE '%"MS"%'
          )
    ),
    benzinga AS (
        SELECT 
            'benzinga_6000stocks' AS source_dataset,
            date,
            text,
            extra_fields->>'source' AS publisher,
            extra_fields->>'url' AS url,
            unnest(from_json(extra_fields->>'stocks', '["VARCHAR"]')) AS ticker
        FROM read_parquet('{DATA_DIR.as_posix()}/benzinga_6000stocks/*.parquet')
        WHERE date >= '2010-01-01' AND date <= '2023-12-31T23:59:59'
          AND (
              extra_fields LIKE '%"JPM"%' OR extra_fields LIKE '%"BAC"%' OR
              extra_fields LIKE '%"WFC"%' OR extra_fields LIKE '%"C"%' OR
              extra_fields LIKE '%"GS"%' OR extra_fields LIKE '%"MS"%'
          )
    ),
    sentarl AS (
        SELECT 
            'sentarl_combined' AS source_dataset,
            date,
            text,
            extra_fields->>'source' AS publisher,
            extra_fields->>'url' AS url,
            unnest(from_json(extra_fields->>'stocks', '["VARCHAR"]')) AS ticker
        FROM read_parquet('{DATA_DIR.as_posix()}/sentarl_combined/*.parquet')
        WHERE date >= '2010-01-01' AND date <= '2023-12-31T23:59:59'
          AND (
              extra_fields LIKE '%"JPM"%' OR extra_fields LIKE '%"BAC"%' OR
              extra_fields LIKE '%"WFC"%' OR extra_fields LIKE '%"C"%' OR
              extra_fields LIKE '%"GS"%' OR extra_fields LIKE '%"MS"%'
          )
    ),
    yahoo_felix AS (
        SELECT 
            'yahoo_finance_felixdrinkall' AS source_dataset,
            date,
            text,
            extra_fields->>'news_outlet' AS publisher,
            extra_fields->>'url' AS url,
            unnest(from_json(extra_fields->>'stocks', '["VARCHAR"]')) AS ticker
        FROM read_parquet('{DATA_DIR.as_posix()}/yahoo_finance_felixdrinkall/*.parquet')
        WHERE date >= '2010-01-01' AND date <= '2023-12-31T23:59:59'
          AND (
              extra_fields LIKE '%"JPM"%' OR extra_fields LIKE '%"BAC"%' OR
              extra_fields LIKE '%"WFC"%' OR extra_fields LIKE '%"C"%' OR
              extra_fields LIKE '%"GS"%' OR extra_fields LIKE '%"MS"%'
          )
    ),
    unificado AS (
        SELECT * FROM fnspid
        UNION ALL
        SELECT * FROM benzinga
        UNION ALL
        SELECT * FROM sentarl
        UNION ALL
        SELECT * FROM yahoo_felix
    )
    SELECT 
        ticker,
        date,
        source_dataset,
        publisher,
        text,
        url
    FROM unificado
    WHERE ticker IN ({tickers_sql})
    ORDER BY date ASC;
    """

    logger.info("Ejecutando consulta de pushdown predicates sobre Parquets en disco...")
    try:
        df = con.execute(query).df()
    finally:
        con.close()
    logger.info(f"Total filas extraídas brutas: {len(df):,}")

    # Enriquecer con metadatos de TIKETS
    df["company_name"] = df["ticker"].map(lambda t: TIKETS.get(t, ("", ""))[1])
    df["cik"] = df["ticker"].map(lambda t: TIKETS.get(t, ("", ""))[0])

    # Normalizar fechas a Timestamp UTC
    df["datetime"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()
    df["date"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Separación explícita en 'title' (titular) y 'article' (cuerpo)
    logger.info("Separando textos en 'title' (titular) y 'article' (cuerpo extenso)...")
    splits = df["text"].apply(extraer_title_y_article)
    df["title"] = [s[0] for s in splits]
    df["article"] = [s[1] for s in splits]

    # Deduplicación estricta por ticker, fecha y titular
    antes_dedup = len(df)
    df = df.drop_duplicates(subset=["ticker", "date", "title"]).reset_index(drop=True)
    eliminados = antes_dedup - len(df)
    logger.info(f"Duplicados eliminados: {eliminados:,} ({eliminados / antes_dedup * 100:.2f}%)")

    # Filtrar falsos positivos de Citigroup ('C')
    # Descartar registros de 'C' que no contengan 'citi' en el titular ni en el cuerpo del texto
    antes_citi = len(df)
    es_basura_citi = (df["ticker"] == "C") & \
                     (~df["title"].str.contains("citi", case=False, na=False)) & \
                     (~df["text"].str.contains("citi", case=False, na=False))
    df = df[~es_basura_citi].reset_index(drop=True)
    eliminados_citi = antes_citi - len(df)
    logger.info(f"Falsos positivos de Citigroup ('C') eliminados: {eliminados_citi:,}")

    # Conteo de tickers por noticia única (title + datetime) y flag multisujeto
    df["ticker_count"] = df.groupby(["title", "datetime"])["ticker"].transform("nunique")
    df["is_multisubject"] = df["ticker_count"] > 1

    logger.info(f"Total registros limpios consolidados: {len(df):,}")

    con_articulo = (df["article"].str.len() > 0).sum()
    logger.info(f"Registros con cuerpo de artículo ('article'): {con_articulo:,} ({con_articulo / len(df) * 100:.2f}%)")
    logger.info(f"Registros que son titulares / alertas breves: {len(df) - con_articulo:,}")
    logger.info(f"Registros de señal pura (1 ticker exclusivo): {(df['ticker_count'] == 1).sum():,}")
    logger.info(f"Registros multisujeto (compartidos por 2+ tickers): {df['is_multisubject'].sum():,}")

    columnas_finales = [
        "ticker",
        "company_name",
        "cik",
        "date",
        "datetime",
        "title",
        "article",
        "text",
        "publisher",
        "url",
        "source_dataset",
        "ticker_count",
        "is_multisubject"
    ]
    df = df[columnas_finales]

    
    logger.info("Distribución por banco:\n" + df["ticker"].value_counts().to_string())

    df["year"] = df["datetime"].dt.year
    
    logger.info("Distribución anual (2010 - 2023):\n" + pd.crosstab(df["year"], df["ticker"]).to_string())
    df = df.drop(columns=["year"])

    return df


def save_and_upload(df: pd.DataFrame) -> None:
    """Guarda en local como Parquet y sube a Cloudflare R2 usando el Storage Repository."""
    logger.info("2. Guardando archivo local y subiendo a Cloudflare R2...")

    # 1. Guardar local
    LOCAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Guardando Parquet comprimido en: {LOCAL_OUTPUT}")
    df.to_parquet(LOCAL_OUTPUT, compression="zstd", index=False)
    file_size_mb = LOCAL_OUTPUT.stat().st_size / (1024 * 1024)
    logger.info(f"Archivo local generado exitosamente: {file_size_mb:.2f} MB")

    # 2. Conectar a S3/R2 mediante el repositorio centralizado de infraestructura
    logger.info(f"Conectando a R2 mediante repositorio - Destino: {R2_OBJECT_KEY}")
    storage = get_storage_repository()

    logger.info("Subiendo archivo Parquet a R2...")
    parquet_bytes = LOCAL_OUTPUT.read_bytes()
    asyncio.run(storage.save(key=R2_OBJECT_KEY, body=parquet_bytes))
    logger.info("Subida a R2 completada con éxito.")

    logger.info("ETL completada satisfactoriamente.")


if __name__ == "__main__":
    df_clean = extract_and_transform()
    save_and_upload(df_clean)
