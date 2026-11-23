"""
ETL Job: Procesamiento y carga del dataset multisource para los 6 bancos (2010-2023).
Extrae de los parquets locales, separa explícitamente en 'title' (titular) y 'article' (cuerpo),
filtra falsos positivos léxicos en Citigroup ('C'), normaliza fechas a Wall Street (America/New_York),
deduplica, calcula trazabilidad ('record_id', 'ticker_count') y carga a Cloudflare R2.
"""

import asyncio
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from dependencies.dependencies import get_compactor, get_storage_repository
from shared.ids import generar_record_id
from shared.logging import Logger
from shared.tickets import TICKETS

logger = Logger("ETLMultisource")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "data" / "financial_news_multisource" / "data"
LOCAL_OUTPUT = PROJECT_ROOT / "src" / "data" / "multisource_bancos_2010_2023.parquet"
R2_OBJECT_KEY = "raw/huggingface/multisource_bancos_2010_2023.parquet"
MARKET_TZ = "America/New_York"


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


def normalizar_fechas_ny(
    df: pd.DataFrame,
    source_col: str = "date",
    preserve_utc_str_col: str | None = "_date_utc_id",
) -> pd.DataFrame:
    """Convierte timestamps UTC a hora de Wall Street (`America/New_York`) sin desplazar fechas `00:00:00`.

    Para registros con reloj real (`has_exact_time=True`), convierte de UTC a `America/New_York` y
    clasifica `market_session` en `premarket`, `intraday` o `afterhours`. Para registros que solo
    traen fecha (`00:00:00` en origen, ej. `fnspid_news`), conserva exactamente el día calendario
    en `America/New_York` a las `00:00:00` y asigna `market_session="date_only"`.
    """
    out = df.copy()
    raw_input = out["datetime"] if "datetime" in out.columns and source_col not in out.columns else out[source_col]
    dt_utc = pd.to_datetime(raw_input, utc=True, errors="coerce")
    out = out.loc[dt_utc.notna()].copy()
    dt_utc = dt_utc.loc[out.index]

    if preserve_utc_str_col:
        out[preserve_utc_str_col] = dt_utc.dt.strftime("%Y-%m-%d %H:%M:%S")

    has_exact = ~((dt_utc.dt.hour == 0) & (dt_utc.dt.minute == 0) & (dt_utc.dt.second == 0))
    dt_converted = dt_utc.dt.tz_convert(MARKET_TZ)
    dt_midnight_ny = dt_utc.dt.tz_localize(None).dt.tz_localize(MARKET_TZ)
    dt_ny = dt_converted.where(has_exact, dt_midnight_ny)

    mins_ny = dt_ny.dt.hour * 60 + dt_ny.dt.minute
    market_session = np.where(
        ~has_exact,
        "date_only",
        np.where(mins_ny < 570, "premarket", np.where(mins_ny < 960, "intraday", "afterhours")),
    )

    out["datetime"] = dt_ny
    out["date"] = dt_ny.dt.strftime("%Y-%m-%d %H:%M:%S")
    out["has_exact_time"] = has_exact.astype(bool)
    out["market_session"] = market_session
    return out


def extract_and_transform() -> pd.DataFrame:
    """Extrae datos con DuckDB aplicando filtros de fecha, tickers y normalización horaria ET."""
    logger.info("1. Iniciando extracción y transformación con DuckDB...")

    con = duckdb.connect()
    tickers = list(TICKETS.keys())
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

    # Enriquecer con metadatos de TICKETS
    df["company_name"] = df["ticker"].map(lambda t: TICKETS.get(t, {}).get("name", ""))
    df["cik"] = df["ticker"].map(lambda t: TICKETS.get(t, {}).get("cik", ""))

    # Normalizar fechas a Wall Street (America/New_York) preservando fecha UTC para record_id estable
    df = normalizar_fechas_ny(df, source_col="date", preserve_utc_str_col="_date_utc_id")

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
    antes_citi = len(df)
    es_basura_citi = (
        (df["ticker"] == "C")
        & (~df["title"].str.contains("citi", case=False, na=False))
        & (~df["text"].str.contains("citi", case=False, na=False))
    )
    df = df[~es_basura_citi].reset_index(drop=True)
    eliminados_citi = antes_citi - len(df)
    logger.info(f"Falsos positivos de Citigroup ('C') eliminados: {eliminados_citi:,}")

    # Conteo de tickers por noticia única (title + datetime) y flag multisujeto
    df["ticker_count"] = df.groupby(["title", "datetime"])["ticker"].transform("nunique")
    df["is_multisubject"] = df["ticker_count"] > 1

    # Identificador determinista estable usando la cadena UTC original para no invalidar IDs históricos
    df["record_id"] = [
        generar_record_id(str(t), str(d), str(tit))
        for t, d, tit in zip(df["ticker"], df["_date_utc_id"], df["title"])
    ]

    logger.info(f"Total registros limpios consolidados: {len(df):,}")

    con_articulo = (df["article"].str.len() > 0).sum()
    logger.info(f"Registros con cuerpo de artículo ('article'): {con_articulo:,} ({con_articulo / len(df) * 100:.2f}%)")
    logger.info(f"Registros que son titulares / alertas breves: {len(df) - con_articulo:,}")
    logger.info(f"Registros de señal pura (1 ticker exclusivo): {(df['ticker_count'] == 1).sum():,}")
    logger.info(f"Registros multisujeto (compartidos por 2+ tickers): {df['is_multisubject'].sum():,}")

    columnas_finales = [
        "record_id",
        "ticker",
        "company_name",
        "cik",
        "date",
        "datetime",
        "has_exact_time",
        "market_session",
        "title",
        "article",
        "text",
        "publisher",
        "url",
        "source_dataset",
        "ticker_count",
        "is_multisubject",
    ]
    df = df[columnas_finales]

    logger.info("Distribución por banco:\n" + df["ticker"].value_counts().to_string())

    df["year"] = df["datetime"].dt.year
    logger.info("Distribución anual (2010 - 2023):\n" + pd.crosstab(df["year"], df["ticker"]).to_string())
    df = df.drop(columns=["year"])

    return df


def save_and_upload(df: pd.DataFrame) -> None:
    """Compacta en memoria con PyArrow ZSTD, guarda backup local y sube a Cloudflare R2."""
    logger.info("2. Compactando datos con infraestructura centralizada y subiendo a Cloudflare R2...")
    compact = get_compactor()
    records = df.to_dict(orient="records")
    parquet_bytes = compact.compact(records, compression="ZSTD")

    # 1. Guardar local
    LOCAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Guardando copia local en: {LOCAL_OUTPUT}")
    LOCAL_OUTPUT.write_bytes(parquet_bytes)
    file_size_mb = len(parquet_bytes) / (1024 * 1024)
    logger.info(f"Archivo local generado exitosamente: {file_size_mb:.2f} MB")

    # 2. Conectar a S3/R2 mediante el repositorio centralizado de infraestructura
    repo = get_storage_repository()
    logger.info(f"Subiendo dataset consolidado al bucket [{repo.bucket_name}] -> '{R2_OBJECT_KEY}'...")

    async def _upload() -> None:
        try:
            await repo.save(key=R2_OBJECT_KEY, body=parquet_bytes)
            logger.info(f"Dataset subido a R2 exitosamente ({file_size_mb:.2f} MB).")
        finally:
            await repo.close()

    asyncio.run(_upload())


if __name__ == "__main__":
    df_clean = extract_and_transform()
    save_and_upload(df_clean)
