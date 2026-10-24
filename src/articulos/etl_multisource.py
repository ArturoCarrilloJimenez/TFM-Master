"""
ETL Job: Procesamiento y carga del dataset multisource para los 6 bancos (2010-2023).
Extrae de los parquets locales, separa explícitamente en 'title' (titular) y 'article' (cuerpo),
normaliza, enriquece, deduplica y carga a Cloudflare R2 bajo 'raw/huggingface/multisource_bancos_2010_2023.parquet'.
"""

import sys
from pathlib import Path
import duckdb
import pandas as pd
import boto3

# Asegurar path para módulos del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
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
    print("=" * 70)
    print("1. INICIANDO EXTRACCIÓN Y TRANSFORMACIÓN CON DUCKDB")
    print("=" * 70)

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

    print("Ejecutando consulta de pushdown predicates sobre Parquets en disco...")
    df = con.execute(query).df()
    print(f"Total filas extraídas brutas: {len(df):,}")

    # Enriquecer con metadatos de TIKETS
    df["company_name"] = df["ticker"].map(lambda t: TIKETS.get(t, ("", ""))[1])
    df["cik"] = df["ticker"].map(lambda t: TIKETS.get(t, ("", ""))[0])

    # Normalizar fechas a Timestamp UTC
    df["datetime"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()
    df["date"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Separación explícita en 'title' (titular) y 'article' (cuerpo)
    print("Separando textos en 'title' (titular) y 'article' (cuerpo extenso)...")
    splits = df["text"].apply(extraer_title_y_article)
    df["title"] = [s[0] for s in splits]
    df["article"] = [s[1] for s in splits]

    # Deduplicación estricta por ticker, fecha y titular
    antes_dedup = len(df)
    df = df.drop_duplicates(subset=["ticker", "date", "title"]).reset_index(drop=True)
    eliminados = antes_dedup - len(df)
    print(f"Duplicados eliminados: {eliminados:,} ({eliminados / antes_dedup * 100:.2f}%)")
    print(f"Total registros limpios consolidados: {len(df):,}")

    con_articulo = (df["article"].str.len() > 0).sum()
    print(f"• Registros con cuerpo de artículo ('article'): {con_articulo:,} ({con_articulo / len(df) * 100:.2f}%)")
    print(f"• Registros que son titulares / alertas breves: {len(df) - con_articulo:,}")

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
        "source_dataset"
    ]
    df = df[columnas_finales]

    print("\n--- DISTRIBUCIÓN POR BANCO ---")
    print(df["ticker"].value_counts().to_string())

    df["year"] = df["datetime"].dt.year
    print("\n--- DISTRIBUCIÓN ANUAL (2010 - 2023) ---")
    print(pd.crosstab(df["year"], df["ticker"]).to_string())
    df = df.drop(columns=["year"])

    return df


def save_and_upload(df: pd.DataFrame) -> None:
    """Guarda en local como Parquet y sube a Cloudflare R2."""
    print("\n" + "=" * 70)
    print("2. GUARDANDO ARCHIVO LOCAL Y SUBIENDO A CLOUDFLARE R2")
    print("=" * 70)

    # 1. Guardar local
    LOCAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Guardando Parquet comprimido en: {LOCAL_OUTPUT}")
    df.to_parquet(LOCAL_OUTPUT, compression="zstd", index=False)
    file_size_mb = LOCAL_OUTPUT.stat().st_size / (1024 * 1024)
    print(f"-> Archivo local generado exitosamente: {file_size_mb:.2f} MB")

    # 2. Conectar a S3/R2
    print("\nConectando a Cloudflare R2:")
    print(f"  Bucket: {settings.s3_bucket_name}")
    print(f"  Destino: {R2_OBJECT_KEY}")

    s3 = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.s3_region,
    )

    print("Subiendo a R2...")
    s3.upload_file(str(LOCAL_OUTPUT), settings.s3_bucket_name, R2_OBJECT_KEY) # NOSONAR
    print("-> Subida a R2 completada con éxito.")

    # 3. Verificar en R2
    head = s3.head_object(Bucket=settings.s3_bucket_name, Key=R2_OBJECT_KEY) # NOSONAR
    r2_size_mb = head["ContentLength"] / (1024 * 1024)
    print(f"-> Verificación R2: Tamaño={r2_size_mb:.2f} MB | LastModified={head['LastModified']}")
    print("=" * 70)
    print("ETL COMPLETADA SATISFACTORIAMENTE")
    print("=" * 70)


if __name__ == "__main__":
    df_clean = extract_and_transform()
    save_and_upload(df_clean)
