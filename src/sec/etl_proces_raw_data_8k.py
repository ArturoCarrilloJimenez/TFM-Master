"""ETL para procesar reportes SEC 8-K desde el scraper al Data Lake del TFM.

Aprovecha el nuevo esquema enriquecido del scraper deserializando directamente el
campo 'context' para heredar el 100% de los metadatos regulatorios (idénticos a los 10-K/Q):
filing_date, report_date, acceptance_date_time, accession_number, index_url, etc.

Segmenta los eventos en relación 1 a 1 para alimentar modelos NLP (FinBERT) y
utiliza la infraestructura centralizada del proyecto (ParquetCompactor y S3StorageRepository).
"""

import asyncio
from datetime import datetime, timedelta, timezone
import io
import json
import re
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from config.settings import settings
from dependencies.dependencies import get_compactor, get_storage_repository
from infrastructure.compaction.base import BaseDataCompactor
from infrastructure.storage.base import BaseStorageRepository
from shared.logging import Logger

logger = Logger("ETLProcesRawData8K")

ITEM_HEADER_PATTERN = re.compile(r'Item\s+([1-9]\.[0-9]{2})[\.\:\s\-]*', re.IGNORECASE)
SIGNATURE_PATTERN = re.compile(r'\bSIGNATURE[S]?\b', re.IGNORECASE)


def clean_text(text: str) -> str:
    """Limpia caracteres de control y normaliza espacios en blanco."""
    if not text:
        return ""
    return " ".join(re.sub(r'[\x00-\x1f\x7f-\x9f\ufffd]', ' ', text).split()).strip()


def extract_items_from_text(raw_text: str) -> list[dict[str, str]]:
    """Segmenta el texto de un 8-K en sus ítems individuales con relación 1 a 1."""
    if not raw_text:
        return []

    norm_text = raw_text.replace('\xa0', ' ')
    matches = list(ITEM_HEADER_PATTERN.finditer(norm_text))
    if not matches:
        return []

    sig_match = SIGNATURE_PATTERN.search(norm_text)
    end_limit = sig_match.start() if sig_match else len(norm_text)

    items: list[dict[str, str]] = []
    for idx, match in enumerate(matches):
        item_code = match.group(1).strip()
        start = match.end()
        end = min(matches[idx + 1].start() if idx + 1 < len(matches) else end_limit, end_limit)

        if end > start:
            chunk = norm_text[start:end].strip()
            # Separar título de descripción por el primer punto
            title_part, sep, body_part = chunk.partition('.')
            if len(title_part) < 120 and sep:
                title = clean_text(title_part)
                body = clean_text(body_part)
            else:
                title = "Regulatory Event"
                body = clean_text(chunk)

            content = body if body else title
            if len(content) >= 20:
                items.append({
                    "item": f"Item {item_code}",
                    "item_description": title,
                    "content": content,
                })

    return items


class ETLProcesRawData8K:
    """Pipeline para transformar 8-Ks crudos a dataset analítico mediante clases de infraestructura."""

    def __init__(
        self,
        job_id: str = "sec-8k-2010-01-01-to-2023-12-31",
        target_key: str = "raw/sec/8k-2010-2023.parquet",
        days_window: int = 7,
        compact_dates: list[str] | None = None,
        compactor: BaseDataCompactor | None = None,
        storage: BaseStorageRepository | None = None,
        source_bucket: str | None = None,
        target_bucket: str | None = None,
    ):
        self.job_id = job_id
        self.target_key = target_key
        self.days_window = days_window
        self.compact_dates = compact_dates
        self.source_bucket = source_bucket or settings.scraper_s3_bucket_name
        self.target_bucket = target_bucket or settings.s3_bucket_name

        self.compactor = compactor or get_compactor()
        self.storage = storage or get_storage_repository()

    async def _listar_partes(self) -> list[str]:
        """Descubre las partes del job_id acotadas a las fechas válidas (por defecto hoy)."""
        prefix = f"compacted-data/job_id={self.job_id}/"
        if self.compact_dates:
            valid_dates = set(self.compact_dates)
        else:
            today = datetime.now(timezone.utc).date()
            valid_dates = {(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(self.days_window)}

        all_keys = await self.storage.list_keys(prefix=prefix, bucket=self.source_bucket)
        keys: list[str] = []
        for k in all_keys:
            if not k.endswith(".parquet"):
                continue
            date_m = re.search(r'year=(\d{4})/month=(\d{2})/day=(\d{2})', k)
            if date_m and f"{date_m.group(1)}-{date_m.group(2)}-{date_m.group(3)}" in valid_dates:
                keys.append(k)

        return sorted(keys, reverse=True)

    async def run(self) -> pd.DataFrame:
        """Descarga, procesa y publica el dataset 8-K estructurado usando context y la infraestructura."""
        keys = await self._listar_partes()
        if not keys:
            raise FileNotFoundError(f"No se encontraron partes para job_id={self.job_id}")

        logger.info(f"Procesando {len(keys)} partes desde s3://{self.source_bucket}/...")
        dfs: list[pd.DataFrame] = []
        for key in keys:
            data_bytes = await self.storage.get(key=key, bucket=self.source_bucket)
            table = pq.read_table(io.BytesIO(data_bytes))
            dfs.append(table.to_pandas())

        df_source = pd.concat(dfs, ignore_index=True)

        records: list[dict[str, Any]] = []
        for _, row in df_source.iterrows():
            url = row["url"]

            # Deserializar contexto de negocio inyectado por el scraper
            ctx_raw = row.get("context")
            if isinstance(ctx_raw, str):
                try:
                    ctx = json.loads(ctx_raw)
                except Exception:
                    ctx = {}
            elif isinstance(ctx_raw, dict):
                ctx = ctx_raw
            else:
                ctx = {}

            ticker = ctx.get("ticker", "UNKNOWN")
            company_name = ctx.get("company_name", "Unknown Financial Institution")
            cik = ctx.get("cik", 0)
            form = ctx.get("form", "8-K")
            filing_date = ctx.get("filing_date", str(row["date"])[:10])
            report_date = ctx.get("report_date", filing_date)
            acceptance_date_time = ctx.get("acceptance_date_time", "")
            accession_number = ctx.get("accession_number", "")
            primary_document = ctx.get("primary_document", "")
            primary_doc_description = ctx.get("primary_doc_description", "")
            index_url = ctx.get("index_url", "")

            raw_text = " ".join(json.loads(row["data"]).get("body_text", []))

            for item in extract_items_from_text(raw_text):
                records.append({
                    "ticker": ticker,
                    "company_name": company_name,
                    "cik": cik,
                    "form": form,
                    "filing_date": filing_date,
                    "report_date": report_date,
                    "acceptance_date_time": acceptance_date_time,
                    "accession_number": accession_number,
                    "item": item["item"],
                    "item_description": item["item_description"],
                    "content": item["content"],
                    "primary_document": primary_document,
                    "primary_doc_description": primary_doc_description,
                    "document_url": url,
                    "index_url": index_url,
                })

        df_out = pd.DataFrame(records)
        logger.info(f"Dataset generado: {len(df_out)} items desde {len(df_source)} documentos.")

        # 1. Compactación columnar usando ParquetCompactor
        parquet_bytes = self.compactor.compact(records, compression="ZSTD")

        # 2. Persistencia en Data Lake usando S3StorageRepository
        await self.storage.save(key=self.target_key, body=parquet_bytes, bucket=self.target_bucket)
        logger.info(f"Guardado exitosamente en s3://{self.target_bucket}/{self.target_key}")
        return df_out

    def ejecutar(self) -> pd.DataFrame:
        """Punto de entrada síncrono para scripts de ejecución directa."""
        return asyncio.run(self.run())


if __name__ == "__main__":
    ETLProcesRawData8K().ejecutar()
