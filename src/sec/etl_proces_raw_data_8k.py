"""ETL para procesar reportes SEC 8-K y sus anexos desde el scraper al Data Lake del TFM.

Aprovecha el esquema enriquecido del scraper deserializando directamente 'context'
(metadatos regulatorios del 8-K) y 'data' (body_text del 8-K + exhibits extraídos vía follow_links).
Genera dos datasets enlazables por 'accession_number':
1. Ítems segmentados del 8-K (target_key, ej. raw/sec/8k-2010-2023.parquet).
2. Anexos/Exhibits descargados (exhibits_target_key, ej. raw/sec/8k-exhibits-2010-2023.parquet).
"""

import asyncio
from datetime import datetime, timedelta, timezone
import json
import re
from typing import Any
import pandas as pd

from config.settings import settings
from dependencies.dependencies import get_compactor, get_storage_repository
from infrastructure.compaction.base import BaseDataCompactor
from infrastructure.storage.base import BaseStorageRepository
from shared.logging import Logger

logger = Logger("ETLProcesRawData8K")

ITEM_HEADER_PATTERN = re.compile(r"Item\s+([1-9]\.[0-9]{2})[\.\:\s\-]*", re.IGNORECASE)
SIGNATURE_PATTERN = re.compile(r"\bSIGNATURE[S]?\b", re.IGNORECASE)
SENTENCE_SPLIT_PATTERN = re.compile(r"\.(?=\s|$)")


def clean_text(text: str) -> str:
    """Limpia caracteres de control y normaliza espacios en blanco."""
    if not text:
        return ""
    return " ".join(re.sub(r"[\x00-\x1f\x7f-\x9f\ufffd]", " ", text).split()).strip()


def _parse_json_field(value: Any) -> dict[str, Any]:
    """Deserializa de forma segura un campo JSON string o dict a diccionario."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _join_body_text(body_value: Any) -> str:
    """Convierte el campo body_text (lista de strings o string) en texto plano unido."""
    if isinstance(body_value, list):
        return " ".join(str(part) for part in body_value if part is not None)
    return str(body_value or "")


def extract_items_from_text(raw_text: str) -> list[dict[str, str]]:
    """Segmenta el texto de un 8-K en sus ítems individuales con relación 1 a 1."""
    if not raw_text:
        return []

    norm_text = raw_text.replace("\xa0", " ")
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
            # Separar título de descripción solo en punto final de frase (evita cortar en 'Exhibit 99.1')
            parts = SENTENCE_SPLIT_PATTERN.split(chunk, maxsplit=1)
            if len(parts) == 2 and len(parts[0]) < 120:
                title = clean_text(parts[0])
                body = clean_text(parts[1])
            else:
                title = "Regulatory Event"
                body = clean_text(chunk)

            content = body if body else title
            if len(content) >= 20:
                items.append(
                    {
                        "item": f"Item {item_code}",
                        "item_description": title,
                        "content": content,
                    }
                )

    return items


class ETLProcesRawData8K:
    """Pipeline para transformar 8-Ks crudos y sus anexos a datasets analíticos en S3/R2."""

    def __init__(
        self,
        job_id: str = "sec-8k-2010-01-01-to-2023-12-31",
        target_key: str = "raw/sec/8k-2010-2023.parquet",
        exhibits_target_key: str | None = "raw/sec/8k-exhibits-2010-2023.parquet",
        days_window: int = 7,
        compact_dates: list[str] | None = None,
        compactor: BaseDataCompactor | None = None,
        storage: BaseStorageRepository | None = None,
        source_bucket: str | None = None,
        target_bucket: str | None = None,
    ):
        self.job_id = job_id
        self.target_key = target_key
        self.exhibits_target_key = exhibits_target_key
        self.days_window = days_window
        self.compact_dates = compact_dates
        self.source_bucket = source_bucket or settings.scraper_s3_bucket_name
        self.target_bucket = target_bucket or settings.s3_bucket_name

        self.compactor = compactor or get_compactor()
        self.storage = storage or get_storage_repository()
        self.last_exhibits_df: pd.DataFrame | None = None

    async def _listar_partes(self) -> list[str]:
        """Descubre las partes del job_id acotadas a las fechas válidas (por defecto hoy)."""
        prefix = f"compacted-data/job_id={self.job_id}/"
        if self.compact_dates:
            valid_dates = set(self.compact_dates)
        else:
            today = datetime.now(timezone.utc).date()
            valid_dates = {
                (today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(self.days_window)
            }

        all_keys = await self.storage.list_keys(prefix=prefix, bucket=self.source_bucket)
        keys: list[str] = []
        for k in all_keys:
            if not k.endswith(".parquet"):
                continue
            date_m = re.search(r"year=(\d{4})/month=(\d{2})/day=(\d{2})", k)
            if date_m and f"{date_m.group(1)}-{date_m.group(2)}-{date_m.group(3)}" in valid_dates:
                keys.append(k)

        return sorted(keys, reverse=True)

    @staticmethod
    def _extract_base_metadata(row: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        """Construye los metadatos comunes del Form 8-K padre para unir ítems y anexos."""
        filing_date = ctx.get("filing_date", str(row.get("date", ""))[:10])
        return {
            "ticker": ctx.get("ticker", "UNKNOWN"),
            "company_name": ctx.get("company_name", "Unknown Financial Institution"),
            "cik": ctx.get("cik", 0),
            "form": ctx.get("form", "8-K"),
            "filing_date": filing_date,
            "report_date": ctx.get("report_date", filing_date),
            "acceptance_date_time": ctx.get("acceptance_date_time", ""),
            "accession_number": ctx.get("accession_number", ""),
            "filing_items": ctx.get("items", ""),
            "primary_document": ctx.get("primary_document", ""),
            "primary_doc_description": ctx.get("primary_doc_description", ""),
            "document_url": row.get("url", ""),
            "index_url": ctx.get("index_url", ""),
        }

    @staticmethod
    def _build_item_records(
        base_meta: dict[str, Any],
        parsed_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Construye las filas de ítems 8-K manteniendo el esquema histórico de 8k-2010-2023.parquet."""
        raw_text = _join_body_text(parsed_data.get("body_text", []))
        records: list[dict[str, Any]] = []
        for item in extract_items_from_text(raw_text):
            records.append(
                {
                    "ticker": base_meta["ticker"],
                    "company_name": base_meta["company_name"],
                    "cik": base_meta["cik"],
                    "form": base_meta["form"],
                    "filing_date": base_meta["filing_date"],
                    "report_date": base_meta["report_date"],
                    "acceptance_date_time": base_meta["acceptance_date_time"],
                    "accession_number": base_meta["accession_number"],
                    "item": item["item"],
                    "item_description": item["item_description"],
                    "content": item["content"],
                    "primary_document": base_meta["primary_document"],
                    "primary_doc_description": base_meta["primary_doc_description"],
                    "document_url": base_meta["document_url"],
                    "index_url": base_meta["index_url"],
                }
            )
        return records

    @staticmethod
    def _build_exhibit_records(
        base_meta: dict[str, Any],
        parsed_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Construye las filas de anexos enlazadas al Form 8-K padre mediante accession_number."""
        exhibits = parsed_data.get("exhibits") or []
        if not isinstance(exhibits, list):
            return []

        records: list[dict[str, Any]] = []
        for exhibit in exhibits:
            if not isinstance(exhibit, dict):
                continue
            ex_meta = exhibit.get("metadata") or {}
            ex_data = exhibit.get("data") or {}
            ex_text = clean_text(_join_body_text(ex_data.get("body_text", [])))
            records.append(
                {
                    "accession_number": base_meta["accession_number"],
                    "filing_items": base_meta["filing_items"],
                    "ticker": base_meta["ticker"],
                    "company_name": base_meta["company_name"],
                    "cik": base_meta["cik"],
                    "form": base_meta["form"],
                    "filing_date": base_meta["filing_date"],
                    "report_date": base_meta["report_date"],
                    "acceptance_date_time": base_meta["acceptance_date_time"],
                    "exhibit_type": exhibit.get("label") or "",
                    "exhibit_description": (ex_meta.get("description") or "") if isinstance(ex_meta, dict) else "",
                    "exhibit_size": (ex_meta.get("size") or "") if isinstance(ex_meta, dict) else "",
                    "exhibit_url": exhibit.get("url") or "",
                    "content": ex_text,
                    "error": exhibit.get("error") or "",
                    "document_url": base_meta["document_url"],
                    "index_url": base_meta["index_url"],
                }
            )
        return records

    async def run(self) -> pd.DataFrame:
        """Descarga, procesa y publica los datasets 8-K (ítems y anexos separados por accession_number)."""
        keys = await self._listar_partes()
        if not keys:
            raise FileNotFoundError(f"No se encontraron partes para job_id={self.job_id}")

        logger.info(f"Procesando {len(keys)} partes desde s3://{self.source_bucket}/...")

        all_raw_records: list[dict[str, Any]] = []
        for key in keys:
            data_bytes = await self.storage.get(key=key, bucket=self.source_bucket)
            all_raw_records.extend(self.compactor.decompact(data_bytes))

        records: list[dict[str, Any]] = []
        exhibit_records: list[dict[str, Any]] = []
        for row in all_raw_records:
            ctx = _parse_json_field(row.get("context"))
            base_meta = self._extract_base_metadata(row, ctx)
            parsed_data = _parse_json_field(row.get("data"))

            records.extend(self._build_item_records(base_meta, parsed_data))
            exhibit_records.extend(self._build_exhibit_records(base_meta, parsed_data))

        df_out = pd.DataFrame(records)
        self.last_exhibits_df = pd.DataFrame(exhibit_records)
        logger.info(
            f"Dataset generado: {len(df_out):,} ítems y {len(exhibit_records):,} anexos "
            f"desde {len(all_raw_records):,} documentos."
        )

        if records:
            parquet_bytes = self.compactor.compact(records, compression="ZSTD")
            await self.storage.save(key=self.target_key, body=parquet_bytes, bucket=self.target_bucket)
            logger.info(f"Guardado exitosamente en s3://{self.target_bucket}/{self.target_key}")

        if exhibit_records and self.exhibits_target_key:
            exhibits_bytes = self.compactor.compact(exhibit_records, compression="ZSTD")
            await self.storage.save(
                key=self.exhibits_target_key,
                body=exhibits_bytes,
                bucket=self.target_bucket,
            )
            logger.info(
                f"Anexos guardados exitosamente en s3://{self.target_bucket}/{self.exhibits_target_key}"
            )

        return df_out

    def ejecutar(self) -> pd.DataFrame:
        """Punto de entrada síncrono para scripts de ejecución directa."""
        return asyncio.run(self.run())


if __name__ == "__main__":
    ETLProcesRawData8K().ejecutar()
