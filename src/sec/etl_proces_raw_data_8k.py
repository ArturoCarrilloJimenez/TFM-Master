"""ETL simplificado para procesar reportes SEC 8-K desde el scraper al Data Lake del TFM.

Descubre dinámicamente los parquets del scraper bajo el job_id, segmenta los
documentos en ítems individuales (1 a 1), extrae el título y contenido de cada evento,
y consolida el resultado en Parquet ZSTD en el bucket del TFM.
"""

from datetime import datetime, timedelta, timezone
import io
import json
import re
from typing import Any

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.config import Config

from config.settings import settings
from shared.logging import Logger
from shared.tikets import TIKETS

logger = Logger("ETLProcesRawData8K")

ITEM_HEADER_PATTERN = re.compile(r'Item\s+([1-9]\.[0-9]{2})[\.\:\s\-]*', re.IGNORECASE)
SIGNATURE_PATTERN = re.compile(r'\bSIGNATURE[S]?\b', re.IGNORECASE)
DATE_ISO_PATTERN = re.compile(r'\b(20[0-2][0-9]-[0-1][0-9]-[0-3][0-9])\b')

# Mapeo inverso de CIK (int) a (ticker, company_name)
CIK_TO_BANK: dict[int, tuple[str, str]] = {
    int(str(cik).upper().replace("CIK", "")): (ticker, company)
    for ticker, (cik, company) in TIKETS.items()
}


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


def parse_sec_url(url: str) -> tuple[int, str]:
    """Extrae CIK y accession number de la URL oficial de la SEC."""
    match = re.search(r'/data/(\d+)/(\d{10})(\d{2})(\d{6})/', url)
    if match:
        return int(match.group(1)), f"{match.group(2)}-{match.group(3)}-{match.group(4)}"
    return 0, ""


class ETLProcesRawData8K:
    """Pipeline simplificado para transformar 8-Ks crudos a dataset analítico."""

    def __init__(
        self,
        job_id: str = "sec-8k-2010-01-01-to-2023-12-31",
        target_key: str = "raw/sec/8k-2010-2023.parquet",
        days_window: int = 7,
    ):
        self.job_id = job_id
        self.target_key = target_key
        self.days_window = days_window
        self.source_bucket = settings.scraper_s3_bucket_name
        self.target_bucket = settings.s3_bucket_name

        self.s3 = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name="auto",
            config=Config(s3={"addressing_style": "path"}),
        )

    def _listar_partes(self) -> list[str]:
        """Descubre las partes del job_id acotadas a la ventana de la última semana."""
        prefix = f"compacted-data/job_id={self.job_id}/"
        today = datetime.now(timezone.utc).date()
        valid_dates = {(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(-1, self.days_window + 1)}

        resp = self.s3.list_objects_v2(Bucket=self.source_bucket, Prefix=prefix)
        keys: list[str] = []
        for obj in resp.get("Contents", []):
            k = obj["Key"]
            if not k.endswith(".parquet"):
                continue
            date_m = re.search(r'year=(\d{4})/month=(\d{2})/day=(\d{2})', k)
            if not date_m or f"{date_m.group(1)}-{date_m.group(2)}-{date_m.group(3)}" in valid_dates:
                keys.append(k)

        return keys

    def ejecutar(self) -> pd.DataFrame:
        """Descarga, procesa y publica el dataset 8-K estructurado en R2."""
        keys = self._listar_partes()
        if not keys:
            raise FileNotFoundError(f"No se encontraron partes para job_id={self.job_id}")

        logger.info(f"Procesando {len(keys)} partes desde s3://{self.source_bucket}/...")
        dfs = [pq.read_table(io.BytesIO(self.s3.get_object(Bucket=self.source_bucket, Key=k)["Body"].read())).to_pandas() for k in keys]
        df_source = pd.concat(dfs, ignore_index=True)

        records: list[dict[str, Any]] = []
        for _, row in df_source.iterrows():
            url = row["url"]
            cik, accession = parse_sec_url(url)
            ticker, company_name = CIK_TO_BANK.get(cik, ("UNKNOWN", "Unknown Financial Institution"))

            raw_text = " ".join(json.loads(row["data"]).get("body_text", []))
            date_m = DATE_ISO_PATTERN.search(raw_text[:800])
            filing_date = date_m.group(1) if date_m else str(row["date"])[:10]

            for item in extract_items_from_text(raw_text):
                records.append({
                    "ticker": ticker,
                    "company_name": company_name,
                    "cik": cik,
                    "form": "8-K",
                    "filing_date": filing_date,
                    "accession_number": accession,
                    "item": item["item"],
                    "item_description": item["item_description"],
                    "content": item["content"],
                    "document_url": url,
                })

        df_out = pd.DataFrame(records)
        logger.info(f"Dataset generado: {len(df_out)} items desde {len(df_source)} documentos.")

        buffer = io.BytesIO()
        pq.write_table(pa.Table.from_pandas(df_out), buffer, compression="ZSTD")
        self.s3.put_object(Bucket=self.target_bucket, Key=self.target_key, Body=buffer.getvalue())
        logger.info(f"Guardado exitosamente en s3://{self.target_bucket}/{self.target_key}")
        return df_out


if __name__ == "__main__":
    ETLProcesRawData8K().ejecutar()
