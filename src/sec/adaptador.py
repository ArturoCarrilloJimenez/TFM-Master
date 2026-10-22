"""Adaptador descubridor de reportes regulatorios en SEC EDGAR para entidades bancarias."""

import json
import time
import urllib.error
import urllib.request
from typing import Any
from pydantic import BaseModel, Field

from shared.logging import Logger
from shared.tikets import TIKETS

logger = Logger("SECEdgarAdapter")


class SECFiling(BaseModel):
    """Modelo Pydantic V2 que encapsula los metadatos y URLs de un archivo SEC EDGAR."""

    ticker: str = Field(..., description="Ticker bursátil consultado (ej. MS, JPM, NVDA).")
    company_name: str = Field(..., description="Razón social oficial registrada en la SEC.")
    cik: int = Field(..., description="Central Index Key numérico sin ceros a la izquierda.")
    form: str = Field(..., description="Tipo de reporte regulatorio (ej. 8-K, 10-Q, 10-K).")
    filing_date: str = Field(..., description="Fecha de presentación formal ante la SEC (YYYY-MM-DD).")
    report_date: str = Field("", description="Fecha de corte del periodo reportado (YYYY-MM-DD).")
    acceptance_date_time: str = Field("", description="Timestamp exacto de aceptación por el servidor EDGAR.")
    accession_number: str = Field(..., description="Número de registro único de 20 dígitos con guiones.")
    items: str = Field("", description="Códigos de eventos reportados (específico para Form 8-K).")
    primary_document: str = Field(..., description="Nombre del archivo principal (ej. ms-20231208.htm).")
    primary_doc_description: str = Field("", description="Descripción corta del documento en la SEC.")
    document_url: str = Field(..., description="URL directa al archivo HTML/TXT para el scraper.")
    index_url: str = Field(..., description="URL a la ficha de índice de EDGAR con todos los anexos.")


class SECEdgarAdapter:
    """Adaptador descubridor optimizado de filings y URLs de la SEC EDGAR."""

    SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions"
    SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    SEC_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data"
    DEFAULT_USER_AGENT = "TFM_Data_Science_EBIS discovery_scraper@ebis.edu"

    # Mapeo O(1) de los bancos objetivo importado de shared.tikets
    KNOWN_CIKS: dict[str, tuple[str | int, str]] = dict(TIKETS)

    def __init__(self, user_agent: str | None = None, min_request_interval: float = 0.12):
        self.user_agent = user_agent or self.DEFAULT_USER_AGENT
        self.min_request_interval = min_request_interval
        self._last_request_time = 0.0
        self._ticker_cache: dict[str, tuple[int, str]] = dict(self.KNOWN_CIKS)

    def _apply_rate_limit(self) -> None:
        """Aplica una espera preventiva para cumplir el límite de cortesía de la SEC (10 req/s)."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _fetch_json(self, url: str) -> dict[str, Any]:
        """Ejecuta una petición HTTP GET con User-Agent institucional requerido por la SEC."""
        self._apply_rate_limit()
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            logger.error(f"Error HTTP {e.code} al consultar endpoint de la SEC: {url}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado al conectar con {url}: {e}")
            raise RuntimeError(f"Fallo al conectar con la SEC: {e}") from e

    def resolve_cik(self, ticker: str) -> tuple[int, str]:
        """Resuelve el ticker a su CIK numérico y razón social oficial."""
        ticker_clean = ticker.strip().upper()
        if ticker_clean in self._ticker_cache:
            cik_val, name = self._ticker_cache[ticker_clean]
            return int(str(cik_val).replace("CIK", "")), name

        logger.info(f"Ticker '{ticker_clean}' no en caché. Consultando directorio SEC...")
        data = self._fetch_json(self.SEC_TICKERS_URL)
        for entry in data.values():
            sym = entry.get("ticker", "").upper()
            cik_str = entry.get("cik_str")
            if sym and cik_str is not None:
                self._ticker_cache[sym] = (int(cik_str), entry.get("title", ""))

        if ticker_clean in self._ticker_cache:
            return self._ticker_cache[ticker_clean]

        raise ValueError(f"El ticker '{ticker_clean}' no fue localizado en el registro oficial de la SEC.")

    def get_filings(
        self,
        ticker: str,
        forms: list[str] | set[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[SECFiling]:
        """Descubre y devuelve todos los filings y sus URLs directas para un ticker y periodo."""
        cik_int, company_name = self.resolve_cik(ticker)
        ticker_clean = ticker.strip().upper()

        target_forms = {f.upper().strip() for f in (forms or ["8-K", "10-Q", "10-K"])}
        norm_start = f"{start_date}-01-01" if start_date and len(start_date) == 4 else (start_date or "1900-01-01")
        norm_end = f"{end_date}-12-31" if end_date and len(end_date) == 4 else (end_date or "2100-12-31")

        cik_10 = f"{cik_int:010d}"
        submissions_url = f"{self.SEC_SUBMISSIONS_URL}/CIK{cik_10}.json"
        logger.info(f"Descubriendo filings para {ticker_clean} (CIK: {cik_10}) entre {norm_start} y {norm_end}...")

        main_data = self._fetch_json(submissions_url)
        filings: list[SECFiling] = []

        # 1. Procesar bloque de filings recientes
        recent_block = main_data.get("filings", {}).get("recent", {})
        self._extract_filings_from_block(recent_block, ticker_clean, company_name, cik_int, target_forms, norm_start, norm_end, filings)

        # 2. Procesar bloques históricos paginados que intersecten el rango temporal
        for file_info in main_data.get("filings", {}).get("files", []):
            if file_info.get("filingFrom", "") <= norm_end and file_info.get("filingTo", "") >= norm_start:
                sub_url = f"{self.SEC_SUBMISSIONS_URL}/{file_info['name']}"
                sub_data = self._fetch_json(sub_url)
                self._extract_filings_from_block(sub_data, ticker_clean, company_name, cik_int, target_forms, norm_start, norm_end, filings)

        filings.sort(key=lambda x: x.filing_date, reverse=True)
        logger.info(f"Descubrimiento completado: {len(filings)} filings localizados para {ticker_clean}.")
        return filings

    def _extract_filings_from_block(
        self,
        block: dict[str, Any],
        ticker: str,
        company_name: str,
        cik_int: int,
        target_forms: set[str],
        start_date: str,
        end_date: str,
        output: list[SECFiling],
    ) -> None:
        """Extrae y convierte registros de un bloque de la SEC a modelos SECFiling."""
        forms = block.get("form", [])
        accessions = block.get("accessionNumber", [])
        filing_dates = block.get("filingDate", [])
        report_dates = block.get("reportDate", [])
        acceptance_datetimes = block.get("acceptanceDateTime", [])
        primary_docs = block.get("primaryDocument", [])
        primary_doc_descs = block.get("primaryDocDescription", [])
        items_list = block.get("items", [])

        for i, form_val in enumerate(forms):
            if form_val not in target_forms:
                continue

            f_date = filing_dates[i] if i < len(filing_dates) else ""
            if not (start_date <= f_date <= end_date):
                continue

            acc = accessions[i] if i < len(accessions) else ""
            primary_doc = primary_docs[i] if i < len(primary_docs) else ""
            if not acc or not primary_doc:
                continue

            acc_clean = acc.replace("-", "")
            raw_items = items_list[i] if i < len(items_list) else ""
            items_str = ",".join(str(it) for it in raw_items) if isinstance(raw_items, list) else str(raw_items)

            doc_url = f"{self.SEC_ARCHIVE_URL}/{cik_int}/{acc_clean}/{primary_doc}"
            idx_url = f"{self.SEC_ARCHIVE_URL}/{cik_int}/{acc_clean}/{acc}-index.htm"

            output.append(
                SECFiling(
                    ticker=ticker,
                    company_name=company_name,
                    cik=cik_int,
                    form=form_val,
                    filing_date=f_date,
                    report_date=report_dates[i] if i < len(report_dates) else "",
                    acceptance_date_time=acceptance_datetimes[i] if i < len(acceptance_datetimes) else "",
                    accession_number=acc,
                    items=items_str,
                    primary_document=primary_doc,
                    primary_doc_description=primary_doc_descs[i] if i < len(primary_doc_descs) else "",
                    document_url=doc_url,
                    index_url=idx_url,
                )
            )

    @staticmethod
    def to_dict_list(filings: list[SECFiling]) -> list[dict[str, Any]]:
        """Convierte una lista de modelos SECFiling a diccionarios serializables."""
        return [f.model_dump() for f in filings]
