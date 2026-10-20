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

    ticker: str = Field(...,
                        description="Ticker bursátil consultado (ej. MS, JPM, NVDA).")
    company_name: str = Field(...,
                              description="Razón social oficial registrada en la SEC.")
    cik: int = Field(...,
                     description="Central Index Key numérico sin ceros a la izquierda.")
    form: str = Field(...,
                      description="Tipo de reporte regulatorio (ej. 8-K, 10-Q, 10-K).")
    filing_date: str = Field(
        ..., description="Fecha de presentación formal ante la SEC (YYYY-MM-DD).")
    report_date: str = Field(
        "", description="Fecha de corte del periodo reportado (YYYY-MM-DD).")
    acceptance_date_time: str = Field(
        "", description="Timestamp exacto de aceptación por el servidor EDGAR.")
    accession_number: str = Field(
        ..., description="Número de registro único de 20 dígitos con guiones.")
    items: str = Field(
        "", description="Códigos de eventos reportados (específico para Form 8-K, ej. '2.02,8.01').")
    primary_document: str = Field(
        ..., description="Nombre del archivo principal (ej. ms-20231208.htm).")
    primary_doc_description: str = Field(
        "", description="Descripción corta del documento en la SEC.")
    document_url: str = Field(...,
                              description="URL directa al archivo HTML/TXT para el scraper.")
    index_url: str = Field(
        ..., description="URL a la ficha de índice de EDGAR con todos los anexos.")


class SECEdgarAdapter:
    """Adaptador descubridor genérico de filings y URLs de la SEC EDGAR.

    Permite resolver cualquier ticker a su identificador CIK, consultar el historial
    reciente y archivos paginados, y construir deterministamente las URLs directas
    de descarga de los reportes solicitados (8-K, 10-Q, 10-K, etc.).
    """

    SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions"
    SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    SEC_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data"
    DEFAULT_USER_AGENT = "TFM_Data_Science_EBIS discovery_scraper@ebis.edu"

    # Pre-mapeo O(1) de los bancos objetivo del TFM importado de shared.tikets
    KNOWN_CIKS: dict[str, tuple[str | int, str]] = dict(TIKETS)

    def __init__(self, user_agent: str | None = None, min_request_interval: float = 0.12):
        """Inicializa el adaptador con cabecera User-Agent y control de cadencia.

        Args:
            user_agent: Identificación institucional requerida por la SEC
                (formato: 'Nombre / Organización contacto@correo.com').
            min_request_interval: Intervalo mínimo entre peticiones para respetar
                el límite de cortesía de 10 req/s de la SEC (0.12s por defecto).
        """
        self.user_agent = user_agent or self.DEFAULT_USER_AGENT
        self.min_request_interval = min_request_interval
        self._last_request_time = 0.0
        self._ticker_cache: dict[str, tuple[int, str]] = dict(self.KNOWN_CIKS)

    def _apply_rate_limit(self) -> None:
        """Aplica una espera preventiva para cumplir el límite de 10 req/s."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _fetch_json(self, url: str) -> dict[str, Any]:
        """Ejecuta una petición HTTP GET con User-Agent y devuelve el JSON deserializado.

        Args:
            url: Endpoint de la SEC a consultar.

        Returns:
            Diccionario con la respuesta JSON.

        Raises:
            urllib.error.HTTPError: Si la SEC responde con código de error HTTP (ej. 403 o 404).
            RuntimeError: Para fallos de red o de parsing.
        """
        self._apply_rate_limit()
        req = urllib.request.Request(
            url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                content = response.read().decode("utf-8")
                return json.loads(content)
        except urllib.error.HTTPError as e:
            logger.error(
                f"Error HTTP {e.code} al consultar endpoint de la SEC: {url}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado al conectar con {url}: {e}")
            raise RuntimeError(f"Fallo al conectar con la SEC: {e}") from e

    def resolve_cik(self, ticker: str) -> tuple[int, str]:
        """Resuelve cualquier ticker de EE.UU. a su CIK numérico y razón social oficial.

        Args:
            ticker: Símbolo bursátil en mayúsculas o minúsculas (ej. 'NVDA', 'AAPL', 'MS').

        Returns:
            Tupla (cik_int, company_name).

        Raises:
            ValueError: Si el ticker no existe en los registros de la SEC.
        """
        ticker_clean = ticker.strip().upper()
        if ticker_clean in self._ticker_cache:
            cik_val, name = self._ticker_cache[ticker_clean]
            if isinstance(cik_val, str):
                cik_val = int(cik_val.replace("CIK", ""))
            return cik_val, name

        logger.info(
            f"Ticker '{ticker_clean}' no encontrado en caché local. Sincronizando directorio SEC...")
        data = self._fetch_json(self.SEC_TICKERS_URL)
        for entry in data.values():
            sym = entry.get("ticker", "").upper()
            cik_str = entry.get("cik_str")
            title = entry.get("title", "")
            if sym and cik_str is not None:
                self._ticker_cache[sym] = (int(cik_str), title)

        if ticker_clean in self._ticker_cache:
            return self._ticker_cache[ticker_clean]

        raise ValueError(
            f"El ticker '{ticker_clean}' no fue localizado en el registro oficial de la SEC."
        )

    def get_filings(
        self,
        ticker: str,
        forms: list[str] | set[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[SECFiling]:
        """Descubre y devuelve todos los filings y sus URLs directas para un ticker y periodo.

        Args:
            ticker: Símbolo bursátil a consultar (ej. 'MS', 'JPM', 'AAPL').
            forms: Colección de formularios requeridos (ej. ['8-K', '10-Q', '10-K']).
                Si es None, se buscan por defecto ['8-K', '10-Q', '10-K'].
            start_date: Fecha inicial en formato 'YYYY-MM-DD' o 'YYYY' (inclusive).
            end_date: Fecha final en formato 'YYYY-MM-DD' o 'YYYY' (inclusive).

        Returns:
            Lista de objetos SECFiling ordenados descendentemente por filing_date.
        """
        cik_int, company_name = self.resolve_cik(ticker)
        ticker_clean = ticker.strip().upper()

        # Normalización de filtros
        target_forms = {f.upper().strip()
                        for f in (forms or ["8-K", "10-Q", "10-K"])}
        norm_start = f"{start_date}-01-01" if start_date and len(
            start_date) == 4 else (start_date or "1900-01-01")
        norm_end = f"{end_date}-12-31" if end_date and len(
            end_date) == 4 else (end_date or "2100-12-31")

        cik_10 = f"{cik_int:010d}"
        submissions_url = f"{self.SEC_SUBMISSIONS_URL}/CIK{cik_10}.json"
        logger.info(
            f"Descubriendo filings para {ticker_clean} (CIK: {cik_10}) entre {norm_start} y {norm_end}...")

        main_data = self._fetch_json(submissions_url)
        filings: list[SECFiling] = []

        # 1. Procesar bloque de filings recientes
        recent_block = main_data.get("filings", {}).get("recent", {})
        self._extract_filings_from_block(
            block=recent_block,
            ticker=ticker_clean,
            company_name=company_name,
            cik_int=cik_int,
            target_forms=target_forms,
            start_date=norm_start,
            end_date=norm_end,
            output=filings,
        )

        # 2. Procesar bloques históricos paginados si aplica
        paginated_files = main_data.get("filings", {}).get("files", [])
        for file_info in paginated_files:
            file_from = file_info.get("filingFrom", "")
            file_to = file_info.get("filingTo", "")

            # Intersección de intervalos: [file_from, file_to] se solapa con [norm_start, norm_end]
            if file_from <= norm_end and file_to >= norm_start:
                sub_url = f"{self.SEC_SUBMISSIONS_URL}/{file_info['name']}"
                logger.info(
                    f"Consultando bloque histórico paginado SEC: {file_info['name']}...")
                sub_data = self._fetch_json(sub_url)
                self._extract_filings_from_block(
                    block=sub_data,
                    ticker=ticker_clean,
                    company_name=company_name,
                    cik_int=cik_int,
                    target_forms=target_forms,
                    start_date=norm_start,
                    end_date=norm_end,
                    output=filings,
                )

        # Ordenar de más reciente a más antiguo
        filings.sort(key=lambda x: x.filing_date, reverse=True)
        logger.info(
            f"Descubrimiento completado: {len(filings)} filings localizados para {ticker_clean}.")
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
        """Extrae y convierte registros de un bloque crudo de la SEC a modelos SECFiling."""
        forms = block.get("form", [])
        accessions = block.get("accessionNumber", [])
        filing_dates = block.get("filingDate", [])
        report_dates = block.get("reportDate", [])
        acceptance_datetimes = block.get("acceptanceDateTime", [])
        primary_docs = block.get("primaryDocument", [])
        primary_doc_descs = block.get("primaryDocDescription", [])
        items_list = block.get("items", [])

        total = len(forms)
        for i in range(total):
            form_val = forms[i] if i < len(forms) else ""
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
            doc_url = f"{self.SEC_ARCHIVE_URL}/{cik_int}/{acc_clean}/{primary_doc}"
            index_url = f"{self.SEC_ARCHIVE_URL}/{cik_int}/{acc_clean}/{acc}-index.htm"

            items_val = items_list[i] if i < len(
                items_list) and items_list[i] is not None else ""
            if isinstance(items_val, list):
                items_val = ",".join(str(it) for it in items_val)
            else:
                items_val = str(items_val)

            output.append(
                SECFiling(
                    ticker=ticker,
                    company_name=company_name,
                    cik=cik_int,
                    form=form_val,
                    filing_date=f_date,
                    report_date=report_dates[i] if i < len(
                        report_dates) and report_dates[i] else "",
                    acceptance_date_time=acceptance_datetimes[i] if i < len(
                        acceptance_datetimes) else "",
                    accession_number=acc,
                    items=items_val,
                    primary_document=primary_doc,
                    primary_doc_description=primary_doc_descs[i] if i < len(
                        primary_doc_descs) and primary_doc_descs[i] else "",
                    document_url=doc_url,
                    index_url=index_url,
                )
            )

    @staticmethod
    def to_dict_list(filings: list[SECFiling]) -> list[dict[str, Any]]:
        """Convierte una lista de SECFiling a una lista de diccionarios planos lista para tabular."""
        return [f.model_dump() for f in filings]
