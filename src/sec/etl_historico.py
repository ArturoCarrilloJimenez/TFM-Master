import asyncio
from typing import Any

from shared.logging import Logger
from shared.tikets import TIKETS
from sec.adaptador import SECEdgarAdapter, SECFiling
from infrastructure.compaction.base import BaseDataCompactor
from infrastructure.storage.base import BaseStorageRepository
from infrastructure.scraper.base import BaseScraperClient, ScraperTask
from dependencies.dependencies import (
    get_compactor,
    get_storage_repository,
    get_scraper_client,
)

logger = Logger("SECHistoricalETL")

# Universo de los 6 gigantes bancarios derivado de la fuente única de verdad (shared.tikets)
TARGET_BANKS: dict[str, str] = {ticker: meta[1] for ticker, meta in TIKETS.items()}


class SECHistoricalETL:
    """Orquestador ETL para extraer, clasificar, compactar y persistir filings de la SEC en el Data Lake."""

    def __init__(
        self,
        adapter: SECEdgarAdapter | None = None,
        compactor: BaseDataCompactor | None = None,
        storage: BaseStorageRepository | None = None,
        scraper_client: BaseScraperClient | None = None,
    ):
        """Inicializa la ETL con inyección de dependencias para extracción, compactación, almacenamiento y scraper."""
        self.adapter = adapter or SECEdgarAdapter()
        self.compactor = compactor or get_compactor()
        self.storage = storage or get_storage_repository()
        self.scraper_client = scraper_client or get_scraper_client()

    async def run(
        self,
        start_date: str = "2010-01-01",
        end_date: str = "2023-12-31",
        target_s3_key: str = "raw/sec/10k-10q-2010-2023.parquet",
        dispatch_to_scraper: bool = True,
        scraper_batch_size: int = 100,
    ) -> dict[str, Any]:
        """Ejecuta la extracción, compacta 10-Q/10-K a Parquet ZSTD y despacha 8-K al cluster de scraping.

        Acepta una ventana temporal por días exactos (YYYY-MM-DD), permitiendo su
        reutilización tanto para cargas históricas completas (2010-2023) como para
        ingestas diarias operativas en producción (día t y t-1).

        Args:
            start_date: Fecha inicial en formato YYYY-MM-DD ("2010-01-01" por defecto).
            end_date: Fecha final en formato YYYY-MM-DD ("2023-12-31" por defecto).
            target_s3_key: Coordenada en S3/Cloudflare R2 donde persistir el archivo Parquet consolidado.
            dispatch_to_scraper: Si es True, despacha las tareas 8-K hacia la API Producer de WebScrapingDistributed.
            scraper_batch_size: Tamaño de lote para el envío de tareas al scraper (100 por defecto).

        Returns:
            dict[str, Any]: Resumen de la operación con estado, métricas y filings 8-K listos para el scraper.
        """
        logger.info(
            f"Iniciando pipeline ETL SEC [{start_date} -> {end_date}] para el Universo US Big 6..."
        )

        all_filings: list[SECFiling] = []

        # 1. Extracción de filings desde la API oficial de la SEC
        for ticker, company_name in TARGET_BANKS.items():
            logger.info(f"Extrayendo filings para {ticker} ({company_name})...")
            filings = self.adapter.get_filings(
                ticker=ticker,
                forms=["8-K", "10-Q", "10-K"],
                start_date=start_date,
                end_date=end_date,
            )
            all_filings.extend(filings)

        # 2. Segmentación de datos según su destino en la arquitectura del TFM
        filings_10k_10q = [f for f in all_filings if f.form in {"10-Q", "10-K"}]
        filings_8k = [f for f in all_filings if f.form == "8-K"]

        logger.info(
            f"Segmentación completada: {len(filings_10k_10q):,} reportes 10-Q/10-K (Parquet S3) "
            f"y {len(filings_8k):,} reportes 8-K (Scraper & FinBERT)."
        )

        # 3. Compactación columnar en memoria RAM de los 10-Q y 10-K (si existen)
        parquet_bytes_size = 0
        if filings_10k_10q:
            logger.info(
                f"Compactando {len(filings_10k_10q):,} registros a formato Apache Parquet (ZSTD)..."
            )
            parquet_bytes = self.compactor.compact(filings_10k_10q, compression="ZSTD")
            parquet_bytes_size = len(parquet_bytes)

            # 4. Persistencia directa en el Data Lake (S3 / Cloudflare R2)
            logger.info(
                f"Subiendo binario Parquet ({parquet_bytes_size:,} bytes) a S3 -> [{target_s3_key}]..."
            )
            await self.storage.save(key=target_s3_key, body=parquet_bytes)
            logger.info(
                f"Archivo persistido en [{target_s3_key}] con {len(filings_10k_10q):,} informes 10-Q/10-K."
            )
        else:
            logger.info("No se encontraron filings 10-Q o 10-K en la ventana temporal especificada.")

        # 5. Despacho masivo o diario de las tareas 8-K al cluster de WebScrapingDistributed
        scraper_result: dict[str, Any] | None = None
        if dispatch_to_scraper and filings_8k:
            logger.info(
                f"Transformando {len(filings_8k):,} filings 8-K en tareas ScraperTask para envío al Producer..."
            )
            scraper_tasks = [
                ScraperTask(
                    url=f.document_url,
                    parser_type="static_css",
                    selectors={"body_text": "body"},
                    priority=1,
                    max_retries=3,
                    context={
                        "ticker": f.ticker,
                        "cik": f.cik,
                        "company_name": f.company_name,
                        "form": f.form,
                        "filing_date": f.filing_date,
                        "report_date": f.report_date,
                        "accession_number": f.accession_number,
                        "items": f.items,
                        "index_url": f.index_url,
                    },
                )
                for f in filings_8k
            ]

            job_id = f"sec-8k-{start_date}-to-{end_date}"
            scraper_result = await self.scraper_client.dispatch_tasks(
                job_id=job_id,
                tasks=scraper_tasks,
                batch_size=scraper_batch_size,
                global_context={
                    "start_date": start_date,
                    "end_date": end_date,
                    "source": "TFM-Master",
                },
            )

        logger.info("Pipeline ETL finalizado con éxito.")

        return {
            "status": "SUCCESS",
            "s3_key": target_s3_key if filings_10k_10q else None,
            "total_universe_discovered": len(all_filings),
            "total_10k_10q_compacted": len(filings_10k_10q),
            "total_8k_for_scraper": len(filings_8k),
            "parquet_bytes_size": parquet_bytes_size,
            "scraper_result": scraper_result,
            "filings_8k": filings_8k,
        }


def run_sec_historical_etl(
    start_date: str = "2010-01-01",
    end_date: str = "2023-12-31",
    target_s3_key: str = "raw/sec/10k-10q-2010-2023.parquet",
    dispatch_to_scraper: bool = True,
    scraper_batch_size: int = 100,
) -> dict[str, Any]:
    """Punto de entrada síncrono para scripts de ejecución directa o tareas cron."""
    etl = SECHistoricalETL()
    return asyncio.run(
        etl.run(
            start_date=start_date,
            end_date=end_date,
            target_s3_key=target_s3_key,
            dispatch_to_scraper=dispatch_to_scraper,
            scraper_batch_size=scraper_batch_size,
        )
    )


if __name__ == "__main__":
    run_sec_historical_etl()
