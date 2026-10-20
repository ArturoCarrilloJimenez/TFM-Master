import asyncio
from typing import Any

from shared.logging import Logger
from shared.tikets import TIKETS
from sec.adaptador import SECEdgarAdapter, SECFiling
from infrastructure.compaction.base import BaseDataCompactor
from infrastructure.storage.base import BaseStorageRepository
from dependencies.dependencies import get_compactor, get_storage_repository

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
    ):
        """Inicializa la ETL con inyección de dependencias para extracción, compactación y almacenamiento."""
        self.adapter = adapter or SECEdgarAdapter()
        self.compactor = compactor or get_compactor()
        self.storage = storage or get_storage_repository()

    async def run(
        self,
        start_date: str = "2010-01-01",
        end_date: str = "2023-12-31",
        target_s3_key: str = "raw/sec/10k-10q-2010-2023.parquet",
    ) -> dict[str, Any]:
        """Ejecuta la extracción, compacta 10-Q/10-K a Parquet ZSTD y los sube a S3/R2.

        Acepta una ventana temporal por días exactos (YYYY-MM-DD), permitiendo su
        reutilización tanto para cargas históricas completas (2010-2023) como para
        ingestas diarias operativas en producción (día t y t-1).

        Args:
            start_date: Fecha inicial en formato YYYY-MM-DD ("2010-01-01" por defecto).
            end_date: Fecha final en formato YYYY-MM-DD ("2023-12-31" por defecto).
            target_s3_key: Coordenada en S3/Cloudflare R2 donde persistir el archivo Parquet consolidado.

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

        # 3. Compactación columnar en memoria RAM de los 10-Q y 10-K
        logger.info(
            f"Compactando {len(filings_10k_10q):,} registros a formato Apache Parquet (ZSTD)..."
        )
        parquet_bytes = self.compactor.compact(filings_10k_10q, compression="ZSTD")

        # 4. Persistencia directa en el Data Lake (S3 / Cloudflare R2)
        logger.info(
            f"Subiendo binario Parquet ({len(parquet_bytes):,} bytes) a S3 -> [{target_s3_key}]..."
        )
        await self.storage.save(key=target_s3_key, body=parquet_bytes)

        logger.info(
            f"Pipeline ETL finalizado con éxito. Archivo persistido en [{target_s3_key}] "
            f"con {len(filings_10k_10q):,} informes 10-Q/10-K."
        )

        return {
            "status": "SUCCESS",
            "s3_key": target_s3_key,
            "total_universe_discovered": len(all_filings),
            "total_10k_10q_compacted": len(filings_10k_10q),
            "total_8k_for_scraper": len(filings_8k),
            "parquet_bytes_size": len(parquet_bytes),
            "filings_8k": filings_8k,
        }


def run_sec_historical_etl(
    start_date: str = "2010-01-01",
    end_date: str = "2023-12-31",
    target_s3_key: str = "raw/sec/10k-10q-2010-2023.parquet",
) -> dict[str, Any]:
    """Punto de entrada síncrono para scripts de ejecución directa o tareas cron."""
    etl = SECHistoricalETL()
    return asyncio.run(
        etl.run(
            start_date=start_date,
            end_date=end_date,
            target_s3_key=target_s3_key,
        )
    )


if __name__ == "__main__":
    run_sec_historical_etl()
