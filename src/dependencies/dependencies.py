from config.settings import settings
from infrastructure.storage.base import BaseStorageRepository
from infrastructure.storage.s3.adapter import S3StorageRepository
from infrastructure.compaction.base import BaseDataCompactor
from infrastructure.compaction.parquet import ParquetCompactor
from infrastructure.scraper.base import BaseScraperClient
from infrastructure.scraper.producer_adapter import ScraperProducerAdapter


# Instancias Únicas (Singletons de Infraestructura)
_adapter_s3_instance = S3StorageRepository(
    endpoint_url=settings.s3_endpoint_url,
    bucket_name=settings.s3_bucket_name,
    prefix_raw_data=settings.s3_prefix_raw_data,
    region=settings.s3_region
)

_compactor_instance = ParquetCompactor()
_scraper_client_instance = ScraperProducerAdapter()


def get_storage_repository() -> BaseStorageRepository:
    return _adapter_s3_instance


def get_compactor() -> BaseDataCompactor:
    return _compactor_instance


def get_scraper_client() -> BaseScraperClient:
    return _scraper_client_instance
