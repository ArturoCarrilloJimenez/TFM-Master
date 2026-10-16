from config.settings import settings
from infrastructure.storage.base import BaseStorageRepository
from infrastructure.storage.s3.adapter import S3StorageRepository


# Instancias Únicas (Singletons de Infraestructura)
_adapter_s3_instance = S3StorageRepository(
    endpoint_url=settings.s3_endpoint_url,
    bucket_name=settings.s3_bucket_name,
    prefix_raw_data=settings.s3_prefix_raw_data,
    region=settings.s3_region
)


def get_storage_repository() -> BaseStorageRepository:
    return _adapter_s3_instance
