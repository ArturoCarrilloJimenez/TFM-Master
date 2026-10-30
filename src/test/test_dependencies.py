from infrastructure.storage.base import BaseStorageRepository
from infrastructure.storage.s3.adapter import S3StorageRepository
from infrastructure.compaction.base import BaseDataCompactor
from infrastructure.compaction.parquet import ParquetCompactor
from dependencies.dependencies import (
    get_storage_repository,
    _adapter_s3_instance,
    get_compactor,
    _compactor_instance,
)


def test_get_storage_repository_returns_singleton():
    """Verifica que el inyector devuelva la instancia singleton de almacenamiento."""
    repo = get_storage_repository()
    assert isinstance(repo, BaseStorageRepository)
    assert isinstance(repo, S3StorageRepository)
    assert repo is _adapter_s3_instance


def test_get_compactor_returns_singleton():
    """Verifica que el inyector devuelva la instancia singleton del compactador."""
    compactor = get_compactor()
    assert isinstance(compactor, BaseDataCompactor)
    assert isinstance(compactor, ParquetCompactor)
    assert compactor is _compactor_instance
