from abc import ABC, abstractmethod


class BaseStorageRepository(ABC):
    @abstractmethod
    async def save(self, key: str, body: str | bytes, bucket: str | None = None) -> None:
        """Persiste un payload de forma asíncrona e idempotente en una coordenada (Key) específica."""
        pass

    @abstractmethod
    async def get(self, key: str, bucket: str | None = None) -> bytes:
        """Recupera los bytes de un objeto almacenado en S3/R2."""
        pass

    @abstractmethod
    async def list_keys(self, prefix: str, bucket: str | None = None) -> list[str]:
        """Lista las coordenadas de objetos bajo un prefijo."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Libera los recursos del pool y cierra los sockets de conexión de forma limpia."""
        pass
