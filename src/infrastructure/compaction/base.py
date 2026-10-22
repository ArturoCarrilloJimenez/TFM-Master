from abc import ABC, abstractmethod
from typing import Any


class BaseDataCompactor(ABC):
    """Interfaz abstracta (puerto) para servicios de compactación y serialización columnar."""

    @abstractmethod
    def compact(
        self,
        data: list[dict[str, Any]] | dict[str, Any] | str | list[Any],
        compression: str = "ZSTD",
    ) -> bytes:
        """Convierte una colección de registros o JSON en un binario Parquet comprimido en memoria.

        Args:
            data: Datos en formato lista de diccionarios, diccionario individual,
                cadena JSON o lista de objetos con método `model_dump()` (Pydantic).
            compression: Códec de compresión columnar ('ZSTD', 'SNAPPY', 'GZIP', 'NONE').

        Returns:
            bytes: Contenido binario del archivo Parquet resultante listo para subir a S3.
        """
        pass

    @abstractmethod
    def decompact(self, payload: bytes) -> list[dict[str, Any]]:
        """Deserializa un binario Parquet devolviendo la lista original de diccionarios.

        Args:
            payload: Bytes en memoria del archivo Parquet.

        Returns:
            list[dict[str, Any]]: Lista de registros reconstituidos.
        """
        pass
