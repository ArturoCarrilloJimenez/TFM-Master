import io
import json
from typing import Any
import pyarrow as pa
import pyarrow.parquet as pq

from shared.logging import Logger
from .base import BaseDataCompactor

logger = Logger("ParquetCompactor")


class ParquetCompactor(BaseDataCompactor):
    """Adaptador de infraestructura para compactar y serializar datasets a formato Apache Parquet en memoria."""

    def __init__(self, default_compression: str = "ZSTD"):
        """Inicializa el compactador con el códec de compresión por defecto.

        Args:
            default_compression: Algoritmo de compresión columnar por defecto ('ZSTD', 'SNAPPY', 'GZIP').
        """
        self.default_compression = default_compression

    def _normalize_data(
        self,
        data: list[dict[str, Any]] | dict[str, Any] | str | list[Any],
    ) -> list[dict[str, Any]]:
        """Normaliza diferentes estructuras de entrada (JSON, Dict, Pydantic) a lista de diccionarios planos."""
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
                if isinstance(parsed, dict):
                    return [parsed]
                if isinstance(parsed, list):
                    return parsed
                raise ValueError("El JSON provisto debe ser un objeto o una lista de objetos.")
            except json.JSONDecodeError as e:
                logger.error(f"Error decodificando string JSON para compactar: {e}")
                raise ValueError(f"Cadena JSON inválida: {e}") from e

        if isinstance(data, dict):
            return [data]

        if isinstance(data, list):
            if not data:
                raise ValueError("No se puede compactar una lista vacía de registros.")

            records: list[dict[str, Any]] = []
            for item in data:
                if hasattr(item, "model_dump"):  # Pydantic V2
                    records.append(item.model_dump())
                elif hasattr(item, "dict"):  # Pydantic V1
                    records.append(item.dict())
                elif isinstance(item, dict):
                    records.append(item)
                else:
                    raise TypeError(
                        f"Tipo de elemento no soportado en la lista a compactar: {type(item)}"
                    )
            return records

        raise TypeError(
            f"Tipo de entrada no soportado para compactación: {type(data)}. "
            "Se requiere list[dict], dict, str (JSON) o list[BaseModel]."
        )

    def compact(
        self,
        data: list[dict[str, Any]] | dict[str, Any] | str | list[Any],
        compression: str | None = None,
    ) -> bytes:
        """Convierte una colección de registros en un buffer binario Parquet comprimido en memoria.

        Args:
            data: Colección de datos (lista de diccionarios, JSON string o modelos Pydantic).
            compression: Códec de compresión ('ZSTD', 'SNAPPY', etc.). Si es None, usa `self.default_compression`.

        Returns:
            bytes: Contenido binario del archivo Parquet listo para persistencia directa en S3/R2.
        """
        records = self._normalize_data(data)
        codec = (compression or self.default_compression).lower()

        logger.info(
            f"Compactando {len(records):,} registros a formato Parquet con códec [{codec.upper()}] en RAM..."
        )

        try:
            table = pa.Table.from_pylist(records)
            buffer = io.BytesIO()
            pq.write_table(table, buffer, compression=codec)
            raw_bytes = buffer.getvalue()

            logger.info(
                f"Compactación completada con éxito. Tamaño binario resultante: {len(raw_bytes):,} bytes."
            )
            return raw_bytes
        except Exception as e:
            logger.error(f"Fallo durante la compactación Parquet con PyArrow: {e}")
            raise RuntimeError(f"Error al generar binario Parquet: {e}") from e

    def decompact(self, payload: bytes) -> list[dict[str, Any]]:
        """Lee un binario Parquet desde la memoria y devuelve la lista de diccionarios original.

        Args:
            payload: Bytes en memoria del archivo Parquet.

        Returns:
            list[dict[str, Any]]: Lista de diccionarios reconstruida a partir de la tabla columnar.
        """
        if not payload:
            raise ValueError("El payload binario Parquet está vacío.")

        try:
            buffer = io.BytesIO(payload)
            table = pq.read_table(buffer)
            records = table.to_pylist()
            logger.info(f"Descompactados {len(records):,} registros desde buffer Parquet ({len(payload):,} bytes).")
            return records
        except Exception as e:
            logger.error(f"Fallo durante la descompactación Parquet: {e}")
            raise RuntimeError(f"Error al leer binario Parquet: {e}") from e
