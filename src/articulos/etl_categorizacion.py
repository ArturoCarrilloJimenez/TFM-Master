"""Pipeline ETL para la categorización semántica y persistencia de noticias financieras en el Data Lake."""

import asyncio
import time
from pathlib import Path

from articulos.models.categorizador import CategorizadorArticulos
from dependencies.dependencies import get_compactor, get_storage_repository
from shared.logging import Logger

logger = Logger("ETLCategorizacion")


async def process_data(
    input_s3_key: str = "raw/huggingface/multisource_bancos_2010_2023.parquet",
    output_s3_key: str = "processed/huggingface/multisource_bancos_2010_2023.parquet",
    local_backup_path: str = "data/processed_multisource_bancos_2010_2023.parquet",
    batch_size: int = 256,
) -> None:
    """Orquesta la descarga del dataset crudo, clasificación vectorial en memoria y subida a Cloudflare R2."""
    repo = get_storage_repository()
    compact = get_compactor()

    logger.info(f"Descargando dataset crudo del Data Lake: [{repo.bucket_name}/{input_s3_key}]...")
    t_start = time.time()
    try:
        data = await repo.get(input_s3_key)
    finally:
        # Cierre explícito del canal para evitar timeout de sockets inactivos durante el cómputo del modelo
        await repo.close()

    # Descompactación directa a lista de diccionarios en memoria sin overhead de DataFrames
    records = compact.decompact(data)
    n_records = len(records)
    logger.info(f"Dataset descompactado con éxito ({n_records:,} registros) en {time.time() - t_start:.2f} s.")

    # 1. Instanciación e inferencia con el modelo desacoplado
    model = CategorizadorArticulos()
    data_process = model.clasificar_dataset(records, batch_size=batch_size)

    # 2. Compactación columnar Parquet con compresión ZSTD
    logger.info(f"Compactando {len(data_process):,} registros a formato Apache Parquet (ZSTD)...")
    data_compact = compact.compact(data_process)

    # 3. Guardado de seguridad local
    backup_file = Path(local_backup_path)
    backup_file.parent.mkdir(parents=True, exist_ok=True)
    backup_file.write_bytes(data_compact)
    logger.info(f"Backup de seguridad guardado en disco: [{backup_file}] ({len(data_compact):,} bytes).")

    # 4. Persistencia fresca en Cloudflare R2
    logger.info(f"Abriendo canal nuevo e independiente con Cloudflare R2 -> [{output_s3_key}]...")
    repo_out = get_storage_repository()
    try:
        await repo_out.save(output_s3_key, data_compact)
        logger.info(f"¡Persistencia completada y confirmada en R2: [{output_s3_key}]!")
    finally:
        await repo_out.close()

    total_time = time.time() - t_start
    logger.info(f"Pipeline ETL finalizado exitosamente en {total_time:.1f} s.")


if __name__ == "__main__":
    asyncio.run(process_data())
