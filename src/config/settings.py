from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # aws
    # Clave de acceso de AWS (puede ser cualquier valor para pruebas locales)
    aws_access_key_id: str = "test"
    # Clave secreta de acceso de AWS (puede ser cualquier valor para pruebas locales)
    aws_secret_access_key: str = "test"
    default_region_aws: str = "us-east-1"
    # s3
    s3_endpoint_url: str = "http://localhost:9000"  # URL del endpoint de S3 local
    s3_bucket_name: str = "scraping-data-lake"  # Nombre del bucket S3 por defecto (destino TFM)
    scraper_s3_bucket_name: str = "scraping-data-lake"  # Nombre del bucket S3 donde el scraper guarda datos
    # Prefijo de las claves de S3 para los datos crudos
    s3_prefix_raw_data: str = "raw-data"
    s3_region: str = default_region_aws

    # WebScrapingDistributed API Producer
    scraper_api_url: str = "https://webscraping-distributed.duckdns.org"
    scraper_basic_auth_user: str = ""
    scraper_basic_auth_password: str = ""

    # Se usa model_config con SettingsConfigDict
    model_config = SettingsConfigDict(
        env_file=(_ROOT_DIR / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",  # Opcional: ignora variables extra en el .env
    )


# Creamos la instancia aquí
settings = Settings()
