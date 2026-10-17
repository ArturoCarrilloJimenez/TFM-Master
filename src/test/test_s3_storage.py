import os
import pytest
import asyncio
import boto3
from unittest.mock import AsyncMock, MagicMock, patch
from moto.server import ThreadedMotoServer

from infrastructure.storage.base import BaseStorageRepository
from infrastructure.storage.s3.adapter import S3StorageRepository
from config.settings import settings

pytestmark = pytest.mark.asyncio


# ==============================================================================
# 1. FIXTURES DE ENTORNO E INTEGRACIÃ“N (MOTO S3 LOCAL)
# ==============================================================================

@pytest.fixture(scope="session", autouse=True)
def aws_dummy_credentials():
    """Garantiza credenciales ficticias para prevenir cualquier acceso accidental a AWS."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


@pytest.fixture(scope="session")
def moto_s3_server():
    """Levanta un servidor HTTP local en un puerto dinÃ¡mico simulando AWS S3."""
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0)
    server.start()
    yield server
    server.stop()


@pytest.fixture(scope="function")
def s3_test_env(moto_s3_server):
    """
    Crea el bucket de pruebas en el servidor Moto local y entrega
    el endpoint_url y bucket_name listos para inyectar al repositorio.
    """
    _, port = moto_s3_server.get_host_and_port()
    endpoint_url = f"http://127.0.0.1:{port}"
    bucket_name = "test-bucket"

    # Inicializamos el bucket usando boto3 sÃ­ncrono
    sync_client = boto3.client(
        "s3",
        region_name="us-east-1",
        endpoint_url=endpoint_url,
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    sync_client.create_bucket(Bucket=bucket_name)

    yield {"endpoint_url": endpoint_url, "bucket_name": bucket_name, "sync_client": sync_client}

    # Limpieza post-test
    try:
        objs = sync_client.list_objects_v2(Bucket=bucket_name)
        if "Contents" in objs:
            for obj in objs["Contents"]:
                sync_client.delete_object(Bucket=bucket_name, Key=obj["Key"])
        sync_client.delete_bucket(Bucket=bucket_name)
    except Exception:
        pass


# ==============================================================================
# 2. PRUEBAS UNITARIAS (CON MOCKS)
# ==============================================================================

async def test_repository_implements_base_interface():
    """Valida que S3StorageRepository herede de BaseStorageRepository y cumpla el contrato."""
    assert issubclass(S3StorageRepository, BaseStorageRepository)
    
    repo = S3StorageRepository(
        endpoint_url="http://localhost:4566",
        bucket_name="my-bucket"
    )
    assert isinstance(repo, BaseStorageRepository)
    assert repo.bucket_name == "my-bucket"
    assert repo.prefix_raw_data == "raw-data"


async def test_save_builds_correct_key_and_invokes_put_object():
    """Verifica que el mÃ©todo save() construya la ruta adecuada y llame a put_object."""
    repo = S3StorageRepository(
        endpoint_url="http://localhost:4566",
        bucket_name="my-bucket",
        prefix_raw_data="custom-prefix"
    )
    
    # Mockeamos el cliente aioboto3 interno
    mock_client = AsyncMock()
    repo._client = mock_client

    payload = '{"test": 123}\n'
    await repo.save("job_01/batch_01.jsonl", payload)

    # Debe invocar put_object con el prefijo correcto
    mock_client.put_object.assert_called_once_with(
        Bucket="my-bucket",
        Key="custom-prefix/job_01/batch_01.jsonl",
        Body=payload
    )


async def test_save_raises_exception_on_network_failure():
    """Verifica que si S3 falla en put_object, la excepciÃ³n se capture y se re-lance."""
    repo = S3StorageRepository(
        endpoint_url="http://localhost:4566",
        bucket_name="my-bucket"
    )
    mock_client = AsyncMock()
    mock_client.put_object.side_effect = ConnectionError("Fallo de red simulado")
    repo._client = mock_client

    with pytest.raises(ConnectionError, match="Fallo de red simulado"):
        await repo.save("fail.jsonl", b"datos")


async def test_close_exits_context_manager():
    """Verifica que close() invoque limpiamente el __aexit__ del cliente de aioboto3."""
    repo = S3StorageRepository(
        endpoint_url="http://localhost:4566",
        bucket_name="my-bucket"
    )
    mock_client = AsyncMock()
    repo._client = mock_client

    await repo.close()
    mock_client.__aexit__.assert_called_once_with(None, None, None)


# ==============================================================================
# 3. PRUEBAS DE INTEGRACIÃ“N (CONEXIÃ“N Y PERSISTENCIA REAL CON MOTO HTTP)
# ==============================================================================

async def test_integration_s3_save_and_read_back(s3_test_env):
    """
    Test End-to-End: Conecta a S3, guarda un archivo JSONL real y luego
    verifica usando el cliente sÃ­ncrono que el archivo existe con el contenido idÃ©ntico.
    """
    endpoint_url = s3_test_env["endpoint_url"]
    bucket_name = s3_test_env["bucket_name"]
    sync_client = s3_test_env["sync_client"]

    repo = S3StorageRepository(
        endpoint_url=endpoint_url,
        bucket_name=bucket_name,
        prefix_raw_data="raw-data",
        region="us-east-1"
    )

    # 1. Guardar contenido
    file_key = "job_test_100/part_001.jsonl"
    file_content = '{"url": "https://example.com", "status": 200}\n'

    await repo.save(key=file_key, body=file_content)

    # 2. Comprobar existencia real en el bucket de Moto
    expected_full_key = f"raw-data/{file_key}"
    response = sync_client.get_object(Bucket=bucket_name, Key=expected_full_key)
    persisted_body = response["Body"].read().decode("utf-8")

    assert persisted_body == file_content

    # 3. Cerrar el pool de conexiones del repositorio
    await repo.close()


async def test_integration_s3_save_binary_payload(s3_test_env):
    """Verifica que el repositorio soporte y persista correctamente bytes binarios."""
    endpoint_url = s3_test_env["endpoint_url"]
    bucket_name = s3_test_env["bucket_name"]
    sync_client = s3_test_env["sync_client"]

    repo = S3StorageRepository(
        endpoint_url=endpoint_url,
        bucket_name=bucket_name,
        prefix_raw_data="binary-data"
    )

    binary_data = b"\x1f\x8b\x08\x00\x00\x00\x00\x00"  # Cabecera gzip ficticia
    file_key = "archive.gz"

    await repo.save(key=file_key, body=binary_data)

    response = sync_client.get_object(Bucket=bucket_name, Key=f"binary-data/{file_key}")
    assert response["Body"].read() == binary_data

    await repo.close()


async def test_get_client_raises_exception_on_bootstrap_failure():
    """Valida que si aioboto3 falla en client.__aenter__, el error se registre y se relance."""
    repo = S3StorageRepository(
        endpoint_url="http://invalid-endpoint",
        bucket_name="test-bucket"
    )
    with patch.object(repo.session, "client") as mock_client_factory:
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.side_effect = RuntimeError("Fallo de arranque S3")
        mock_client_factory.return_value = mock_ctx

        with pytest.raises(RuntimeError, match="Fallo de arranque S3"):
            await repo._get_client()


async def test_close_handles_exception_gracefully():
    """Valida que si __aexit__ falla al cerrar conexiones, se capture el error y no reviente."""
    repo = S3StorageRepository(
        endpoint_url="http://localhost:4566",
        bucket_name="test-bucket"
    )
    mock_client = AsyncMock()
    mock_client.__aexit__.side_effect = RuntimeError("Error en socket close")
    repo._client = mock_client

    # No debe propagar la excepción
    await repo.close()
    mock_client.__aexit__.assert_called_once()
