"""Adaptador concreto para el envío masivo y recurrente a la API Producer de WebScrapingDistributed."""

import asyncio
import base64
import aiohttp
from typing import Any

from config.settings import settings
from infrastructure.scraper.base import BaseScraperClient, ScraperTask
from shared.logging import Logger

logger = Logger("ScraperProducerAdapter")


class ScraperProducerAdapter(BaseScraperClient):
    """Cliente HTTP asíncrono para enviar lotes de tareas a WebScrapingDistributed.

    Soporta autenticación HTTP Basic (requerida por proxies inversos como OpenResty/Nginx),
    fragmentación automática en lotes (batching), reintentos automáticos con backoff exponencial
    y trazabilidad mediante job_id.
    """

    def __init__(
        self,
        base_url: str | None = None,
        auth_user: str | None = None,
        auth_password: str | None = None,
        timeout_seconds: float = 30.0,
        max_batch_retries: int = 3,
        backoff_factor: float = 1.0,
    ) -> None:
        """Inicializa el adaptador con las coordenadas de la API Producer.

        Args:
            base_url: URL base del servicio Producer (por defecto desde settings).
            auth_user: Usuario HTTP Basic Auth opcional.
            auth_password: Password HTTP Basic Auth opcional.
            timeout_seconds: Tiempo límite de espera para cada llamada HTTP.
            max_batch_retries: Número máximo de reintentos por cada lote en caso de error transitorio.
            backoff_factor: Factor multiplicador para el tiempo de espera entre reintentos en segundos.
        """
        raw_url = base_url or settings.scraper_api_url
        self.base_url = raw_url.rstrip("/")
        self.auth_user = auth_user if auth_user is not None else settings.scraper_basic_auth_user
        self.auth_password = (
            auth_password if auth_password is not None else settings.scraper_basic_auth_password
        )
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.max_batch_retries = max_batch_retries
        self.backoff_factor = backoff_factor

    def _get_headers(self) -> dict[str, str]:
        """Construye las cabeceras HTTP, incluyendo Authorization Basic estándar si está configurado."""
        headers = {"Content-Type": "application/json"}
        if self.auth_user and self.auth_password:
            user_pass = f"{self.auth_user}:{self.auth_password}".encode("utf-8")
            encoded = base64.b64encode(user_pass).decode("ascii")
            headers["Authorization"] = f"Basic {encoded}"
        return headers

    async def _send_batch_with_retry(
        self,
        session: aiohttp.ClientSession,
        endpoint_url: str,
        payload: dict[str, Any],
        batch_idx: int,
        total_batches: int,
    ) -> dict[str, Any]:
        """Envía un único lote con reintentos y retroceso exponencial ante fallos transitorios."""
        last_error: Exception | None = None

        for attempt in range(1, self.max_batch_retries + 1):
            try:
                async with session.post(endpoint_url, json=payload) as response:
                    status_code = response.status
                    if status_code in (200, 202):
                        resp_data = await response.json()
                        return {
                            "batch_index": batch_idx,
                            "status_code": status_code,
                            "tasks_count": len(payload.get("tasks", [])),
                            "response": resp_data,
                            "attempts": attempt,
                        }

                    resp_text = await response.text()
                    last_error = RuntimeError(
                        f"HTTP {status_code} en lote {batch_idx}/{total_batches}: {resp_text}"
                    )

                    # Si es error de cliente 4xx no reintentable (ej: 400 Bad Request o 401 Unauthorized), salir rápido
                    if status_code in (400, 401, 403, 422):
                        logger.error(f"Fallo no recuperable en lote {batch_idx} ({last_error})")
                        raise last_error

            except (aiohttp.ClientError, asyncio.TimeoutError) as net_err:
                last_error = net_err

            if attempt < self.max_batch_retries:
                sleep_s = self.backoff_factor * (2 ** (attempt - 1))
                logger.warning(
                    f"Intento {attempt}/{self.max_batch_retries} falló para lote {batch_idx}/{total_batches} "
                    f"({last_error}). Reintentando en {sleep_s:.1f}s..."
                )
                await asyncio.sleep(sleep_s)

        raise RuntimeError(
            f"Lote {batch_idx}/{total_batches} falló tras {self.max_batch_retries} intentos: {last_error}"
        )

    async def dispatch_tasks(
        self,
        job_id: str,
        tasks: list[ScraperTask],
        batch_size: int = 100,
        global_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Despacha un conjunto de tareas hacia la API Producer dividiéndolas en chunks con reintentos automáticos.

        Args:
            job_id: Identificador global del trabajo.
            tasks: Lista de tareas ScraperTask a procesar.
            batch_size: Tamaño de cada lote (por defecto 100).
            global_context: Metadatos compartidos opcionales.

        Returns:
            dict[str, Any]: Resumen con métricas de envío y respuestas del servidor.
        """
        if not tasks:
            logger.warning(f"No se proporcionaron tareas para despachar en job_id='{job_id}'.")
            return {
                "job_id": job_id,
                "status": "SKIPPED",
                "total_tasks": 0,
                "total_batches": 0,
                "accepted_tasks": 0,
            }

        endpoint_url = f"{self.base_url}/v1/scraping/tasks"
        headers = self._get_headers()
        serialized_tasks = [t.to_dict() for t in tasks]

        # Particionar en lotes de tamaño batch_size
        chunks = [
            serialized_tasks[i : i + batch_size]
            for i in range(0, len(serialized_tasks), batch_size)
        ]
        total_batches = len(chunks)

        logger.info(
            f"Despachando {len(tasks):,} tareas a [{endpoint_url}] en {total_batches} lotes "
            f"(job_id='{job_id}', batch_size={batch_size})..."
        )

        sent_count = 0
        batch_results: list[dict[str, Any]] = []

        async with aiohttp.ClientSession(timeout=self.timeout, headers=headers) as session:
            for idx, chunk in enumerate(chunks, start=1):
                payload: dict[str, Any] = {
                    "job_id": job_id,
                    "tasks": chunk,
                }
                if global_context:
                    payload["context"] = global_context

                logger.info(
                    f"Enviando lote {idx}/{total_batches} ({len(chunk)} tareas) para job_id='{job_id}'..."
                )

                batch_res = await self._send_batch_with_retry(
                    session=session,
                    endpoint_url=endpoint_url,
                    payload=payload,
                    batch_idx=idx,
                    total_batches=total_batches,
                )
                sent_count += len(chunk)
                batch_results.append(batch_res)

        logger.info(
            f"Despacho completado con éxito: {sent_count:,}/{len(tasks):,} tareas aceptadas "
            f"en {total_batches} lotes."
        )

        return {
            "job_id": job_id,
            "status": "SUCCESS",
            "total_tasks": len(tasks),
            "total_batches": total_batches,
            "accepted_tasks": sent_count,
            "batches": batch_results,
        }
