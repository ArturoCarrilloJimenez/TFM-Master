"""Motor de análisis de sentimiento financiero dirigido al ticker mediante Claude API (Batch y Realtime)."""

import json
import time
from typing import Any

from articulos.models.sentiment.base_sentiment import (
    BaseSentimentEngine,
    EntradaSentimiento,
    clean_secret,
)
from config.settings import settings
from shared.logging import Logger

logger = Logger("AnthropicSentimentEngine")

# Modelos recientes de Claude que ya no admiten el parámetro `temperature`.
_NO_TEMP_PREFIXES: tuple[str, ...] = (
    "claude-haiku-5",
    "claude-sonnet-4-6",
    "claude-sonnet-5",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-5",
)


class AnthropicSentimentEngine(BaseSentimentEngine):
    """Adaptador de sentimiento sobre Anthropic Messages API (Realtime y Batch -50%)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        prompt_version: str | None = None,
        system_prompt: str | None = None,
        api_version: str | None = None,
        max_tokens: int = 4096,
        timeout_s: int = 180,
    ) -> None:
        super().__init__(
            model=model,
            prompt_version=prompt_version,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
        self.api_key = clean_secret(api_key if api_key is not None else settings.anthropic_api_key)
        self.api_version = api_version or settings.anthropic_api_version or "2023-06-01"

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY no configurada en .env / entorno.")
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
        }

    def _request_json(
        self,
        url: str,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return json.loads(self._http_call(url, headers=self._headers(), method=method, payload=payload))

    def _download_text(self, url: str) -> str:
        return self._http_call(url, headers=self._headers(), method="GET")

    def _build_message_params(self, lote: list[EntradaSentimiento]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.system_prompt,
            "messages": [{"role": "user", "content": self._build_user_prompt(lote)}],
        }
        if not self.model.startswith(_NO_TEMP_PREFIXES):
            params["temperature"] = 0.0
        return params

    def parsear_respuesta_message(
        self,
        message_obj: dict[str, Any],
        lote_hashes: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Extrae el texto y uso de tokens de Anthropic Messages y delega al parser base."""
        bloques = message_obj.get("content", [])
        txt = "\n".join(
            b["text"] for b in bloques if isinstance(b, dict) and b.get("type") == "text" and "text" in b
        )
        if not txt:
            raise RuntimeError(f"Respuesta Anthropic sin bloque text: {[b.get('type') for b in bloques]}")

        usage = message_obj.get("usage", {})
        return self._parsear_lote_texto(
            raw_text=txt,
            lote_hashes=lote_hashes,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
        )

    # ------------------------------------------------------------------
    # MODO REALTIME (AWS Lambda / inferencia diaria síncrona)
    # ------------------------------------------------------------------
    def clasificar_realtime(
        self,
        entradas: list[EntradaSentimiento],
        batch_size: int = 10,
        max_retries: int = 3,
    ) -> dict[str, dict[str, Any]]:
        """Ejecuta inferencia síncrona por lotes contra POST /v1/messages (ideal para AWS Lambda)."""
        salida: dict[str, dict[str, Any]] = {}
        for start in range(0, len(entradas), batch_size):
            lote = entradas[start : start + batch_size]
            hashes = [e.input_hash for e in lote]
            params = self._build_message_params(lote)

            for intento in range(max_retries):
                try:
                    data = self._request_json(
                        "https://api.anthropic.com/v1/messages",
                        method="POST",
                        payload=params,
                    )
                    salida.update(self.parsear_respuesta_message(data, hashes))
                    break
                except Exception as exc:
                    msg = str(exc)
                    if any(t in msg for t in ("HTTP 400", "HTTP 401", "HTTP 403", "HTTP 404")):
                        raise
                    if intento == max_retries - 1:
                        logger.warning(f"Lote realtime {start // batch_size + 1} agotó reintentos: {msg}")
                        raise
                    wait_s = (2**intento) * 2
                    logger.warning(f"Lote realtime reintento {intento + 1} tras {wait_s}s: {msg}")
                    time.sleep(wait_s)
        return salida

    # ------------------------------------------------------------------
    # MODO BATCH (-50% coste, procesamiento histórico asíncrono)
    # ------------------------------------------------------------------
    def construir_batch_requests(
        self,
        entradas: list[EntradaSentimiento],
        batch_size: int = 10,
    ) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
        """Construye el payload de solicitudes Batch API y el mapa custom_id -> hashes."""
        requests_list: list[dict[str, Any]] = []
        chunk_map: dict[str, list[str]] = {}

        for idx_chunk, start in enumerate(range(0, len(entradas), batch_size)):
            lote = entradas[start : start + batch_size]
            custom_id = f"chunk_{idx_chunk:05d}"
            chunk_map[custom_id] = [e.input_hash for e in lote]
            requests_list.append({
                "custom_id": custom_id,
                "params": self._build_message_params(lote),
            })
        return requests_list, chunk_map

    def enviar_batch(self, requests_list: list[dict[str, Any]]) -> dict[str, Any]:
        """Envía el lote completo a POST /v1/messages/batches (-50% coste)."""
        return self._request_json(
            "https://api.anthropic.com/v1/messages/batches",
            method="POST",
            payload={"requests": requests_list},
        )

    def consultar_batch(self, batch_id: str) -> dict[str, Any]:
        """Consulta el estado actual de un Message Batch."""
        return self._request_json(f"https://api.anthropic.com/v1/messages/batches/{batch_id}", method="GET")

    def descargar_resultados_batch(
        self,
        results_url: str,
        chunk_map: dict[str, list[str]],
    ) -> tuple[dict[str, dict[str, Any]], int, int]:
        """Descarga el JSONL de resultados de Anthropic y valida cada chunk recuperado.

        Returns:
            Tupla (resultados_por_input_hash, chunks_ok, chunks_fallidos).
        """
        raw_jsonl = self._download_text(results_url)
        resultados: dict[str, dict[str, Any]] = {}
        chunks_ok = 0
        chunks_err = 0

        for line in raw_jsonl.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            custom_id = str(item.get("custom_id", ""))
            lote_hashes = chunk_map.get(custom_id)
            if not lote_hashes:
                continue

            res_obj = item.get("result", {})
            if res_obj.get("type") != "succeeded":
                chunks_err += 1
                logger.warning(f"Chunk {custom_id} no completado: type={res_obj.get('type')}")
                continue

            try:
                parsed = self.parsear_respuesta_message(res_obj["message"], lote_hashes)
                resultados.update(parsed)
                chunks_ok += 1
            except Exception as exc:
                chunks_err += 1
                logger.warning(f"Error parseando chunk {custom_id}: {exc}")

        return resultados, chunks_ok, chunks_err
