"""Base abstracta y contratos comunes para motores de análisis de sentimiento financiero."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import re
from typing import Any
import urllib.error
import urllib.request

import numpy as np
from pydantic import BaseModel, Field

from articulos.models.base_nlp import BaseProcesadorArticulo
from articulos.models.sentiment.system_prompt import JSON_INSTR, PROMPTS
from config.settings import settings
from shared.ids import generar_input_hash
from shared.tickets import TICKETS


class SentimientoItem(BaseModel):
    """Contrato Pydantic para la evaluación individual de sentimiento por ticker."""

    idx: int = Field(description="Índice numérico dentro del lote")
    label: int = Field(description="Polaridad: 1 (Positivo), 0 (Neutral), -1 (Negativo)")
    sentiment_score: float = Field(description="Puntuación continua en [-1.0, +1.0]")
    p_pos: float = Field(default=0.0, description="Probabilidad positiva [0.0, 1.0]")
    p_neg: float = Field(default=0.0, description="Probabilidad negativa [0.0, 1.0]")
    p_neu: float = Field(default=0.0, description="Probabilidad neutral [0.0, 1.0]")
    motivo: str = Field(default="ok", description="Justificación breve (máx 10 palabras)")


class LoteSentimientoLLM(BaseModel):
    """Contrato Pydantic para la respuesta por lote del LLM."""

    resultados: list[SentimientoItem]


@dataclass(frozen=True)
class EntradaSentimiento:
    """Unidad mínima deduplicada y lista para inferencia LLM."""

    input_hash: str
    ticker: str
    title: str
    contexto: str


def clean_secret(val: str) -> str:
    """Limpia comentarios inline o espacios sobrantes en credenciales de entorno."""
    return (val or "").split("//")[0].split("#")[0].strip()


def preparar_entradas_sentimiento(
    records: list[dict[str, Any]],
    model: str | None = None,
    prompt_version: str | None = None,
    max_chars_ctx: int = 1000,
) -> tuple[list[str], dict[str, EntradaSentimiento]]:
    """Extrae contextos RE_ADAPTATIVA con base_nlp y genera hashes inmutables de entrada.

    Args:
        records: Lista de registros relevantes ya categorizados (is_ruido == False).
        model: Modelo objetivo (por defecto `settings.claude_model_default`).
        prompt_version: Versión del prompt (por defecto `settings.anthropic_prompt_version`).
        max_chars_ctx: Límite máximo de caracteres del contexto enviado al LLM.

    Returns:
        Tupla (lista de input_hash alineada 1:1 con records, diccionario de entradas únicas por input_hash).
    """
    resolved_model = model or settings.claude_model_default or "claude-haiku-5-5"
    resolved_prompt_ver = prompt_version or settings.anthropic_prompt_version or "v1_en"
    row_hashes: list[str] = []
    unicas: dict[str, EntradaSentimiento] = {}

    for row in records:
        ticker = str(row.get("ticker", ""))
        title = str(row.get("title", ""))
        article = str(row.get("article") or row.get("process_text") or "")

        ctx, _ = BaseProcesadorArticulo.extraer_contexto_banco(
            titular=title,
            cuerpo=article,
            ticker_banco=ticker,
            titular_condicional=True,
            ventana_auto_multiticker=True,
        )
        ctx_trim = ctx[:max_chars_ctx]
        h = generar_input_hash(
            ticker=ticker,
            model=resolved_model,
            prompt_version=resolved_prompt_ver,
            contexto=ctx_trim,
        )
        row_hashes.append(h)

        if h not in unicas:
            unicas[h] = EntradaSentimiento(
                input_hash=h,
                ticker=ticker,
                title=title,
                contexto=ctx_trim,
            )

    return row_hashes, unicas


class BaseSentimentEngine(ABC):
    """Clase base agnóstica de proveedor para inferencia de sentimiento financiero por lotes."""

    def __init__(
        self,
        model: str | None = None,
        prompt_version: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        timeout_s: int = 180,
    ) -> None:
        self.model = model or settings.claude_model_default or "claude-haiku-5-5"
        self.prompt_version = prompt_version or settings.anthropic_prompt_version or "v1_en"
        base_prompt = system_prompt if system_prompt is not None else PROMPTS[self.prompt_version]
        self.system_prompt = base_prompt + JSON_INSTR
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def _build_user_prompt(self, lote: list[EntradaSentimiento]) -> str:
        """Construye el prompt de usuario indexado para el lote de noticias."""
        bloques: list[str] = []
        for idx, item in enumerate(lote):
            ticker_info = TICKETS.get(item.ticker, {})
            banco = ticker_info.get("short_name") or ticker_info.get("name", item.ticker)
            bloques.append(
                f"[idx={idx}] BANCO: {banco} ({item.ticker}) | TITULAR: {item.title} | CONTEXTO: {item.contexto}"
            )
        return "Evaluate the following news items:\n\n" + "\n\n".join(bloques)

    def _http_call(
        self,
        url: str,
        headers: dict[str, str],
        method: str = "GET",
        payload: dict[str, Any] | None = None,
    ) -> str:
        """Ejecuta una petición HTTP síncrona sin dependencias externas."""
        raw_body = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=raw_body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="ignore")[:400]
            raise RuntimeError(f"HTTP {exc.code}: {err_body}") from exc

    def _parsear_lote_texto(
        self,
        raw_text: str,
        lote_hashes: list[str],
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> dict[str, dict[str, Any]]:
        """Valida el JSON estructurado del LLM, recorta cotas numéricas y mapea por input_hash."""
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        parsed = LoteSentimientoLLM.model_validate_json(match.group(0) if match else raw_text)

        n_lote = max(len(lote_hashes), 1)
        in_tok = round(float(input_tokens) / n_lote, 1)
        out_tok = round(float(output_tokens) / n_lote, 1)

        resultados: dict[str, dict[str, Any]] = {}
        for item in parsed.resultados:
            if 0 <= item.idx < len(lote_hashes):
                h = lote_hashes[item.idx]
                resultados[h] = {
                    "input_hash": h,
                    "sentiment_label": int(np.clip(item.label, -1, 1)),
                    "sentiment_score": round(float(np.clip(item.sentiment_score, -1.0, 1.0)), 4),
                    "p_pos": round(float(np.clip(item.p_pos, 0.0, 1.0)), 4),
                    "p_neg": round(float(np.clip(item.p_neg, 0.0, 1.0)), 4),
                    "p_neu": round(float(np.clip(item.p_neu, 0.0, 1.0)), 4),
                    "sentiment_motivo": str(item.motivo or "ok")[:120],
                    "sentiment_model": self.model,
                    "sentiment_prompt_ver": self.prompt_version,
                    "in_tokens": in_tok,
                    "out_tokens": out_tok,
                }
        return resultados

    @abstractmethod
    def clasificar_realtime(
        self,
        entradas: list[EntradaSentimiento],
        batch_size: int = 10,
        max_retries: int = 3,
    ) -> dict[str, dict[str, Any]]:
        """Ejecuta inferencia síncrona por lotes para un proveedor concreto."""
