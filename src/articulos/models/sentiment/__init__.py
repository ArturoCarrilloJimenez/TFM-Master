"""Subpaquete de inferencia de sentimiento financiero (base agnóstica y adaptadores LLM)."""

from articulos.models.sentiment.base_sentiment import (
    BaseSentimentEngine,
    EntradaSentimiento,
    LoteSentimientoLLM,
    SentimientoItem,
    clean_secret,
    preparar_entradas_sentimiento,
)
from articulos.models.sentiment.sentiment_llm import AnthropicSentimentEngine

__all__ = [
    "AnthropicSentimentEngine",
    "BaseSentimentEngine",
    "EntradaSentimiento",
    "LoteSentimientoLLM",
    "SentimientoItem",
    "clean_secret",
    "preparar_entradas_sentimiento",
]
