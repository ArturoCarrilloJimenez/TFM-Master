"""Generación determinista de identificadores de trazabilidad y hash de caché LLM."""

import hashlib


def generar_record_id(ticker: str, date: str, title: str) -> str:
    """Genera un identificador determinista de trazabilidad para un registro Bronze/Silver.

    Args:
        ticker: Símbolo bursátil canónico del banco (ej. 'JPM').
        date: Fecha normalizada del artículo en formato string.
        title: Titular limpio del artículo.

    Returns:
        Prefijo hexadecimal SHA-256 de 24 caracteres (96 bits).
    """
    payload = f"{(ticker or '').strip()}|{(date or '').strip()}|{(title or '').strip()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def generar_input_hash(
    ticker: str,
    model: str,
    prompt_version: str,
    contexto: str,
) -> str:
    """Genera el hash inmutable de la entrada exacta enviada al motor de sentimiento LLM.

    Aísla la caché de sentimiento frente a cambios posteriores del ETL (nuevas columnas,
    reordenado de filas o metadatos) mientras no cambie el contexto analizado.

    Args:
        ticker: Símbolo bursátil objetivo.
        model: Identificador del modelo LLM (ej. 'claude-haiku-5-5').
        prompt_version: Versión del prompt utilizado (ej. 'v1_en').
        contexto: Fragmento textual extraído para el banco objetivo.

    Returns:
        Hash SHA-256 hexadecimal completo (64 caracteres).
    """
    payload = (
        f"{(ticker or '').strip()}|{(model or '').strip()}|"
        f"{(prompt_version or '').strip()}|{(contexto or '').strip()}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generar_categorizer_hash(
    ticker: str,
    title: str,
    article: str = "",
    ticker_count: int = 1,
    is_multisubject: bool = False,
    version: str = "v1",
) -> str:
    """Genera el hash inmutable de entrada para CategorizadorArticulos.

    Evita re-ejecutar el Bi-Encoder en CPU sobre artículos ya categorizados
    cuando solo cambian metadatos o columnas externas (p. ej. record_id).

    Args:
        ticker: Símbolo bursátil objetivo.
        title: Titular limpio del artículo.
        article: Cuerpo textual del artículo.
        ticker_count: Número de tickers asociados al texto.
        is_multisubject: Flag de noticia multisujeto.
        version: Versión de lógica/umbrales del categorizador.

    Returns:
        Hash SHA-256 hexadecimal completo (64 caracteres).
    """
    payload = (
        f"{version}|{(ticker or '').strip()}|{(title or '').strip()}|"
        f"{(article or '').strip()}|{int(ticker_count or 1)}|{bool(is_multisubject)}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
