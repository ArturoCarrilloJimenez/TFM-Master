"""Módulo para la clasificación estructural de eventos regulatorios SEC Form 8-K.

Distingue entre carátulas procedimentales (announcement-only) y cuerpos
con contenido económico sustantivo (substantive-body), aplicando una poda
limpia del boilerplate legal (Safe Harbor y cláusulas de firma) para maximizar
la calidad de entrada a modelos de NLP financiero como FinBERT.
"""

import re
from typing import Any

# 1. Patrones de referencia explícita a anexos (Exhibits)
EXHIBIT_REF_PATTERN = re.compile(
    r"\b("
    r"attached (hereto )?as exhibit|"
    r"furnished (herewith|as exhibit|pursuant to)|"
    r"incorporated (herein )?by reference|"
    r"is annexed as exhibit|"
    r"is included as exhibit|"
    r"filed as exhibit|"
    r"exhibit 99\.[1-9]|"
    r"copy of the (related )?(press|news) release|"
    r"quarterly (financial )?data supplement"
    r")\b",
    re.IGNORECASE,
)

# 2. Patrones de contenido y magnitudes económicas sustantivas
SUBSTANTIVE_METRICS_PATTERN = re.compile(
    r"("
    r"\$\s*\d+[\d\.,]*\s*(billion|million|trillion|m|b)?|"
    r"\b\d+[\d\.,]*\s*percent|\b\d+[\d\.,]*\s*%|"
    r"\bnet income\b|\bnet earnings\b|\beps\b|\bearnings per share\b|"
    r"\btotal revenue\b|\bnet interest income\b|\boperating expenses\b|"
    r"\bprovisions for credit losses\b|\bcommon stock repurchase\b|"
    r"\bdividend (increase|decrease|raised|cut|paid|per share)\b|"
    r"\bcivil (money )?penalty\b|\bsettlement\b|\bdisgorgement\b|"
    r"\bconsent order\b|\bpleaded guilty\b|\bcease and desist\b"
    r")",
    re.IGNORECASE,
)

# 3. Patrón de Safe Harbor y Boilerplate legal estándar
SAFE_HARBOR_PATTERN = re.compile(
    r"("
    r"Private Securities Litigation Reform Act|"
    r"forward-looking statements|"
    r"these statements are based on (the )?current beliefs|"
    r"factors that could cause (our |actual )?results to differ materially|"
    r"does not undertake to update|"
    r"can be found in (the company|the firm|the registrant|[A-Z\s]+)\'s (annual report on )?form 10-k|"
    r"shall not be (deemed to be |considered )?[\“\"]filed[\”\"] for purposes of section 18|"
    r"securities exchange act of 1934"
    r")",
    re.IGNORECASE,
)


def clasificar_item_8k(item_code: str, titular: str, content: str) -> dict[str, Any]:
    """Clasifica un ítem de Form 8-K en announcement_only o substantive_body.

    Aísla las oraciones con contenido financiero real y poda las oraciones que
    contienen advertencias de Safe Harbor o meras referencias procedimentales a exhibits.

    Args:
        item_code: Código del ítem SEC (ej. "Item 2.02", "Item 8.01").
        titular: Titular compuesto para FinBERT (ej. "[JPM] Item 2.02: Results...").
        content: Texto sin procesar del cuerpo del ítem.

    Returns:
        dict[str, Any]: Metadatos de clasificación estructural:
            - is_announcement_only (bool): True si es una carátula sin sustancia.
            - category (str): "announcement_only" o "substantive_body".
            - has_metrics (bool): Si contiene magnitudes económicas o sanciones.
            - words_substantive (int): Cantidad de palabras en oraciones sustantivas.
            - texto_limpio_finbert (str): Texto recortado sin boilerplate listo para FinBERT.
    """
    raw_text = (content or "").strip()
    if not raw_text:
        return {
            "is_announcement_only": True,
            "category": "announcement_only",
            "has_metrics": False,
            "words_substantive": 0,
            "texto_limpio_finbert": titular.strip(),
        }

    # Segmentación oracional preservando delimitadores
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw_text) if s.strip()]

    substantive_sentences: list[str] = []
    exhibit_sentences: list[str] = []

    for s in sentences:
        is_safe_harbor = bool(SAFE_HARBOR_PATTERN.search(s))
        is_exhibit_ref = bool(EXHIBIT_REF_PATTERN.search(s))

        if is_safe_harbor:
            continue
        if is_exhibit_ref:
            exhibit_sentences.append(s)
        else:
            substantive_sentences.append(s)

    has_metrics = bool(SUBSTANTIVE_METRICS_PATTERN.search(raw_text))
    has_metrics_in_substantive = bool(
        SUBSTANTIVE_METRICS_PATTERN.search(" ".join(substantive_sentences))
    )
    words_substantive = len(" ".join(substantive_sentences).split())

    # Criterio estructural:
    # 1. Si no hay oraciones sustantivas fuera del boilerplate.
    # 2. Si hay menos de 25 palabras sustantivas y carece de métricas financieras.
    # 3. Si es un ítem de carátula típica (2.02, 7.01, 9.01) sin métricas en el texto.
    is_announcement = False
    if not substantive_sentences:
        is_announcement = True
    elif words_substantive < 25 and not has_metrics_in_substantive:
        is_announcement = True
    elif item_code in {"Item 2.02", "Item 7.01", "Item 9.01"} and not has_metrics:
        is_announcement = True

    # Generación del texto optimizado para FinBERT
    if substantive_sentences:
        texto_limpio_finbert = f"{titular.strip()} [SEP] " + " ".join(substantive_sentences)
    elif exhibit_sentences:
        texto_limpio_finbert = f"{titular.strip()} [SEP] {exhibit_sentences[0]}"
    else:
        texto_limpio_finbert = titular.strip()

    return {
        "is_announcement_only": is_announcement,
        "category": "announcement_only" if is_announcement else "substantive_body",
        "has_metrics": has_metrics,
        "words_substantive": words_substantive,
        "texto_limpio_finbert": texto_limpio_finbert.strip(),
    }
