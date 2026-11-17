"""Módulo base con utilidades comunes de procesamiento de texto y resolución de entidades bancarias.

Centraliza la lógica transversal de limpieza de artículos, detección de roles de analista,
mitigación de ticker bleed y extracción guiada de contexto tanto para categorización (embeddings)
como para modelos de polaridad y análisis de sentimiento (FinBERT).
"""

import re
from typing import Any
import pandas as pd

from shared.tickets import (
    ALIAS_BANCOS,
    ANALYST_ROLE_EXTENDED_PATTERNS,
    ANALYST_ROLE_TERMS,
    BOILERPLATE_SECTOR_PATTERNS,
    MULTI_TICKER_LIST_PATTERNS,
    PORTFOLIO_13F_PATTERNS,
)

# Regex estándar para segmentación de oraciones en inglés financiero
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


class BaseProcesadorArticulo:
    """Clase base transversal con heurísticas de dominio financiero y extracción contextual."""

    @staticmethod
    def contiene_alias(texto: str, alias: str) -> bool:
        """Frontera de palabra regex estricta para evitar colisiones léxicas accidentales (\b)."""
        return bool(re.search(r"\b" + re.escape(alias) + r"\b", texto.lower()))

    @staticmethod
    def resolver_cuerpo_noticia(row: pd.Series | dict[str, Any], titular: str) -> str:
        """Resuelve el cuerpo del artículo descartando vacíos y disclaimers estándar."""
        art = str(row.get("article") or "").strip()
        candidato = (
            art
            if art and art.lower() not in ["none", "nan", ""]
            else str(row.get("text") or row.get("summary") or "").strip()
        )

        if not candidato or candidato.lower() in ["none", "nan", ""] or candidato.strip() == titular.strip():
            return ""

        if len(candidato) < 160 and any(
            w in candidato.lower()
            for w in ["views and opinions", "do not necessarily reflect", "terms not disclosed"]
        ):
            return ""

        return candidato

    @classmethod
    def extraer_fragmento_y_presencia(
        cls, titular: str, cuerpo: str, ticker_banco: str, max_chars: int = 400
    ) -> tuple[str, bool]:
        """Extrae el fragmento guiado por caracteres (para categorización semántica) y si fue localizado."""
        if not cuerpo or cuerpo.lower() in ["none", "nan", ""]:
            return "", False

        aliases = ALIAS_BANCOS.get(ticker_banco, [ticker_banco.lower()])
        parrafos = [p.strip() for p in re.split(r"\n+|, \n+", cuerpo) if len(p.strip()) > 30]

        # 1. Buscar en párrafos legítimos donde aparezca el banco
        for p in parrafos:
            p_lower = p.lower()
            if any(bp in p_lower for bp in BOILERPLATE_SECTOR_PATTERNS):
                continue
            for a in aliases:
                match = re.search(r"\b" + re.escape(a) + r"\b", p_lower)
                if match:
                    start_pos = max(0, match.start() - 150)
                    end_pos = min(len(p), match.end() + 250)
                    return p[start_pos:end_pos].strip(), True

        # 2. Si no aparece en párrafos limpios, buscar en cuerpo completo libre de boilerplate
        p_limpio = cuerpo
        for bp in BOILERPLATE_SECTOR_PATTERNS:
            p_limpio = re.sub(bp + r".*?(\n|$)", "", p_limpio, flags=re.IGNORECASE)

        p_limpio_lower = p_limpio.lower()
        for a in aliases:
            match = re.search(r"\b" + re.escape(a) + r"\b", p_limpio_lower)
            if match:
                start_pos = max(0, match.start() - 150)
                end_pos = min(len(p_limpio), match.end() + 250)
                return p_limpio[start_pos:end_pos].strip(), True

        # 3. Fallback: Mantener el inicio del texto limpio
        fallback = p_limpio[:max_chars].strip() if p_limpio.strip() else cuerpo[:max_chars].strip()
        return fallback, False

    @classmethod
    def extraer_fragmento_relevante(
        cls, titular: str, cuerpo: str, ticker_banco: str, max_chars: int = 400
    ) -> str:
        """Extrae el fragmento guiado donde el banco es mencionado."""
        frag, _ = cls.extraer_fragmento_y_presencia(titular, cuerpo, ticker_banco, max_chars=max_chars)
        return frag

    @classmethod
    def extraer_oraciones_banco(
        cls, titular: str, cuerpo: str, ticker_banco: str, ventana_frases: int = 1
    ) -> tuple[str, bool]:
        """Extrae las oraciones completas donde se nombra al banco ± ventana_frases (diseñado para FinBERT).

        Args:
            titular: Titular del artículo.
            cuerpo: Texto completo del cuerpo de la noticia.
            ticker_banco: Ticker del banco (ej. 'JPM', 'GS').
            ventana_frases: Número de frases adyacentes antes y después a incluir como contexto.

        Returns:
            Tupla (texto_concatenado, localizado_en_cuerpo).
        """
        cuerpo_limpio = cuerpo.strip() if cuerpo else ""
        if not cuerpo_limpio or cuerpo_limpio.lower() in ["none", "nan", ""] or cuerpo_limpio == titular.strip():
            return titular.strip(), False

        # Segmentar en oraciones limpias
        oraciones = [s.strip() for s in SENTENCE_SPLIT_REGEX.split(cuerpo_limpio) if s.strip()]
        if not oraciones:
            return titular.strip(), False

        aliases = ALIAS_BANCOS.get(ticker_banco, [ticker_banco.lower()])
        pat_alias = re.compile(r"\b(" + "|".join(re.escape(a) for a in aliases) + r")\b", re.IGNORECASE)

        # Localizar índices de oraciones que nombran al banco
        indices_acierto = [i for i, s in enumerate(oraciones) if pat_alias.search(s)]

        if not indices_acierto:
            # Fallback: no se encontró al banco en el cuerpo -> Titular + primera oración / párrafo
            fallback_text = f"{titular.strip()}. {oraciones[0]}" if oraciones else titular.strip()
            return fallback_text, False

        # Expandir la ventana de oraciones adyacentes (+/- ventana_frases)
        indices_expandidos: set[int] = set()
        for idx in indices_acierto:
            for offset in range(-ventana_frases, ventana_frases + 1):
                j = idx + offset
                if 0 <= j < len(oraciones):
                    indices_expandidos.add(j)

        oraciones_seleccionadas = [oraciones[i] for i in sorted(indices_expandidos)]
        cuerpo_banco = " ".join(oraciones_seleccionadas)
        texto_final = f"{titular.strip()}. {cuerpo_banco}"

        return texto_final, True

    @classmethod
    def es_rol_analista_tercero(cls, titular: str, cuerpo: str, ticker_banco: str) -> bool:
        """Detecta si el banco actúa únicamente como analista u opinador sobre un tercero o divisa."""
        t_lower = titular.lower()
        c_lower = cuerpo.lower()
        aliases = ALIAS_BANCOS.get(ticker_banco, [ticker_banco.lower()])
        banco_en_titular = any(cls.contiene_alias(t_lower, a) for a in aliases)

        if not banco_en_titular:
            for a in aliases:
                for term in ANALYST_ROLE_TERMS:
                    if (
                        f"{a} {term}" in c_lower
                        or f"{term} at {a}" in c_lower
                        or f"{a}'s {term}" in c_lower
                        or f"{a} reiterated" in c_lower
                    ):
                        return True
                for pat in ANALYST_ROLE_EXTENDED_PATTERNS:
                    if f"{pat} {a}" in c_lower or f"{a} {pat}" in c_lower:
                        return True
        else:
            for a in aliases:
                if re.search(
                    r"\b" + re.escape(a) + r"\b.*?\b(cuts forecast|raises forecast|warns|warns traders)\b",
                    t_lower,
                ):
                    return True
            if "note to clients" in c_lower and any(
                term in c_lower for term in ["bullish conviction", "forecast", "maintains conviction"]
            ):
                return True

        return False

    @classmethod
    def es_contaminacion_competidor(
        cls, titular: str, ticker_banco: str, banco_en_cuerpo: bool = False
    ) -> bool:
        """Detecta si el titular habla explícitamente de un banco competidor del Big 6 y no del banco asignado."""
        t_lower = titular.lower()
        aliases_propios = ALIAS_BANCOS.get(ticker_banco, [ticker_banco.lower()])

        if any(cls.contiene_alias(t_lower, a) for a in aliases_propios):
            return False

        if any(
            w in t_lower
            for w in [
                "big bank",
                "big banks",
                "banks",
                "peers",
                "rivals",
                "wall street",
                "banking sector",
            ]
        ):
            return False

        if banco_en_cuerpo:
            return False

        for competidor, aliases_comp in ALIAS_BANCOS.items():
            if competidor == ticker_banco:
                continue
            if any(cls.contiene_alias(t_lower, ac) for ac in aliases_comp):
                return True

        return False

    @classmethod
    def es_plantilla_irrelevante(cls, titular: str, cuerpo: str) -> bool:
        """Detecta declaraciones patrimoniales de fondos (13F) o listas multi-ticker sin cuerpo."""
        t_lower = titular.lower()
        c_lower = cuerpo.lower()

        if any(re.search(pat, c_lower) for pat in PORTFOLIO_13F_PATTERNS):
            return True

        if not cuerpo and any(re.search(pat, t_lower) for pat in MULTI_TICKER_LIST_PATTERNS):
            return True

        return False
