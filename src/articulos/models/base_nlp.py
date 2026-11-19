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

# Regex estándar para segmentación de oraciones en inglés financiero (protege abreviaturas y meses)
SENTENCE_SPLIT_REGEX = re.compile(
    r"(?<=[.!?])"
    r"(?<!\bJ\.P\.)(?<!\bU\.S\.)(?<!\bU\.K\.)(?<!\bL\.L\.C\.)(?<!\bL\.P\.)(?<!\bN\.A\.)"
    r"(?<!\bInc\.)(?<!\bCorp\.)(?<!\bLtd\.)(?<!\bCo\.)(?<!\bLLC\.)"
    r"(?<!\bJan\.)(?<!\bFeb\.)(?<!\bMar\.)(?<!\bApr\.)(?<!\bAug\.)(?<!\bSept\.)(?<!\bSep\.)(?<!\bOct\.)(?<!\bNov\.)(?<!\bDec\.)"
    r"(?<!\bNo\.)(?<!\bvs\.)(?<!\bEst\.)(?<!\bMr\.)(?<!\bMrs\.)(?<!\bMs\.)(?<!\bDr\.)"
    r"\s+(?=[A-Z0-9\"'(])"
)


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

    @staticmethod
    def _normalizar_texto(texto: str) -> str:
        """Normaliza puntos y guiones para capturar variantes como 'J.P. Morgan' -> 'jp morgan'."""
        return re.sub(r"\s+", " ", texto.replace(".", "").replace("-", " ")).strip()

    @staticmethod
    def limpiar_boilerplate(cuerpo: str) -> str:
        """Elimina cabeceras publicitarias y disclaimers estándar de Zacks/reportes."""
        out = re.sub(r"^.*Free Stock Analysis Report.*$", "", cuerpo, flags=re.IGNORECASE | re.MULTILINE)
        out = re.sub(
            r"^.*(Zacks Investment Research|To read this article on Zacks).*$",
            "",
            out,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        return out.strip()

    @classmethod
    def extraer_contexto_banco(
        cls,
        titular: str,
        cuerpo: str,
        ticker_banco: str,
        ventana_frases: int = 0,
        incluir_titular: bool = True,
        separador: str = " [SEP] ",
        max_chars: int | None = None,
    ) -> tuple[str, bool]:
        """Extrae las oraciones limpias donde se menciona al banco (unificado para Categorizador y FinBERT).

        Args:
            titular: Titular del artículo.
            cuerpo: Texto completo del cuerpo de la noticia.
            ticker_banco: Ticker del banco (ej. 'JPM', 'GS').
            ventana_frases: Frases adyacentes antes y después a incluir (0 = foco estricto en la entidad).
            incluir_titular: Si True, antepone el titular separado por `separador`.
            separador: Separador entre titular y cuerpo extraído (por defecto ' [SEP] ' para FinBERT).
            max_chars: Límite opcional de caracteres sobre el cuerpo extraído (útil para categorizador).

        Returns:
            Tupla (texto_extraido, localizado_en_cuerpo).
        """
        cuerpo_limpio = cls.limpiar_boilerplate(cuerpo or "")
        if not cuerpo_limpio or cuerpo_limpio.lower() in ["none", "nan", ""] or cuerpo_limpio == titular.strip():
            return (titular.strip() if incluir_titular else ""), False

        oraciones = [
            s.strip()
            for s in SENTENCE_SPLIT_REGEX.split(cuerpo_limpio)
            if s.strip() and not any(bp in s.lower() for bp in BOILERPLATE_SECTOR_PATTERNS)
        ]
        if not oraciones:
            return (titular.strip() if incluir_titular else ""), False

        aliases = ALIAS_BANCOS.get(ticker_banco, [ticker_banco.lower()])
        pat_alias = re.compile(r"\b(" + "|".join(re.escape(a) for a in aliases) + r")\b", re.IGNORECASE)

        indices_acierto = [
            i for i, s in enumerate(oraciones)
            if pat_alias.search(s) or pat_alias.search(cls._normalizar_texto(s))
        ]

        if not indices_acierto:
            fallback = oraciones[0][:max_chars].strip() if max_chars is not None else oraciones[0]
            if not incluir_titular:
                return fallback, False
            return (f"{titular.strip()}{separador}{fallback}" if fallback else titular.strip()), False

        indices_expandidos: set[int] = set()
        for idx in indices_acierto:
            for offset in range(-ventana_frases, ventana_frases + 1):
                j = idx + offset
                if 0 <= j < len(oraciones):
                    indices_expandidos.add(j)

        cuerpo_banco = " ".join(oraciones[i] for i in sorted(indices_expandidos))
        if max_chars is not None:
            cuerpo_banco = cuerpo_banco[:max_chars].strip()

        texto_final = f"{titular.strip()}{separador}{cuerpo_banco}" if incluir_titular else cuerpo_banco
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
