"""Módulo transversal de datos maestros para entidades bancarias (US Big 6) y constantes semánticas."""

from typing import Any

TICKETS: dict[str, dict[str, Any]] = {
    "JPM": {
        "cik": "CIK0000019617",
        "name": "JPMorgan Chase & Co.",
        "aliases": ["jpmorgan", "j.p. morgan", "chase", "jamie dimon", "jpm"],
    },
    "BAC": {
        "cik": "CIK0000070858",
        "name": "Bank of America Corp /DE/",
        "aliases": ["bank of america", "b of a", "bofa", "merrill lynch", "brian moynihan", "bac"],
    },
    "WFC": {
        "cik": "CIK0000072971",
        "name": "Wells Fargo & Company/MN",
        "aliases": ["wells fargo", "charlie scharf", "wfc"],
    },
    "C": {
        "cik": "CIK0000831001",
        "name": "Citigroup Inc.",
        "aliases": ["citigroup", "citi", "jane fraser"],
    },
    "GS": {
        "cik": "CIK0000886982",
        "name": "Goldman Sachs Group Inc.",
        "aliases": ["goldman sachs", "goldman", "david solomon", "gs"],
    },
    "MS": {
        "cik": "CIK0000895421",
        "name": "Morgan Stanley",
        "aliases": ["morgan stanley", "ted pick", "james gorman", "ms"],
    },
}

# Diccionario mapeado de alias por ticker
ALIAS_BANCOS: dict[str, list[str]] = {ticker: data["aliases"] for ticker, data in TICKETS.items()}

# Términos macroeconómicos y de banca central (afectan transversalmente al sistema financiero)
MACRO_KEYWORDS: list[str] = [
    "fed", "fomc", "federal reserve", "interest rates", "interest rate",
    "rate hike", "rate cut", "inflation", "cpi", "powell", "treasury",
    "yield curve", "central bank", "monetary policy", "deficit",
    "banking sector", "financial sector", "stress test", "stress tests",
    "liquidity", "gdp", "recession",
]

# Patrones estándar de párrafos boilerplate de industria
BOILERPLATE_SECTOR_PATTERNS: list[str] = [
    "is a part of the finance sector, which includes",
    "part of the financial sector, which includes",
    "which includes companies such as",
    "peer companies in this sector include",
]

# Términos de rol sintáctico de analista (cuando el banco emite la opinión y no es el sujeto corporativo)
ANALYST_ROLE_TERMS: list[str] = [
    "analyst", "analysts", "strategist", "strategists",
    "economist", "economists", "equity research", "note to clients", "notes to clients",
]

# Patrones sintácticos extendidos de rol de analista / casa de estudios sobre terceros
ANALYST_ROLE_EXTENDED_PATTERNS: list[str] = [
    "downgrade from", "downgraded by", "downgrades from", "downgrades by",
    "upgrade from", "upgraded by", "upgrades from", "upgrades by",
    "price target from", "price target by", "lowered its price target",
    "raised its price target", "cuts price target", "raises price target",
    "initiates coverage", "initiated coverage", "maintains conviction",
    "bullish conviction", "note to clients", "notes to clients",
    "cuts forecast", "raises forecast", "lowers forecast",
]

# Términos de impacto bancario sistémico para validar la categoría macro
SYSTEMIC_BANKING_TERMS: list[str] = [
    "bank stocks", "banking sector", "financial sector", "big banks",
    "stress test", "stress tests", "capital requirements", "bank liquidity",
    "fed stress", "deposit outflows", "deposit insurance", "fdic",
    "bank failure", "regional banks", "wall street banks",
]

# Patrones de declaraciones patrimoniales de fondos (13F)
PORTFOLIO_13F_PATTERNS: list[str] = [
    r"% of the total portfolio",
    r"shares,\s*\d+\.\d+%\s*of",
    r"holdings in",
    r"position in",
    r"new stake in",
]

# Patrones de titulares de listas multi-ticker de TV / resúmenes de apertura sin cuerpo
MULTI_TICKER_LIST_PATTERNS: list[str] = [
    r"fast money picks",
    r"stocks to watch",
    r"stocks in focus",
    r"movers and shakers",
    r"top picks for",
]
