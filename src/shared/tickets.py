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
    "fed", "fomc", "federal reserve", "interest rates", "inflation",
    "powell", "treasury", "yield curve", "central bank", "monetary policy", "deficit",
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
