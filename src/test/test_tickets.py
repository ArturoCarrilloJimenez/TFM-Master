"""Tests unitarios para el módulo transversal shared.tickets."""

from shared.tickets import (
    TICKETS,
    ALIAS_BANCOS,
    MACRO_KEYWORDS,
    BOILERPLATE_SECTOR_PATTERNS,
    ANALYST_ROLE_TERMS,
)


def test_tickets_master_structure():
    """Valida que TICKETS contenga los 6 bancos canónicos con su estructura enriquecida."""
    assert len(TICKETS) == 6
    expected_tickers = ["JPM", "BAC", "WFC", "C", "GS", "MS"]

    for ticker in expected_tickers:
        assert ticker in TICKETS
        entry = TICKETS[ticker]
        assert isinstance(entry, dict)
        assert "cik" in entry and entry["cik"].startswith("CIK")
        assert "name" in entry and len(entry["name"]) > 0
        assert "aliases" in entry and isinstance(entry["aliases"], list) and len(entry["aliases"]) > 0

    assert TICKETS["JPM"]["name"] == "JPMorgan Chase & Co."
    assert TICKETS["MS"]["cik"] == "CIK0000895421"


def test_alias_bancos_consistency():
    """Valida que ALIAS_BANCOS coincida con los alias definidos en TICKETS."""
    assert len(ALIAS_BANCOS) == 6
    for ticker, aliases in ALIAS_BANCOS.items():
        assert aliases == TICKETS[ticker]["aliases"]
        assert len(aliases) > 0


def test_semantic_constants():
    """Valida que las listas de constantes semánticas no estén vacías y contengan tipos correctos."""
    assert len(MACRO_KEYWORDS) > 0
    assert "fed" in MACRO_KEYWORDS
    assert len(BOILERPLATE_SECTOR_PATTERNS) > 0
    assert len(ANALYST_ROLE_TERMS) > 0
