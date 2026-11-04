"""Tests unitarios para el módulo de alias semántico shared.tickets."""

from shared.tikets import TIKETS
from shared.tickets import TICKETS, TIKETS as ALIAS_TIKETS


def test_tickets_alias_coherence():
    """Valida que TICKETS y TIKETS apunten a la misma estructura de datos canónica."""
    assert TICKETS == TIKETS
    assert ALIAS_TIKETS == TIKETS
    assert "JPM" in TICKETS
    assert TICKETS["JPM"] == ("CIK0000019617", "JPMorgan Chase & Co.")
    assert len(TICKETS) == 6
