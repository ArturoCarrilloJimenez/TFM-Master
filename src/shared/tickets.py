"""Módulo de alias semántico para compatibilidad de importación con shared.tikets."""

from .tikets import TIKETS

TICKETS = TIKETS

__all__ = ["TIKETS", "TICKETS"]
