"""Módulo de infraestructura para el cliente del scraper distribuido."""

from infrastructure.scraper.base import BaseScraperClient, ScraperTask
from infrastructure.scraper.producer_adapter import ScraperProducerAdapter

__all__ = ["BaseScraperClient", "ScraperTask", "ScraperProducerAdapter"]
