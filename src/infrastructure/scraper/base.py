"""Contratos y modelos base para la integración con sistemas de scraping."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScraperTask:
    """Modelo de dominio que representa una tarea individual de extracción web.

    Attributes:
        url: Dirección URL válida a raspar.
        parser_type: Motor de scraping destino ("static_css" o "dynamic_playwright").
        selectors: Diccionario de selectores CSS o FieldSpecs a extraer.
        priority: Nivel de prioridad de 1 a 10 (por defecto 1).
        max_retries: Intentos máximos de reintento ante fallos temporales (por defecto 3).
        context: Metadatos adicionales de negocio (ticker, filing_date, accession_number, etc.).
        headers: Cabeceras HTTP personalizadas opcionales para la petición de scraping.
        exclude_selectors: Selectores CSS o etiquetas a excluir del DOM antes de extraer.
        follow_links: Reglas declarativas opcionales para descubrir y extraer sub-enlaces.
    """

    url: str
    parser_type: str = "static_css"
    selectors: dict[str, Any] = field(default_factory=lambda: {"body_text": "body"})
    priority: int = 1
    max_retries: int = 3
    context: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    exclude_selectors: list[str] | None = None
    follow_links: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serializa la tarea al formato esperado por la API Producer."""
        parser_config: dict[str, Any] = {
            "selectors": self.selectors,
        }
        if self.exclude_selectors:
            parser_config["exclude_selectors"] = self.exclude_selectors
        if self.follow_links:
            parser_config["follow_links"] = self.follow_links

        return {
            "url": self.url,
            "parser_type": self.parser_type,
            "parser_config": parser_config,
            "priority": self.priority,
            "max_retries": self.max_retries,
            "context": self.context,
            "headers": self.headers,
        }


class BaseScraperClient(ABC):
    """Interfaz abstracta que define las operaciones del cliente de scraping."""

    @abstractmethod
    async def dispatch_tasks(
        self,
        job_id: str,
        tasks: list[ScraperTask],
        batch_size: int = 100,
        global_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Despacha un conjunto de tareas hacia el sistema de scraping distribuyendo en lotes.

        Args:
            job_id: Identificador global para la agrupación y trazabilidad del trabajo.
            tasks: Lista de tareas individuales a ejecutar.
            batch_size: Cantidad máxima de tareas por petición HTTP.
            global_context: Metadatos contextuales compartidos por todo el lote.

        Returns:
            dict[str, Any]: Resumen consolidado del envío (total de tareas, lotes y estado).
        """
        pass
