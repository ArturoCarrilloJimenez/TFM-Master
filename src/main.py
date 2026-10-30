"""Punto de entrada principal para la orquestación del proyecto TFM-Master."""

from shared.logging import Logger

logger = Logger("Main")


def main() -> None:
    """Función de inicio del sistema."""
    logger.info("Iniciando TFM-Master...")


if __name__ == "__main__":
    main()
