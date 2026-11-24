import datetime
import json
import logging
import re
import sys
from decimal import Decimal
from typing import Any, Dict, Optional
from uuid import UUID


class StyleFormatter(logging.Formatter):
    # (Tus colores se mantienen igual...)
    LEVEL_COLORS = {
        logging.DEBUG: "\x1b[38;5;244m",
        logging.INFO: "\x1b[32m",
        logging.WARNING: "\x1b[33m",
        logging.ERROR: "\x1b[31m",
        logging.CRITICAL: "\x1b[1;31m",
    }
    RESET = "\x1b[0m"
    MAGENTA = "\x1b[35m"
    CYAN = "\x1b[36m"

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, self.RESET)
        level_name = f"{color}{record.levelname:<8s}{self.RESET}"
        now = datetime.datetime.now()
        logger_name = f"{self.MAGENTA}{record.name}{self.RESET}"

        return f" {level_name} | {now} | {logger_name} | {record.getMessage()}"


class Logger:
    def __init__(self, name: str, level: int = logging.INFO):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.propagate = False

        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(StyleFormatter())
            self._logger.addHandler(handler)

    @staticmethod
    def _universal_serializer(obj: Any) -> Any:
        """Manejador de seguridad para Pydantic, fechas y objetos complejos."""
        if hasattr(obj, "model_dump"):  # Pydantic v2
            return obj.model_dump()
        if hasattr(obj, "dict"):  # Pydantic v1
            return obj.dict()
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        if isinstance(obj, (UUID, Decimal)):
            return str(obj)
        try:
            return repr(obj)  # Último recurso: representación en string
        except Exception:
            return "<Unserializable Object>"

    _CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")

    @classmethod
    def _sanitize_log_text(cls, value: Any) -> str:
        """Neutraliza saltos de línea y caracteres de control para evitar Log Injection (CWE-117)."""
        clean = str(value).replace("\r", "\\r").replace("\n", "\\n")
        return cls._CONTROL_CHARS_RE.sub(" ", clean)

    def _build_msg(self, msg: str, context: Optional[Dict[str, Any]]) -> str:
        """Serialización segura con sanitización CRLF y fallback ante fallos."""
        safe_msg = self._sanitize_log_text(msg)
        if not context:
            return safe_msg

        try:
            # Usamos el serializador universal como 'default'
            ctx_json = self._sanitize_log_text(
                json.dumps(context, default=self._universal_serializer, ensure_ascii=False)
            )
        except Exception:
            # Si incluso el serializador falla (ej. recursión infinita),
            # enviamos un mensaje de emergencia seguro para no tirar el proceso.
            raw_safe = self._sanitize_log_text(str(context)[:100])
            ctx_json = json.dumps(
                {"log_error": "Serialization failed", "raw": raw_safe},
                ensure_ascii=False,
            )

        return f"{safe_msg} | \x1b[36m{ctx_json}\x1b[0m"

    # Métodos de log (info, error, etc.) se mantienen igual llamando a self._build_msg
    def info(self, msg: str, context: Optional[Dict[str, Any]] = None):
        self._logger.info(self._build_msg(msg, context))

    def error(self, msg: str, context: Optional[Dict[str, Any]] = None):
        self._logger.error(self._build_msg(msg, context))

    def warning(self, msg: str, context: Optional[Dict[str, Any]] = None):
        self._logger.warning(self._build_msg(msg, context))

    def debug(self, msg: str, context: Optional[Dict[str, Any]] = None):
        self._logger.debug(self._build_msg(msg, context))
