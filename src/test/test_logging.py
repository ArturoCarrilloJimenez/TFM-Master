import logging
import datetime
from decimal import Decimal
from uuid import uuid4

from shared.logging import Logger, StyleFormatter


def test_style_formatter():
    """Valida el formateo de registros con colores ANSI y timestamps."""
    formatter = StyleFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Test message",
        args=(),
        exc_info=None
    )
    formatted = formatter.format(record)
    assert "INFO" in formatted
    assert "test_logger" in formatted
    assert "Test message" in formatted


def test_logger_levels_and_messages(capsys):
    """Verifica la invocación de todos los métodos de severidad de Logger."""
    custom_logger = Logger("unit_test_logger", level=logging.DEBUG)

    custom_logger.debug("Debug msg")
    custom_logger.info("Info msg")
    custom_logger.warning("Warning msg")
    custom_logger.error("Error msg")

    captured = capsys.readouterr().out
    assert "Debug msg" in captured
    assert "Info msg" in captured
    assert "Warning msg" in captured
    assert "Error msg" in captured


def test_universal_serializer_pydantic_and_custom_types():
    """Verifica que el serializador universal gestione Pydantic, fechas, UUID y Decimales."""
    # Objeto con model_dump (Pydantic v2)
    class DummyPydanticV2:
        def model_dump(self):
            return {"key": "val_v2"}

    # Objeto con dict (Pydantic v1)
    class DummyPydanticV1:
        def dict(self):
            return {"key": "val_v1"}

    # Objeto con repr fallido
    class Unrepresentable:
        def __repr__(self):
            raise RuntimeError("Cannot represent")

    now_dt = datetime.datetime(2026, 10, 16, 12, 0, 0)
    now_date = datetime.date(2026, 10, 16)
    test_uuid = uuid4()
    test_decimal = Decimal("123.45")

    assert Logger._universal_serializer(DummyPydanticV2()) == {"key": "val_v2"}
    assert Logger._universal_serializer(DummyPydanticV1()) == {"key": "val_v1"}
    assert Logger._universal_serializer(now_dt) == now_dt.isoformat()
    assert Logger._universal_serializer(now_date) == now_date.isoformat()
    assert Logger._universal_serializer(test_uuid) == str(test_uuid)
    assert Logger._universal_serializer(test_decimal) == "123.45"
    assert Logger._universal_serializer(12345) == "12345"
    assert Logger._universal_serializer(Unrepresentable()) == "<Unserializable Object>"


def test_build_msg_with_and_without_context():
    """Verifica _build_msg con y sin contexto, y la recuperación ante fallos de serialización."""
    logger_instance = Logger("msg_builder_logger")

    # Sin contexto
    raw_msg = logger_instance._build_msg("Hello World", None)
    assert raw_msg == "Hello World"

    # Con contexto estándar
    ctx = {"user_id": 42, "status": "active"}
    formatted_msg = logger_instance._build_msg("Task dispatched", ctx)
    assert "Task dispatched" in formatted_msg
    assert '"user_id": 42' in formatted_msg
    assert '"status": "active"' in formatted_msg

    # Con contexto que provoca fallo (recursión cíclica de diccionario)
    cyclic_dict = {}
    cyclic_dict["self"] = cyclic_dict
    fallback_msg = logger_instance._build_msg("Failure case", cyclic_dict)
    assert "Serialization failed" in fallback_msg


def test_logger_sanitizes_log_injection(capsys):
    """Verifica que saltos de línea y caracteres de control no inyecten líneas de log falsas."""
    logger_instance = Logger("security_logger")

    malicious_msg = "Batch [123\r\nFAKE | INFO | Admin logged in]\x00 ok"
    malicious_ctx = {"batch_id": "abc\nFAKE_KEY=1", "note": "line1\r\nline2"}

    sanitized_plain = logger_instance._build_msg(malicious_msg, None)
    assert "\n" not in sanitized_plain
    assert "\r" not in sanitized_plain
    assert "\x00" not in sanitized_plain
    assert "\\nFAKE" in sanitized_plain

    sanitized_ctx = logger_instance._build_msg(malicious_msg, malicious_ctx)
    assert "\n" not in sanitized_ctx
    assert "\r" not in sanitized_ctx
    assert sanitized_ctx.endswith("\x1b[0m")

    logger_instance.info(malicious_msg, context=malicious_ctx)
    out_lines = capsys.readouterr().out.strip().splitlines()
    assert len(out_lines) == 1
