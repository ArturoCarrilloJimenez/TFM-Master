"""Tests unitarios para el motor AnthropicSentimentEngine (Batch API y Realtime)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from articulos.models.sentiment import (
    AnthropicSentimentEngine,
    BaseSentimentEngine,
    preparar_entradas_sentimiento,
)


def test_preparar_entradas_deduplica_por_input_hash():
    """Verifica que noticias repetidas con mismo ticker y contexto compartan input_hash."""
    records = [
        {
            "ticker": "JPM",
            "date": "2022-01-01 09:00:00",
            "title": "JPMorgan beats Q4 EPS estimates",
            "article": "JPMorgan Chase posted strong net income growth in Q4.",
        },
        {
            "ticker": "JPM",
            "date": "2022-01-02 09:00:00",
            "title": "JPMorgan beats Q4 EPS estimates",
            "article": "JPMorgan Chase posted strong net income growth in Q4.",
        },
        {
            "ticker": "BAC",
            "date": "2022-01-02 09:00:00",
            "title": "Bank of America faces margin pressure",
            "article": "Bank of America saw lower net interest income.",
        },
    ]

    row_hashes, unicas = preparar_entradas_sentimiento(records)
    assert len(row_hashes) == 3
    assert row_hashes[0] == row_hashes[1]
    assert row_hashes[0] != row_hashes[2]
    assert len(unicas) == 2


def test_construir_batch_y_descargar_resultados():
    """Valida construcción de chunks de 10 y parseo JSONL con bloque thinking previo."""
    records = [
        {"ticker": "WFC", "title": f"Wells Fargo update {i}", "article": "Wells Fargo raised dividend."}
        for i in range(12)
    ]
    _, unicas = preparar_entradas_sentimiento(records)
    entradas = list(unicas.values())

    engine = AnthropicSentimentEngine(api_key="sk-ant-test")
    requests_list, chunk_map = engine.construir_batch_requests(entradas, batch_size=10)

    assert len(requests_list) == 2
    assert len(chunk_map["chunk_00000"]) == 10
    assert len(chunk_map["chunk_00001"]) == 2
    assert "temperature" not in requests_list[0]["params"]

    # Simular respuesta JSONL con bloque thinking antes del bloque text y chunk_00001 fallido
    payload_c0 = {
        "resultados": [
            {"idx": idx, "label": 1, "sentiment_score": 0.7, "motivo": "Dividend hike"}
            for idx in range(10)
        ]
    }
    jsonl_mock = "\n".join([
        json.dumps({
            "custom_id": "chunk_00000",
            "result": {
                "type": "succeeded",
                "message": {
                    "content": [
                        {"type": "thinking", "thinking": "Analyzing..."},
                        {"type": "text", "text": f"```json\n{json.dumps(payload_c0)}\n```"},
                    ],
                    "usage": {"input_tokens": 3000, "output_tokens": 1500},
                },
            },
        }),
        json.dumps({"custom_id": "chunk_00001", "result": {"type": "errored"}}),
    ])

    with patch.object(engine, "_download_text", return_value=jsonl_mock):
        res, ok_cnt, err_cnt = engine.descargar_resultados_batch("https://example.com/res.jsonl", chunk_map)

    assert ok_cnt == 1
    assert err_cnt == 1
    assert len(res) == 10
    first_hash = chunk_map["chunk_00000"][0]
    assert res[first_hash]["sentiment_label"] == 1
    assert res[first_hash]["sentiment_score"] == 0.7
    assert res[first_hash]["in_tokens"] == 300.0


def test_clasificar_realtime_claude():
    """Valida el modo realtime síncrono usado en AWS Lambda."""
    records = [
        {"ticker": "GS", "title": "Goldman Sachs profit jumps", "article": "Goldman Sachs beat consensus."}
    ]
    _, unicas = preparar_entradas_sentimiento(records)
    entradas = list(unicas.values())

    engine = AnthropicSentimentEngine(api_key="sk-ant-test")
    mock_resp = {
        "content": [
            {
                "type": "text",
                "text": json.dumps({
                    "resultados": [
                        {"idx": 0, "label": 1, "sentiment_score": 0.85, "p_pos": 0.9, "p_neg": 0.05, "p_neu": 0.05, "motivo": "Earnings beat"}
                    ]
                }),
            }
        ],
        "usage": {"input_tokens": 310, "output_tokens": 120},
    }

    with patch.object(engine, "_request_json", return_value=mock_resp) as mock_req:
        out = engine.clasificar_realtime(entradas, batch_size=10)

    mock_req.assert_called_once()
    h = entradas[0].input_hash
    assert out[h]["sentiment_label"] == 1
    assert out[h]["sentiment_score"] == 0.85


def test_base_sentiment_contract_and_clipping():
    """Verifica la herencia de BaseSentimentEngine y el recorte de cotas en _parsear_lote_texto."""
    assert issubclass(AnthropicSentimentEngine, BaseSentimentEngine)
    engine = AnthropicSentimentEngine(api_key="sk-ant-test")
    raw_json = json.dumps({
        "resultados": [
            {
                "idx": 0,
                "label": 5,
                "sentiment_score": 2.4,
                "p_pos": 1.3,
                "p_neg": -0.2,
                "p_neu": 0.1,
                "motivo": "Strong rally",
            }
        ]
    })
    parsed = engine._parsear_lote_texto(raw_json, ["h0"], input_tokens=100, output_tokens=50)
    assert parsed["h0"]["sentiment_label"] == 1
    assert parsed["h0"]["sentiment_score"] == 1.0
    assert parsed["h0"]["p_pos"] == 1.0
    assert parsed["h0"]["p_neg"] == 0.0
    assert parsed["h0"]["in_tokens"] == 100.0
    assert parsed["h0"]["out_tokens"] == 50.0


@pytest.mark.parametrize(
    "bad_batch_id",
    [
        "",
        "../admin",
        "msgbatch_1/../../secret",
        "msgbatch_1?foo=bar",
        "msgbatch_1#frag",
        "msgbatch%2e%2e",
        "msgbatch 1",
    ],
)
def test_consultar_batch_rechaza_traversal(bad_batch_id: str):
    """Bloquea identificadores con caracteres de traversal o inyección antes de construir la URL."""
    engine = AnthropicSentimentEngine(api_key="sk-ant-test")
    with patch("urllib.request.urlopen") as mock_urlopen:
        with pytest.raises(ValueError, match="Identificador de batch inválido"):
            engine.consultar_batch(bad_batch_id)
        mock_urlopen.assert_not_called()


def test_consultar_batch_valido_codifica_url():
    """Permite batch_id válidos y ejecuta la petición HTTPS esperada."""
    engine = AnthropicSentimentEngine(api_key="sk-ant-test")
    mock_resp = MagicMock()
    mock_resp.read.return_value = b'{"id": "msgbatch_01ABC_xyz-1", "processing_status": "ended"}'
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        out = engine.consultar_batch("msgbatch_01ABC_xyz-1")

    assert out["id"] == "msgbatch_01ABC_xyz-1"
    req_arg = mock_urlopen.call_args.args[0]
    assert req_arg.full_url == "https://api.anthropic.com/v1/messages/batches/msgbatch_01ABC_xyz-1"


@pytest.mark.parametrize(
    "bad_url",
    [
        "file:///etc/passwd",
        "http://api.anthropic.com/v1/messages",
        "https://evil.com/v1/messages",
        "https://api.anthropic.com/v1/messages/../../admin",
        "https://api.anthropic.com/v1/messages/%2e%2e/admin",
        "https://api.anthropic.com/v1\\messages",
    ],
)
def test_http_call_rechaza_urls_inseguras(bad_url: str):
    """Valida esquema HTTPS, host permitido y ausencia de path traversal en _http_call."""
    engine = AnthropicSentimentEngine(api_key="sk-ant-test")
    with patch("urllib.request.urlopen") as mock_urlopen:
        with pytest.raises(ValueError):
            engine._http_call(bad_url, headers={"x-api-key": "test"})
        mock_urlopen.assert_not_called()
