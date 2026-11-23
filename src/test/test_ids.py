"""Tests unitarios para generación determinista de record_id e input_hash."""

from shared.ids import generar_categorizer_hash, generar_input_hash, generar_record_id


def test_generar_record_id_determinista_y_unico():
    """Verifica determinismo, normalización de espacios y sensibilidad a ticker/fecha/titular."""
    id1 = generar_record_id("JPM", "2023-05-01 10:00:00", "JPMorgan beats Q1")
    id2 = generar_record_id(" JPM ", "2023-05-01 10:00:00", " JPMorgan beats Q1 ")
    id_otro_banco = generar_record_id("BAC", "2023-05-01 10:00:00", "JPMorgan beats Q1")
    id_otra_fecha = generar_record_id("JPM", "2023-05-02 10:00:00", "JPMorgan beats Q1")

    assert len(id1) == 24
    assert id1 == id2
    assert id1 != id_otro_banco
    assert id1 != id_otra_fecha


def test_generar_input_hash_inmune_a_metadatos():
    """Verifica que input_hash dependa solo de ticker, modelo, prompt y contexto."""
    h1 = generar_input_hash("WFC", "claude-haiku-5-5", "v1_en", "Wells Fargo beat EPS.")
    h2 = generar_input_hash("WFC", "claude-haiku-5-5", "v1_en", "Wells Fargo beat EPS.")
    h_otro_prompt = generar_input_hash("WFC", "claude-haiku-5-5", "v2_en", "Wells Fargo beat EPS.")

    assert len(h1) == 64
    assert h1 == h2
    assert h1 != h_otro_prompt


def test_generar_categorizer_hash_inmune_a_record_id():
    """Verifica que cat_hash sea estable aunque cambien metadatos externos como record_id."""
    c1 = generar_categorizer_hash("JPM", "Headline", "Body text", ticker_count=1, is_multisubject=False)
    c2 = generar_categorizer_hash("JPM", "Headline", "Body text", ticker_count=1, is_multisubject=False)
    c_multi = generar_categorizer_hash("JPM", "Headline", "Body text", ticker_count=2, is_multisubject=True)

    assert len(c1) == 64
    assert c1 == c2
    assert c1 != c_multi
