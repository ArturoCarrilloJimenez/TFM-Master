"""Módulo de modelo y clasificación semántica de artículos financieros."""

import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from articulos.categories import CATEGORIES
from shared.logging import Logger
from shared.tickets import (
    ALIAS_BANCOS,
    ANALYST_ROLE_EXTENDED_PATTERNS,
    ANALYST_ROLE_TERMS,
    BOILERPLATE_SECTOR_PATTERNS,
    MACRO_KEYWORDS,
    MULTI_TICKER_LIST_PATTERNS,
    PORTFOLIO_13F_PATTERNS,
    SYSTEMIC_BANKING_TERMS,
)

logger = Logger("CategorizadorArticulos")


def is_running_in_lambda() -> bool:
    """Detecta de forma determinista si el entorno de ejecución actual es AWS Lambda."""
    return "AWS_LAMBDA_FUNCTION_NAME" in os.environ or "LAMBDA_TASK_ROOT" in os.environ


def get_optimal_workers(max_workers: int | None = None) -> int:
    """Calcula el número óptimo de procesos concurrentes según el entorno y hardware."""
    if max_workers is not None:
        return max(1, max_workers)
    if is_running_in_lambda():
        return 1
    cpu_count = os.cpu_count() or 1
    return min(4, max(1, cpu_count // 4))


# Variable global de proceso para aislar la instancia del categorizador por worker sin re-serializar
_worker_categorizador: "CategorizadorArticulos | None" = None


def _init_worker(model_name: str, device: str) -> None:
    """Inicializador por proceso hijo: carga la instancia del modelo con hilos CPU acotados."""
    global _worker_categorizador
    import torch

    torch.set_num_threads(2)
    _worker_categorizador = CategorizadorArticulos(model_name=model_name, device=device)
    _worker_categorizador.bi_model.max_seq_length = 128


def _procesar_chunk_worker(
    chunk: list[dict[str, Any]], batch_size: int = 128
) -> list[dict[str, Any]]:
    """Función de nivel de módulo para ejecución en procesos paralelos."""
    global _worker_categorizador
    if _worker_categorizador is None:
        _worker_categorizador = CategorizadorArticulos()
        _worker_categorizador.bi_model.max_seq_length = 128
    return _worker_categorizador._clasificar_dataset_secuencial(chunk, batch_size=batch_size)


@dataclass(slots=True)
class ArticuloContexto:
    """Contenedor ligero de atributos precalculados para la inferencia semántica."""

    titular: str
    fragmento: str
    banco_prominente: bool
    banco_en_titular: bool
    es_macro: bool
    rol_analista: bool
    es_corta: bool
    es_ticker_puro: bool
    es_contaminacion: bool
    es_plantilla: bool


class CategorizadorArticulos:
    """Clasificador semántico vectorial de noticias financieras con desambiguación de entidad y mitigación de ticker bleed."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        categories: dict[str, list[str]] | None = None,
    ):
        """Inicializa el Bi-Encoder y precalcula los centroides semánticos normalizados por categoría."""
        self.model_name = model_name
        self.device = device
        self.categories_dict = categories or CATEGORIES
        logger.info(f"Cargando Bi-Encoder ({model_name}) en dispositivo [{self.device}]...")
        t0 = time.time()
        self.bi_model = SentenceTransformer(model_name, device=self.device)
        self.bi_model.max_seq_length = 128
        self._normalizar_matriz_centroides()
        logger.info(
            f"Modelo y centroides normalizados inicializados en {time.time() - t0:.2f} s. "
            f"Matriz de categorías: {self.matriz_centroides.shape}"
        )

    def _normalizar_matriz_centroides(self) -> None:
        """Precalcula y normaliza los centroides en espacio euclídeo para inferencia matricial O(1)."""
        centroids: dict[str, np.ndarray] = {}
        for cat, phrases in self.categories_dict.items():
            embs = self.bi_model.encode(phrases, normalize_embeddings=True)
            c = np.mean(embs, axis=0)
            centroids[cat] = c / np.linalg.norm(c)

        self.matriz_centroides = np.array(list(centroids.values()))
        self.nombres_cats = list(centroids.keys())

    @staticmethod
    def contiene_alias(texto: str, alias: str) -> bool:
        """Frontera de palabra regex estricta para evitar colisiones léxicas accidentales."""
        return bool(re.search(r"\b" + re.escape(alias) + r"\b", texto.lower()))

    @staticmethod
    def resolver_cuerpo_noticia(row: pd.Series | dict[str, Any], titular: str) -> str:
        """Resuelve el cuerpo del artículo descartando vacíos y disclaimers estándar."""
        art = str(row.get("article") or "").strip()
        candidato = art if art and art.lower() not in ["none", "nan", ""] else str(row.get("text") or row.get("summary") or "").strip()

        if not candidato or candidato.lower() in ["none", "nan", ""] or candidato.strip() == titular.strip():
            return ""

        if len(candidato) < 160 and any(
            w in candidato.lower()
            for w in ["views and opinions", "do not necessarily reflect", "terms not disclosed"]
        ):
            return ""

        return candidato

    @classmethod
    def extraer_fragmento_y_presencia(
        cls, titular: str, cuerpo: str, ticker_banco: str, max_chars: int = 400
    ) -> tuple[str, bool]:
        """Extrae el fragmento guiado donde el banco es mencionado y un booleano indicando si fue localizado."""
        if not cuerpo or cuerpo.lower() in ["none", "nan", ""]:
            return "", False

        aliases = ALIAS_BANCOS.get(ticker_banco, [ticker_banco.lower()])
        parrafos = [p.strip() for p in re.split(r"\n+|, \n+", cuerpo) if len(p.strip()) > 30]

        # 1. Buscar en párrafos legítimos donde aparezca el banco
        for p in parrafos:
            p_lower = p.lower()
            if any(bp in p_lower for bp in BOILERPLATE_SECTOR_PATTERNS):
                continue
            for a in aliases:
                match = re.search(r"\b" + re.escape(a) + r"\b", p_lower)
                if match:
                    start_pos = max(0, match.start() - 150)
                    end_pos = min(len(p), match.end() + 250)
                    return p[start_pos:end_pos].strip(), True

        # 2. Si no aparece en párrafos limpios, buscar en cuerpo completo libre de boilerplate
        p_limpio = cuerpo
        for bp in BOILERPLATE_SECTOR_PATTERNS:
            p_limpio = re.sub(bp + r".*?(\n|$)", "", p_limpio, flags=re.IGNORECASE)

        p_limpio_lower = p_limpio.lower()
        for a in aliases:
            match = re.search(r"\b" + re.escape(a) + r"\b", p_limpio_lower)
            if match:
                start_pos = max(0, match.start() - 150)
                end_pos = min(len(p_limpio), match.end() + 250)
                return p_limpio[start_pos:end_pos].strip(), True

        # 3. Fallback: Mantener el inicio del texto limpio
        fallback = p_limpio[:max_chars].strip() if p_limpio.strip() else cuerpo[:max_chars].strip()
        return fallback, False

    @classmethod
    def extraer_fragmento_relevante(
        cls, titular: str, cuerpo: str, ticker_banco: str, max_chars: int = 400
    ) -> str:
        """Extrae el fragmento guiado donde el banco es mencionado."""
        frag, _ = cls.extraer_fragmento_y_presencia(titular, cuerpo, ticker_banco, max_chars=max_chars)
        return frag

    @classmethod
    def es_rol_analista_tercero(cls, titular: str, cuerpo: str, ticker_banco: str) -> bool:
        """Detecta si el banco actúa únicamente como analista de un tercero o divisa."""
        t_lower = titular.lower()
        c_lower = cuerpo.lower()
        aliases = ALIAS_BANCOS.get(ticker_banco, [ticker_banco.lower()])
        banco_en_titular = any(cls.contiene_alias(t_lower, a) for a in aliases)

        if not banco_en_titular:
            for a in aliases:
                for term in ANALYST_ROLE_TERMS:
                    if (
                        f"{a} {term}" in c_lower
                        or f"{term} at {a}" in c_lower
                        or f"{a}'s {term}" in c_lower
                        or f"{a} reiterated" in c_lower
                    ):
                        return True
                for pat in ANALYST_ROLE_EXTENDED_PATTERNS:
                    if f"{pat} {a}" in c_lower or f"{a} {pat}" in c_lower:
                        return True
        else:
            for a in aliases:
                if re.search(r"\b" + re.escape(a) + r"\b.*?\b(cuts forecast|raises forecast|warns|warns traders)\b", t_lower):
                    return True
            if "note to clients" in c_lower and any(
                term in c_lower for term in ["bullish conviction", "forecast", "maintains conviction"]
            ):
                return True

        return False

    @classmethod
    def es_contaminacion_competidor(
        cls, titular: str, ticker_banco: str, banco_en_cuerpo: bool = False
    ) -> bool:
        """Detecta si el titular habla explícitamente de un banco competidor del Big 6 y no del banco asignado."""
        t_lower = titular.lower()
        aliases_propios = ALIAS_BANCOS.get(ticker_banco, [ticker_banco.lower()])

        if any(cls.contiene_alias(t_lower, a) for a in aliases_propios):
            return False

        if any(w in t_lower for w in ["big bank", "big banks", "banks", "peers", "rivals", "wall street", "banking sector"]):
            return False

        if banco_en_cuerpo:
            return False

        for competidor, aliases_comp in ALIAS_BANCOS.items():
            if competidor == ticker_banco:
                continue
            if any(cls.contiene_alias(t_lower, ac) for ac in aliases_comp):
                return True

        return False

    @classmethod
    def es_plantilla_irrelevante(cls, titular: str, cuerpo: str) -> bool:
        """Detecta declaraciones patrimoniales de fondos (13F) o listas multi-ticker sin cuerpo."""
        t_lower = titular.lower()
        c_lower = cuerpo.lower()

        if any(re.search(pat, c_lower) for pat in PORTFOLIO_13F_PATTERNS):
            return True

        if not cuerpo and any(re.search(pat, t_lower) for pat in MULTI_TICKER_LIST_PATTERNS):
            return True

        return False

    def _extraer_contexto(self, row: dict[str, Any] | pd.Series) -> ArticuloContexto:
        """Precalcula los metadatos léxicos y de saliencia de un artículo en un objeto estructurado."""
        titular = str(row.get("title") or "").strip()
        cuerpo = self.resolver_cuerpo_noticia(row, titular)
        ticker = str(row.get("ticker") or "").strip()
        t_count = int(row.get("ticker_count") or 1)
        is_multi = bool(
            row.get("is_multisubject")
            if row.get("is_multisubject") is not None
            else (t_count > 1)
        )

        frag, banco_en_cuerpo = self.extraer_fragmento_y_presencia(titular, cuerpo, ticker, max_chars=400)
        t_lower = titular.lower()
        c_lower = frag.lower() if frag else ""
        aliases = ALIAS_BANCOS.get(ticker, [ticker.lower()])

        banco_en_titular = any(self.contiene_alias(t_lower, a) for a in aliases)
        banco_prominente = banco_en_titular or banco_en_cuerpo

        tiene_termino_macro = any(k in t_lower for k in MACRO_KEYWORDS) or any(k in c_lower for k in MACRO_KEYWORDS)
        impacto_banca_sistemica = any(k in t_lower for k in SYSTEMIC_BANKING_TERMS) or any(k in c_lower for k in SYSTEMIC_BANKING_TERMS)
        if banco_prominente:
            es_macro_kw = tiene_termino_macro
        else:
            es_macro_kw = impacto_banca_sistemica

        rol_analista = self.es_rol_analista_tercero(titular, frag or cuerpo, ticker)
        es_contaminacion = self.es_contaminacion_competidor(titular, ticker, banco_en_cuerpo=banco_en_cuerpo)
        es_plantilla = self.es_plantilla_irrelevante(titular, cuerpo)
        es_corta = not bool(cuerpo and cuerpo.strip())
        es_ticker_puro = (t_count == 1) or (not is_multi)

        return ArticuloContexto(
            titular=titular,
            fragmento=frag,
            banco_prominente=banco_prominente,
            banco_en_titular=banco_en_titular,
            es_macro=es_macro_kw,
            rol_analista=rol_analista,
            es_corta=es_corta,
            es_ticker_puro=es_ticker_puro,
            es_contaminacion=es_contaminacion,
            es_plantilla=es_plantilla,
        )

    @staticmethod
    def evaluar_es_ruido(
        cat: str,
        score: float,
        es_corta: bool,
        banco_prominente: bool,
        banco_en_titular: bool,
        es_macro: bool,
        rol_analista: bool,
        es_ticker_puro: bool,
        es_contaminacion: bool = False,
        es_plantilla: bool = False,
    ) -> bool:
        """Determina de forma centralizada si un artículo es clasificado como ruido o señal financiera."""
        es_cat_macro = "macro" in cat.lower()
        macro_activo = es_macro or es_cat_macro

        if rol_analista or es_contaminacion or es_plantilla or "ruido" in cat.lower():
            return True

        if es_corta:
            tau = 0.30 if (banco_en_titular or macro_activo) else 0.38
        else:
            tau = 0.38 if (banco_prominente or macro_activo) else 0.48

        if score < tau:
            return True

        # Regla de anclaje estricta: si la categoría no es macro, el banco DEBE ser prominente
        if not es_cat_macro and not banco_prominente:
            return True

        # Para categorías macro, si el banco no es prominente, debe tener impacto bancario sistémico
        if es_cat_macro and not banco_prominente and not es_macro:
            return True

        # Regla de anclaje obligatorio en titulares huérfanos sin cuerpo
        if es_corta and not banco_en_titular and not es_macro:
            return True

        return False

    def clasificar_noticia(
        self,
        titular: str,
        cuerpo: str,
        ticker_banco: str,
        ticker_count: int = 1,
        is_multisubject: bool = False,
    ) -> dict[str, Any]:
        """Clasifica un artículo individual aplicando desambiguación, embeddings y umbrales adaptativos."""
        ctx = self._extraer_contexto({
            "title": titular,
            "article": cuerpo,
            "ticker": ticker_banco,
            "ticker_count": ticker_count,
            "is_multisubject": is_multisubject,
        })

        emb_t = self.bi_model.encode(ctx.titular, normalize_embeddings=True)
        if ctx.fragmento:
            emb_c = self.bi_model.encode(ctx.fragmento, normalize_embeddings=True)
            emb_doc = 0.65 * emb_t + 0.35 * emb_c
            emb_doc = emb_doc / np.linalg.norm(emb_doc)
        else:
            emb_doc = emb_t

        sims = np.dot(self.matriz_centroides, emb_doc)
        idx_max = int(np.argmax(sims))
        score = float(sims[idx_max])
        cat = self.nombres_cats[idx_max]

        es_ruido = self.evaluar_es_ruido(
            cat=cat,
            score=score,
            es_corta=ctx.es_corta,
            banco_prominente=ctx.banco_prominente,
            banco_en_titular=ctx.banco_en_titular,
            es_macro=ctx.es_macro,
            rol_analista=ctx.rol_analista,
            es_ticker_puro=ctx.es_ticker_puro,
            es_contaminacion=ctx.es_contaminacion,
            es_plantilla=ctx.es_plantilla,
        )

        return {
            "categoria": cat,
            "confianza": round(score * 100, 2),
            "is_ruido": es_ruido,
            "score": score,
        }

    def _clasificar_dataset_secuencial(
        self, data: list[dict[str, Any]] | pd.DataFrame, batch_size: int = 128
    ) -> list[dict[str, Any]]:
        """Ejecuta inferencia semántica matricial de alto rendimiento por lotes."""
        records: list[dict[str, Any]] = (
            data.to_dict(orient="records") if isinstance(data, pd.DataFrame) else data
        )
        n_rows = len(records)
        t_start = time.time()
        logger.info(f"Iniciando clasificación masiva de {n_rows:,} registros por lotes (batch_size={batch_size})...")

        # 1. Extracción contextual en memoria
        contextos = [self._extraer_contexto(row) for row in records]
        titulares = [ctx.titular for ctx in contextos]

        # 2. Vectorización masiva de titulares
        logger.info(f"Vectorizando {n_rows:,} titulares por lotes...")
        embs_t = self.bi_model.encode(
            titulares,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        # 3. Vectorización de fragmentos relevantes no vacíos
        frag_items = [(i, ctx.fragmento) for i, ctx in enumerate(contextos) if ctx.fragmento]
        embs_doc = embs_t.copy()

        if frag_items:
            logger.info(f"Vectorizando {len(frag_items):,} fragmentos relevantes por lotes...")
            frags_to_encode = [item[1] for item in frag_items]
            embs_c = self.bi_model.encode(
                frags_to_encode,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            for k, (orig_idx, _) in enumerate(frag_items):
                combined = 0.65 * embs_t[orig_idx] + 0.35 * embs_c[k]
                embs_doc[orig_idx] = combined / np.linalg.norm(combined)

        # 4. Inferencia matricial de similitud coseno
        logger.info(f"Ejecutando producto matricial ({embs_doc.shape} x {self.matriz_centroides.shape})...")
        sims = np.dot(embs_doc, self.matriz_centroides.T)
        idx_max = np.argmax(sims, axis=1)
        scores = np.max(sims, axis=1)

        # 5. Aplicar evaluación de ruido e inyectar resultados
        data_process: list[dict[str, Any]] = []
        relevantes_count = 0
        ruido_count = 0

        for i, ctx in enumerate(contextos):
            cat = self.nombres_cats[idx_max[i]]
            score = float(scores[i])
            es_ruido = self.evaluar_es_ruido(
                cat=cat,
                score=score,
                es_corta=ctx.es_corta,
                banco_prominente=ctx.banco_prominente,
                banco_en_titular=ctx.banco_en_titular,
                es_macro=ctx.es_macro,
                rol_analista=ctx.rol_analista,
                es_ticker_puro=ctx.es_ticker_puro,
                es_contaminacion=ctx.es_contaminacion,
                es_plantilla=ctx.es_plantilla,
            )

            if es_ruido:
                ruido_count += 1
            else:
                relevantes_count += 1

            row_out = dict(records[i])
            row_out["process_text"] = ctx.fragmento if ctx.fragmento else "(Titular directo sin cuerpo)"
            row_out["category"] = cat
            row_out["confianza"] = round(score * 100, 2)
            row_out["is_ruido"] = es_ruido
            data_process.append(row_out)

        elapsed = time.time() - t_start
        logger.info(
            f"Categorización secuencial completada en {elapsed:.1f} s (~{n_rows / max(elapsed, 0.001):.1f} registros/s). "
            f"Relevantes: {relevantes_count:,} ({relevantes_count / n_rows * 100:.1f}%) | "
            f"Ruido: {ruido_count:,} ({ruido_count / n_rows * 100:.1f}%)"
        )
        return data_process

    def clasificar_dataset(
        self,
        data: list[dict[str, Any]] | pd.DataFrame,
        batch_size: int = 128,
        max_workers: int | None = None,
    ) -> list[dict[str, Any]]:
        """Ejecuta inferencia semántica matricial de alto rendimiento por lotes."""
        records: list[dict[str, Any]] = (
            data.to_dict(orient="records") if isinstance(data, pd.DataFrame) else data
        )
        n_rows = len(records)
        workers = get_optimal_workers(max_workers)

        if workers <= 1 or n_rows < 2000:
            return self._clasificar_dataset_secuencial(records, batch_size=batch_size)

        logger.info(
            f"Ejecutando inferencia paralela para {n_rows:,} registros usando {workers} workers (Entorno: Local)..."
        )
        t_start = time.time()
        chunk_size = (n_rows + workers - 1) // workers
        chunks = [records[i : i + chunk_size] for i in range(0, n_rows, chunk_size)]

        results: list[dict[str, Any]] = []
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(self.model_name, self.device),
        ) as executor:
            futures = [executor.submit(_procesar_chunk_worker, ch, batch_size) for ch in chunks]
            for f in futures:
                results.extend(f.result())

        elapsed = time.time() - t_start
        relevantes = sum(1 for r in results if not r["is_ruido"])
        ruido = n_rows - relevantes
        logger.info(
            f"Inferencia paralela completada en {elapsed:.1f} s (~{n_rows / max(elapsed, 0.001):.1f} registros/s). "
            f"Relevantes: {relevantes:,} ({relevantes / n_rows * 100:.1f}%) | "
            f"Ruido: {ruido:,} ({ruido / n_rows * 100:.1f}%)"
        )
        return results


# Alias de retrocompatibilidad
ModelCategorizar = CategorizadorArticulos
