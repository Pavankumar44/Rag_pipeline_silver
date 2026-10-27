"""Hybrid retrieval -> cross-encoder rerank -> procedure-level aggregation.

Pages/chunks are only evidence; the unit that gets ranked and returned is the procedure.

    S(P) = w_max * max_c ce(c) + w_top2 * mean(top2 ce(c)) + w_title * ce(q, P.summary) + w_cov * coverage(P)
    S(P) *= component_mismatch_penalty   if query names components and P shares none
    S(P) *= reference_section_penalty    if query is a how-to and P has no numbered steps
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..config import Config
from ..index.models import embed_key, get_embedder, get_reranker, rerank_key
from ..index.store import Store
from ..taxonomy import HOWTO_RE, query_components


@dataclass
class ProcedureHit:
    procedure_id: str
    document_id: str
    doc_title: str
    title: str
    pages: list[int]
    unit_type: str
    components: list[str]
    score: float
    s_max: float
    s_top2: float
    s_title: float
    coverage: float
    n_hits: int
    best_chunk_id: str
    penalties: list[str] = field(default_factory=list)
    row: dict = field(default_factory=dict, repr=False)


@dataclass
class RetrievalResult:
    query: str
    procedures: list[ProcedureHit]
    chunk_scores: dict[str, float]
    timings: dict[str, float]


def _where(filters: dict | None) -> str | None:
    if not filters:
        return None
    clauses = []
    if ids := filters.get("document_ids"):
        clauses.append("document_id IN (" + ",".join(f"'{i}'" for i in ids) + ")")
    if comp := filters.get("component"):
        clauses.append(f"array_has_any(components, ['{comp}'])")
    return " AND ".join(clauses) or None


class Retriever:
    def __init__(self, cfg: Config, store: Store | None = None):
        self.cfg = cfg
        self.store = store or Store(cfg.index_dir)
        self.embedder = get_embedder(embed_key(cfg))
        self.reranker = get_reranker(rerank_key(cfg))

    def retrieve(self, query: str, filters: dict | None = None) -> RetrievalResult:
        cfg, T = self.cfg, {}
        t = time.perf_counter()
        qvec = self.embedder.embed_query(query)
        T["embed_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        where = _where(filters)
        dense = self.store.dense_chunks(qvec, cfg.k_dense, where)
        sparse = self.store.bm25.search(query, cfg.k_sparse)
        proc_dense = self.store.dense_procedures(qvec, cfg.k_proc_dense, where)
        T["search_ms"] = (time.perf_counter() - t) * 1000

        # ---- reciprocal rank fusion over the two chunk rankings -------------
        rrf: dict[str, float] = {}
        for ranking in ([r["chunk_id"] for r in dense], [cid for cid, _ in sparse]):
            for rank, cid in enumerate(ranking):
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (cfg.rrf_k + rank + 1)
        cand_ids = [cid for cid, _ in sorted(rrf.items(), key=lambda kv: -kv[1])[: cfg.n_rerank]]
        # procedure-title matches contribute their first chunks so a well-titled procedure is never missed
        for p in proc_dense[:5]:
            for cid in p["chunk_ids"][:2]:
                if cid not in cand_ids:
                    cand_ids.append(cid)

        rows = {r["chunk_id"]: r for r in dense}
        rows.update(self.store.chunks_by_ids([c for c in cand_ids if c not in rows]))
        if filters:  # BM25 is unfiltered; enforce filters on fetched rows
            allowed_docs = set(filters.get("document_ids") or [])
            comp = filters.get("component")
            rows = {k: r for k, r in rows.items()
                    if (not allowed_docs or r["document_id"] in allowed_docs) and (not comp or comp in r["components"])}
        cand_ids = [c for c in cand_ids if c in rows]

        t = time.perf_counter()
        ce = dict(zip(cand_ids, self.reranker.score(query, [rows[c]["embed_text"] for c in cand_ids])))
        T["rerank_ms"] = (time.perf_counter() - t) * 1000

        # ---- aggregate chunk evidence per procedure -----------------------
        t = time.perf_counter()
        by_proc: dict[str, list[tuple[str, float]]] = {}
        for cid, s in ce.items():
            by_proc.setdefault(rows[cid]["procedure_id"], []).append((cid, s))
        prow = {p["procedure_id"]: p for p in proc_dense}
        prow.update(self.store.procedures_by_ids([pid for pid in by_proc if pid not in prow]))

        prelim = sorted(by_proc.items(), key=lambda kv: -max(s for _, s in kv[1]))
        title_targets = [pid for pid, _ in prelim[: cfg.n_title_rerank]]
        title_scores = dict(zip(title_targets, self.reranker.score(
            query, [f"{prow[p]['title']}. {prow[p]['heading_path']}. {prow[p]['summary_text']}" for p in title_targets])))

        q_comps = set(query_components(query))
        howto = bool(HOWTO_RE.search(query))
        hits: list[ProcedureHit] = []
        for pid, ev in by_proc.items():
            p = prow[pid]
            ev.sort(key=lambda x: -x[1])
            scores = [s for _, s in ev]
            s_max = scores[0]
            if p["n_chunks"] <= 1:
                s_top2 = s_max
            else:
                s_top2 = (scores[0] + (scores[1] if len(scores) > 1 else 0.0)) / 2
            hit_count = sum(1 for s in scores if s >= 0.05)
            coverage = min(1.0, hit_count / max(p["n_chunks"], 1))
            s_title = title_scores.get(pid, 0.0)
            score = cfg.w_max * s_max + cfg.w_top2 * s_top2 + cfg.w_title * s_title + cfg.w_cov * coverage
            penalties = []
            if q_comps and p["components"] and not (q_comps & set(p["components"])):
                score *= cfg.component_mismatch_penalty
                penalties.append("component_mismatch")
            if howto and p["unit_type"] != "procedure":
                score *= cfg.reference_section_penalty
                penalties.append("not_a_procedure")
            hits.append(ProcedureHit(pid, p["document_id"], p["doc_title"], p["title"], list(p["pages"]),
                                     p["unit_type"], list(p["components"]), round(score, 4), round(s_max, 4),
                                     round(s_top2, 4), round(s_title, 4), round(coverage, 3), len(ev), ev[0][0],
                                     penalties, p))
        hits.sort(key=lambda h: -h.score)
        T["aggregate_ms"] = (time.perf_counter() - t) * 1000
        return RetrievalResult(query, hits, ce, T)
