"""Dynamic context expansion: winning procedure -> every chunk/step/page it owns, in order.

Illustrated manuals often split one job across caption-sized 'procedures' ('2', '1 2').
Those are merged back onto the parent section so the UI always has the page text.

If the PDF has no numbered steps, body paragraphs on the retrieved pages become
the repair instructions — the LLM never has to invent them.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..ingest.structure import CALLOUT_RE
from ..index.store import Store
from .retriever import ProcedureHit

_INSTR = re.compile(
    r"\b(remove|removed|install|loosen|tighten|replace|slide|push|pull|undo|refit|clean|apply|"
    r"torque|align|squeeze|insert|lift|check|inspect|fit|unbolt|disconnect|reconnect|pry|"
    r"thread|screw|unscrew|press|hold|rotate|turn|open|close|release|clip|unclip|bed|adjust)\b",
    re.I,
)
_PAGE_MARK = re.compile(r"\s*\[p+\.\d+(?:\s*-\s*\d+)?\]")
MAX_MERGE_PROCS = 14
MAX_MERGE_PAGES = 8


@dataclass
class ExpandedProcedure:
    procedure_id: str
    document_id: str
    doc_title: str
    source_path: str
    title: str
    heading_path: str
    pages: list[int]
    overview: list[dict]           # [{"text", "pages"}]
    steps: list[dict]              # [{"sid", "n", "phase", "phase_title", "text", "pages"}]
    warnings: list[dict]           # [{"label", "text", "pages"}]
    figures: list[str]
    merged_procedure_ids: list[str] = field(default_factory=list)
    missing_step_numbers: list[str] = field(default_factory=list)
    duplicates_removed: int = 0
    truncated: bool = False


def _norm(t: str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()


def section_key(p: dict) -> str:
    parts = [x.strip() for x in str(p.get("heading_path") or "").split(">") if x.strip()]
    while parts and CALLOUT_RE.match(parts[-1]):
        parts.pop()
    title = (p.get("title") or "").strip()
    if title and not CALLOUT_RE.match(title) and title not in parts:
        parts.append(title)
    return " > ".join(parts)


def display_title(p: dict) -> str:
    parts = [x.strip() for x in str(p.get("heading_path") or "").split(">") if x.strip()]
    while parts and CALLOUT_RE.match(parts[-1]):
        parts.pop()
    title = (p.get("title") or "").strip()
    if title and not CALLOUT_RE.match(title):
        return title
    return parts[-1] if parts else title or "Repair procedure"


def _collect(rows: list[dict]):
    overview, steps, warnings, figures = [], [], [], []
    for r in rows:
        if r["chunk_type"] in ("overview", "reference"):
            overview.append({"text": r["text"], "pages": list(r["pages"])})
        steps.extend(json.loads(r["steps_json"]))
        warnings.extend(json.loads(r["warnings_json"]))
        figures.extend(r["figures"])
    return overview, steps, warnings, figures


def _continues(prev_nums: list[int], next_nums: list[int]) -> bool:
    return bool(prev_nums and next_nums and next_nums[0] > 1 and next_nums[0] == prev_nums[-1] + 1)


def _same_section(a: dict, b: dict) -> bool:
    return a["document_id"] == b["document_id"] and section_key(a) == section_key(b)


def _walk_section(store: Store, start: dict) -> list[dict]:
    """Include neighbouring caption-fragments that belong to the same parent section."""
    chain = [start]
    pages = set(start["pages"])
    cur = start
    while cur["prev_procedure_id"] and len(chain) < MAX_MERGE_PROCS:
        prev = store.procedures_by_ids([cur["prev_procedure_id"]]).get(cur["prev_procedure_id"])
        if not prev:
            break
        numbering = _continues(list(prev["step_numbers"]), list(cur["step_numbers"]))
        if not (numbering or _same_section(prev, cur)):
            break
        nxt_pages = pages | set(prev["pages"])
        if max(nxt_pages) - min(nxt_pages) + 1 > MAX_MERGE_PAGES and not numbering:
            break
        chain.insert(0, prev)
        pages = nxt_pages
        cur = prev
    cur = start
    while cur["next_procedure_id"] and len(chain) < MAX_MERGE_PROCS:
        nxt = store.procedures_by_ids([cur["next_procedure_id"]]).get(cur["next_procedure_id"])
        if not nxt:
            break
        numbering = _continues(list(cur["step_numbers"]), list(nxt["step_numbers"]))
        if not (numbering or _same_section(cur, nxt)):
            break
        nxt_pages = pages | set(nxt["pages"])
        if max(nxt_pages) - min(nxt_pages) + 1 > MAX_MERGE_PAGES and not numbering:
            break
        chain.append(nxt)
        pages = nxt_pages
        cur = nxt
    return chain


def _body_to_steps(overview: list[dict], existing: list[dict]) -> list[dict]:
    if existing:
        return existing
    steps, sid = [], 0
    for o in overview:
        paras = [p.strip() for p in re.split(r"\n+", o["text"] or "") if p.strip()]
        widened = []
        for para in paras:
            if len(para.split()) > 45:
                widened.extend([s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()])
            else:
                widened.append(para)
        for para in widened:
            text = _PAGE_MARK.sub("", para).strip()
            text = re.sub(r"^(?:step\s*)?\d{1,2}\s*[.):]\s*", "", text, flags=re.I)
            if len(text.split()) < 6:
                continue
            if text.lower().startswith("tools"):
                continue
            if not _INSTR.search(text):
                continue
            sid += 1
            steps.append({"sid": sid, "n": sid, "phase": 1, "phase_title": "",
                          "text": text, "pages": list(o["pages"])})
    return steps


def expand(store: Store, hit: ProcedureHit, max_words: int = 3500) -> ExpandedProcedure:
    p = hit.row
    chain = _walk_section(store, p)

    rows = [r for proc in chain for r in store.chunks_of_procedure(proc["procedure_id"])]
    overview, raw_steps, raw_warn, figures = _collect(rows)

    seen, steps, dupes = set(), [], 0
    for s in raw_steps:
        key = (s.get("phase", 1), s["n"], _norm(s["text"])[:80])
        if key in seen:
            dupes += 1
            continue
        seen.add(key)
        steps.append(s)
    steps = _body_to_steps(overview, steps)
    for i, s in enumerate(steps, 1):
        s["sid"] = i

    wseen, warnings = set(), []
    for w in raw_warn:
        k = _norm(w["text"])
        if k not in wseen:
            wseen.add(k)
            warnings.append(w)

    missing = []
    by_phase: dict[int, list[int]] = {}
    for s in steps:
        by_phase.setdefault(s.get("phase", 1), []).append(s["n"])
    for ph, nums in by_phase.items():
        if not nums:
            continue
        for n in range(1, max(nums) + 1):
            if n not in nums:
                missing.append(f"{ph}.{n}" if len(by_phase) > 1 else str(n))

    truncated = False

    def words() -> int:
        return sum(len(x["text"].split()) for x in steps + overview + warnings) + sum(len(f.split()) for f in figures)

    if words() > max_words:
        overview = [{"text": " ".join(o["text"].split()[:150]), "pages": o["pages"]} for o in overview[:1]]
        truncated = True
    if words() > max_words:
        figures = []

    pages = sorted({pg for proc in chain for pg in proc["pages"]})
    title = display_title(p)
    return ExpandedProcedure(
        procedure_id=p["procedure_id"], document_id=p["document_id"], doc_title=p["doc_title"],
        source_path=p["source_path"], title=title, heading_path=p["heading_path"], pages=pages,
        overview=overview, steps=steps, warnings=warnings, figures=figures,
        merged_procedure_ids=[c["procedure_id"] for c in chain if c["procedure_id"] != p["procedure_id"]],
        missing_step_numbers=missing, duplicates_removed=dupes, truncated=truncated,
    )
