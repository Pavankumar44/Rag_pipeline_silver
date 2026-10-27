"""Web UI + JSON API on the Python standard library (no web framework).

  python -m bikerag serve [--host 127.0.0.1] [--port 8000]

  GET  /                         UI
  GET  /api/status               manuals in the index, backend info
  POST /api/ask                  {"question": str, "document_ids": [str]?} -> cited repair answer
  GET  /api/page?doc=ID&page=N   PNG render of one manual page (cached)
  GET  /api/pdf?doc=ID           the source PDF
"""
from __future__ import annotations

import json
import mimetypes
import threading
import time
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..config import Config
from ..generation.llm import resolve_backend

STATIC = Path(__file__).parent / "static"
MAX_BODY = 16 * 1024


class App:
    def __init__(self, cfg: Config):
        from ..pipeline import RAGPipeline

        self.cfg = cfg
        self.pipeline = RAGPipeline(cfg)
        self.lock = threading.Lock()
        self.documents = {d["document_id"]: d for d in self.pipeline.store.table("documents").to_arrow().to_pylist()}
        self.log_path = Path(cfg.index_dir).parent / "logs" / "ask.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def status(self) -> dict:
        backend, model = resolve_backend(self.cfg)
        return {
            "ready": True,
            "n_manuals": len(self.documents),
            "manuals": [{"document_id": d["document_id"], "title": d["title"], "n_pages": d["n_pages"],
                         "n_procedures": d["n_procedures"]} for d in self.documents.values()],
            "llm_backend": backend,
            "llm_model": None if backend == "extractive" else model,
            "embed_model": self.cfg.embed_model,
            "rerank_model": self.cfg.rerank_model,
            "retrieval": "hybrid",
        }

    def ask(self, question: str, document_ids: list[str] | None) -> dict:
        filters = {"document_ids": [d for d in document_ids if d in self.documents]} if document_ids else None
        t0 = time.perf_counter()
        with self.lock:
            ans = self.pipeline.ask(question, filters)
        elapsed = (time.perf_counter() - t0) * 1000
        backend, model = resolve_backend(self.cfg)
        answer = {
            "query": ans.query,
            "answerable": ans.answerable,
            "in_scope": ans.in_scope,
            "title": ans.title,
            "summary": ans.summary,
            "document_id": ans.document_id,
            "doc_title": ans.doc_title,
            "pages": ans.pages,
            "tools": ans.tools,
            "warnings": ans.warnings,
            "steps": ans.steps,
            "figures": ans.figures,
            "ranked_manuals": ans.ranked_manuals,
            "missing_info": ans.missing_info,
        }
        result = {"elapsed_ms": round(elapsed), "llm_backend": backend, "llm_model": model,
                  "retrieval": "hybrid", "answer": answer}
        with self.log_path.open("a") as f:
            f.write(json.dumps({"ts": time.time(), **result}) + "\n")
        return result

    @lru_cache(maxsize=256)
    def page_png(self, doc_id: str, page: int, zoom: float) -> bytes:
        import pymupdf

        doc = pymupdf.open(self.documents[doc_id]["source_path"])
        if not 1 <= page <= doc.page_count:
            raise KeyError(page)
        return doc[page - 1].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "bikerag"

        def log_message(self, fmt, *args):
            if "/api/ask" in (args[0] if args else ""):
                super().log_message(fmt, *args)

        def _send(self, status: int, body: bytes, ctype: str, cache: bool = False):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def _json(self, obj, status: int = 200):
            self._send(status, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/api/status":
                    return self._json(app.status())
                if url.path == "/api/page":
                    doc, page = q.get("doc", ""), int(q.get("page", "0"))
                    zoom = min(max(float(q.get("zoom", "1.6")), 0.5), 3.0)
                    if doc not in app.documents:
                        return self._json({"error": "unknown document"}, 404)
                    return self._send(200, app.page_png(doc, page, zoom), "image/png", cache=True)
                if url.path == "/api/pdf":
                    doc = q.get("doc", "")
                    if doc not in app.documents:
                        return self._json({"error": "unknown document"}, 404)
                    return self._send(200, Path(app.documents[doc]["source_path"]).read_bytes(), "application/pdf")
                name = "index.html" if url.path in ("/", "/index.html") else url.path.lstrip("/")
                path = (STATIC / name).resolve()
                if STATIC.resolve() not in path.parents or not path.is_file():
                    return self._json({"error": "not found"}, 404)
                ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                return self._send(200, path.read_bytes(), ctype)
            except (KeyError, ValueError):
                return self._json({"error": "bad request"}, 400)

        def do_POST(self):
            if urlparse(self.path).path != "/api/ask":
                return self._json({"error": "not found"}, 404)
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY:
                return self._json({"error": "body too large or empty"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            try:
                body = json.loads(self.rfile.read(length))
                question = str(body.get("question", "")).strip()[:500]
                docs = body.get("document_ids") or None
            except (json.JSONDecodeError, AttributeError):
                return self._json({"error": "invalid JSON"}, 400)
            if not question:
                return self._json({"error": "question is empty"}, 400)
            try:
                return self._json(app.ask(question, docs))
            except Exception as e:
                return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    return Handler


def serve(cfg: Config, host: str = "127.0.0.1", port: int = 8000) -> None:
    print("Loading models and index ...")
    app = App(cfg)
    try:
        app.pipeline.retriever.retrieve("replace brake pads")
    except Exception as e:
        print(f"retriever warmup skipped: {e}")
    backend, model = resolve_backend(cfg)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    print(f"Shop bench: http://{host}:{port}  ({len(app.documents)} manuals, LLM: {backend}"
          f"{'' if backend == 'extractive' else ' · ' + model})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
