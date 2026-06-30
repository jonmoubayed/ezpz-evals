"""Local viewer server — Python stdlib only (no web framework, no extra deps).

`ezpz view` launches this: it serves the static SPA (ui/static/index.html), a small JSON API, and
the raw source files behind the drill-down. All *read* logic lives in the framework-free
`ezpz.ui.data`; launching a run lives in `ezpz.ui.launch`.

`api_route` is a pure function (store, path, query) -> (status, payload) so the read API is
unit-testable without binding a socket. Reads touch SQLite only. The single write path is
POST /api/run (re-run an experiment, budget-gated) — see `launch`.
"""
from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from ezpz.store.sqlite import SqliteStore
from ezpz.ui import data as D
from ezpz.ui import launch as L

STATIC_DIR = Path(__file__).resolve().parent / "static"

_IMAGE_MIME = ("image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp")


def _default_run(store: SqliteStore, requested: Optional[str]) -> Optional[str]:
    runs = [r.run_id for r in store.list_runs()]
    if requested and requested in runs:
        return requested
    return runs[-1] if runs else None  # newest (list_runs is oldest-first)


def _default_base(store: SqliteStore, run_id: str, requested: Optional[str]) -> str:
    runs = store.list_runs()
    ids = [r.run_id for r in runs]
    if requested and requested in ids:
        return requested
    cur = next((r for r in runs if r.run_id == run_id), None)
    # prefer the newest OTHER run on the same dataset × task (a diff across cohorts is meaningless)
    same = [
        r.run_id for r in reversed(runs)
        if r.run_id != run_id and cur is not None
        and r.dataset_ref == cur.dataset_ref and r.task_ref == cur.task_ref
    ]
    if same:
        return same[0]
    others = [r for r in reversed(ids) if r != run_id]
    return others[0] if others else run_id


def api_route(store: SqliteStore, path: str, query: dict[str, str]) -> tuple[int, dict]:
    """Resolve one GET API request to (http_status, json_payload). Pure + socket-free for testing."""
    run_id = _default_run(store, query.get("run"))
    if path == "/api/state":
        return 200, {"runs": D.run_menu(store), "current": run_id}

    if path == "/api/run_status":
        return 200, L.job_status(query.get("run", ""))

    if run_id is None:
        return 404, {"error": "no runs found — run `ezpz run <experiment>` first"}

    if path == "/api/leaderboard":
        slice_tag = query.get("slice", "all")
        return 200, {"run": run_id, **D.leaderboard_board(store, run_id, slice_tag)}

    if path == "/api/documents":
        return 200, {"run": run_id, "docs": D.documents_in_run(store, run_id)}

    if path == "/api/doc":
        docs = D.documents_in_run(store, run_id)
        if not docs:
            return 404, {"error": "no documents in run"}
        doc_id = query.get("doc") or ""
        if doc_id not in {d["doc_id"] for d in docs}:
            doc_id = str(docs[0]["doc_id"])
        return 200, {"run": run_id, **D.drilldown(store, run_id, doc_id)}

    if path == "/api/diff":
        base = _default_base(store, run_id, query.get("base"))
        return 200, D.diff_view(store, run_id, base)

    if path == "/api/failures":
        return 200, {"run": run_id, **D.failure_rows(store, run_id)}

    if path == "/api/analyze":
        return 200, {"run": run_id, **D.analyze(store, run_id)}

    if path == "/api/estimate":
        try:
            sample = int(query.get("sample", "10"))
            cap = float(query.get("cap", "25"))
        except ValueError:
            return 400, {"error": "sample/cap must be numeric"}
        return 200, {"run": run_id, **D.estimate(store, run_id, sample, cap)}

    return 404, {"error": f"unknown endpoint {path}"}


def _source_bytes(store: SqliteStore, run_id: str, doc_id: str) -> tuple[bytes, str] | None:
    """Raw bytes + content-type for a document's source file, or None. Serves only the exact
    source_path the store recorded for a doc in this run (no arbitrary path access)."""
    if doc_id not in {r.doc_id for r in store.load_results(run_id)}:
        return None
    doc = store.load_documents([doc_id]).get(doc_id)
    if not doc or not doc.source_path:
        return None
    path = Path(doc.source_path)
    if not path.is_file():
        return None
    mime = doc.mime or "application/octet-stream"
    return path.read_bytes(), mime


def _make_handler(db_path: str, root: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet; the CLI prints the URL itself
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict) -> None:
            self._send(status, json.dumps(payload, default=str).encode(), "application/json")

        def _store(self) -> SqliteStore:
            store = SqliteStore(db_path)
            store.init_db()
            return store

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            if path == "/api/source":
                got = _source_bytes(self._store(), query.get("run", ""), query.get("doc", ""))
                if got is None:
                    self._send(404, b"not found", "text/plain")
                else:
                    self._send(200, got[0], got[1])
                return
            if path.startswith("/api/"):
                status, payload = api_route(self._store(), path, query)
                self._json(status, payload)
                return
            self._serve_static(path)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json(400, {"error": "invalid JSON body"})
                return
            if parsed.path == "/api/run":
                run = body.get("run")
                try:
                    sample = int(body.get("sample", 0))
                    cap = float(body.get("cap", 0))
                except (TypeError, ValueError):
                    self._json(400, {"error": "sample/cap must be numeric"})
                    return
                if not run:
                    self._json(400, {"error": "missing 'run'"})
                    return
                job = L.launch(db_path, root, run, sample, cap)
                self._json(200 if job.get("status") in ("running",) else 409, job)
                return
            self._json(404, {"error": f"unknown endpoint {parsed.path}"})

        def _serve_static(self, path: str) -> None:
            rel = "index.html" if path in ("/", "") else path.lstrip("/")
            target = (STATIC_DIR / rel).resolve()
            if not str(target).startswith(str(STATIC_DIR)) or not target.is_file():
                self._send(404, b"not found", "text/plain")
                return
            ctype = {
                ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
                ".svg": "image/svg+xml", ".ico": "image/x-icon",
            }.get(target.suffix, "application/octet-stream")
            self._send(200, target.read_bytes(), ctype + "; charset=utf-8")

    return Handler


def serve(
    db_path: str, host: str = "127.0.0.1", port: int = 8501,
    open_browser: bool = True, root: Optional[str] = None,
) -> None:
    """Blocking: start the viewer server and (optionally) open a browser at it.

    `root` is the eval project directory (datasets/ + tasks/) used to launch re-runs; defaults to
    the current working directory."""
    project_root = root or str(Path.cwd())
    httpd = ThreadingHTTPServer((host, port), _make_handler(db_path, project_root))
    url = f"http://{host}:{port}/"
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    print(f"ezpz viewer → {url}  (reads {db_path} · runs resolve from {project_root} · Ctrl-C)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        httpd.server_close()
