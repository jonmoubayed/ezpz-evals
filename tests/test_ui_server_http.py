"""Socket-level smoke test: boot the real ThreadingHTTPServer and drive it over HTTP.

`test_ui_server.py` covers the pure `api_route`; this covers the bits that only exist once a real
server is bound — `do_GET`/`do_POST`, static serving, the export download — plus a guard that the
SPA file is actually present (a wheel-packaging regression would otherwise ship a broken viewer).
"""
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from ezpz.adapters.registry import get_adapter
from ezpz.config.experiment import ExperimentConfig, resolve_dataset, resolve_task
from ezpz.core.run import Run, RunStatus
from ezpz.engine.cache import RawResponseCache
from ezpz.engine.executor import Executor
from ezpz.store.blobs import BlobStore
from ezpz.store.sqlite import SqliteStore
from ezpz.ui.server import STATIC_DIR, _make_handler

REPO = Path(__file__).resolve().parents[1]
ROOT = str(REPO / "examples")


def test_static_spa_ships_with_the_package():
    # the wheel includes ui/static/* (pyproject artifacts) — if this path breaks, `ezpz view` 404s
    assert (STATIC_DIR / "index.html").is_file()


def _serve(tmp_path):
    db = str(tmp_path / "db.sqlite")
    store = SqliteStore(db)
    store.init_db()
    cache = RawResponseCache(store, BlobStore(str(tmp_path / "blobs")))
    exp = ExperimentConfig.load_yaml(str(REPO / "examples" / "experiments" / "fake_invoice_smoke.yaml"))
    dataset = resolve_dataset(ROOT, exp.dataset)
    task = resolve_task(ROOT, exp.task)
    pipelines = [get_adapter(p.adapter)(p) for p in exp.pipelines]
    run = Run(run_id="r1", dataset_ref=exp.dataset, task_ref=exp.task, pipelines=exp.pipelines,
              scorers=exp.scorers, options=exp.options, status=RunStatus.RUNNING)
    Executor(store, cache).run(run, dataset, task, pipelines, exp.scorers)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, ROOT))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.read(), r.headers


def test_server_serves_spa_api_and_export(tmp_path):
    httpd, base = _serve(tmp_path)
    try:
        status, body, _ = _get(base + "/")
        assert status == 200 and b"<!DOCTYPE html>" in body and b"ezpz" in body  # the SPA

        status, body, _ = _get(base + "/api/state")
        assert status == 200 and json.loads(body)["current"] == "r1"

        for ep in ("/api/leaderboard", "/api/fields", "/api/cost", "/api/analyze"):
            status, body, _ = _get(base + ep + "?run=r1")
            assert status == 200, ep

        # export sends a JSON attachment
        status, body, headers = _get(base + "/api/export?run=r1")
        assert status == 200 and "attachment" in headers.get("Content-Disposition", "")
        assert json.loads(body)  # valid run export
    finally:
        httpd.shutdown()


def test_server_launches_a_run_over_http(tmp_path):
    httpd, base = _serve(tmp_path)
    try:
        req = urllib.request.Request(
            base + "/api/run", method="POST",
            data=json.dumps({"run": "r1", "sample": 1, "cap": 10}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            job = json.loads(r.read())
        assert job["status"] == "running" and job["run_id"]
    finally:
        httpd.shutdown()
