"""Streamlit viewer. Reads SQLite ONLY (never re-runs) — all logic lives in ezpz.ui.data.

Three views + a failure explorer:
  1. Leaderboard   — pipelines x metrics, slice toggle, CI shown so small gaps read as ties.
  2. Per-document  — pick a doc; predicted vs GT per pipeline, color-coded; source page alongside.
  3. Run diff      — baseline vs new: per-pipeline delta + which cells improved/regressed.
  4. Failures      — every failed cell, filterable by case (hallucination/missing/parse_error/...).

Launched by `ezpz view` (which runs `streamlit run` on this file); DB path comes from $EZPZ_DB.
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from ezpz.store.sqlite import SqliteStore
from ezpz.ui import data as D

_STATUS_EMOJI = {
    "correct": "✅", "wrong": "❌", "missing": "⬜", "hallucinated": "⚠️",
    "absent_ok": "✅", "parse_error": "🛑", "not_labeled": "·", "unscored": "·",
}


def _store() -> SqliteStore:
    return SqliteStore(os.environ.get("EZPZ_DB", ".ezpz/ezpz.sqlite"))


def _leaderboard_view(store: SqliteStore, run_id: str) -> None:
    tags = D.slice_tags(store, run_id)
    slice_tag = st.selectbox("Slice", ["(all)", *tags], index=0)
    rows = D.leaderboard(store, run_id, None if slice_tag == "(all)" else slice_tag)
    table = [
        {
            "pipeline": f"{r['pipeline']} ({r['adapter']})",
            "accuracy": f"{r['accuracy']:.0%}",
            "95% CI": f"[{r['ci_low']:.0%}–{r['ci_high']:.0%}]",
            "macro": f"{r['accuracy_macro']:.0%}",
            "halluc": r["hallucinations"], "missed": r["missed"],
            "fields": r["fields_scored"], "errors": r["errors"],
            "cost $": round(r["cost_usd"], 4), "latency ms": round(r["latency_ms"], 1),
        }
        for r in rows
    ]
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)


def _drilldown_view(store: SqliteStore, run_id: str) -> None:
    docs = D.documents_in_run(store, run_id)
    if not docs:
        st.info("No documents in this run.")
        return
    slug = st.selectbox("Document", [d["slug"] for d in docs])
    doc_id = next(d["doc_id"] for d in docs if d["slug"] == slug)
    view = D.drilldown(store, run_id, doc_id)

    left, right = st.columns([3, 2])
    with left:
        labels = {p["pipeline_id"]: p["label"] for p in view["pipelines"]}
        table = []
        for f in view["fields"]:
            row = {"field": f["field"], "type": f["type"], "ground truth": f["gt"]}
            for pid, label in labels.items():
                cell = f["cells"][pid]
                row[label] = f"{_STATUS_EMOJI.get(cell['status'], '?')} {cell['value']}"
            table.append(row)
        st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
    with right:
        st.caption("Source")
        if view["doc"]["source_text"]:
            st.text(view["doc"]["source_text"])
        else:
            st.info("Source preview unavailable (non-text or missing file).")


def _diff_view(store: SqliteStore, run_id: str, runs: list[dict]) -> None:
    others = [r["run_id"] for r in runs if r["run_id"] != run_id]
    if not others:
        st.info("Need a second run to diff against.")
        return
    baseline = st.selectbox("Baseline run (B)", others)
    diff = D.run_diff(store, run_id, baseline)
    st.subheader("Per-pipeline paired delta (A − B)")
    st.dataframe(pd.DataFrame([
        {
            "pipeline": p["label"], "Δ accuracy": f"{p['mean_delta']:+.1%}",
            "95% CI": f"[{p['ci_low']:+.2f}, {p['ci_high']:+.2f}]",
            "n": p["n"], "verdict": "within noise" if p["within_noise"] else "real",
        }
        for p in diff["pipelines"]
    ]), hide_index=True, use_container_width=True)
    c1, c2 = st.columns(2)
    c1.metric("Improved cells", len(diff["improvements"]))
    c2.metric("Regressed cells", len(diff["regressions"]))
    if diff["regressions"]:
        st.subheader("Regressions")
        st.dataframe(pd.DataFrame(diff["regressions"]), hide_index=True, use_container_width=True)


def _failures_view(store: SqliteStore, run_id: str) -> None:
    cases = sorted({f["case"] for f in D.failures(store, run_id) if f["case"]})
    case = st.selectbox("Case", ["(all)", *cases])
    rows = D.failures(store, run_id, None if case == "(all)" else case)
    if not rows:
        st.success("No failures 🎉")
        return
    st.dataframe(
        pd.DataFrame([{k: v for k, v in r.items() if k != "detail"} for r in rows]),
        hide_index=True, use_container_width=True,
    )


def main() -> None:
    st.set_page_config(page_title="ezpz evals", layout="wide")
    st.title("ezpz evals — viewer")
    store = _store()
    runs = D.list_runs(store)
    if not runs:
        st.warning("No runs found. Run `ezpz run <experiment>` first.")
        return

    labels = {f"{r['run_id']} · {r['dataset']} · {r['task']}": r["run_id"] for r in runs}
    run_id = labels[st.sidebar.selectbox("Run", list(labels))]

    tabs = st.tabs(["Leaderboard", "Per-document", "Run diff", "Failures"])
    with tabs[0]:
        _leaderboard_view(store, run_id)
    with tabs[1]:
        _drilldown_view(store, run_id)
    with tabs[2]:
        _diff_view(store, run_id, runs)
    with tabs[3]:
        _failures_view(store, run_id)


if __name__ == "__main__":
    main()
