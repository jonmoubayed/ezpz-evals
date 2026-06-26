# Local UI redesign — faithful port of `ezpz Evals.dc.html`

**Status:** approved (user: "keep it light but actually useful" + "be faithful to the design").
**Source design:** claude.ai/design project `ac34bdf7…` → `ezpz Evals.dc.html` (dark/light terminal-aesthetic SPA).

## Goal
Replace the Streamlit viewer with a faithful implementation of the design: a dense, themeable
(dark/light) single-page terminal UI with five views, served by a tiny local web server. Render
**real** data from SQLite throughout — faithful *layout/aesthetic*, honest *content* (never fabricate
numbers; where the data model lacks a field, render the slot gracefully rather than invent).

## Architecture (decided)
- **Standalone app, Python stdlib only.** No new dependencies; the `[ui]` extra (streamlit, pandas)
  is dropped. `ezpz view` launches `http.server` (ThreadingHTTPServer) and opens the browser.
- **Server is thin.** `src/ezpz/ui/server.py` exposes a pure, unit-testable
  `api_route(store, path, query) -> (status, dict)` plus an HTTP handler that serves the static
  `index.html` and the JSON API. All read logic stays in the framework-free `ezpz.ui.data`.
- **Display logic lives in the browser**, ported verbatim from the design's `renderVals` (theme
  palettes, colour thresholds, CI-bar math, glyph/case maps). The API returns *semantic* view-models;
  the JS computes styling exactly as the mockup does → maximum fidelity, thin server.
- **Read-only invariant preserved.** The budget modal is **estimate-only**: it computes a real
  estimate from the current run's observed cost/doc and shows the exact `ezpz run …` command to
  copy. No run is launched from the browser (structured so a future POST /api/run is a small add).

## Views (all backed by real data)
1. **Leaderboard** — `data.leaderboard`: rank, dot, name, STRAT badge (cascade/ensemble/verify),
   cfg (derived from `pc.config`), accuracy + ±CI + macro, CI whisker bar (domain computed from
   data), hal, miss, $/doc, p50 latency (new), err; slice toggles (tags); paired-compare footer.
2. **Documents** — `data.drilldown` (extended): doc list with per-doc accuracy bar + tags; field ×
   pipeline grid with case glyph/colour + value + confidence sub-label; source panel = real source
   text styled as the design's "page", with provenance (page/bbox/text_span) shown when an adapter
   emits it. (No page-count in the model → the `Np` badge is omitted, not faked.)
3. **Run diff** — `data.run_diff`: base→compare selectors, +improved/−regressed/net, filter chips,
   per-cell fail↔pass change rows (coloured via case map).
4. **Failures** — `data.failures` (extended with expected/GT): case chips with counts
   (wrong/missing/hallucinated/parse_error normalised from score detail), filterable table.
5. **Analyze** — confidence-calibration histogram (new `aggregate.calibration_buckets`), paired
   comparison (top vs others via `paired_compare`), strategy-flag rates (`extras_rate`).

## Backend additions (small, derivable, honest)
- `aggregate.calibration_buckets(scores, results, n_bins)` — per-bin accuracy + count from
  (confidence, pass) pairs. `aggregate.latency_p50` (or inline) for the p50 column.
- `data.py` new view-models: `run_menu`, per-doc accuracy/tags in `documents_in_run`, confidence +
  provenance in `drilldown`, expected/GT + normalised case in `failures`, `analyze`, `estimate`.
- `data.py` helpers: `_cfg(pc)` short config string, `_is_strategy(pc)`.

## Files
- new: `src/ezpz/ui/server.py`, `src/ezpz/ui/static/index.html`
- edit: `src/ezpz/ui/data.py`, `src/ezpz/engine/aggregate.py`, `src/ezpz/cli/main.py` (view),
  `pyproject.toml` (drop streamlit/pandas; mypy overrides), `CLAUDE.md` (ui line + commands)
- remove: `src/ezpz/ui/app.py` (Streamlit)
- tests: extend `tests/test_ui_data.py`; add `tests/test_ui_server.py` (route smoke, no socket)

## Out of scope (deferred)
Browser-triggered run execution; document image rendering / true bbox overlay for non-image docs.

## Definition of done
`pytest` + `ruff check` green; `ezpz view` serves all five views against the example DB with real
numbers, dark/light toggle, slice/diff/failure filters, and the estimate-only budget modal.
