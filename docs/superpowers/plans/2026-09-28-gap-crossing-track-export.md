# Gap-Crossing Track Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional validated Joblib export with continuous samples and ordered attempt states for each retained gap-crossing track.

**Architecture:** Keep the export settings and functions in `gap_cross_db_vial.py`. Build one plain dictionary from the existing `tracks` list and `events` table, validate it whether saving is enabled or disabled, and write it only when `SAVE_TRACKS_PATH` is a `Path`.

**Tech Stack:** Python, NumPy, pandas, Joblib, pytest

**Spec:** `docs/superpowers/specs/2026-09-28-gap-crossing-track-export-design.md`

## Global Constraints

- Keep the existing database query, event detection, session summaries, and plots unchanged.
- Put all export behavior in `gap_crossing/gap_cross_db_vial.py`.
- Use only dictionaries, lists, strings, numbers, and NumPy arrays in the artifact.
- Do not add transition tables, CTMC intervals, custom classes, or per-track metadata.
- Use ASD-STE100 Simplified Technical English for new prose.

## Review Focus

- A continuous array length mismatch must raise `ValueError`; test it in Task 1.
- Non-increasing continuous or attempt times must raise `ValueError`; test both in Task 1.
- An unsupported attempt state must raise `ValueError`; test it in Task 1.
- An attempt time outside its continuous track range must raise `ValueError`; test it in Task 1.
- A track with no retained attempts must not enter the export; test retained-track selection in Task 1.

---

### Task 1: Build, validate, and optionally save the collaborator export

**Files:**
- Modify: `gap_crossing/gap_cross_db_vial.py`
- Modify: `gap_crossing/tests/test_gap_cross_db_vial.py`

**Interfaces:**
- Consumes: the existing track dictionaries from `gap_cross_db.make_tracks()` and the filtered event table from `gap_cross_track.make_event_table()` after recording metadata is merged.
- Produces: `make_track_export(tracks: list[dict], events: pd.DataFrame) -> dict`, `validate_track_export(export: dict) -> None`, and `save_track_export(export: dict, output_path: Path | None) -> Path | None`.

- [ ] **Step 1: Add failing structure and selection tests**

Add `TrackExportTests.test_export_contains_query_parameters_and_retained_track_arrays`. Use two small track dictionaries and events for only one track. Assert exact top-level keys, query and analysis parameter values, the six exact track keys, continuous array values, ordered attempt times, ordered states, and one exported track.

- [ ] **Step 2: Run the structure test and confirm the missing interface failure**

Run: `python -m pytest gap_crossing/tests/test_gap_cross_db_vial.py::TrackExportTests::test_export_contains_query_parameters_and_retained_track_arrays -v`

Expected: FAIL because `make_track_export` does not exist.

- [ ] **Step 3: Implement export construction**

In `gap_crossing/gap_cross_db_vial.py`, import `Path`, `joblib`, and `deepcopy`; define `SAVE_TRACKS_PATH: Path | None = None`; and implement `make_track_export(tracks: list[dict], events: pd.DataFrame) -> dict`. Sort each track's matching events by `attempt_time_s`, include only track IDs present in `events`, copy the continuous arrays, split `xy` into `x_mm` and `y_mm`, and create `attempt_t_s` and `attempt_state` arrays. Copy the query settings from `pooled` and the specified analysis settings from `analysis` into `metadata`.

- [ ] **Step 4: Run the structure test and confirm it passes**

Run: `python -m pytest gap_crossing/tests/test_gap_cross_db_vial.py::TrackExportTests::test_export_contains_query_parameters_and_retained_track_arrays -v`

Expected: PASS.

- [ ] **Step 5: Add failing validation tests**

Add tests that pass an otherwise valid export to `validate_track_export`, then independently change it to have mismatched continuous lengths, non-increasing `t_s`, non-increasing `attempt_t_s`, an unsupported state, an out-of-range attempt time, and an empty attempt sequence. Assert `ValueError` for each case.

- [ ] **Step 6: Run the validation tests and confirm the missing interface failure**

Run: `python -m pytest gap_crossing/tests/test_gap_cross_db_vial.py::TrackExportTests -v`

Expected: FAIL because `validate_track_export` does not exist.

- [ ] **Step 7: Implement export validation**

Implement `validate_track_export(export: dict) -> None`. Require the exact six track fields, equal continuous array lengths, equal non-empty attempt array lengths, strictly increasing `t_s` and `attempt_t_s`, attempt times within the continuous time range, and states contained in `analysis.OUTCOME_ORDER`.

- [ ] **Step 8: Run the export tests and confirm they pass**

Run: `python -m pytest gap_crossing/tests/test_gap_cross_db_vial.py::TrackExportTests -v`

Expected: PASS.

- [ ] **Step 9: Add failing disabled and enabled save tests**

Add one test that calls `save_track_export(export, None)`, asserts validation occurred, and asserts that no path was returned. Add one test that saves to `tmp_path / "gap_crossing_tracks.joblib"`, reloads it with `joblib.load`, and compares its metadata and arrays with the input export.

- [ ] **Step 10: Run the save tests and confirm the missing interface failure**

Run: `python -m pytest gap_crossing/tests/test_gap_cross_db_vial.py::TrackExportTests -v`

Expected: FAIL because `save_track_export` does not exist.

- [ ] **Step 11: Implement optional saving and connect it to `run()`**

Implement `save_track_export(export: dict, output_path: Path | None) -> Path | None` so it always calls `validate_track_export`, returns `None` without writing when the path is `None`, creates the configured parent directory, writes with `joblib.dump`, and returns the path. In `run()`, build the export after the event metadata merge and call `save_track_export(export, SAVE_TRACKS_PATH)` before session summaries and plots.

- [ ] **Step 12: Run focused and full gap-crossing tests**

Run: `python -m pytest gap_crossing/tests/test_gap_cross_db_vial.py -v`

Expected: PASS.

Run: `python -m pytest gap_crossing/tests -v`

Expected: PASS.

- [ ] **Step 13: Review and verify the change**

Run: `git diff --check`

Review the complete diff and confirm that no current user changes outside `gap_cross_db_vial.py` and `test_gap_cross_db_vial.py` changed during implementation.

- [ ] **Step 14: Commit the implementation**

```bash
git add gap_crossing/gap_cross_db_vial.py gap_crossing/tests/test_gap_cross_db_vial.py
git commit -m "feat: export gap crossing tracks"
```
