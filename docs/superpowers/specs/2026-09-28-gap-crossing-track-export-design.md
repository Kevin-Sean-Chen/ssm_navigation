# Gap-Crossing Track Export Design

## Goal

Add an optional Joblib export to `gap_crossing/gap_cross_db_vial.py`. The export gives a Python collaborator the continuous trajectory samples and the ordered gap-crossing attempt states for each analyzed track.

## Output control

Define `SAVE_TRACKS_PATH` near the other settings in `gap_cross_db_vial.py`. A value of `None` disables the file write. A `Path` value enables the write to that path. The script builds and validates the export in both modes so the disabled mode still checks the export logic.

## Artifact structure

The saved Joblib object is a dictionary with `metadata` and `tracks` keys. It uses only dictionaries, lists, strings, numbers, and NumPy arrays. It does not use a custom class.

`metadata["query"]` records `DATABASE_LOCATION`, `DATA_LOCATION`, `QUERY_FILTERS`, `QUERY_PERIODS`, and `MAX_EXPERIMENTS` from `gap_cross_db.py`.

`metadata["analysis_parameters"]` records the parameters that define track selection, attempt detection, geometry, and outcome classification: `FRAME_RATE_HZ`, `MIN_TRACK_S`, `PRE_SIGNAL_S`, `OUTCOME_S`, `DISTANCE_MM`, `MIN_ATTEMPTS_PER_TRACK`, `ATTEMPT_RIBBON_HALF_WIDTH_MM`, and `GAP_GEOMETRY_METHOD`.

`tracks` is a list with one dictionary for each track that remains in the event table after the repeated-attempt filter. Each track dictionary has these arrays:

- `t_s`: continuous real time in seconds.
- `x_mm`: continuous x position in millimeters.
- `y_mm`: continuous y position in millimeters.
- `signal`: continuous signal values.
- `attempt_t_s`: ordered attempt times in seconds, from the same time base as `t_s`.
- `attempt_state`: ordered labels. Each value is `cross`, `regain`, or `abort`.

The export does not add transition rows, CTMC dwell intervals, covariates, custom classes, or duplicate summary tables. A collaborator can calculate intervals with `numpy.diff(attempt_t_s)` and pair adjacent attempt states.

## Data flow

After `gap_cross_db_vial.run()` builds `tracks`, gap geometry, and `events`, one pure function combines each retained track with its matching ordered event rows. A second function validates the complete export. The script then writes the object only when `SAVE_TRACKS_PATH` is not `None`. The existing session summaries and plots continue to use the same in-memory inputs.

## Validation

Validation rejects an export when continuous arrays have different lengths, attempt arrays have different lengths, continuous or attempt times are not strictly increasing, an attempt time is outside the continuous track time range, or an attempt state is not one of the three supported labels. It also checks that all exported tracks have at least one attempt.

## Tests

Unit tests will verify the exact output structure and array values, event ordering, retained-track selection, metadata values, validation failures, no file write when saving is disabled, and a readable Joblib file when saving is enabled. Existing gap-crossing tests must continue to pass.
