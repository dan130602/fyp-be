# Exporting the prepared NLP dataset

Run from `fyp-be/evaluation-dataset` in CMD or PowerShell:

```bat
python -B scripts/export_dataset.py --dataset-root "datasets/legacy_pilot_v1" --destination "../../fyp-ai/evaluation/datasets/pilot"
```

The script creates all destination folders. It does not create Git branches,
commits, pushes or PRs. Review and commit its output in `fyp-ai` normally.
The evaluator itself is not implemented by this change.

The snapshot contains only:

```text
pilot/
  manifest.json
  checklist.json
  calls/
    CALL_001/
      ground_truth.json
      transcript.json
      speaker_roles.json
    ...every discovered call directory...
```

No audio, authoring inputs, Python code, credentials or predictions are copied.
JSON is serialized deterministically without altering string values. Source
ground truth is copied as data, never regenerated or edited. The exporter checks
schemas, checklist coverage/types, call identity, exact transcript text, segment
IDs, speakers and full-transcript consistency using existing validators. These
checks do not prove semantic correctness of gold answers or regulatory reasoning.

The selected dataset's `dataset_manifest.json`, source JSON, normalization
config and schemas must match committed `HEAD` data. The source manifest's
ordered `call_ids` must exactly match the discovered call directories.
Commit prepared-data corrections before exporting. Uncommitted exporter/docs
changes do not block export; commit the exporter too before sharing the tooling.
Git must trust the source checkout under your user account. No dependencies are
added beyond the project's existing Python requirements and Git.

## Releases compiled by G8, and uncommitted exports

`--dataset-root` may be a G8 release such as `datasets/advisory_execution_pilot_v1`. The destination folder name is whatever you pass to
`--destination`; use the release name (for example `../../fyp-ai/evaluation/datasets/advisory_execution_pilot_v1`) and keep the legacy
`pilot/` folder untouched. For such a release the exporter uses the release's own `normalization.json` (never
`configs/normalization.json`), re-runs the G8 compilation in memory and refuses any `ground_truth.json`, `transcript.json` or
`speaker_roles.json` that differs from a fresh compilation of the current accepted inputs (so a stale or hand-edited snapshot cannot be
exported), and writes `dataset_status` (G7 verification and timing state) into `manifest.json`.

`--allow-uncommitted` skips the "source must equal committed HEAD" check, for a provisional export. The manifest then carries
`source.uncommitted: true`, because `source.git_commit` no longer identifies the exported bytes. Prefer committing first for anything
others will rely on.

## Manifest

`format` versions the snapshot contract. `source.git_commit`,
`source.dataset_path`, `source.dataset_id` and `source.checklist_id` identify
the release inside `fyp-be`.
`call_ids` lists the calls. `timing_status` explicitly says `placeholder_zero`.
`files` maps each data file to its SHA-256 hash of canonical JSON; formatting,
object-key ordering and Windows Git line-ending conversion do not change hashes.
`dataset_sha256` identifies the complete data-file inventory and hashes.
There is no creation timestamp or machine-specific path, so unchanged inputs
produce identical output. Hashes detect changes; they are not signatures or
proof of data quality. Record the source commit and dataset hash in future runs.

This initial exporter rejects non-placeholder timestamps. Real timing export
needs explicit timing provenance support later, not inferred timing status.

## Verify or update

Read-only verification, without needing the original source checkout state:

```bat
python -B scripts/export_dataset.py --destination "../../fyp-ai/evaluation/datasets/pilot" --check
```

After committing source updates, deliberately replace a verified snapshot:

```bat
python -B scripts/export_dataset.py --dataset-root "datasets/legacy_pilot_v1" --destination "../../fyp-ai/evaluation/datasets/pilot" --overwrite
```

Existing destinations require a supported manifest with matching files/hashes.
Unrelated files/directories, modified data and symlinks/junctions are refused.
An empty hand-created destination is also refused: let the exporter create it.
The complete snapshot is staged and verified before directory replacement.
On an install error the old snapshot is restored; if restoration also fails,
the backup is retained and its path is printed. This is a local single-writer
operation: do not run concurrent exports or edit the destination during export.
Do not hand-edit generated snapshots; correct the source and export again.

`--check` checks snapshot integrity, not whether it matches the latest source.
Teammates obtain the prepared files by cloning/pulling `fyp-ai`; they do not
need this Python package. Consumer-side manifest checking can be added with
the future evaluator; no dataset-generation package dependency is required.

## Consumer boundary

Export only prepares files. It does not make HTTP requests, import LangGraph,
or execute the answering workflow. The future evaluator in `fyp-ai` owns
loading these files and calling the actual graph directly. Ground truth is
for scoring; the manifest is for dataset identity. Neither is model input.

`datasets/advisory_execution_pilot_v1` is currently a release scaffold, not an
exportable snapshot: it has no approved calls, and its new extraction fields
still require explicit normalization rules before checklist validation can pass.
