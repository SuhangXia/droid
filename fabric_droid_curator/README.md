# Fabric-DROID Dataset Curator

A local FastAPI + React application for timestamp-synchronized review, event
correction, versioned annotation, QC, segment preview, split-safe manifest
export, and LeRobot smoke validation.

Captured episode directories are immutable inputs. The curator writes only to
the configured `curation_root`:

```text
curation/
├── curator.sqlite
├── annotations/
├── exports/
├── logs/
├── proxies/
├── signals/
└── thumbnails/
```

## Install

From `/home/suhang/projects/droid`:

```bash
uv venv /home/suhang/datasets2/.venvs/fabric-curator --python 3.11
uv pip install \
  --python /home/suhang/datasets2/.venvs/fabric-curator/bin/python \
  -r fabric_droid_curator/requirements.txt

source /home/suhang/datasets2/.venvs/fabric-curator/bin/activate
```

The standalone requirements file avoids installing robot-control dependencies.
Use the venv's Python explicitly, or activate it.

## Index and run

```bash
python -m fabric_droid_curator.index \
  --data-root /home/suhang/datasets2/frabric_pi \
  --config configs/curator.yaml

uvicorn fabric_droid_curator.backend.main:app \
  --host 0.0.0.0 \
  --port 8000

cd fabric_droid_curator/frontend
npm install
npm run dev
```

The index is incremental. Pass `--skip-proxies` to precompute DB, signals,
events, QC, and thumbnails while generating browser-compatible H.264 proxies
on first use. Pass `--force` only to recompute derived state.

## QC and export

```bash
python -m fabric_droid_curator.qc \
  --data-root /home/suhang/datasets2/frabric_pi \
  --config configs/curator.yaml

python -m fabric_droid_curator.export \
  --config configs/curator.yaml \
  --version fabric_droid_v001

python -m fabric_droid_curator.export_lerobot \
  --manifest curation/exports/fabric_droid_v001/manifest.parquet \
  --smoke-only
```

Exports are immutable. Reusing an export version fails. Heldout segments are
excluded by default, and any swatch assigned to more than one split blocks the
export.

## Review workflow

The main decision is deliberately binary: **抓取并拿下 / Remove** or
**抓取后不拿 / Leave**. Clicking either large button highlights it and writes
a new annotation version immediately. The shared timeline exposes one
unlabelled draggable cut point (`branch_point`); automatic detector events stay
in the backend for QC but do not clutter manual review. The cut produces a
common grasp-probe segment and one branch-specific segment.

A separate **要这个视频 / 不要这个视频** decision is also saved immediately.
Dropped episodes produce no active segments and cannot enter an export.
Indexer and API eligibility require finalized metadata with `success=true`
and an empty `failure_reason`; unfinished recordings are never analyzed or
shown in the review queue.

Leading idle frames are removed logically, without touching raw files. A
`joint-angle-trim-v1` detector finds the first three-frame sustained joint
departure from the initial pose and keeps 0.20 s of pre-roll. Playback,
timeline coordinates, segment generation, and export start at this
`trim_start`. The second signal chart intentionally displays only Nano17 Fz.

For the inclusive range `episode_20260728_203423` through
`episode_20260728_233726`, the incorrectly named `external` and `wrist` MP4,
timestamp, and proxy files were physically renamed once. The API therefore
reads their corrected names directly and marks the range as camera-corrected.

Accept, Reject, Recovery, Verified, swatch ID, quality fields, curve zoom,
segment preview, version history, and Undo/Redo remain available.

Keyboard shortcuts:

```text
1 Accept        2 Reject       3 Recovery
R Remove (save) L Leave (save) U Unknown
K Keep (save)   X Drop (save)
N Next          P Previous
Space Play/Pause
Left/Right Previous/next policy frame
```

See [the implementation and audit report](../docs/FABRIC_DROID_CURATOR_REPORT.md)
for the real-data schema, validation results, algorithms, and known risks.
