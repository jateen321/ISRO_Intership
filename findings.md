# GeoShield AI — Engineering Findings Log

Living record of what was built, what broke, and what's still open. This
project is a research-prototype "before/after satellite imagery → building
damage" tool, built to an explicit 16-step gated plan (Sites/Next.js frontend
+ PyTorch → ONNX ML pipeline, browser-only inference, no server upload).

Repo history note: this working directory previously held an unrelated
project ("SENTINEL-OPS" / "IncidentIQ", commits `5697303`..`8853b8a`). GeoShield
started fresh at `ab15daa: chore: scaffold geoshield application` and everything
below is scoped to GeoShield only.

## Handoff

Steps 1–5 (Sites scaffold, product shell, ML package scaffold, xBD
preprocessing, deterministic event-held-out splits — commits `ab15daa` through
`ee6101e`) were built by a different agent ("Luna", GPT-5.6) in an earlier,
now-unavailable session. This log picks up from Step 6 onward, where that
session had left `train.py`, `xbd.py`, `dataset.py`, `losses.py` staged but
uncommitted, with the full ML test suite passing (10/10) but two real bugs
still latent and undetected (found and fixed below).

## Findings

### F-01 (High, fixed) — `history.json` was destroyed on checkpoint resume
`train.py` initialized `history: list = []` fresh on every call to `train()`
and overwrote `{model}_history.json` unconditionally at the end. Resuming
from a checkpoint at epoch 20 and running 10 more epochs left `history.json`
containing only epochs 20–29 — the first 20 epochs of loss/metric curve were
silently gone. The plan requires saving "training curves"; a preempted
Colab/Kaggle run (the plan's intended real-training environment) resuming
after a disconnect would have lost most of its history on every resume.
**Fix:** `train()` now loads the existing `history.json` when `--resume` is
set and appends to it, so the curve is continuous across resumes.
Caught by: advisor review before the Step 6 commit, confirmed by
`ml/tests/test_train.py::test_checkpoint_resume_continues_training`
(asserts epochs `[0..7]` after a resume mid-run, not just the post-resume
tail).

### F-02 (High, fixed) — augmentation was a fixed one-time transform, not per-epoch
`PreparedTileDataset.__getitem__` seeded its augmentation RNG as
`random.Random(self.seed + index)` — a function of the sample index only.
Every epoch, tile 5 got the *identical* flip/rotation, for all 30 epochs.
This wasn't caught by the existing test suite because the smoke-test path
uses `_SyntheticDataset`, which bypasses `PreparedTileDataset` entirely.
**Fix:** added a settable `dataset.epoch` counter threaded from the training
loop (`train_dataset.epoch = epoch` each epoch) and mixed into the seed:
`self.seed + index * 100_003 + self.epoch`. Verified with an asymmetric
gradient-image fixture — a solid-color image with a centered mask (the first
fixture tried) is geometrically invariant under every flip/rotation, so it
couldn't actually detect this bug; an off-center mask on a gradient image
can. Regression test:
`ml/tests/test_dataset.py::test_augmentation_varies_across_epochs_and_is_reproducible`.

### F-03 (High, fixed) — smoke-test overfit gate wasn't actually overfitting
The eight-tile synthetic smoke test capped epochs at `min(args.epochs, 12)`.
At 12 epochs, validation `damage_macro_f1` reached only ~0.44 — still
climbing, not converged — which is weak evidence the training loop is
correct (Step 6's actual gate: "a small overfit test learns eight tiles").
**Fix:** raised the cap to 60 epochs; val macro-F1 reaches 0.94
(post-only) / 0.98 (siamese, after F-04 below). Runtime is ~20–30s per
smoke run on Apple Silicon MPS, acceptable for a gating test.

### F-04 (Critical, fixed) — Siamese model's shared encoder broke BatchNorm at eval time
The most serious bug found. `SiameseUNet.forward` called the shared
ResNet-18 encoder as two **separate** forward passes — once on the
"before" image batch, once on "after". With the plan's small batch size
(2 in the smoke test, 4 in the real config), each call's BatchNorm layers
computed statistics from a tiny batch **from a different image
distribution** (pre- vs post-disaster) and folded both into the same
running-mean/var buffers via momentum updates. The buffers never
stabilized.

Measured effect: after 60 smoke epochs, train-mode (batch statistics)
`damage_macro_f1` was 0.976 — the weights had genuinely memorized the
fixture — but eval-mode (running statistics) `damage_macro_f1` was
**0.10**. The model was correct; only inference-time normalization was
broken. This is a classic small-batch-BatchNorm failure mode, and it's
exactly the kind of bug that a metrics-only check (train loss going down)
would never surface — it only shows up by comparing train-mode vs
eval-mode inference on the same data, which is what exposed it.

**Fix:** concatenate `before_image` and `after_image` along the batch
dimension and run the shared encoder **once** per training step, then
split the returned features back into before/after halves before fusion.
This is standard practice for Siamese networks specifically to avoid this
failure mode, and is mathematically neutral for eval-mode inference
(running stats don't depend on the batch that produced them). After the
fix: eval-mode val macro-F1 = 0.98, matching train-mode.
Regression test: `ml/tests/test_train.py::test_siamese_smoke_test_trains_through_fused_branch`.

### F-05 (Medium, fixed) — `torch.onnx.export`'s new default exporter needs `onnxscript`
Torch 2.13's `torch.onnx.export` defaults to the dynamo-based exporter
(`dynamo=True`), which imports `onnxscript` at call time. It isn't installed
by default and isn't a dependency the ML package scaffold declared.
**Fix:** added `onnxscript` to `ml/pyproject.toml` dependencies.

### F-06 (Critical, fixed) — fp32 ONNX export exceeded Cloudflare's per-asset limit
The first ONNX export (Step 9) produced a 53MB fp32 file — the real size of
a ResNet-18 U-Net in float32, nothing export-specific. This was committed to
`public/models/` as the "shipped browser model." Cloudflare Workers Static
Assets caps individual files at **25 MiB**
(developers.cloudflare.com/workers/platform/limits/#static-assets, verified
directly, not assumed) — the committed file would have failed at Step 16
deployment, and the failure wouldn't have surfaced until after four more
steps of browser-inference UI work were built on top of it. Caught by
advisor review before starting Step 10.

Also found in the same pass: `torch.onnx.export`'s dynamo exporter defaults
to `external_data=True` for larger models, splitting weights into a second
`.onnx.data` sidecar file (53MB main graph → tiny `.onnx` + 53MB `.data`).
Disabled (`external_data=False`) since the browser Web Worker fetches the
model by URL and has no simple way to resolve a second relative-path fetch
for the sidecar.

**Fix:** `export_onnx.py` now runs the fp32 export and its 99.9%-pixel-parity
check exactly as before (this remains the real correctness gate — it proves
the ONNX graph translation matches PyTorch), then int8-dynamic-quantizes the
verified fp32 graph before writing anything to `--output`. Quantization
targets `Conv/MatMul/Gemm` explicitly — `onnxruntime.quantization
.quantize_dynamic`'s default op set targets MatMul/LSTM (built for
RNN/Transformer workloads), which does essentially nothing for a
Conv-dominated CNN. Result: 53.6MB → 12.9MB (well under 25 MiB), with
99.86% argmax agreement between the fp32 and quantized graphs (measured,
not assumed — this is a sanity floor at 50%, not a hard gate, since
quantization is expected to shift some predictions). Both the fp32-parity
and quantization-agreement numbers are recorded in the exported metadata
JSON so neither is hidden.

### F-07 (High, fixed) — vinext's import.meta.url rewriter breaks Vite's module-worker pattern
The standard, textbook Vite pattern `new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })`
failed at runtime with `SecurityError: Failed to construct 'Worker': Script at
'file:///l…' cannot be accessed from origin 'http://localhost:3000'`.
Root cause, found by reading the actual transformed source the dev server
served (`curl http://localhost:3000/lib/geoshield/client.ts`): Vite's own
worker-URL transform rewrites the second `new URL(...)` argument into `'' +
import.meta.url` (a `BinaryExpression`) as part of producing its output.
vinext's `import.meta.url` rewriter
(`node_modules/vinext/dist/plugins/import-meta-url.js`,
`isNewUrlExpression`/`collectImportMetaUrlRanges`) only recognizes a
*direct* `import.meta.url` as the second argument to `new URL(...)` as the
pattern to leave alone; wrapped inside a `BinaryExpression` it no longer
matches, so vinext's rewriter descends into it anyway and replaces the
nested `import.meta.url` with a literal `file:///ROOT/...` string — which
then fails in the browser, since that's not a valid script origin. This is
a real incompatibility between two pieces of this project's own tooling
(Vite's worker plugin + vinext's import-meta-url plugin), not something
fixable by editing `node_modules`. **Fix:** switched to Vite's `?worker`
import suffix (`import GeoShieldWorker from './worker?worker'`, then `new
GeoShieldWorker()`) — a different code path resolved through the module
graph rather than a runtime `import.meta.url` string, so it isn't affected.
Needed a small ambient module declaration
(`lib/geoshield/worker-module.d.ts`) since this project's `tsconfig.json`
doesn't include `vite/client` types, and an oxlint disable comment on that
one import line since oxlint's import resolver doesn't read wildcard
ambient `declare module '*?worker'` types the way `tsc` does.

**This bug is dev-server-only** — verified by checking the actual
production build output (`dist/client/_next/static/worker-*.js`) contained
zero references to `window`/HMR client code even before this fix, and
`npm run build` always succeeded. Only `npm run dev`'s live HMR path was
broken. If testing this project's worker code locally, prefer `npm run
build && npm start` (`wrangler dev` against the real build) over `npm run
dev` for anything touching `lib/geoshield/worker.ts`.

### F-08 (Critical, fixed) — onnxruntime-web's own WebGPU wasm binary exceeds Cloudflare's asset limit
Separate from F-06 (our model file): once the Worker construction bug
above was fixed, `npm start` (`wrangler dev` against the real production
build) failed outright with a Cloudflare-verified error, not a guess:
`Asset too large. Cloudflare Workers supports assets with sizes of up to 25
MiB. We found a file
.../ort-wasm-simd-threaded.jsep-D-icqfN-.wasm with a size of 26.5 MiB.`
This is ONNX Runtime Web's own WebGPU-capable wasm runtime binary
(`onnxruntime-web`'s default import discovers it via its own internal
`new URL(..., import.meta.url)`, which Vite's asset pipeline then
auto-bundles as a local static asset) — nothing to do with our model or
our code, and not something achievable to shrink by quantization the way
F-06 was, since it's third-party runtime code we don't control.

Checked the available onnxruntime-web wasm variants
(`node_modules/onnxruntime-web/dist/*.wasm`): the plain (non-jsep, no
WebGPU) build is 13.3MB — well under the limit — but dropping WebGPU
support entirely would violate the plan's explicit "prefer WebGPU; fall
back to WASM" requirement for Step 10. **Fix:** `onnxruntime-web` ships a
package.json `"onnxruntime-web-use-extern-wasm"` import condition
specifically for this situation — it resolves to a build
(`ort.min.mjs`) that expects the wasm runtime to be loaded from an
external URL (via `ort.env.wasm.wasmPaths`) instead of being bundled
locally. Added `resolve: { conditions: ['onnxruntime-web-use-extern-wasm'] }`
to `vite.config.ts` and set `ort.env.wasm.wasmPaths` to a version-pinned
jsDelivr CDN URL in `worker.ts`. This only fetches generic runtime engine
code (no user data, no model weights) from the CDN — the privacy
guarantee (no image upload) is unaffected, and after this fix `wrangler
dev`'s asset-size check passed with the model file (13MB) as the largest
shipped asset. Verified end-to-end afterward: full assessment completed
via WebGPU (confirmed by the UI's own runtime badge, not assumed), cancel
mid-run recovered cleanly, and two more assessments ran without a page
reload. Not yet explicitly cross-verified: WASM-path output vs WebGPU-path
output numerical equivalence (Step 10's gate asks for this) — both paths
share the identical tiling/postprocessing code and differ only in which
ONNX Runtime execution provider is requested, but this wasn't forced and
directly diffed in a real browser due to no straightforward way to disable
`navigator.gpu` for a spawned worker without adding test-only code.

### F-09 (Low, fixed) — `class_buildings` audit counter was silently always zero
Found while auditing the first real-data `geoshield.prepare` run against the
downloaded xBD Challenge training set (2,799 pairs, all accepted, no missing
pairs or invalid records — the pipeline itself checked out). The emitted
`audit.json`'s `class_buildings` field (a per-building damage-class count,
distinct from `class_pixels`, a per-pixel count) came back `{"1": 0, "2": 0,
"3": 0, "4": 0}` even though `class_pixels` showed a real, plausible xBD class
distribution (126M undamaged / 15.7M minor / 19.8M major / 9.3M destroyed
pixels across ~2,800 tiles) — i.e. real damage polygons were being parsed and
rasterized correctly, but the separate building-count tally wasn't moving at
all, on real data or synthetic.
Root cause (`ml/geoshield/xbd.py`, `prepare_dataset`): `class_buildings` was
initialized with **string** keys (`{str(label): 0 for label in range(1, 5)}`)
but the per-polygon loop checked membership with the **int** label
`read_label_features` returns (`if label in class_buildings`). `1 in
{"1": 0, ...}` is always `False` in Python (dict membership is exact-match on
keys, and `int` never equals `str`), so the increment line never ran,
regardless of what the input data contained. `class_pixels` was unaffected
because it's computed independently by byte-scanning the rendered mask
image, not from this loop.
**Fix:** compare with `str(label) in class_buildings` (keeping string keys,
consistent with `class_pixels`' convention) instead of switching to int keys.
Regression test: `ml/tests/test_xbd.py::test_prepare_writes_one_tile_for_512_input`
now asserts `class_buildings == {"1": 1, "2": 0, "3": 0, "4": 1}` against a
fixture with one no-damage and one destroyed polygon — a dict comparing
against all-zero would have caught this before it ever reached real data.
This field is audit-only metadata (not read by `splits.py` or `train.py`),
so it did not affect any model, split, or export correctness — only the
printed/saved audit summary was wrong.

### F-10 (Medium, fixed) — `history.json` was only written after the *entire* training run finished
Found while about to launch the first real, multi-hour, multi-epoch training
run on the downloaded xBD data (previously only smoke-tested with a fast
eight-tile synthetic fixture, where this never mattered). `train()` appended
each epoch's record to an in-memory `history` list inside the epoch loop, but
only called `history_path.write_text(...)` once, **after** the loop
completed. Per-epoch checkpoints (`{model}_last.pt`, `{model}_best.pt`) *are*
written inside the loop, so a crash or interruption mid-run wouldn't break
resumability — but it would silently lose the entire loss/metric curve
collected up to that point (worse than F-01, which was specifically about
losing history across a `--resume`; this is about never persisting it at
all until a run completes cleanly), and it left no way to observe progress
on a run in progress by reading the file.
**Fix:** moved the `history_path.write_text(...)` call inside the epoch
loop, after each epoch's record is appended, so the file is current after
every epoch. `ml/tests/test_train.py`'s existing history assertions check
final-state content only, so this needed no test changes — a redundant
final write was simply removed rather than added.

### F-11 (Critical, fixed) — `geoshield.evaluate` was never implemented
Found while assembling the commands to run Step 8's real evaluation now
that real xBD data and a split manifest exist. `ml/geoshield/evaluate.py`'s
`main()` printed a "scaffold ready" placeholder when `--checkpoint`/`--data`
were absent, but unconditionally raised `NotImplementedError("Evaluation
loop is implemented after the training gate.")` the moment both were
supplied — i.e. the entry point could never actually evaluate anything,
independent of whether real data was available. This is a bigger gap than
OQ-01 (blocked on the dataset download): even with the dataset in hand,
there was no way to run Step 7's acceptance check ("Siamese beats
post-only baseline on held-out damage macro-F1") or Step 8's evaluation
artifacts, because the code path itself didn't exist yet. No test file
(`ml/tests/test_evaluate.py`) previously existed either, so nothing in the
suite could have caught this.

**Fix:** implemented `evaluate()` by reusing `train.py`'s own
already-tested `_run_epoch`/`_loader`/`select_device` (the same
confusion-matrix-based metrics computation the training loop uses every
epoch, run once over a held-out split instead) rather than duplicating
that logic. Model architecture and image size are read from the
checkpoint's own recorded `config` (written by `train.py`), not re-passed
via CLI flags, so an evaluation run can't silently mismatch what a
checkpoint was actually trained as. Output is the same
`summarize_metrics()` shape `train.py` already produces
(`damage_macro_f1`, per-class F1/IoU, confusion matrix, localization F1)
plus run metadata (checkpoint path/epoch, split, record/tile counts),
written to `--output` as JSON. Added
`ml/tests/test_evaluate.py` (4 new tests: post-only metrics shape, siamese
dual-input path, missing-checkpoint error, empty-split error) — full
suite now 23/23 passing (19 before F-09/F-10/F-11 this session, +4 new
evaluate tests; F-09's fix only added an assertion to an existing test).

### F-12 (Medium, fixed) — synthetic overfit gate failed on the CPU backend
The deterministic post-only smoke test did not reliably meet its own
`damage_macro_f1 > 0.85` acceptance threshold on CPU. With the smoke-only
learning rate of `1e-3`, the 60-epoch run reached 0.8254 on the Linux CPU
backend even though its loss remained finite and decreased normally. The
same fixture had originally been tuned on Apple MPS, where it exceeded the
gate, making the regression test backend-dependent. **Fix:** raised only the
automatic post-only smoke-test learning rate to `2e-3`; the identical CPU run
now reaches 0.8623. The Siamese fixture remains at `1e-3` (0.9858 on the same
CPU), since it is less stable at the higher rate. Explicit `--learning-rate`
values and the real-data default remain unchanged. Caught by running the
complete ML suite in a clean Linux environment:
`ml/tests/test_train.py::test_smoke_test_overfits_eight_tiles`.

### F-13 (High, fixed) — smoke checkpoints could silently replace the shipped model
The training documentation distinguished local PyTorch checkpoints from the shipped
ONNX artifact, but the export command accepted either a real xBD checkpoint or a
synthetic smoke-test checkpoint without an explicit acknowledgement. That made it too
easy to overwrite the Git-tracked browser weights with a meaningless pipeline fixture,
and it obscured that `public/models/geoshield-siamese.onnx` is the canonical weight
artifact users receive from the repository. **Fix:** checkpoints now record explicit
`training_data` provenance (`xbd` or `synthetic_smoke`), export refuses smoke
checkpoints unless `--allow-placeholder` is supplied, and the publishing instructions
now include staging and committing both the ONNX file and checksum metadata. Existing
older checkpoints remain readable through the prior `data` field. Regression test:
`ml/tests/test_export_onnx.py::test_export_rejects_smoke_checkpoint_without_explicit_override`.

## Environment

- macOS, Apple Silicon (arm64), MPS acceleration available and used for
  training/smoke tests (`torch.device("mps")` — auto-selected in
  `select_device()`).
- Python 3.14.7, project-local `.venv` (not the system/main environment,
  per explicit instruction).
- `torch==2.13.0`, `torchvision==0.28.0`, `onnx==1.22.0`,
  `onnxruntime==1.29.0`, `onnxscript==0.7.1`.
- Node per `package.json` `engines.node: ">=22.13.0"`; frontend stack is
  Vinext/Sites (not plain Next.js) + Cloudflare Workers via `wrangler`, React
  19, Tailwind 4, shadcn components, `onnxruntime-web` + `zod` already
  present from the Step 1 scaffold.

## Decisions

- **D-01**: ML dependencies live only in the project-local `ml/.venv` (later
  consolidated to the repo-root `.venv`), never the system Python.
- **D-02**: `rasterio`, `albumentations`, and `scikit-learn` are declared in
  `ml/pyproject.toml` (per the plan's suggested stack) but are **not
  actually used** by any code — the Luna-authored implementation chose a
  lighter dependency-free approach (PIL + Shapely for rasterization, plain
  torch tensor ops for augmentation) instead. Left as-is; not scope for this
  pass. Worth pruning from `pyproject.toml` at some point so the declared
  dependencies match reality.
- **D-03**: the ONNX model committed to `public/models/` is an untrained
  placeholder (from the `--smoke-test` synthetic eight-tile fixture, not
  real xBD data) — necessary to unblock Steps 10–13 (browser inference,
  post-processing, UI, reports), all of which need *some* validated model
  artifact to build and test against. Its metadata explicitly records
  `"trained_on_real_data": false` and `checkpoint_config.data == "."`.
  **This must not be presented to a user as a working detector** — Step 12's
  UI work must carry the same explicit "not a working detector" disclosure
  used elsewhere in this kind of build, until it's replaced by a real
  xBD-trained export.

## Open questions

- **OQ-01 (blocking, not resolvable by me)**: the real xBD training dataset
  (~7.8GB, official challenge training split) is not present locally. The
  plan explicitly requires a **manual, registered download** — Step 4:
  "Require the user to manually register and download the dataset. Do not
  automate sign-in or commit dataset content." This blocks: real training
  runs, Step 7's actual acceptance criterion (Siamese beats post-only
  baseline on held-out damage macro-F1 — currently unfulfillable, and not
  reported as fulfilled), Step 8's real evaluation artifacts, and a
  genuinely-trained ONNX export. Everything built so far proves the
  *pipeline* is correct (training loop, loss, checkpointing, export, parity)
  using the synthetic eight-tile fixture — none of it is a claim about real
  detection quality.
- **OQ-02**: Step 14 requires "legally redistributable, attributed pre/post
  examples" for the sample selector. I can't independently verify xBD
  redistribution terms with confidence. Per the plan's own fallback ("If
  redistribution is not permitted, do not commit xBD images. Replace the
  sample selector with a guided 'download official sample' instruction and
  retain upload functionality."), that's the path to take, not committing
  imagery on my own judgment.
- **OQ-03**: `public/models/geoshield-siamese.onnx` (12.9MB, untrained) is
  the first binary asset committed into this repo. Once a real trained
  model lands, the old placeholder commit stays in git history unless
  someone explicitly asks for history to be rewritten — not doing that
  without being asked.
- **OQ-04**: WASM-path vs WebGPU-path output equivalence (F-08) wasn't
  directly diffed — see F-08 for why. Worth forcing both paths and
  comparing masks before calling Step 10 fully closed.

## Planned verification (for what's still open)

- Once xBD is downloaded: re-run `python -m geoshield.prepare`, inspect the
  visual audit grid and `audit.json` for missing pairs / invalid polygons /
  class balance, then re-run splits, training (both models, real 30-epoch
  config), evaluation, and a real ONNX export — replacing the placeholder.
- OQ-04 above: explicit WASM-vs-WebGPU mask diff.

## How to reproduce the audit

```bash
# ML pipeline (project-local venv, never the system Python)
python3 -m venv .venv
.venv/bin/pip install -e ml/[dev]
PYTHONPATH=ml .venv/bin/python -m pytest ml/tests -q   # 19 passed

# Frontend
npm run lint
npx tsc --noEmit
npx vitest run                                          # 35 passed, incl. lib/geoshield
npm run build

# Manually verifying anything in lib/geoshield/worker.ts (Web Worker +
# onnxruntime-web): use the real production build, not `npm run dev` — see
# F-07. `npm run dev`'s HMR client breaks module workers in this project.
npm run build && npm start   # wrangler dev on the real build, http://localhost:8787
```

## Tooling

- `.venv/` at repo root, gitignored, holds all Python deps (torch,
  torchvision, onnx, onnxruntime, onnxscript, pillow, shapely, pytest).
- ML entry points: `python -m geoshield.{prepare,train,evaluate,export_onnx}`
  (run with `PYTHONPATH=ml`, or from inside `ml/` directly).
- `.claude/launch.json` runs `npm run dev` for the Browser-pane preview —
  fine for everything except the Worker/onnxruntime-web path (F-07); use
  `npm start` against a real build for that instead.
- No Claude Code hook is wired to this file in this pass — not set up
  automatically here since it wasn't asked for this time; a SessionStart
  hook injecting this file's contents (as was done in an earlier, unrelated
  project) would be a reasonable follow-up if this file is meant to keep
  being updated across sessions.

## Changelog

- **2026-08-31**: created this log, covering Steps 1–9 (F-01 through F-06)
  plus the Step 11/13 pure-function work. Steps 6, 7, 9 (with the
  quantization fix) committed as `5b87ba1`, `706d548`, `3b75f42`, `322c68d`.
- **2026-08-31 (later)**: Steps 10 and 12 (Web Worker inference, full UI
  integration) built and manually verified end-to-end against a real
  `wrangler dev` build — full assessment completing via WebGPU, cancel
  recovering cleanly, repeat runs without reload, no image bytes leaving
  the browser. Found and fixed two more real bugs in the process (F-07,
  F-08), both tooling/deployment issues rather than application logic.
  Committed as `a44b270`, `fd30411`, `1bd07db`.
  35 vitest tests passing, typecheck/lint/build all clean.
- **2026-08-31 (evening)**: user downloaded and verified the official xBD
  Challenge training set (~7.8GB, SHA1-verified against xview2.org's
  published hash) and extracted it into the repo working directory; moved
  into the already-gitignored `data/xbd/raw/train` so it can't be
  accidentally committed (see `.gitignore`'s `/data/` rule). Ran
  `geoshield.prepare` against real data for the first time — this
  surfaced F-09 (`class_buildings` audit counter silently always zero) — and,
  in preparing to launch the first real multi-hour training run, also found
  and fixed F-10 (`history.json` only written at end of run, not
  per-epoch). Both fixes verified: full `pytest ml/tests` suite (19/19,
  now 20/20 after F-09's added assertion) still passes. `geoshield.splits`
  run against the real 2,799-record manifest — 10 distinct events, heavily
  size-skewed (`socal-fire`: 823 pairs down to `guatemala-volcano`: 18), so
  the resulting train/val/test split is far from an even 70/15/15 by record
  count (1,170 / 630 / 999) even though it's balanced by event — expected
  behavior of event-held-out splitting with this few, this uneven a set of
  events, not a bug.
- **2026-09-01**: clean Linux/CPU verification exposed and fixed F-12, a
  backend-dependent failure in the synthetic overfit gate. The real xBD data
  is not present in this checkout, so real-data training was not attempted;
  both synthetic model variants were trained as pipeline checks instead.
