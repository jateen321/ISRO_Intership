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

### F-12 (Low, fixed) — early-stopping's stale-epoch counter didn't survive `--resume`
Found immediately after the real `post_only` training run was killed
unexpectedly at epoch 2 (see the changelog entry below — no Python
traceback, consistent with the process being terminated by something
outside the training loop rather than crashing on its own) and needed to
be resumed. While preparing the resume, noticed `train()` initialized
`stale_epochs = 0` unconditionally, never restoring it from a checkpoint's
recorded value the way `best_metric` already is (`best_metric =
float(checkpoint.get("best_metric", best_metric))`). `_save_checkpoint`
didn't even write a `stale_epochs` field to the checkpoint dict to restore
from. Effect: resuming a run that was, say, 6 non-improving epochs into a
7-epoch early-stopping patience window would silently reset that count to
zero, giving the model a fresh full patience window instead of the one
epoch it actually had left — not a crash or data-loss bug like F-01, but a
real behavioral inconsistency between an uninterrupted run and a
resumed-after-interruption run that should be identical.
**Fix:** `_save_checkpoint` now takes and writes `stale_epochs`; `train()`
restores it from the checkpoint on `--resume`, the same way `best_metric`
already was. Regression: extended
`ml/tests/test_train.py::test_checkpoint_resume_continues_training` to
assert the checkpoint carries an integer `stale_epochs` field (a full
early-stopping-boundary integration test would need to force deterministic
non-improving epochs via mocking `_run_epoch`, which felt like more
machinery than this fix's severity warranted; this at least catches the
field being silently dropped again).

### F-13 (Low, fixed) — unquoted `pip install -e ml/[dev]` fails under zsh
Found while writing `README.md`'s setup instructions and cross-checking
this file's own "How to reproduce the audit" snippet against it (this
project's shell is zsh, per environment info, not bash). `ml/[dev]` is a
valid pip extras specifier, but zsh treats an unquoted `[dev]` as a glob
character-class pattern (matching a single literal `d`, `e`, or `v`
character) and fails outright with `no matches found:
.../ml/[dev]` since no such file exists — the command never reaches pip
at all. Reproduced directly (`echo .../ml/[dev]` → `no matches found`;
quoted, it resolves and installs correctly). This exact unquoted form was
already present in this file's own reproduction steps below, so it would
have failed for anyone actually running it under zsh (the project's own
documented shell) rather than bash.
**Fix:** quoted to `pip install -e "ml/[dev]"` everywhere it appears
(`README.md`, `AGENTS.md`, and this file's reproduction steps below).

### F-14 (Medium, fixed) — README's Kaggle recipe used a `<placeholder>` inside a real shell command
Found when the user actually ran the Kaggle instructions from `README.md`
(added this session) and hit a cascading failure: `geoshield.prepare
--input /kaggle/input/<dataset-slug>/train ...` failed immediately with
`dataset-slug: No such file or directory` — not a Python error at all.
Root cause: bash reads an unescaped `<name>` as input redirection ("read
from a file called `name`"), so `<dataset-slug>` was never a valid
placeholder syntax to put inside a real `!shell` command in a notebook —
it needed to be replaced textually, and even then angle brackets in a
shell command are a footgun regardless. Because `prepare` never ran,
`records.json` was never created, and every subsequent step in the recipe
(`splits`, `train` ×2, `evaluate` ×2, `export_onnx`) failed in turn with
its own `FileNotFoundError` — five cascading failures traced back to one
bad line, plus a separate, likely copy-paste-driven issue where cell
boundaries got merged (`export_onnx.py: error: unrecognized arguments: 3
— train both models` — a stray comment fragment ended up as a CLI arg)
and a `geoshield/geoshield` double-nested clone from Jupyter's `%cd`
persisting statefully across a cell re-run.
**Fix:** rewrote the Kaggle section as separate, independently-runnable
cells, each asserting its own expected output file exists before printing
`OK:` and letting the user move on — a bad `--input` (or any other step)
now fails immediately and locally instead of silently cascading through
five later steps. Replaced the bracketed placeholder with a Python
variable (`INPUT_PATH = "..."`) referenced via `"{INPUT_PATH}"` — no raw
angle brackets in a shell command anywhere in the recipe — and added a
discovery cell (`glob.glob('/kaggle/input/*/images')`) so the user finds
the real mount path instead of guessing it. The clean-clone cell now
resets to an absolute path (`os.chdir('/kaggle/working')` +
`rm -rf geoshield`) every time, so re-running it can't stack a nested
clone the way a bare `%cd geoshield` can.

### F-15 (Medium, fixed) — DataLoader used `num_workers=0`, serializing CPU decode against GPU compute
Found while explaining to the user why the real training run was taking
~12-13 min/epoch on Apple Silicon MPS — checked whether hardware was the
only factor before answering, and found `_loader()` in `train.py`
hardcoded `num_workers=0`. Every tile's image decode (2 PNGs),
ImageNet normalization, and (for training) geometric augmentation ran on
a single CPU thread, synchronously, between each GPU batch — the GPU sat
idle waiting for the next batch to be prepared rather than the next
batch being loaded in parallel while the GPU worked on the current one.
This compounds specifically on a fast GPU: moving training to a hosted
GPU (the plan's own preference, and what the user was about to do on
Kaggle) makes per-batch compute much faster, which would have made this
single-threaded loading the new bottleneck, quietly eating into the
expected speedup instead of realizing it.
**Fix:** `_loader()` now sets `num_workers = min(4, os.cpu_count() or 1)`
whenever the dataset has more than 16 samples (the `--smoke-test` fixture
and small `ml/tests/` fixtures stay at `num_workers=0`, where
multiprocessing worker startup cost would dominate rather than help), and
`pin_memory=torch.cuda.is_available()` (a no-op on MPS/CPU, a real win on
a hosted CUDA GPU). `persistent_workers` was deliberately left at its
default (`False`): `PreparedTileDataset.epoch` is mutated on the
main-process dataset object every epoch (`train_dataset.epoch = epoch`,
the F-02 fix that makes augmentation vary per epoch) and workers are
plain `Dataset` copies pickled fresh per `DataLoader.__iter__()` call —
`persistent_workers=True` would keep worker processes alive across
epochs with a stale copy of `.epoch` frozen at whatever it was when they
were first spawned, silently reintroducing F-02's exact bug for any run
with `num_workers>0`. Verified this reasoning holds and nothing broke:
full `pytest ml/tests` suite (23/23) unaffected, including
`test_augmentation_varies_across_epochs_and_is_reproducible` (that test's
determinism comes from a seed that's a pure function of index+epoch, not
process-global RNG state, so it's unaffected by worker process
boundaries either way).

### F-16 (High, fixed) — automatic DataLoader workers broke the local MPS/sandbox path
The first attempt to resume real training on the Mac used the F-15 CUDA
optimization unconditionally for datasets larger than 16 tiles. PyTorch then
spawned four workers, and every worker failed to start `torch_shm_manager`
with `Operation not permitted` in the restricted macOS environment. The
training process could not fetch its next batch and was stopped before the
next epoch completed; no model-quality result was produced.
**Fix:** automatic worker selection now enables up to four workers only when
CUDA is available (the hosted-GPU target), stays at zero on MPS/CPU, and
accepts an explicit `--num-workers` override. Added
`ml/tests/test_train.py::test_loader_stays_serial_without_cuda` to prevent a
future change from reintroducing the local failure. The user's laptop run
was stopped and must not be resumed there; use the documented CUDA notebook
or VM workflow for real training.

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

- **OQ-01 (blocking, requires hosted GPU)**: the official xBD Challenge
  training split has now been manually downloaded, verified, extracted, and
  prepared locally under the gitignored `data/` tree (2,799 accepted records,
  deterministic event-held-out manifest). The remaining work is to run both
  real 30-epoch models on a hosted CUDA GPU. A local attempt was explicitly
  stopped at the user's request and also hit the restricted macOS
  `torch_shm_manager` path when multiprocessing workers were enabled. The
  plan requires a manual, registered download and does not permit committing
  dataset content. Until the hosted run completes, Step 7's actual acceptance
  criterion (Siamese beats post-only on held-out damage macro-F1), Step 8's
  real evaluation artifacts, and a genuinely-trained ONNX export remain
  unfulfilled. The current code and committed ONNX asset prove only that the
  *pipeline* works on the synthetic eight-tile fixture; they make no claim
  about real detection quality.
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
- **OQ-05 (blocking, abandoned this session — see below)**: attempted
  `gcloud colab runtime-templates create` / `gcloud colab runtimes create`
  (Colab Enterprise, the Vertex AI-managed alternative to a raw Compute
  Engine VM — see README's Google Cloud section) as the hosted environment
  for OQ-01, in the `geoshield-training` project (billing enabled,
  `aiplatform.googleapis.com`/`compute.googleapis.com` already on). Three
  separate quota/capacity walls were hit, in this order:
  1. **GPU**: per-region/per-type quotas (e.g. `NVIDIA_V100_GPUS`) report a
     limit of `1.0` in every region checked (`us-central1`, `us-east1`,
     `us-west1`, `europe-west4`, `asia-east1`, `southamerica-east1`), but
     that figure is misleading — it's gated by a separate global quota,
     `GPUS_ALL_REGIONS`, hard-capped at `0.0` for this project (confirmed
     via `gcloud compute project-info describe --format="json(quotas)"`).
     No GPU runtime could be created anywhere until that's raised via
     Console → IAM & Admin → Quotas → "GPUs (all regions)". The user
     confirmed this quota increase is **not allowed** for this account, so
     the GPU path is closed, not just pending.
  2. **Disk**: switching to a CPU-only template, a `200GB`/`140GB`
     `PD_SSD` disk collided with `SSD_TOTAL_GB` (limit `250` in
     `us-central1`) — a pre-existing, auto-created "Default" Colab
     Enterprise runtime (`e2-standard-4`, `pd-standard`, silently
     provisioned the first time `gcloud colab` touched this project) was
     already consuming headroom. Fixed by switching the template's disk
     type to `PD_STANDARD` (`DISKS_TOTAL_GB` limit `2048`, effectively
     unconstrained) — this is what actually unblocked runtime creation.
  3. **CPU**: `gcloud colab executions create` (the ephemeral,
     fire-and-forget notebook-execution API — the CLI-only equivalent of
     Kaggle's "Save & Run All") failed 4/4 times across two regions and two
     machine sizes (`e2-standard-8`, `e2-standard-4`), alternating between
     `CPUS_ALL_REGIONS` exceeded (limit `12` globally) and genuine
     zone-capacity errors. Root cause: **`gcloud colab runtimes create`
     (persistent) and `gcloud colab executions create` (ephemeral) draw
     from the same tiny 12-vCPU global cap but provision *separately* — an
     execution never reuses an already-running runtime's compute.** With a
     persistent `e2-standard-8` runtime already up, there was no quota left
     for an execution's own VM; deleting the runtime to free quota for an
     execution never once succeeded anyway (capacity or quota errors every
     time), so that trade was a net loss — a runtime that worked was
     deleted to make room for a job that never ran.
  **What's actually confirmed**: `gcloud colab runtimes create` (the
  persistent-runtime path) reliably succeeds — twice, with `e2-standard-8`
  and `e2-standard-4`, both `PD_STANDARD` disk, both went `RUNNING`/
  `HEALTHY`. The template that's known-good: `us-central1`,
  `notebookRuntimeTemplates/8872968676198842368` (`e2-standard-8`, 200GB
  `PD_STANDARD`, `idle-shutdown-timeout=24h`). **What's unconfirmed**:
  whether that runtime's Jupyter proxy (`proxyUri`, e.g.
  `https://<id>-dot-us-central1.aiplatform-notebook.googleusercontent.com`)
  accepts authenticated REST/kernel-gateway calls from outside a browser
  (`gcloud auth print-identity-token` + `POST /api/kernels`) — that's the
  discriminating test for whether this path can be driven CLI-only at all,
  versus needing the browser-based JupyterLab terminal, versus falling
  back to the README's raw Compute Engine VM + `tmux` path (real SSH,
  no managed-capacity lottery, and CPU-only GCE instances don't touch the
  `GPUS_ALL_REGIONS` quota that blocked step 1 above).
  **Also unconfirmed**: whether CPU training is even worth pursuing at any
  of this — no CPU-vs-MPS speed measurement was taken before this session's
  work stopped. The MPS estimate (~12 min/epoch, see the "Training on
  Colab / Kaggle" README section) suggests CPU could plausibly run into
  multi-day territory for a full 30-epoch × 2-model run; that should be
  measured locally (force `device = torch.device("cpu")`, time a handful
  of real batches from `data/prepared`) before spending more effort on
  hosted-CPU infrastructure.
  **Housekeeping**: 8 runtime templates were created across this session
  (3 GPU — dead, quota denied; 5 CPU, of varying disk-type/size/machine
  combinations working through the walls above). Only
  `8872968676198842368` (`us-central1`) is worth keeping; the rest are
  free-standing config (no cost while unused) but clutter — safe to
  `gcloud colab runtime-templates delete` the other 7 next time this is
  picked up. `gs://geoshield-training-prepared-data/prepared.tar` (7.9GB,
  the tarred `data/prepared/`) is uploaded and ready to pull onto whatever
  compute ends up running the real job.

## Planned verification (for what's still open)

- On a hosted CUDA runtime, inspect the existing `data/prepared/audit.json`
  and `visual_audit.png`, then run the documented supervisor for both real
  models, evaluate the untouched test split, and export the winning checkpoint
  to replace the synthetic placeholder model.
- OQ-04 above: explicit WASM-vs-WebGPU mask diff.
- OQ-05 above: (1) measure real CPU-vs-MPS per-batch speed locally before
  investing more effort in hosted-CPU infra; (2) test whether
  `notebookRuntimes/<id>`'s `proxyUri` accepts CLI-driven kernel-gateway
  calls, and if not, fall back to the README's raw Compute Engine VM +
  `tmux` path; (3) either way, `gcloud colab runtimes create
  --runtime-template=8872968676198842368` is the known-good starting
  point, and `gs://geoshield-training-prepared-data/prepared.tar` is
  ready to pull onto whatever compute ends up running the job.

## How to reproduce the audit

```bash
# ML pipeline (project-local venv, never the system Python)
python3 -m venv .venv
.venv/bin/pip install -e "ml/[dev]"
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
