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

## Planned verification (for what's still open)

- Once xBD is downloaded: re-run `python -m geoshield.prepare`, inspect the
  visual audit grid and `audit.json` for missing pairs / invalid polygons /
  class balance, then re-run splits, training (both models, real 30-epoch
  config), evaluation, and a real ONNX export — replacing the placeholder.
- Step 10 (browser inference) needs manual verification in an actual browser
  (WebGPU vs WASM equivalence, memory release on cancel) — this can't be
  meaningfully unit-tested under vitest/jsdom, which has no WebGPU/WASM
  execution.

## How to reproduce the audit

```bash
# ML pipeline (project-local venv, never the system Python)
python3 -m venv .venv
.venv/bin/pip install -e ml/[dev]
PYTHONPATH=ml .venv/bin/python -m pytest ml/tests -q   # 19 passed

# Frontend
npm run lint
npx tsc --noEmit
npx vitest run                                          # includes lib/geoshield (15 passed)
npm run build
```

## Tooling

- `.venv/` at repo root, gitignored, holds all Python deps (torch,
  torchvision, onnx, onnxruntime, onnxscript, pillow, shapely, pytest).
- ML entry points: `python -m geoshield.{prepare,train,evaluate,export_onnx}`
  (run with `PYTHONPATH=ml`, or from inside `ml/` directly).
- No Claude Code hook is wired to this file in this pass — not set up
  automatically here since it wasn't asked for this time; a SessionStart
  hook injecting this file's contents (as was done in an earlier, unrelated
  project) would be a reasonable follow-up if this file is meant to keep
  being updated across sessions.

## Changelog

- **2026-08-31**: created this log, covering Steps 1–9 (F-01 through F-06)
  plus the in-progress Step 11/13 pure-function work (`lib/geoshield/postprocess.ts`,
  `lib/geoshield/reports.ts`, 15 vitest tests passing). Steps 6, 7, 9 (with the
  quantization fix) committed as `5b87ba1`, `706d548`, `3b75f42`, `322c68d`.
  Steps 10 and 12 (Web Worker inference wiring, full UI integration) not
  started yet at time of writing.
