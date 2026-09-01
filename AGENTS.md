# AGENTS.md — GeoShield AI

Operating notes for any agent (Claude Code or otherwise) working in this
repo. For the full narrative of what's been built, what broke, and why —
including every fix referenced below — read [`findings.md`](findings.md)
first. This file is the short version: what to do, what not to do, where
things are.

## What this is

A browser-only "before/after satellite imagery → building damage"
assessment tool. Sites/Next.js frontend (`app/`, `lib/geoshield/`) +
PyTorch → ONNX ML pipeline (`ml/geoshield/`). Inference runs entirely
client-side via ONNX Runtime Web (WebGPU, falling back to WASM) — no image
is ever uploaded to a server. This is a research prototype, not a
validated detector; the UI carries explicit "not a working detector" /
"placeholder model" disclosures until a real xBD-trained export replaces
the current one.

## Where things live

- `app/page.tsx` — the entire UI (single-page app).
- `lib/geoshield/` — pure TS logic: tensor prep, postprocessing, the Web
  Worker (`worker.ts`), the main-thread client (`client.ts`). Each has a
  matching `.test.ts`.
- `ml/geoshield/` — the ML package: `xbd.py` (dataset discovery +
  rasterization), `splits.py` (event-held-out splitting), `dataset.py` /
  `train.py` (training loop, two model variants), `export_onnx.py`
  (ONNX export + quantization), `evaluate.py`.
- `public/models/geoshield-siamese.{onnx,json}` — the shipped browser
  model + its metadata (check `trained_on_real_data` in the JSON before
  claiming anything about model quality).
- `data/` — gitignored. Dataset downloads and prepared tiles go here
  (`data/xbd/raw/...`, `data/prepared/...`). **Never** commit anything
  under `data/` — it's real xBD imagery/derived tiles, and redistribution
  terms aren't confirmed (see `findings.md` OQ-02).

## Environment

- Python deps live only in the repo-root `.venv/` (gitignored), never the
  system Python. Set up: `python3 -m venv .venv && .venv/bin/pip install -e "ml/[dev]"`.
  (Quote it — zsh treats unquoted `ml/[dev]` as a glob character class and
  fails with "no matches found".)
- Run any `ml/geoshield` entry point with `PYTHONPATH=ml`:
  `PYTHONPATH=ml .venv/bin/python -m geoshield.{prepare,splits,train,evaluate,export_onnx}`.
- Node deps are normal (`npm install`); `engines.node >= 22.13.0`.

## Gotchas that will cost you time if you don't know them

- **Don't use `npm run dev` to test `lib/geoshield/worker.ts`.** vinext's
  `import.meta.url` rewriter breaks Vite's module-worker pattern in dev
  mode only (F-07). Use `npm run build && npm start` (wrangler dev against
  the real build, `http://localhost:8787`) for anything touching the
  Worker or `onnxruntime-web`.
- **The ONNX model must stay under 25 MiB** — Cloudflare Workers Static
  Assets' per-file limit. This is why `export_onnx.py` int8-quantizes
  after verifying fp32 parity (F-06), and why `onnxruntime-web`'s own
  WebGPU wasm binary is loaded from a pinned CDN URL instead of bundled
  (F-08, see `vite.config.ts` + `worker.ts`'s `ort.env.wasm.wasmPaths`).
  Don't "fix" the CDN fetch by trying to bundle it locally — it'll blow
  the asset limit again.
- **Training real data takes hours, smoke tests take seconds.**
  `--smoke-test` trains an 8-tile synthetic fixture (fast, used as a CI
  gate — proves the training loop works, says nothing about real
  detection quality). Real training needs `--data` pointing at a
  `geoshield.prepare` output directory and `--split-manifest`.
  `{model}_history.json` is written after every epoch (not just at the
  end, F-10) — tail it to check progress on a long-running job instead of
  waiting for the process to exit.
- **`SiameseUNet` must run its shared encoder once on a concatenated
  before/after batch**, not as two separate forward passes — two separate
  calls corrupts BatchNorm running stats at small batch sizes (F-04). If
  you touch `ml/geoshield/models.py`, don't reintroduce that pattern.
- Every fix above has a regression test in `ml/tests/`. Run the full
  suite after touching `ml/geoshield/` or `lib/geoshield/`:
  ```bash
  PYTHONPATH=ml .venv/bin/python -m pytest ml/tests -q
  npx vitest run
  npx tsc --noEmit
  npm run lint
  ```

## When you find a new bug

Log it in `findings.md` (pattern: root cause, measured effect, fix,
regression test — follow the existing F-01..F-15 entries) before or
alongside fixing it. This file has been the project's actual audit trail
across sessions; keep it that way rather than starting a separate one.
