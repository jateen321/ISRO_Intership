# GeoShield AI

Satellite before/after building-damage assessment that runs entirely in
your browser. Upload a pre-disaster and post-disaster image of the same
location; a paired-image segmentation model highlights building damage by
severity — no image is ever uploaded to a server.

This is a research prototype built as an internship project, not a
validated detector. See [Model status & limitations](#model-status--limitations)
before drawing any conclusions from its output, and [`findings.md`](findings.md)
for the full engineering history (every bug found, how it was caught, and
how it was fixed).

## Contents

- [What this is](#what-this-is)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Local development](#local-development)
- [Dataset preparation](#dataset-preparation)
- [Training](#training)
  - [Locally](#training-locally)
  - [On Colab / Kaggle](#training-on-colab--kaggle)
  - [On Google Cloud](#training-on-google-cloud)
- [Evaluation protocol](#evaluation-protocol)
- [ONNX export](#onnx-export)
- [Deployment](#deployment)
- [Privacy](#privacy)
- [Model status & limitations](#model-status--limitations)
- [Testing](#testing)
- [Technical writeup](#technical-writeup)

## What this is

Two models are trained on the [xBD dataset](https://xview2.org/dataset)
(the xView2 challenge's building-damage-assessment benchmark) to turn a
pre/post-disaster image pair into a per-pixel, 5-class damage map:

| Class | Meaning |
|---|---|
| 0 | Background (no building) |
| 1 | Undamaged |
| 2 | Minor damage |
| 3 | Major damage |
| 4 | Destroyed |

- **Post-only baseline** (`PostOnlyUNet`) — a ResNet-18 U-Net that only
  sees the post-disaster image. Establishes a floor: how much damage
  classification is possible from a single image alone.
- **Siamese model** (`SiameseUNet`, the one shipped to the browser) — a
  shared-encoder U-Net that reads the pre- and post-disaster images
  together, fusing multi-scale features (concatenation + absolute
  difference) before decoding. The premise is that a *change* signal
  between two aligned images should beat single-image damage
  classification — this is the hypothesis the evaluation protocol below
  is designed to check, not an assumed result.

Everything downstream of "have a trained ONNX model" — tiling, inference,
post-processing, the overlay UI, JSON/CSV export — runs as a static
site with a Web Worker, entirely on the client device.

## Architecture

Two independent pipelines meet at one artifact:
`public/models/geoshield-siamese.onnx`.

```
Offline ML pipeline (Python, ml/geoshield/) — run once, ahead of time
──────────────────────────────────────────────────────────────────────
  xBD dataset (manual, registered download — see Dataset preparation)
        │
        ▼
  geoshield.prepare      tiles xBD pairs to 512px, rasterizes JSON polygon
        │                labels to masks, writes records.json + audit.json
        ▼
  geoshield.splits       deterministic event-held-out 70/15/15 split —
        │                a disaster event never appears in more than one split
        ▼
  geoshield.train        trains post_only and siamese (ResNet-18 U-Net,
        │                combined CE + soft-dice loss, cosine LR, early
        │                stopping) — checkpoints + per-epoch history.json
        ▼
  geoshield.evaluate     held-out damage-macro-F1, per-class F1/IoU,
        │                confusion matrix — the "siamese beats baseline" check
        ▼
  geoshield.export_onnx  fp32 ONNX export → PyTorch/ONNX parity check
                         (must agree on ≥99.9% of pixels) → int8 dynamic
                         quantization → public/models/geoshield-siamese.{onnx,json}


Browser inference pipeline (TypeScript, lib/geoshield/) — every request
──────────────────────────────────────────────────────────────────────
  app/page.tsx (React UI)
        │  before.png, after.png (File objects — never sent over the network)
        ▼
  InferenceClient (client.ts)  ── owns one persistent Web Worker ──▶  worker.ts
                                                                         │
                                            decode + letterbox to 1024×1024,
                                            ImageNet-normalize, split into
                                            four 512×512 tiles
                                                                         │
                                                                         ▼
                                  ONNX Runtime Web session — WebGPU first,
                                  falls back to WASM — running the model
                                  fetched from public/models/
                                                                         │
                                          per-tile softmax → argmax,
                                          stitched back to a 1024×1024 mask
                                                                         │
                                                                         ▼
                                  postprocessMask(): 8-connected component
                                  labeling → estimated building regions +
                                  per-class pixel statistics
                                                                         │
        ◀── {mask, result} ───────────────────────────────────────────┘
        ▼
  overlay canvas + before/after swipe compare + stats grid + region table
  + JSON/CSV export (reports.ts)
```

Every arrow inside the browser pipeline crosses a Web Worker boundary via
`postMessage`, not a network request — the "your imagery stays on this
device" claim in the UI is a structural property of the code (no `fetch`
call anywhere in the pipeline takes image bytes as a body), not just a
sentence in the privacy banner.

## Repository layout

```
app/page.tsx              the entire UI (single-page app)
lib/geoshield/
  tensor.ts                image decode/normalize/tile/stitch (pure functions)
  postprocess.ts           mask → regions + class statistics
  overlay.ts               mask → RGBA overlay for canvas rendering
  reports.ts               JSON/CSV export builders
  types.ts                 AssessmentResult schema (zod-validated)
  worker.ts                the Web Worker: owns the ONNX Runtime session
  client.ts                main-thread handle to the worker
ml/geoshield/
  xbd.py                   xBD discovery + polygon rasterization + tiling
  splits.py                event-held-out split manifest
  dataset.py               PreparedTileDataset, class weighting, augmentation
  models.py                PostOnlyUNet, SiameseUNet (ResNet-18 U-Net)
  losses.py                weighted cross-entropy + soft dice
  metrics.py               confusion-matrix-derived F1/IoU
  train.py                 training loop (real data + --smoke-test gate)
  evaluate.py              held-out evaluation entry point
  export_onnx.py           ONNX export, parity check, int8 quantization
public/models/             the shipped browser model (.onnx + metadata .json)
findings.md                engineering log: every bug found, how, and the fix
AGENTS.md                  operating notes for agents working in this repo
```

## Local development

```bash
npm install
npm run dev
```

Opens the Sites/Next.js dev server on `http://localhost:3000`.

> **One caveat:** the dev server's HMR breaks `lib/geoshield/worker.ts`
> (a real incompatibility between two build-tool plugins — see
> `findings.md` F-07). To exercise the Web Worker / ONNX Runtime path
> locally, use a real production build instead:
> ```bash
> npm run build && npm start   # wrangler dev, http://localhost:8787
> ```

## Dataset preparation

The xBD dataset requires a manual, registered download — this project
does not automate sign-in or bundle any dataset imagery (redistribution
terms aren't confirmed; see `findings.md` OQ-02).

1. Register and download the **Challenge training set (~7.8GB)** at
   [xview2.org/dataset](https://xview2.org/dataset). (The full GeoTIFF
   bundle on that page is a different format this code doesn't parse —
   get the Challenge-format tarball specifically.)
2. Verify and extract it:
   ```bash
   shasum -a 1 train_images_labels_targets.tar   # compare to the published SHA1
   tar -xvzf train_images_labels_targets.tar
   ```
3. Prepare tiles + rasterized masks + an audit report:
   ```bash
   PYTHONPATH=ml .venv/bin/python -m geoshield.prepare \
     --input  data/xbd/raw/train \
     --output data/prepared \
     --tile-size 512
   ```
   This writes `data/prepared/{records.json, audit.json, visual_audit.png}`
   plus `data/prepared/tiles/{before,after,mask}/`. Check `audit.json` for
   `missing_pairs`/`invalid_records` (should be empty) and `visual_audit.png`
   (a thumbnail sheet with damage polygons overlaid) before trusting the
   output — this is the step that would surface a wrong dataset layout or
   a broken label parser.
4. Build the split manifest:
   ```bash
   PYTHONPATH=ml .venv/bin/python -m geoshield.splits \
     --records data/prepared/records.json \
     --output  data/prepared/split_manifest.json
   ```
   Splitting is done by **disaster event**, not by image, so no event
   leaks across train/val/test — a model can't get credit for having
   already seen a slightly different tile from the same disaster. Because
   xBD's training tier has only 10 distinct events of very uneven size
   (823 pairs for `socal-fire` down to 18 for `guatemala-volcano`), the
   resulting split is balanced *by event*, not by an even 70/15/15 record
   count — that's expected, not a bug.

`data/` is gitignored end to end — nothing from this step is ever committed.

## Training

Both `PostOnlyUNet` and `SiameseUNet` share one CLI
(`ml/geoshield/train.py`): ResNet-18 (ImageNet-pretrained) encoder, U-Net
decoder, combined 0.5×cross-entropy + 0.5×soft-dice loss, AdamW +
cosine LR schedule, early stopping (patience 7) on held-out
`damage_macro_f1`. Checkpoints and a per-epoch `history.json` are written
after every epoch, so a killed or crashed run loses at most the epoch in
progress, never the curve so far.

### Training locally

```bash
PYTHONPATH=ml .venv/bin/python -m geoshield.train \
  --model post_only \
  --data data/prepared \
  --split-manifest data/prepared/split_manifest.json \
  --output ml/checkpoints

PYTHONPATH=ml .venv/bin/python -m geoshield.train \
  --model siamese \
  --data data/prepared \
  --split-manifest data/prepared/split_manifest.json \
  --output ml/checkpoints
```

30 epochs over the full training tier is a multi-hour run per model on a
laptop GPU (Apple Silicon MPS auto-selected via `select_device()`, falling
back to CUDA or CPU). Resume an interrupted run with `--resume
ml/checkpoints/<model>_last.pt`.

Before touching real data at all, there's a fast correctness gate that
doesn't need the dataset:

```bash
PYTHONPATH=ml .venv/bin/python -m geoshield.train --model siamese --smoke-test --output /tmp/smoke
```

This overfits an 8-tile synthetic fixture in under a minute and asserts
`damage_macro_f1 > 0.85` — it proves the training loop, loss, and (for
the Siamese model) the shared-encoder fusion path are wired correctly.
It says nothing about real detection quality; see `findings.md` F-03/F-04
for two real bugs this exact gate caught before real data was ever
involved.

### Training on Colab / Kaggle

The plan this project follows explicitly prefers a hosted GPU for the
real training run over a laptop — a ResNet-18 U-Net at 512px is slow on
Apple Silicon MPS (~12 min/epoch observed; 30 epochs × 2 models is a
multi-hour laptop commitment) and considerably faster on a free Kaggle/Colab
GPU. There's no bundled notebook file, but the CLI is designed to drop
into one directly. **Kaggle is the more convenient of the two** — dataset
hosting and GPU attachment are built into the notebook UI, no Drive
mounting needed:

1. Search [Kaggle Datasets](https://www.kaggle.com/datasets) for an
   existing "xView2"/"xBD" upload first — it saves re-uploading the 7.8GB
   Challenge tarball. Otherwise upload your own copy as a new (private)
   Kaggle Dataset.
2. New Notebook → right panel → **Accelerator: GPU T4 x2** (or P100) →
   **Internet: On** → **Add Input**: attach the dataset from step 1.
3. Run these as **separate cells, one at a time** — each one checks its
   own output before you move to the next, so a failure stops immediately
   instead of silently cascading into every step after it (a real
   `<dataset-slug>`-style placeholder left in a shell command, or a
   Jupyter `%cd` re-run stacking a `geoshield/geoshield` nested clone,
   are exactly the kind of mistake this is designed to catch early):
   ```python
   # Cell 0 — clean clone (safe to re-run: always resets to one clone, not a nested one)
   import os
   os.chdir('/kaggle/working')
   !rm -rf geoshield
   !git clone https://github.com/jateen321/ISRO_Intership.git geoshield
   os.chdir('/kaggle/working/geoshield')
   print("cwd:", os.getcwd())
   !pip install -e "ml/[dev]" -q
   ```
   ```python
   # Cell 1 — find where the dataset actually mounted; don't guess the folder name
   import glob
   candidates = glob.glob('/kaggle/input/*/images') + glob.glob('/kaggle/input/*/*/images')
   print("Found images/ under:", candidates)
   # Your --input for Cell 2 is the PARENT of whichever images/ path prints here.
   ```
   ```python
   # Cell 2 — prepare (replace INPUT_PATH with what Cell 1 printed — no angle brackets;
   # bash reads a literal "<name>" as input redirection from a file called "name")
   INPUT_PATH = "/kaggle/input/PASTE-THE-REAL-PATH-HERE/train"
   !PYTHONPATH=ml python -m geoshield.prepare --input "{INPUT_PATH}" --output /kaggle/working/prepared --tile-size 512
   assert os.path.exists('/kaggle/working/prepared/records.json'), "prepare failed — scroll up for the real error"
   print("OK: records.json exists")
   ```
   ```python
   # Cell 3 — splits
   !PYTHONPATH=ml python -m geoshield.splits --records /kaggle/working/prepared/records.json --output /kaggle/working/prepared/split_manifest.json
   assert os.path.exists('/kaggle/working/prepared/split_manifest.json')
   print("OK: split_manifest.json exists")
   ```
   ```python
   # Cell 4 — train post_only
   !PYTHONPATH=ml python -m geoshield.train --model post_only --data /kaggle/working/prepared --split-manifest /kaggle/working/prepared/split_manifest.json --output /kaggle/working/checkpoints
   assert os.path.exists('/kaggle/working/checkpoints/post_only_best.pt')
   print("OK: post_only_best.pt exists")
   ```
   ```python
   # Cell 5 — train siamese
   !PYTHONPATH=ml python -m geoshield.train --model siamese --data /kaggle/working/prepared --split-manifest /kaggle/working/prepared/split_manifest.json --output /kaggle/working/checkpoints
   assert os.path.exists('/kaggle/working/checkpoints/siamese_best.pt')
   print("OK: siamese_best.pt exists")
   ```
   ```python
   # Cell 6 — evaluate both
   !PYTHONPATH=ml python -m geoshield.evaluate --checkpoint /kaggle/working/checkpoints/siamese_best.pt   --data /kaggle/working/prepared --split test --output /kaggle/working/siamese_eval.json
   !PYTHONPATH=ml python -m geoshield.evaluate --checkpoint /kaggle/working/checkpoints/post_only_best.pt --data /kaggle/working/prepared --split test --output /kaggle/working/post_only_eval.json
   ```
   ```python
   # Cell 7 — export
   !PYTHONPATH=ml python -m geoshield.export_onnx --model siamese --checkpoint /kaggle/working/checkpoints/siamese_best.pt --output /kaggle/working/geoshield-siamese.onnx
   assert os.path.exists('/kaggle/working/geoshield-siamese.onnx')
   print("OK: exported")
   ```
4. Once Cells 0–3 (the cheap, CPU-only steps) have each printed their
   `OK:` line, switch to **Save Version → Save & Run All (Commit)** for
   the rest — this executes the whole notebook in the background on
   Kaggle's servers even after the tab is closed, up to Kaggle's session
   limit. Running the cheap steps interactively first catches a bad
   `INPUT_PATH` in seconds instead of after a multi-hour background run.
5. Once it finishes, download everything under `/kaggle/working/` from
   the notebook's **Output** tab: the two `*_eval.json` files, the
   checkpoints, and `geoshield-siamese.onnx` + its metadata. Bring those
   back into this repo (`ml/checkpoints/`, `artifacts/metrics/`,
   `public/models/`).

Colab works the same way with one difference — no built-in dataset
hosting, so mount Drive instead and point `--input` at the extracted
tarball there:
```python
from google.colab import drive
drive.mount('/content/drive')
```

`select_device()` picks CUDA automatically when available, so nothing in
`train.py` needs to change for either hosted runtime. Because a preempted
session is exactly the "disconnect mid-run" scenario `history.json`'s
per-epoch write and `--resume`'s checkpoint restore (including the
early-stopping patience counter — see `findings.md` F-01/F-12) are built
to survive, re-running the train cell with `--resume
<output>/<model>_last.pt` after a disconnect (or a deliberate pause, e.g.
`ml/train_supervisor.sh` locally does exactly this automatically) picks
up where it left off rather than restarting.

### Training on Google Cloud

An alternative to a notebook environment: a persistent Compute Engine VM,
not bound by a notebook session's time limit. Unlike Colab/Kaggle, this
requires a **paid (billable) Google Cloud account** — a fresh account's
Free Trial explicitly blocks attaching a GPU to any VM and blocks
requesting GPU quota at all (verified against Google's own docs, not
assumed) until you upgrade; the $300/90-day trial credit still applies
against GPU usage afterward, but the upgrade step (adding a real payment
method) has to happen first.

1. **Upgrade off Free Trial**: [console.cloud.google.com](https://console.cloud.google.com)
   → the "Upgrade" banner → add a payment method.
2. **Enable Compute Engine + request GPU quota** (the only step with
   unpredictable timing — start it first):
   ```bash
   gcloud services enable compute.googleapis.com
   ```
   Console → **IAM & Admin → Quotas** → filter "NVIDIA T4 GPUs" for a
   region (e.g. `us-central1`) → **Edit Quotas** → request **1** → submit.
   Often approved within minutes on a billing-enabled account, but budget
   up to ~1-2 business days on a brand-new one.
3. **Install the CLI** (while waiting on quota):
   ```bash
   brew install --cask google-cloud-sdk
   gcloud init
   ```
4. **Launch the VM** once quota is approved — a Deep Learning VM image
   ships CUDA/drivers pre-configured, no manual driver install:
   ```bash
   gcloud compute instances create geoshield-train \
     --zone=us-central1-a \
     --machine-type=n1-standard-8 \
     --accelerator="type=nvidia-tesla-t4,count=1" \
     --image-family=pytorch-latest-gpu \
     --image-project=deeplearning-platform-release \
     --maintenance-policy=TERMINATE \
     --boot-disk-size=200GB \
     --metadata="install-nvidia-driver=True"
   ```
5. **SSH in and set up**:
   ```bash
   gcloud compute ssh geoshield-train --zone=us-central1-a
   ```
   ```bash
   git clone https://github.com/jateen321/ISRO_Intership.git geoshield
   cd geoshield
   python3 -m venv .venv && source .venv/bin/activate
   pip install -e "ml/[dev]"
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"   # must print True
   ```
6. **Get the dataset onto the VM** — from your local machine, in a
   separate terminal:
   ```bash
   gcloud compute scp data/xbd/train_images_labels_targets.tar geoshield-train:~/xbd.tar --zone=us-central1-a
   ```
   Back on the VM:
   ```bash
   mkdir -p ~/geoshield/data/xbd/raw
   tar -xf ~/xbd.tar -C ~/geoshield/data/xbd/raw
   ls ~/geoshield/data/xbd/raw/train    # confirm images/ and labels/ are there
   ```
7. **Run it inside `tmux`**, so it survives SSH disconnects — `Ctrl+B`
   then `D` to detach, `tmux attach -t geoshield` to reattach later:
   ```bash
   tmux new -s geoshield
   cd ~/geoshield && source .venv/bin/activate
   PYTHONPATH=ml python -m geoshield.prepare --input data/xbd/raw/train --output data/prepared --tile-size 512
   ls data/prepared/records.json    # confirm before continuing
   PYTHONPATH=ml python -m geoshield.splits --records data/prepared/records.json --output data/prepared/split_manifest.json
   ./ml/train_supervisor.sh
   ```
8. **Evaluate + export** once training finishes (same commands as
   [Evaluation protocol](#evaluation-protocol) / [ONNX export](#onnx-export) below).
9. **Pull results back**, from your local machine:
   ```bash
   gcloud compute scp --recurse geoshield-train:~/geoshield/ml/checkpoints ./ml/ --zone=us-central1-a
   gcloud compute scp --recurse geoshield-train:~/geoshield/artifacts/metrics ./artifacts/ --zone=us-central1-a
   gcloud compute scp geoshield-train:~/geoshield/public/models/geoshield-siamese.onnx ./public/models/ --zone=us-central1-a
   gcloud compute scp geoshield-train:~/geoshield/public/models/geoshield-siamese.json ./public/models/ --zone=us-central1-a
   ```
10. **Delete the VM — don't skip this.** A running GPU VM bills
    continuously whether or not it's actively training:
    ```bash
    gcloud compute instances delete geoshield-train --zone=us-central1-a
    ```

## Evaluation protocol

```bash
PYTHONPATH=ml .venv/bin/python -m geoshield.evaluate \
  --checkpoint ml/checkpoints/siamese_best.pt \
  --data data/prepared \
  --split test \
  --output artifacts/metrics/siamese_eval.json

PYTHONPATH=ml .venv/bin/python -m geoshield.evaluate \
  --checkpoint ml/checkpoints/post_only_best.pt \
  --data data/prepared \
  --split test \
  --output artifacts/metrics/post_only_eval.json
```

Model architecture and input size are read from the checkpoint's own
recorded config, not re-specified on the command line, so an evaluation
run can't accidentally mismatch what a checkpoint was trained as.
`evaluate.py` reuses the exact same confusion-matrix-based metric
computation `train.py` runs every epoch (`damage_macro_f1`, per-class
F1/IoU, localization F1, the full confusion matrix), just run once over
the **test** split instead of a validation mini-batch — the two never
diverge because there's only one implementation.

The project's Step 7 acceptance criterion is: **the Siamese model's
held-out `damage_macro_f1` beats the post-only baseline's.** Compare the
two `artifacts/metrics/*.json` files' `damage_macro_f1` fields to check
it — this is not assumed anywhere in the codebase, it's a number you
compute from the two eval runs above.

## ONNX export

```bash
PYTHONPATH=ml .venv/bin/python -m geoshield.export_onnx \
  --model siamese \
  --checkpoint ml/checkpoints/siamese_best.pt \
  --output public/models/geoshield-siamese.onnx
```

What this does, in order, aborting (and deleting any partial output) if
any step fails:

1. Exports the checkpoint to fp32 ONNX (opset 18, weights embedded in one
   file — no external-data sidecar, since the browser Worker fetches this
   by URL with no simple way to resolve a second relative fetch).
2. Runs the exported graph through ONNX Runtime and diffs its argmax
   predictions against the original PyTorch model on random inputs — this
   must agree on **≥99.9% of pixels** or the export is rejected. This is
   the real PyTorch↔ONNX correctness gate.
3. int8-dynamic-quantizes the *verified* fp32 graph (targeting
   `Conv`/`MatMul`/`Gemm` — the ops a CNN actually uses; the
   quantization library's default target list is tuned for
   RNN/Transformer workloads and would do almost nothing here). This is
   required, not optional: the unquantized fp32 model is ~53MB, over
   Cloudflare Workers Static Assets' 25 MiB per-file limit — the
   quantized model is ~13MB.
4. Sanity-checks fp32-vs-quantized agreement (informational floor at 50%
   agreement — quantization is expected to shift some predictions, this
   just catches a badly broken quantization).
5. Writes `<output>.json` alongside the model with the sha256, both
   parity numbers, the source checkpoint's config, and a
   `trained_on_real_data` flag — `false` whenever the checkpoint came from
   `--smoke-test` rather than real xBD tiles. **Nothing in the UI or the
   Web Worker should ever claim real-model performance without checking
   this field first.**

## Deployment

This project deploys as a Cloudflare Workers static site (via `wrangler`,
wired through `@cloudflare/vite-plugin` in `vite.config.ts`):

```bash
npm run build
npm start   # wrangler dev against the real build — verify locally first
```

Two things had to be worked around to make this deployment target work at
all with an ONNX model in the browser (both documented in detail in
`findings.md` F-06/F-08, both already handled by the code as it stands —
noted here so they're not "fixed" a second time):

- The model file must stay under Cloudflare's 25 MiB per-asset limit —
  handled by the quantization step above.
- `onnxruntime-web`'s own WebGPU-capable wasm runtime binary is 26.5MB —
  over the same limit, and not something quantization can shrink since
  it's third-party runtime code. `vite.config.ts` resolves
  `onnxruntime-web` to its "extern wasm" build instead of bundling the
  runtime locally, and `worker.ts` points `ort.env.wasm.wasmPaths` at a
  version-pinned CDN URL. This fetches generic runtime engine code only —
  no user imagery, no model weights — so the "nothing uploaded" privacy
  guarantee is unaffected.

Actually publishing this publicly (`wrangler deploy` or equivalent) is a
step this project deliberately treats as requiring explicit human
go-ahead, not something to run as part of a build — ask before doing it.

## Privacy

Inference runs entirely client-side via ONNX Runtime Web. No code path in
`lib/geoshield/` sends image bytes anywhere: `worker.ts` only ever
`fetch()`es the model file and its CDN-hosted wasm runtime (both static,
public, non-user assets), never a `File` the user uploaded. The exported
JSON/CSV reports (`reports.ts`) intentionally never include the original
image data. If you're auditing this claim rather than taking it on faith,
`lib/geoshield/worker.ts` is short enough to read end to end.

## Model status & limitations

**The model currently shipped in `public/models/` may not yet reflect a
completed real-data training run — check
`public/models/geoshield-siamese.json`'s `trained_on_real_data` field
before relying on anything below.** When it's `false`, the model was
fit only to an 8-tile synthetic fixture to validate the pipeline
end-to-end, and its predictions carry no information about real damage.

Even once trained on real xBD data, keep in mind:

- **Not validated for operational or emergency use.** This is an
  internship research prototype evaluated on a held-out split of one
  public benchmark, not a certified detector.
- **Region counts are estimates.** `postprocessMask`'s connected-component
  regions are a segmentation-derived proxy for "a building," not a
  verified building count or footprint.
- **Aligned inputs are assumed.** The model has no registration step; a
  before/after pair that isn't spatially aligned will produce a
  meaningless change signal.
- **10 training events, heavily skewed in size.** The Challenge-tier xBD
  download alone (no Tier3 supplement) spans 10 disasters from 18 to 823
  pairs each — generalization to disaster types or geographies outside
  this set is untested.

## Testing

```bash
# ML pipeline
PYTHONPATH=ml .venv/bin/python -m pytest ml/tests -q

# Frontend
npx vitest run
npx tsc --noEmit
npm run lint
npm run build
npx playwright test   # e2e
```

Every bug fix referenced above has a regression test named for it in
`ml/tests/`; see `findings.md` for the full list.

## Technical writeup

A few of the more interesting engineering problems this project ran into,
for anyone reading this as a portfolio piece rather than a repo:

**Small-batch BatchNorm silently breaking a Siamese network.** The
Siamese model's shared encoder was originally called twice per step —
once on the before-image batch, once on after — which is the natural way
to write it, and also wrong: at small batch sizes, each call's BatchNorm
layers accumulate running statistics from two *different* image
distributions (pre- vs. post-disaster) into the same buffers, and they
never converge. Train-mode accuracy looked fine (0.976 macro-F1) because
train-mode BatchNorm uses the current batch's own statistics; only
eval-mode inference — which uses the corrupted running statistics — was
broken (0.10 macro-F1). The fix (concatenate before/after along the batch
dimension, run the encoder once, split the features back apart) is
standard Siamese-network practice specifically for this failure mode, but
finding it required comparing train-mode vs. eval-mode inference on
identical data, not just watching the loss curve go down. Full writeup in
`findings.md` F-04.

**A metadata counter that was wrong on every single run, and why nothing
caught it.** An audit counter (`class_buildings`) was initialized with
string dict keys but checked against an int label with `if label in
class_buildings` — a comparison that's `False` for every possible input,
so the counter stayed at zero regardless of what data ran through it. It
shipped invisibly because a *different*, correctly-computed counter
(`class_pixels`, built by byte-scanning the rendered mask rather than the
same buggy loop) sat right next to it in the same audit output and looked
plausible on its own. Found only by scrutinizing a real audit report
number by number instead of checking "did it run without error." Full
writeup in `findings.md` F-09.

**Two build-tool plugins that were each correct in isolation.** Vite's
worker-import transform rewrites `new URL(x, import.meta.url)` into
`'' + import.meta.url` as part of producing dev-server output. A separate
plugin (vinext's `import.meta.url` rewriter) only recognizes the
*original*, unwrapped form as the pattern to leave alone — so it descends
into Vite's rewritten expression and replaces the nested
`import.meta.url` with a literal filesystem path, which then fails at
runtime. Neither plugin has a bug on its own; the interaction between them
does. Diagnosing it meant reading the actual transformed source the dev
server served, not just the error message. Full writeup in `findings.md`
F-07.

The full log — fifteen findings, what broke, how each was caught, and
the regression test for each — is in [`findings.md`](findings.md).
