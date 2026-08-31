/// <reference lib="webworker" />
import * as ort from 'onnxruntime-web';
import { postprocessMask } from './postprocess';
import { normalizeImageNet, softmaxArgmax, splitIntoTiles, stitchTiles, TILE_SIZE, validateImagePair, WORKING_SIZE } from './tensor';
import type { AssessmentResult } from './types';

declare const self: DedicatedWorkerGlobalScope;

const MODEL_URL = '/models/geoshield-siamese.onnx';
const METADATA_URL = '/models/geoshield-siamese.json';

// The WebGPU-capable ONNX Runtime Web wasm binary is 26.5MB — over
// Cloudflare Workers' 25 MiB per-asset limit — so it can't be bundled as a
// static asset (see vite.config.ts). Load the runtime engine itself from a
// CDN, pinned to the exact installed onnxruntime-web version, instead. This
// only fetches generic runtime code, never user imagery or results.
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/';

interface ModelMetadata {
  model: string;
  sha256: string;
  trained_on_real_data: boolean;
}

type RunMessage = { type: 'run'; before: File; after: File; runId: number };
type CancelMessage = { type: 'cancel'; runId: number };
type InboundMessage = RunMessage | CancelMessage;

let sessionPromise: Promise<{ session: ort.InferenceSession; runtime: 'webgpu' | 'wasm' }> | null = null;
let metadataPromise: Promise<ModelMetadata> | null = null;
let cancelledRunId: number | null = null;

function loadMetadata(): Promise<ModelMetadata> {
  metadataPromise ??= fetch(METADATA_URL).then((response) => {
    if (!response.ok) throw new Error(`Failed to load model metadata (${response.status})`);
    return response.json() as Promise<ModelMetadata>;
  });
  return metadataPromise;
}

async function loadSession(): Promise<{ session: ort.InferenceSession; runtime: 'webgpu' | 'wasm' }> {
  sessionPromise ??= (async () => {
    if ('gpu' in self.navigator) {
      try {
        const session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ['webgpu'] });
        return { session, runtime: 'webgpu' as const };
      } catch {
        // WebGPU present but session creation failed (unsupported op, driver
        // issue, etc.) — fall through to WASM below.
      }
    }
    const session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'] });
    return { session, runtime: 'wasm' as const };
  })();
  return sessionPromise;
}

async function decodeAndResize(file: File, size: number): Promise<{ imageData: ImageData; width: number; height: number }> {
  const bitmap = await createImageBitmap(file);
  const { width, height } = bitmap;
  const canvas = new OffscreenCanvas(size, size);
  const context = canvas.getContext('2d');
  if (!context) {
    bitmap.close();
    throw new Error('Could not create a 2D canvas context for image decoding');
  }
  context.drawImage(bitmap, 0, 0, size, size);
  bitmap.close();
  const imageData = context.getImageData(0, 0, size, size);
  return { imageData, width, height };
}

async function runAssessment(before: File, after: File, runId: number): Promise<{ result: AssessmentResult; mask: Uint8Array }> {
  const startTime = performance.now();

  const beforeDecoded = await decodeAndResize(before, WORKING_SIZE);
  const afterDecoded = await decodeAndResize(after, WORKING_SIZE);
  const validation = validateImagePair(
    { width: beforeDecoded.width, height: beforeDecoded.height },
    { width: afterDecoded.width, height: afterDecoded.height },
  );
  if (!validation.ok) throw new Error(validation.error);

  const beforeChw = normalizeImageNet(beforeDecoded.imageData.data, WORKING_SIZE, WORKING_SIZE);
  const afterChw = normalizeImageNet(afterDecoded.imageData.data, WORKING_SIZE, WORKING_SIZE);
  const beforeTiles = splitIntoTiles(beforeChw, 3, WORKING_SIZE, TILE_SIZE, Float32Array);
  const afterTiles = splitIntoTiles(afterChw, 3, WORKING_SIZE, TILE_SIZE, Float32Array);

  const { session, runtime } = await loadSession();
  const metadata = await loadMetadata();

  const maskTiles: Uint8Array[] = [];
  const probabilityTiles: Float32Array[] = [];
  const totalTiles = beforeTiles.length;
  for (let tileIndex = 0; tileIndex < totalTiles; tileIndex++) {
    if (cancelledRunId === runId) throw new DOMException('Assessment cancelled', 'AbortError');

    const beforeTensor = new ort.Tensor('float32', beforeTiles[tileIndex], [1, 3, TILE_SIZE, TILE_SIZE]);
    const afterTensor = new ort.Tensor('float32', afterTiles[tileIndex], [1, 3, TILE_SIZE, TILE_SIZE]);
    const outputs = await session.run({ before: beforeTensor, after: afterTensor });
    const logits = outputs.logits.data as Float32Array;
    const { mask, probabilities } = softmaxArgmax(logits, 5, TILE_SIZE, TILE_SIZE);
    maskTiles.push(mask);
    probabilityTiles.push(probabilities);

    beforeTensor.dispose();
    afterTensor.dispose();
    for (const output of Object.values(outputs)) output.dispose();

    self.postMessage({ type: 'progress', runId, completedTiles: tileIndex + 1, totalTiles });
  }

  const mask = stitchTiles(maskTiles, 1, WORKING_SIZE, TILE_SIZE, Uint8Array);
  const probabilities = stitchTiles(probabilityTiles, 5, WORKING_SIZE, TILE_SIZE, Float32Array);
  const { regions, classStatistics } = postprocessMask(mask, probabilities, WORKING_SIZE, WORKING_SIZE);

  const warnings: string[] = [];
  if (!metadata.trained_on_real_data) {
    warnings.push('This model has not been trained on real satellite imagery (development placeholder). Predictions are not meaningful.');
  }

  const result: AssessmentResult = {
    schemaVersion: '1.0',
    modelVersion: metadata.model,
    modelHash: metadata.sha256,
    runtime,
    inputDimensions: [WORKING_SIZE, WORKING_SIZE],
    processingTimeMs: performance.now() - startTime,
    classStatistics,
    regions,
    warnings,
  };

  return { result, mask };
}

self.onmessage = async (event: MessageEvent<InboundMessage>) => {
  const message = event.data;
  if (message.type === 'cancel') {
    cancelledRunId = message.runId;
    return;
  }
  if (message.type !== 'run') return;

  const { runId } = message;
  try {
    const { result, mask } = await runAssessment(message.before, message.after, runId);
    if (cancelledRunId === runId) {
      self.postMessage({ type: 'cancelled', runId });
      return;
    }
    self.postMessage({ type: 'complete', runId, result, mask }, [mask.buffer]);
  } catch (error) {
    if (cancelledRunId === runId || (error instanceof DOMException && error.name === 'AbortError')) {
      self.postMessage({ type: 'cancelled', runId });
      return;
    }
    self.postMessage({ type: 'error', runId, message: error instanceof Error ? error.message : String(error) });
  }
};
