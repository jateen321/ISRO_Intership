// The runtime `new Worker(new URL('./worker.ts', import.meta.url))` pattern
// is broken under this project's dev server: vinext's import-meta-url
// rewriter (node_modules/vinext/dist/plugins/import-meta-url.js) only
// recognizes a *direct* `import.meta.url` argument to `new URL(...)` as the
// pattern to leave alone for Vite's own worker transform to handle: it
// special-cases `new URL(specifier, import.meta.url)`. But Vite's worker
// transform itself first rewrites that second argument to `'' +
// import.meta.url` (a BinaryExpression) as part of its own output, and
// vinext's rewriter no longer recognizes that wrapped form, so it descends
// into it and replaces the nested `import.meta.url` with a literal
// `file:///ROOT/...` path — which then fails at runtime with
// "SecurityError: Failed to construct 'Worker'". The `?worker` import
// form below is a different Vite code path (resolved through the module
// graph, not a runtime import.meta.url string) and isn't affected.
// oxlint's import resolver doesn't read the `declare module '*?worker'`
// wildcard ambient type in worker-module.d.ts the way tsc does; tsc has
// already verified this import has a default export.
// oxlint-disable-next-line import/default
import GeoShieldWorker from './worker?worker';
import type { AssessmentResult } from './types';

export interface AssessmentProgress {
  completedTiles: number;
  totalTiles: number;
}

export interface AssessmentOutcome {
  result: AssessmentResult;
  mask: Uint8Array;
}

export interface RunHandle {
  promise: Promise<AssessmentOutcome>;
  cancel: () => void;
}

/** Owns one Web Worker across multiple assessments, so the ONNX Runtime
 * session (WebGPU/WASM init + model load) only pays its startup cost once
 * per page load, not once per assessment. */
export class InferenceClient {
  private worker: Worker | null = null;
  private nextRunId = 1;

  private ensureWorker(): Worker {
    this.worker ??= new GeoShieldWorker();
    return this.worker;
  }

  runAssessment(before: File, after: File, onProgress: (progress: AssessmentProgress) => void): RunHandle {
    const worker = this.ensureWorker();
    const runId = this.nextRunId++;
    let settled = false;

    const promise = new Promise<AssessmentOutcome>((resolve, reject) => {
      const handleMessage = (event: MessageEvent) => {
        const message = event.data;
        if (message.runId !== runId) return;
        if (message.type === 'progress') {
          onProgress({ completedTiles: message.completedTiles, totalTiles: message.totalTiles });
          return;
        }
        if (message.type === 'complete') {
          settled = true;
          cleanup();
          resolve({ result: message.result, mask: message.mask });
          return;
        }
        if (message.type === 'cancelled') {
          settled = true;
          cleanup();
          reject(new DOMException('Assessment cancelled', 'AbortError'));
          return;
        }
        if (message.type === 'error') {
          settled = true;
          cleanup();
          reject(new Error(message.message));
        }
      };
      const handleWorkerError = (event: ErrorEvent) => {
        settled = true;
        cleanup();
        reject(new Error(event.message || 'Inference worker crashed'));
      };
      const cleanup = () => {
        worker.removeEventListener('message', handleMessage);
        worker.removeEventListener('error', handleWorkerError);
      };
      worker.addEventListener('message', handleMessage);
      worker.addEventListener('error', handleWorkerError);
      worker.postMessage({ type: 'run', before, after, runId });
    });

    const cancel = () => {
      if (settled) return;
      worker.postMessage({ type: 'cancel', runId });
    };

    return { promise, cancel };
  }

  /** Tears the worker down outright — releases the loaded model session and
   * any in-flight buffers immediately, rather than waiting on a cancel flag.
   * A later runAssessment() call transparently spawns a fresh worker. */
  dispose(): void {
    this.worker?.terminate();
    this.worker = null;
  }
}
