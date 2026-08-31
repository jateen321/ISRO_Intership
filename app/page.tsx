/* oxlint-disable next/no-img-element */
'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  ArrowUpRight,
  BarChart3,
  Check,
  CircleHelp,
  CloudUpload,
  Download,
  FileImage,
  Fingerprint,
  Info,
  Layers3,
  LockKeyhole,
  MapPinned,
  MousePointer2,
  RefreshCcw,
  Satellite,
  ScanSearch,
  ShieldCheck,
  Sparkles,
  X,
  XCircle,
} from 'lucide-react';
import { InferenceClient, type AssessmentOutcome } from '@/lib/geoshield/client';
import { buildOverlayRgba } from '@/lib/geoshield/overlay';
import { buildAssessmentJson, buildRegionsCsv, reportFilename } from '@/lib/geoshield/reports';
import { WORKING_SIZE } from '@/lib/geoshield/tensor';

type SlotName = 'before' | 'after';

type ImageSlot = {
  file: File;
  url: string;
};

type AssessmentPhase = 'idle' | 'running' | 'complete' | 'error' | 'cancelled';

const DAMAGE_CLASSES: Array<{ classId: 1 | 2 | 3 | 4; key: string; name: string; color: string; description: string }> = [
  { classId: 1, key: 'undamaged', name: 'Undamaged', color: '#59d39b', description: 'No visible change' },
  { classId: 2, key: 'minor', name: 'Minor damage', color: '#f6c85f', description: 'Light structural impact' },
  { classId: 3, key: 'major', name: 'Major damage', color: '#ef8b4e', description: 'Severe structural impact' },
  { classId: 4, key: 'destroyed', name: 'Destroyed', color: '#ee6571', description: 'Building loss likely' },
];

function downloadText(filename: string, contents: string, mimeType: string) {
  const blob = new Blob([contents], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

const SAMPLE_PAIRS = [
  { id: 'moore', label: 'Moore tornado', meta: 'United States · tornado' },
  { id: 'socal', label: 'SoCal wildfire', meta: 'United States · wildfire' },
  { id: 'palus', label: 'Palu tsunami', meta: 'Indonesia · tsunami' },
];

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  return `${(bytes / 1024).toFixed(1)} KB`;
}

function UploadCard({
  name,
  slot,
  onFile,
  onClear,
}: {
  name: string;
  slot: ImageSlot | null;
  onFile: (file: File) => void;
  onClear: () => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const acceptFile = (file?: File) => {
    if (!file || !file.type.startsWith('image/')) return;
    onFile(file);
  };

  return (
    <div className={`upload-card ${isDragging ? 'is-dragging' : ''} ${slot ? 'has-file' : ''}`}>
      <input
        ref={inputRef}
        className="visually-hidden"
        type="file"
        accept="image/png,image/jpeg,image/webp"
        aria-label={`Choose ${name.toLowerCase()} image`}
        onChange={(event) => acceptFile(event.target.files?.[0])}
      />
      {slot ? (
        <div className="file-selected">
          <img src={slot.url} alt={`${name} preview`} />
          <div className="file-selected-copy">
            <div className="file-title-row">
              <span className="file-type"><FileImage size={13} /> Image ready</span>
              <button type="button" className="clear-file" onClick={onClear} aria-label={`Remove ${name.toLowerCase()} image`}>
                <X size={14} />
              </button>
            </div>
            <strong title={slot.file.name}>{slot.file.name}</strong>
            <span>{formatBytes(slot.file.size)} · {slot.file.type.replace('image/', '').toUpperCase()}</span>
          </div>
        </div>
      ) : (
        <button
          type="button"
          className="upload-action"
          onClick={() => inputRef.current?.click()}
          onDragEnter={(event) => { event.preventDefault(); setIsDragging(true); }}
          onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(event) => { event.preventDefault(); setIsDragging(false); acceptFile(event.dataTransfer.files[0]); }}
        >
          <span className="upload-icon"><CloudUpload size={20} /></span>
          <span className="upload-heading">Drop {name.toLowerCase()} image</span>
          <span className="upload-subheading">or click to browse · PNG, JPG, WEBP</span>
        </button>
      )}
    </div>
  );
}

export default function Home() {
  const [before, setBefore] = useState<ImageSlot | null>(null);
  const [after, setAfter] = useState<ImageSlot | null>(null);
  const [sample, setSample] = useState('');
  const [showMethodology, setShowMethodology] = useState(false);
  const [notice, setNotice] = useState('');

  const [phase, setPhase] = useState<AssessmentPhase>('idle');
  const [progress, setProgress] = useState<{ completedTiles: number; totalTiles: number } | null>(null);
  const [outcome, setOutcome] = useState<AssessmentOutcome | null>(null);
  const [errorMessage, setErrorMessage] = useState('');
  const [overlayOpacity, setOverlayOpacity] = useState(0.6);
  const [visibleClasses, setVisibleClasses] = useState<Set<number>>(new Set([1, 2, 3, 4]));
  const [swipePosition, setSwipePosition] = useState(50);

  const clientRef = useRef<InferenceClient | null>(null);
  const cancelRef = useRef<(() => void) | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);

  const replaceSlot = useCallback((slot: SlotName, file: File) => {
    const url = URL.createObjectURL(file);
    if (slot === 'before') {
      setBefore((current) => { if (current) URL.revokeObjectURL(current.url); return { file, url }; });
    } else {
      setAfter((current) => { if (current) URL.revokeObjectURL(current.url); return { file, url }; });
    }
    setSample('');
    setPhase('idle');
    setOutcome(null);
    setErrorMessage('');
    setNotice('');
  }, []);

  const clearSlot = useCallback((slot: SlotName) => {
    if (slot === 'before') {
      setBefore((current) => { if (current) URL.revokeObjectURL(current.url); return null; });
    } else {
      setAfter((current) => { if (current) URL.revokeObjectURL(current.url); return null; });
    }
    setPhase('idle');
    setOutcome(null);
    setErrorMessage('');
    setNotice('');
  }, []);

  useEffect(() => () => {
    if (before) URL.revokeObjectURL(before.url);
    if (after) URL.revokeObjectURL(after.url);
  }, [before, after]);

  // One worker (and its loaded model session) persists across multiple
  // assessments; only torn down when the component unmounts.
  useEffect(() => () => clientRef.current?.dispose(), []);

  const pair = before && after ? { before, after } : null;
  const hasPair = pair !== null;

  const runAssessment = () => {
    if (!pair) return;
    clientRef.current ??= new InferenceClient();
    setPhase('running');
    setProgress({ completedTiles: 0, totalTiles: 4 });
    setOutcome(null);
    setErrorMessage('');
    setNotice('');

    const handle = clientRef.current.runAssessment(pair.before.file, pair.after.file, (nextProgress) => setProgress(nextProgress));
    cancelRef.current = handle.cancel;
    handle.promise
      .then((result) => {
        setOutcome(result);
        setPhase('complete');
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === 'AbortError') {
          setPhase('cancelled');
          return;
        }
        setErrorMessage(error instanceof Error ? error.message : 'Assessment failed for an unknown reason.');
        setPhase('error');
      })
      .finally(() => {
        cancelRef.current = null;
      });
  };

  const cancelAssessment = () => {
    cancelRef.current?.();
  };

  const reset = () => {
    cancelRef.current?.();
    clearSlot('before');
    clearSlot('after');
    setSample('');
    setPhase('idle');
    setOutcome(null);
    setErrorMessage('');
    setNotice('');
    setSwipePosition(50);
  };

  const toggleClassVisible = (classId: number) => {
    setVisibleClasses((current) => {
      const next = new Set(current);
      if (next.has(classId)) next.delete(classId);
      else next.add(classId);
      return next;
    });
  };

  useEffect(() => {
    const canvas = overlayCanvasRef.current;
    if (!canvas || !outcome) return;
    canvas.width = WORKING_SIZE;
    canvas.height = WORKING_SIZE;
    const context = canvas.getContext('2d');
    if (!context) return;
    const rgba = buildOverlayRgba(outcome.mask, WORKING_SIZE, WORKING_SIZE, overlayOpacity, visibleClasses);
    // buildOverlayRgba always backs its return value with a plain ArrayBuffer
    // (`new Uint8ClampedArray(n)`); TS's generic TypedArray<ArrayBufferLike>
    // signature just can't express that statically.
    context.putImageData(new ImageData(rgba as Uint8ClampedArray<ArrayBuffer>, WORKING_SIZE, WORKING_SIZE), 0, 0);
  }, [outcome, overlayOpacity, visibleClasses]);

  const exportJson = () => {
    if (!outcome) return;
    downloadText(reportFilename('assessment', outcome.result, 'json'), buildAssessmentJson(outcome.result), 'application/json');
  };

  const exportCsv = () => {
    if (!outcome) return;
    downloadText(reportFilename('regions', outcome.result, 'csv'), buildRegionsCsv(outcome.result), 'text/csv');
  };

  return (
    <main className="app-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="GeoShield AI home">
          <span className="brand-mark"><Satellite size={17} /></span>
          <span className="brand-copy"><strong>GeoShield</strong><span>AI</span></span>
        </a>
        <div className="header-context"><span className="status-dot" /> Research workspace <span className="context-separator">/</span> Damage assessment</div>
        <nav className="header-actions" aria-label="Site navigation">
          <button type="button" className="text-button" onClick={() => setShowMethodology((open) => !open)}>Methodology <ArrowUpRight size={14} /></button>
          <a className="github-button" href="#privacy"><LockKeyhole size={14} /> Private by design</a>
        </nav>
      </header>

      <div className="app-body" id="top">
        <aside className="side-rail" aria-label="Assessment navigation">
          <div className="rail-section">
            <span className="rail-label">Workspace</span>
            <a className="rail-link active" href="#analysis"><ScanSearch size={16} /> Analyze</a>
            <a className="rail-link" href="#how-it-works"><Layers3 size={16} /> How it works</a>
          </div>
          <div className="rail-section rail-bottom">
            <span className="rail-label">Model status</span>
            <div className="model-status">
              <span className="status-dot muted" /> ONNX model · untrained placeholder
            </div>
            {outcome && <div className="rail-version">Runtime: {outcome.result.runtime === 'webgpu' ? 'WebGPU' : 'WASM'}</div>}
            <div className="rail-version">v0.2 · Browser inference stage</div>
          </div>
        </aside>

        <section className="workspace" id="analysis">
          <div className="workspace-intro">
            <div>
              <p className="eyebrow"><MapPinned size={13} /> Remote sensing analysis</p>
              <h1>See what changed<br /><em>after disaster.</em></h1>
              <p className="lede">Compare aligned satellite imagery before and after an event. GeoShield will highlight building damage by severity—entirely in your browser.</p>
            </div>
            <div className="intro-aside">
              <div className="prototype-chip"><Sparkles size={14} /> Research prototype</div>
              <p>Four-class semantic segmentation<br />for rapid visual assessment.</p>
            </div>
          </div>

          <div className="privacy-banner" id="privacy">
            <ShieldCheck size={16} />
            <span><strong>Your imagery stays on this device.</strong> Inference runs entirely in your browser via ONNX Runtime Web (WebGPU, falling back to WASM) — no image is ever uploaded to a server.</span>
            <button type="button" aria-label="Learn about privacy"><CircleHelp size={15} /></button>
          </div>

          <div className="inline-notice placeholder-model-notice">
            <AlertTriangle size={16} />
            <span><strong>Placeholder model.</strong> The bundled ONNX model has not been trained on real satellite imagery — it was fit to a synthetic eight-tile fixture to validate the pipeline. Results below are not meaningful. See <code>findings.md</code>.</span>
          </div>

          <section className="analysis-card" aria-labelledby="input-heading">
            <div className="card-heading">
              <div><span className="step-number">01</span><div><p className="eyebrow">Input imagery</p><h2 id="input-heading">Add a before / after pair</h2></div></div>
              <span className="alignment-note"><Fingerprint size={14} /> Images must be aligned</span>
            </div>
            <div className="upload-grid">
              <UploadCard name="Before disaster" slot={before} onFile={(file) => replaceSlot('before', file)} onClear={() => clearSlot('before')} />
              <div className="pair-connector" aria-hidden="true"><span>+</span></div>
              <UploadCard name="After disaster" slot={after} onFile={(file) => replaceSlot('after', file)} onClear={() => clearSlot('after')} />
            </div>
            <div className="sample-row">
              <div className="sample-label"><Sparkles size={14} /><span>Or explore a sample pair</span></div>
              <div className="sample-options">
                {SAMPLE_PAIRS.map((item) => (
                  <button key={item.id} type="button" className={`sample-button ${sample === item.id ? 'selected' : ''}`} onClick={() => { setSample(item.id); setNotice('Sample imagery will be bundled after dataset licensing is verified.'); }}>
                    <span>{item.label}</span><small>{item.meta}</small>{sample === item.id && <Check size={13} />}
                  </button>
                ))}
              </div>
            </div>
          </section>

          <section className="action-row" aria-label="Assessment actions">
            <div className="readiness">
              <span className={`readiness-dot ${hasPair ? 'ready' : ''}`} />
              {phase === 'running'
                ? `Assessing… tile ${progress?.completedTiles ?? 0} of ${progress?.totalTiles ?? 4}`
                : hasPair
                  ? 'Pair ready for assessment'
                  : 'Waiting for two images'}
            </div>
            <div className="action-buttons">
              <button type="button" className="secondary-button" onClick={reset} disabled={!before && !after && !sample}><RefreshCcw size={15} /> Reset</button>
              {phase === 'running' ? (
                <button type="button" className="secondary-button" onClick={cancelAssessment}><XCircle size={15} /> Cancel</button>
              ) : (
                <button type="button" className="primary-button" onClick={runAssessment} disabled={!hasPair}><ScanSearch size={16} /> Run assessment <span className="button-key">⌘ ↵</span></button>
              )}
            </div>
          </section>

          {phase === 'running' && progress && (
            <progress className="progress-track" value={progress.completedTiles} max={progress.totalTiles} aria-label="Assessment progress" />
          )}

          {phase === 'error' && (
            <output className="inline-notice error-notice"><AlertTriangle size={16} /><span>{errorMessage}</span><button type="button" aria-label="Dismiss error" onClick={() => setPhase('idle')}><X size={14} /></button></output>
          )}

          {phase === 'cancelled' && (
            <output className="inline-notice"><Info size={16} /><span>Assessment cancelled.</span><button type="button" aria-label="Dismiss notice" onClick={() => setPhase('idle')}><X size={14} /></button></output>
          )}

          {notice && <output className="inline-notice"><Info size={16} /><span>{notice}</span><button type="button" aria-label="Dismiss notice" onClick={() => setNotice('')}><X size={14} /></button></output>}

          <section className="results-card" aria-labelledby="results-heading">
            <div className="card-heading results-heading">
              <div><span className="step-number">02</span><div><p className="eyebrow">Assessment output</p><h2 id="results-heading">Damage overview</h2></div></div>
              <div className="export-row">
                <button type="button" className="export-button" onClick={exportJson} disabled={!outcome}><Download size={14} /> JSON</button>
                <button type="button" className="export-button" onClick={exportCsv} disabled={!outcome}><Download size={14} /> CSV</button>
              </div>
            </div>
            {hasPair ? (
              <div className="comparison-grid">
                <div className="comparison-view">
                  <div className="comparison-toolbar">
                    <span>{outcome ? 'Before / after (drag to compare)' : 'Input pair'}</span>
                    <span className="toolbar-hint"><MousePointer2 size={12} /> {outcome ? 'Overlay on the after image' : 'Overlay available after inference'}</span>
                  </div>
                  {outcome ? (
                    <>
                      <div className="swipe-viewer">
                        <img src={pair.after.url} alt="After disaster" className="swipe-base" />
                        <canvas ref={overlayCanvasRef} className="swipe-overlay-canvas" aria-hidden="true" />
                        <div className="swipe-clip" style={{ clipPath: `inset(0 ${100 - swipePosition}% 0 0)` }}>
                          <img src={pair.before.url} alt="Before disaster" className="swipe-base" />
                        </div>
                        <input
                          type="range"
                          className="swipe-slider"
                          min={0}
                          max={100}
                          value={swipePosition}
                          onChange={(event) => setSwipePosition(Number(event.target.value))}
                          aria-label="Before/after comparison position"
                        />
                      </div>
                      <div className="overlay-controls">
                        <label className="opacity-control">
                          <span>Overlay opacity</span>
                          <input type="range" min={0} max={1} step={0.05} value={overlayOpacity} onChange={(event) => setOverlayOpacity(Number(event.target.value))} />
                        </label>
                        <div className="class-toggle-row">
                          {DAMAGE_CLASSES.map((item) => (
                            <button
                              key={item.key}
                              type="button"
                              className={`class-toggle ${visibleClasses.has(item.classId) ? 'active' : ''}`}
                              style={{ '--toggle-color': item.color } as React.CSSProperties}
                              onClick={() => toggleClassVisible(item.classId)}
                            >
                              <i /> {item.name}
                            </button>
                          ))}
                        </div>
                      </div>
                    </>
                  ) : (
                    <div className="image-pair-preview">
                      <figure><img src={pair.before.url} alt="Before disaster uploaded preview" /><figcaption>Before disaster</figcaption></figure>
                      <figure><img src={pair.after.url} alt="After disaster uploaded preview" /><figcaption>After disaster</figcaption></figure>
                    </div>
                  )}
                </div>
                {outcome ? (
                  <div className="confidence-panel">
                    <h3>Confidence &amp; limitations</h3>
                    <ul>
                      <li>Processed at {outcome.result.inputDimensions[0]}×{outcome.result.inputDimensions[1]}px in {Math.round(outcome.result.processingTimeMs)}ms via {outcome.result.runtime === 'webgpu' ? 'WebGPU' : 'WASM'}.</li>
                      <li>Region counts are estimates from connected-component analysis, not a verified building count.</li>
                      <li>Not validated for operational or emergency decisions.</li>
                    </ul>
                    {outcome.result.warnings.map((warning) => (
                      <div className="confidence-warning" key={warning}><AlertTriangle size={13} /><span>{warning}</span></div>
                    ))}
                  </div>
                ) : (
                  <div className="pending-output">
                    <div className="pending-icon"><BarChart3 size={22} /></div>
                    <h3>{phase === 'running' ? 'Assessment in progress' : 'Ready for model inference'}</h3>
                    <p>{phase === 'running' ? 'Tiling, running the model, and stitching results — this happens entirely in your browser.' : 'Run an assessment to see a damage overlay, severity breakdown, and exportable regions.'}</p>
                    <div className="pending-meta"><span><span className={`status-dot ${phase === 'running' ? '' : 'muted'}`} /> ONNX model</span><span>{phase === 'running' ? 'Running' : 'Ready (placeholder)'}</span></div>
                  </div>
                )}
              </div>
            ) : (
              <div className="empty-results">
                <div className="empty-graphic"><span /><span /><span /><span /></div>
                <div><h3>Your assessment will appear here</h3><p>Upload a matching before and after pair to preview the comparison and prepare an analysis.</p></div>
              </div>
            )}
            <div className="damage-legend" aria-label="Damage class legend">
              <span className="legend-title">Damage classes</span>
              {DAMAGE_CLASSES.map((item) => <span className="legend-item" key={item.name}><i style={{ background: item.color }} /> <span>{item.name}</span></span>)}
            </div>
          </section>

          <section className="stats-grid" aria-label="Assessment statistics">
            {DAMAGE_CLASSES.map((item) => {
              const stat = outcome?.result.classStatistics[item.key];
              return (
                <article className="stat-card" key={item.key} style={{ borderLeft: `3px solid ${item.color}` }}>
                  <p>{item.name}</p>
                  <strong>{stat ? `${stat.percentage.toFixed(1)}%` : '—'}</strong>
                  <span>{stat ? `${stat.estimatedRegions} estimated region${stat.estimatedRegions === 1 ? '' : 's'}` : 'Available after inference'}</span>
                </article>
              );
            })}
          </section>

          {outcome && outcome.result.regions.length > 0 && (
            <section className="region-table-card" aria-label="Estimated building regions">
              <div className="card-heading"><p className="eyebrow">Detected regions</p><h2>{outcome.result.regions.length} estimated building region{outcome.result.regions.length === 1 ? '' : 's'}</h2></div>
              <div className="region-table-scroll">
                <table className="region-table">
                  <thead><tr><th>ID</th><th>Class</th><th>Confidence</th><th>Pixel area</th></tr></thead>
                  <tbody>
                    {outcome.result.regions.map((region) => {
                      const classInfo = DAMAGE_CLASSES.find((item) => item.classId === region.damageClass);
                      return (
                        <tr key={region.id}>
                          <td>{region.id}</td>
                          <td><span className="region-class-chip" style={{ '--toggle-color': classInfo?.color } as React.CSSProperties}>{classInfo?.name ?? region.damageClass}</span></td>
                          <td>{(region.confidence * 100).toFixed(1)}%</td>
                          <td>{region.pixelArea.toLocaleString()}px</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </section>
          )}
          {outcome && outcome.result.regions.length === 0 && (
            <section className="region-table-card" aria-label="Estimated building regions">
              <p className="no-regions-message">No building-sized regions were detected in this pair.</p>
            </section>
          )}

          <section className="how-it-works" id="how-it-works">
            <div><p className="eyebrow">Designed for clarity</p><h2>From two images to a<br /><em>damage map.</em></h2></div>
            <div className="how-steps">
              <div><span>01</span><strong>Align</strong><p>Use the same footprint before and after the event.</p></div>
              <div><span>02</span><strong>Segment</strong><p>A paired vision model identifies building regions.</p></div>
              <div><span>03</span><strong>Understand</strong><p>Review severity, confidence, and exportable evidence.</p></div>
            </div>
          </section>

          {showMethodology && <dialog open className="methodology-panel" aria-label="Methodology"><div><p className="eyebrow">Methodology</p><h2>Multi-temporal segmentation</h2><p>GeoShield is being built around a shared-encoder U-Net that reads pre- and post-disaster image features together. The model will be evaluated on disaster events it has not seen during training.</p></div><button type="button" aria-label="Close methodology" onClick={() => setShowMethodology(false)}><X size={16} /></button></dialog>}

          <footer className="site-footer"><span>GeoShield AI · Satellite damage assessment</span><span>Built as an internship research prototype · No operational claims</span></footer>
        </section>
      </div>
    </main>
  );
}
