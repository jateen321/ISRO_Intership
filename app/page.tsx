/* oxlint-disable next/no-img-element */
'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import {
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
} from 'lucide-react';

type SlotName = 'before' | 'after';

type ImageSlot = {
  file: File;
  url: string;
};

const DAMAGE_CLASSES = [
  { name: 'Undamaged', color: '#59d39b', description: 'No visible change' },
  { name: 'Minor damage', color: '#f6c85f', description: 'Light structural impact' },
  { name: 'Major damage', color: '#ef8b4e', description: 'Severe structural impact' },
  { name: 'Destroyed', color: '#ee6571', description: 'Building loss likely' },
];

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
  const [runRequested, setRunRequested] = useState(false);

  const replaceSlot = useCallback((slot: SlotName, file: File) => {
    const url = URL.createObjectURL(file);
    if (slot === 'before') {
      setBefore((current) => { if (current) URL.revokeObjectURL(current.url); return { file, url }; });
    } else {
      setAfter((current) => { if (current) URL.revokeObjectURL(current.url); return { file, url }; });
    }
    setSample('');
    setRunRequested(false);
    setNotice('');
  }, []);

  const clearSlot = useCallback((slot: SlotName) => {
    if (slot === 'before') {
      setBefore((current) => { if (current) URL.revokeObjectURL(current.url); return null; });
    } else {
      setAfter((current) => { if (current) URL.revokeObjectURL(current.url); return null; });
    }
    setRunRequested(false);
    setNotice('');
  }, []);

  useEffect(() => () => {
    if (before) URL.revokeObjectURL(before.url);
    if (after) URL.revokeObjectURL(after.url);
  }, [before, after]);

  const pair = before && after ? { before, after } : null;
  const hasPair = pair !== null;

  const runAssessment = () => {
    if (!hasPair) return;
    setRunRequested(true);
    setNotice('Inference is not connected yet. The validated ONNX model will be added in the next build stage.');
  };

  const reset = () => {
    clearSlot('before');
    clearSlot('after');
    setSample('');
    setRunRequested(false);
    setNotice('');
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
            <div className="model-status"><span className="status-dot muted" /> ONNX pending</div>
            <div className="rail-version">v0.1 · Interface stage</div>
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
            <span><strong>Your imagery stays on this device.</strong> No upload is sent to a server. Browser inference will be enabled after the model checkpoint is validated.</span>
            <button type="button" aria-label="Learn about privacy"><CircleHelp size={15} /></button>
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
            <div className="readiness"><span className={`readiness-dot ${hasPair ? 'ready' : ''}`} />{hasPair ? 'Pair ready for assessment' : 'Waiting for two images'}</div>
            <div className="action-buttons">
              <button type="button" className="secondary-button" onClick={reset} disabled={!before && !after && !sample}><RefreshCcw size={15} /> Reset</button>
              <button type="button" className="primary-button" onClick={runAssessment} disabled={!hasPair}><ScanSearch size={16} /> Run assessment <span className="button-key">⌘ ↵</span></button>
            </div>
          </section>

          {notice && <output className="inline-notice"><Info size={16} /><span>{notice}</span><button type="button" aria-label="Dismiss notice" onClick={() => setNotice('')}><X size={14} /></button></output>}

          <section className="results-card" aria-labelledby="results-heading">
            <div className="card-heading results-heading">
              <div><span className="step-number">02</span><div><p className="eyebrow">Assessment output</p><h2 id="results-heading">Damage overview</h2></div></div>
              <button type="button" className="export-button" disabled><Download size={14} /> Export report</button>
            </div>
            {hasPair ? (
              <div className="comparison-grid">
                <div className="comparison-view">
                  <div className="comparison-toolbar"><span>Input pair</span><span className="toolbar-hint"><MousePointer2 size={12} /> Overlay available after inference</span></div>
                  <div className="image-pair-preview">
                    <figure><img src={pair.before.url} alt="Before disaster uploaded preview" /><figcaption>Before disaster</figcaption></figure>
                    <figure><img src={pair.after.url} alt="After disaster uploaded preview" /><figcaption>After disaster</figcaption></figure>
                  </div>
                </div>
                <div className="pending-output">
                  <div className="pending-icon"><BarChart3 size={22} /></div>
                  <h3>{runRequested ? 'Model connection pending' : 'Ready for model inference'}</h3>
                  <p>{runRequested ? 'This interface is wired for the ONNX runtime, but no validated model artifact is included yet.' : 'Run an assessment once the ONNX model is connected. Results will appear here without leaving your browser.'}</p>
                  <div className="pending-meta"><span><span className="status-dot muted" /> ONNX model</span><span>Not connected</span></div>
                </div>
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
            {['Buildings detected', 'Affected area', 'Model confidence'].map((label) => <article className="stat-card" key={label}><p>{label}</p><strong>—</strong><span>Available after inference</span></article>)}
          </section>

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
