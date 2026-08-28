'use client';

import { useEffect, useRef, useState, useCallback } from 'react';

interface AgentStep {
  step: number;
  thought: string;
  action: string;
  tool: string;
  observation: string;
}

interface ModelMetrics {
  threat_score: number;
  threat_percentage: string;
  tensor_shape: number[];
  raw_byte_size: number;
  model_architecture: string;
  device: string;
  status: string;
}

interface AgentResponse {
  success: boolean;
  incidentId: string;
  defcon: string;
  threatScore: number;
  tacticalBriefing: string;
  toolsCalled: string[];
  chainOfThought: AgentStep[];
  modelMetrics?: ModelMetrics;
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [autoAgentScan, setAutoAgentScan] = useState(false);
  const [defcon, setDefcon] = useState('DEFCON-5');
  const [threatScore, setThreatScore] = useState(0.08);
  const [agentThinking, setAgentThinking] = useState(false);
  const [agentLogs, setAgentLogs] = useState<AgentStep[]>([]);
  const [tacticalBriefing, setTacticalBriefing] = useState<string>('');
  const [operatorPrompt, setOperatorPrompt] = useState('');
  const [toast, setToast] = useState('');
  const [activeNav, setActiveNav] = useState('Surveillance & AI Agent');
  const [lastFrameSize, setLastFrameSize] = useState<number>(0);
  const [modelMetrics, setModelMetrics] = useState<ModelMetrics | null>(null);
  const [telemetryLogs, setTelemetryLogs] = useState([
    ['10:30:42.184', 'INFO', 'pytorch-engine', 'ThreatSeverityNet initialized [1, 3, 224, 224]', '200', '12ms'],
    ['10:30:39.721', 'INFO', 'camera-feed', 'Webcam stream ready on HTML5 Canvas', '200', '33ms'],
    ['10:30:36.402', 'WARN', 'agent-loop', 'Passive perimeter monitoring active in Sector Alpha', '—', '—'],
    ['10:30:31.097', 'INFO', 'threat-db', 'Knowledge base connected: 4 tactical signatures', '200', '5ms'],
  ]);

  // Webcam activation
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setCameraActive(true);
        setToast('Laptop camera stream active (Tactical Sector Alpha)');
      }
    } catch (err) {
      setToast('Unable to access webcam: ' + (err as Error).message);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
      setCameraActive(false);
      setAutoAgentScan(false);
      setToast('Camera feed stopped');
    }
  };

  // Capture real frame from video element to Base64 JPEG
  const captureFrameBase64 = (): string => {
    if (!videoRef.current || !cameraActive) {
      return '';
    }
    try {
      const canvas = document.createElement('canvas');
      canvas.width = 224;
      canvas.height = 224;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0, 224, 224);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.75);
        return dataUrl;
      }
    } catch (e) {
      console.error('Frame capture failed:', e);
    }
    return '';
  };

  // Trigger AI Agent ReAct Loop with REAL captured frame
  const triggerAgentEvaluation = useCallback(async (promptText?: string) => {
    setAgentThinking(true);
    const frameBase64 = captureFrameBase64();
    setLastFrameSize(frameBase64 ? frameBase64.length : 0);

    try {
      const res = await fetch('/api/agent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: promptText || 'Autonomous Sector Threat Evaluation',
          sector: 'Sector Alpha (Perimeter East)',
          frameBase64,
          hasMotion: cameraActive
        })
      });
      const data: AgentResponse = await res.json();
      if (data.success) {
        setDefcon(data.defcon);
        setThreatScore(data.threatScore);
        setAgentLogs(data.chainOfThought);
        setTacticalBriefing(data.tacticalBriefing);
        if (data.modelMetrics) {
          setModelMetrics(data.modelMetrics);
        }
        setToast(`PyTorch processed ${data.modelMetrics?.raw_byte_size || frameBase64.length} bytes · Posture: ${data.defcon}`);
        
        // Add to telemetry log
        setTelemetryLogs((prev) => [
          [new Date().toLocaleTimeString(), data.defcon === 'DEFCON-1' ? 'ERROR' : data.defcon === 'DEFCON-2' ? 'WARN' : 'INFO', 'agent-core', `Event ${data.incidentId} evaluated -> ${data.defcon}`, '200', '38ms'],
          ...prev.slice(0, 7)
        ]);
      }
    } catch (err) {
      setToast('Agent execution error: ' + (err as Error).message);
    } finally {
      setAgentThinking(false);
    }
  }, [cameraActive]);

  // Automated agent scanning loop
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (autoAgentScan && cameraActive) {
      interval = setInterval(() => {
        triggerAgentEvaluation('Automated Sentinel Loop Routine Scan');
      }, 6000);
    }
    return () => clearInterval(interval);
  }, [autoAgentScan, cameraActive, triggerAgentEvaluation]);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(''), 3500);
    return () => window.clearTimeout(timer);
  }, [toast]);

  return (
    <main className="shell">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark" style={{ background: 'linear-gradient(135deg, #2563eb, #7c3aed)' }}>SO</span>
          <span>SENTINEL<span>-OPS</span></span>
        </div>
        <p className="nav-label">ISRO / BSERC DEFENCE OPS</p>
        <nav aria-label="Primary navigation">
          {['Surveillance & AI Agent', 'PyTorch Model Metrics', 'Tactical Threat DB', 'Live Telemetry'].map((item, index) => (
            <button
              className={activeNav === item ? 'nav-link active' : 'nav-link'}
              onClick={() => setActiveNav(item)}
              key={item}
            >
              <span className="nav-symbol">{['🛡️', '⚡', '🛰️', '📊'][index]}</span>
              {item}
              {item === 'Surveillance & AI Agent' && defcon === 'DEFCON-1' && <span className="nav-badge" style={{ background: '#dc2626' }}>ALERT</span>}
            </button>
          ))}
        </nav>
        
        <div className="analyst-card" style={{ border: defcon === 'DEFCON-1' ? '1px solid #ef4444' : '1px solid #3b82f6' }}>
          <span className="spark">✦</span>
          <strong>AI Tactical Commander</strong>
          <p>ReAct loop actively calling ThreatSeverityNet.py</p>
          <span className="online" style={{ color: defcon === 'DEFCON-1' ? '#ef4444' : '#10b981' }}>
            <i style={{ background: defcon === 'DEFCON-1' ? '#ef4444' : '#10b981' }} /> {defcon}
          </span>
        </div>

        <div className="profile">
          <span className="avatar" style={{ background: '#4f46e5' }}>IS</span>
          <span><strong>Defence Operator</strong><small>Tactical Command</small></span>
        </div>
      </aside>

      {/* Main Operations Workspace */}
      <section className="workspace">
        <header className="topbar">
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '0.875rem' }}>
            <span style={{ fontWeight: 600, color: '#94a3b8' }}>SYSTEM POSTURE:</span>
            <span style={{ 
              padding: '4px 10px', 
              borderRadius: '9999px', 
              fontWeight: 700, 
              background: defcon === 'DEFCON-1' ? 'rgba(239, 68, 68, 0.2)' : defcon === 'DEFCON-3' ? 'rgba(245, 158, 11, 0.2)' : 'rgba(16, 185, 129, 0.2)',
              color: defcon === 'DEFCON-1' ? '#ef4444' : defcon === 'DEFCON-3' ? '#f59e0b' : '#10b981',
              border: `1px solid ${defcon === 'DEFCON-1' ? '#ef4444' : defcon === 'DEFCON-3' ? '#f59e0b' : '#10b981'}`
            }}>
              {defcon}
            </span>
          </div>
          <span className="environment"><i /> Live Perception Engine</span>
          <button className="icon-button" onClick={() => triggerAgentEvaluation('Manual Threat Audit')}>⚡ Scan</button>
        </header>

        <div className="content">
          {/* Header Banner */}
          <div className="headline">
            <div>
              <p className="eyebrow">REAL END-TO-END PIPELINE (CAMERA ➔ PYTORCH ➔ AI AGENT)</p>
              <h1>SENTINEL-OPS Perimeter Intelligence</h1>
              <p className="subcopy">HTML5 Canvas Frame Capture ➔ Python ThreatSeverityNet.py Subprocess ➔ ReAct Tool Execution</p>
            </div>
            <div className="actions">
              {!cameraActive ? (
                <button className="primary-button" onClick={startCamera} style={{ background: '#2563eb' }}>
                  <span>📷</span> Start Laptop Camera
                </button>
              ) : (
                <button className="ghost-button" onClick={stopCamera} style={{ color: '#ef4444' }}>
                  <span>⏹</span> Stop Camera
                </button>
              )}
              <button 
                className="primary-button" 
                onClick={() => triggerAgentEvaluation(operatorPrompt || 'Tactical Threat Assessment')}
                disabled={agentThinking}
                style={{ background: '#7c3aed' }}
              >
                <span>✦</span> {agentThinking ? 'Executing PyTorch...' : 'Command Agent'}
              </button>
            </div>
          </div>

          {/* Metric Bar */}
          <section className="metric-grid" aria-label="Key Metrics">
            <article className="metric-card">
              <div><p>PyTorch CNN Threat Score</p><span className="trend danger">{(threatScore * 100).toFixed(1)}%</span></div>
              <strong style={{ color: threatScore > 0.6 ? '#ef4444' : '#10b981' }}>{(threatScore * 100).toFixed(1)}<small>%</small></strong>
              <div className="microbar hot" style={{ width: `${Math.min(100, threatScore * 100)}%`, background: threatScore > 0.6 ? '#ef4444' : '#3b82f6' }} />
            </article>

            <article className="metric-card">
              <div><p>PyTorch Subprocess Bridge</p><span className="trend">{modelMetrics ? 'Active (Exit 0)' : 'Connected'}</span></div>
              <strong style={{ fontSize: '1rem' }}>ThreatSeverityNet<small>.py</small></strong>
              <div className="microbar" style={{ width: '100%', background: '#8b5cf6' }} />
            </article>

            <article className="metric-card">
              <div><p>Serialized Frame Size</p><span className="trend">{lastFrameSize > 0 ? `${(lastFrameSize / 1024).toFixed(1)} KB` : 'Idle'}</span></div>
              <strong>{lastFrameSize > 0 ? (lastFrameSize / 1024).toFixed(1) : '0'}<small>KB</small></strong>
              <div className="microbar" style={{ width: lastFrameSize > 0 ? '100%' : '10%', background: '#10b981' }} />
            </article>

            <article className="metric-card">
              <div><p>Camera Stream</p><span className="trend">{cameraActive ? '30 FPS' : 'Standby'}</span></div>
              <strong>{cameraActive ? '640x480' : 'OFF'}<small>{cameraActive ? 'px' : ''}</small></strong>
              <div className="microbar" style={{ width: cameraActive ? '100%' : '10%', background: cameraActive ? '#10b981' : '#64748b' }} />
            </article>
          </section>

          {/* Main Grid: Webcam Perception Feed vs Agent Chain of Thought */}
          <section className="main-grid" style={{ gridTemplateColumns: '1.2fr 1fr', gap: '20px' }}>
            {/* Left: Camera Feed */}
            <article className="panel performance-panel" style={{ position: 'relative', overflow: 'hidden' }}>
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">LIVE CAMERA PERCEPTION (CANVAS SERIALIZED)</p>
                  <h2>Tactical Camera (Sector Alpha)</h2>
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                  {cameraActive && (
                    <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.8rem', color: '#94a3b8', cursor: 'pointer' }}>
                      <input type="checkbox" checked={autoAgentScan} onChange={(e) => setAutoAgentScan(e.target.checked)} />
                      Auto Frame Streaming
                    </label>
                  )}
                </div>
              </div>

              <div style={{ 
                minHeight: '340px', 
                background: '#090d16', 
                borderRadius: '8px', 
                position: 'relative', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center',
                border: '1px solid #1e293b',
                overflow: 'hidden'
              }}>
                <video 
                  ref={videoRef} 
                  autoPlay 
                  playsInline 
                  muted 
                  style={{ 
                    width: '100%', 
                    height: '100%', 
                    objectFit: 'cover', 
                    display: cameraActive ? 'block' : 'none' 
                  }} 
                />
                {!cameraActive && (
                  <div style={{ textAlign: 'center', color: '#64748b', padding: '24px' }}>
                    <div style={{ fontSize: '2.5rem', marginBottom: '8px' }}>📷</div>
                    <p style={{ fontWeight: 600, color: '#cbd5e1' }}>Laptop Camera is Offline</p>
                    <p style={{ fontSize: '0.85rem' }}>Click &quot;Start Laptop Camera&quot; to capture real video frames.</p>
                  </div>
                )}
                {cameraActive && (
                  <div style={{ 
                    position: 'absolute', 
                    top: '12px', 
                    left: '12px', 
                    background: 'rgba(0,0,0,0.6)', 
                    backdropFilter: 'blur(4px)', 
                    padding: '4px 10px', 
                    borderRadius: '4px',
                    fontSize: '0.75rem',
                    color: '#10b981',
                    border: '1px solid rgba(16, 185, 129, 0.4)'
                  }}>
                    ● REAL-TIME FEED [SECTOR ALPHA] ➔ HTML5 CANVAS
                  </div>
                )}
              </div>
            </article>

            {/* Right: AI Agent Chain of Thought */}
            <article className="panel services-panel">
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">RE-ACT AUTONOMOUS REASONING TRACE</p>
                  <h2>Agent Chain of Thought</h2>
                </div>
                <span style={{ fontSize: '0.8rem', color: agentThinking ? '#8b5cf6' : '#10b981' }}>
                  {agentThinking ? '● Running PyTorch Model...' : '● Standby'}
                </span>
              </div>

              <div style={{ maxHeight: '340px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {agentLogs.length === 0 ? (
                  <div style={{ color: '#64748b', fontSize: '0.875rem', textAlign: 'center', padding: '30px' }}>
                    <p>No active agent reasoning trace.</p>
                    <p style={{ fontSize: '0.8rem', marginTop: '6px' }}>Click &quot;Command Agent&quot; to process the current frame through PyTorch.</p>
                  </div>
                ) : (
                  agentLogs.map((step) => (
                    <div key={step.step} style={{ 
                      background: 'rgba(15, 23, 42, 0.6)', 
                      padding: '12px', 
                      borderRadius: '6px', 
                      borderLeft: '3px solid #7c3aed',
                      fontSize: '0.85rem'
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', color: '#a78bfa', fontWeight: 600, marginBottom: '4px' }}>
                        <span>Step {step.step}: {step.action}</span>
                        <code style={{ fontSize: '0.75rem', background: '#1e1b4b', padding: '2px 6px', borderRadius: '4px' }}>{step.tool}</code>
                      </div>
                      <p style={{ color: '#cbd5e1', marginBottom: '4px' }}><strong>Thought:</strong> {step.thought}</p>
                      <p style={{ color: '#94a3b8', fontSize: '0.8rem' }}><strong>Observation:</strong> {step.observation}</p>
                    </div>
                  ))
                )}
              </div>
            </article>
          </section>

          {/* Interactive Agent Operator Console */}
          <section className="panel" style={{ marginTop: '20px', padding: '16px', background: 'rgba(15, 23, 42, 0.4)', borderRadius: '8px', border: '1px solid #1e293b' }}>
            <h3 style={{ fontSize: '1rem', color: '#f8fafc', marginBottom: '8px' }}>💬 Command AI Tactical Agent</h3>
            <div style={{ display: 'flex', gap: '10px' }}>
              <input 
                type="text" 
                placeholder="E.g. Agent, run ThreatSeverityNet.py on current camera frame and cross-reference threat database..." 
                value={operatorPrompt}
                onChange={(e) => setOperatorPrompt(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') triggerAgentEvaluation(operatorPrompt); }}
                style={{ 
                  flex: 1, 
                  background: '#0f172a', 
                  border: '1px solid #334155', 
                  borderRadius: '6px', 
                  padding: '10px 14px', 
                  color: '#fff',
                  fontSize: '0.9rem' 
                }}
              />
              <button 
                className="primary-button"
                onClick={() => triggerAgentEvaluation(operatorPrompt)}
                disabled={agentThinking}
                style={{ background: '#2563eb' }}
              >
                Send Instruction
              </button>
            </div>
          </section>

          {/* Tactical Briefing Report */}
          {tacticalBriefing && (
            <section className="incident-alert" style={{ marginTop: '20px', borderColor: defcon === 'DEFCON-1' ? '#ef4444' : '#3b82f6' }}>
              <div className="severity" style={{ background: defcon === 'DEFCON-1' ? '#ef4444' : '#3b82f6' }}>REPORT</div>
              <div style={{ whiteSpace: 'pre-line', fontSize: '0.9rem' }}>
                <h2>Autonomous Tactical Briefing (Real PyTorch Ingestion)</h2>
                <p style={{ marginTop: '6px', color: '#cbd5e1' }}>{tacticalBriefing}</p>
              </div>
            </section>
          )}
        </div>
      </section>

      {toast && <div className="toast" role="status"><span>✓</span>{toast}</div>}
    </main>
  );
}
