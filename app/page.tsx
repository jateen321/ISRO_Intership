'use client';

import { useEffect, useState } from 'react';

const services = [
  { name: 'Edge Gateway', meta: '23.4k req/min', latency: '42 ms' },
  { name: 'Identity', meta: '8.1k req/min', latency: '86 ms' },
  { name: 'Catalog', meta: '11.7k req/min', latency: '124 ms' },
  { name: 'Payments', meta: '4.2k req/min', latency: '196 ms' },
];

const metrics = [
  { label: 'Throughput', value: '48.2k', unit: 'rpm', trend: '+12.4%' },
  { label: 'P95 latency', value: '221', unit: 'ms', trend: '-8.2%' },
  { label: 'Error rate', value: '0.07', unit: '%', trend: '-0.03%' },
  { label: 'SLO burn', value: '0.4', unit: '×', trend: 'Healthy' },
];

const healthyLogs = [
  ['10:30:42.184', 'INFO', 'edge-gateway', 'GET /v1/catalog completed', '200', '41ms'],
  ['10:30:39.721', 'INFO', 'payment-service', 'Payment intent authorized', '201', '189ms'],
  ['10:30:36.402', 'WARN', 'identity-service', 'Access token nearing expiry', '—', '—'],
  ['10:30:31.097', 'INFO', 'catalog-service', 'Inventory cache refreshed', '200', '92ms'],
];

const incidentLogs = [
  ['10:30:47.902', 'ERROR', 'payment-service', 'Database connection acquisition timed out', '500', '8.4s'],
  ['10:30:45.664', 'ERROR', 'payment-service', 'Payment processing failed', '504', '6.2s'],
  ['10:30:44.210', 'WARN', 'edge-gateway', 'Upstream response exceeded SLO', '—', '2.9s'],
  ...healthyLogs,
];

export default function Home() {
  const [isIncident, setIsIncident] = useState(false);
  const [analysisOpen, setAnalysisOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [activeNav, setActiveNav] = useState('Overview');
  const [timeRange, setTimeRange] = useState('60m');
  const [toast, setToast] = useState('');

  useEffect(() => {
    const handleKeys = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        setSearchOpen(true);
      }
      if (event.key === 'Escape') {
        setSearchOpen(false);
        setAnalysisOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeys);
    return () => window.removeEventListener('keydown', handleKeys);
  }, []);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(''), 3200);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const injectIncident = () => {
    setIsIncident(true);
    setActiveNav('Overview');
    setToast('Incident INC-1042 detected · 6 signals correlated');
  };

  const resolveIncident = () => {
    setIsIncident(false);
    setAnalysisOpen(false);
    setToast('Rollback complete · system health recovering');
  };

  return (
    <main className="shell">
      <aside className="sidebar">
        <a className="brand" href="#overview" aria-label="IncidentIQ home">
          <span className="brand-mark">IQ</span>
          <span>Incident<span>IQ</span></span>
        </a>
        <p className="nav-label">Operations</p>
        <nav aria-label="Primary navigation">
          {['Overview', 'Service map', 'Live traces', 'Incidents', 'Deployments'].map((item, index) => (
            <button className={activeNav === item ? 'nav-link active' : 'nav-link'} onClick={() => { setActiveNav(item); document.getElementById(item === 'Overview' ? 'overview' : item === 'Service map' ? 'service-map' : item === 'Live traces' ? 'live-traces' : item === 'Incidents' ? 'incident-workspace' : 'deployments')?.scrollIntoView(); }} key={item}>
              <span className="nav-symbol">{['⌂', '◇', '⌁', '!', '↗'][index]}</span>{item}
              {item === 'Incidents' && isIncident && <span className="nav-badge">1</span>}
            </button>
          ))}
        </nav>
        <div className="analyst-card">
          <span className="spark">✦</span>
          <strong>AI investigator</strong>
          <p>Correlation engine ready. Watching 42,891 signals.</p>
          <span className="online"><i /> Live</span>
        </div>
        <div className="profile">
          <span className="avatar">AR</span>
          <span><strong>Alex Rivera</strong><small>Platform engineer</small></span>
          <button aria-label="Open profile menu">•••</button>
        </div>
      </aside>

      <section className="workspace" id="overview">
        <header className="topbar">
          <button className="search" onClick={() => setSearchOpen(true)} aria-label="Open global search"><span>⌕</span> Search traces, services, incidents <kbd>⌘ K</kbd></button>
          <span className="environment"><i /> Production</span>
          <button className="icon-button" aria-label="Notifications">♢</button>
        </header>

        <div className="content">
          <div className="headline">
            <div>
              <p className="eyebrow">SYSTEM PULSE / LIVE</p>
              <h1>Your stack, understood.</h1>
              <p className="subcopy">One signal-rich view across every service, trace, and deployment.</p>
            </div>
            <div className="actions">
              <select className="range-select" value={timeRange} onChange={(event) => setTimeRange(event.target.value)} aria-label="Time range"><option value="15m">Last 15m</option><option value="60m">Last 60m</option><option value="24h">Last 24h</option></select>
              {isIncident && <button className="ghost-button" onClick={resolveIncident}>Resolve</button>}
              <button className="primary-button" onClick={injectIncident}><span>⚡</span> Inject incident</button>
            </div>
          </div>

          <section className={isIncident ? 'pulse-card incident-state' : 'pulse-card'} aria-live="polite">
            <div className="pulse-score"><span>{isIncident ? '72' : '98'}</span><small>HEALTH</small></div>
            <div>
              <div className="pulse-title"><span className="pulse-dot" /> {isIncident ? 'Payment path degraded' : 'All systems nominal'}</div>
              <p>{isIncident ? 'Latency and error budgets are burning after deploy pay-api@2.4.1.' : 'All 12 services are operating within their SLOs. No action required.'}</p>
            </div>
            <div className="pulse-meta"><span>12 services</span><span>3 regions</span><span>Updated now</span></div>
          </section>

          {isIncident && <section className="incident-alert" id="incident-workspace">
            <div className="severity">SEV-1</div>
            <div><p><span>INC-1042</span> · DETECTED 34 SECONDS AGO</p><h2>Payment processing failures after deployment</h2><small>Database pool saturation is impacting checkout in us-east-1.</small></div>
            <button onClick={() => setAnalysisOpen(true)}><span>✦</span> Investigate with AI</button>
          </section>}

          <section className="metric-grid" aria-label="Key metrics">
            {metrics.map((metric, index) => (
              <article className="metric-card" key={metric.label}>
                <div><p>{metric.label}</p><span className={isIncident && index > 0 ? 'trend danger' : 'trend'}>{isIncident && index === 1 ? '+733%' : isIncident && index === 2 ? '+14.1%' : isIncident && index === 3 ? '8.9×' : metric.trend}</span></div>
                <strong>{isIncident ? [31.6, 1840, 14.2, 8.9][index] : metric.value}<small>{metric.unit}</small></strong>
                <div className={`microbar m${index} ${isIncident ? 'hot' : ''}`} />
              </article>
            ))}
          </section>

          <section className="main-grid">
            <article className="panel performance-panel">
              <div className="panel-heading"><div><p className="eyebrow">TRAFFIC</p><h2>System performance</h2></div><div className="legend"><span><i className="violet" /> Requests</span><span><i className="cyan" /> Latency</span></div></div>
              <div className="chart" aria-label="Performance over the last hour">
                <div className="chart-labels"><span>10k</span><span>7.5k</span><span>5k</span><span>2.5k</span><span>0</span></div>
                <div className="plot"><div className={isIncident ? 'trace purple failing' : 'trace purple'} /><div className={isIncident ? 'trace blue spiking' : 'trace blue'} /></div>
                <div className="chart-times"><span>{timeRange === '24h' ? '10:30' : timeRange === '15m' ? '10:15' : '09:30'}</span><span>09:45</span><span>10:00</span><span>10:15</span><span>NOW</span></div>
              </div>
            </article>

            <article className="panel services-panel" id="service-map">
              <div className="panel-heading"><div><p className="eyebrow">DEPENDENCIES</p><h2>Service health</h2></div><button className="text-button" onClick={() => setToast('Service topology: 12 nodes · 18 dependencies')}>View map →</button></div>
              <div className="service-list">
                {services.map((service, index) => {
                  const affected = isIncident && index === 3;
                  return <div className="service-row" key={service.name}><span className={affected ? 'service-icon critical' : 'service-icon'}>{service.name[0]}</span><span><strong>{service.name}</strong><small>{service.meta}</small></span><span className="latency">{affected ? '2.3 s' : service.latency}</span><span className={affected ? 'status critical' : 'status'}><i />{affected ? 'Critical' : 'Healthy'}</span></div>;
                })}
              </div>
            </article>
          </section>

          <section className="panel logs-panel" id="live-traces">
            <div className="panel-heading logs-heading"><div><p className="eyebrow">TELEMETRY STREAM</p><h2>Live traces</h2><span>Normalized logs across every production service</span></div><div className="streaming"><i /> Streaming</div></div>
            <div className="table-wrap"><table><thead><tr><th>Timestamp</th><th>Level</th><th>Service</th><th>Event</th><th>Status</th><th>Duration</th></tr></thead><tbody>{(isIncident ? incidentLogs : healthyLogs).map((log, index) => <tr key={`${log[0]}-${index}`}><td><code>{log[0]}</code></td><td><span className={`log-level ${log[1].toLowerCase()}`}>{log[1]}</span></td><td>{log[2]}</td><td>{log[3]}</td><td>{log[4]}</td><td>{log[5]}</td></tr>)}</tbody></table></div>
          </section>

          <section className="portfolio-note" id="deployments"><span>Built for the on-call engineer</span><p>IncidentIQ connects telemetry, deployments, and service dependencies into one explainable incident narrative.</p><div><strong>42,891</strong><small>signals / min</small><strong>34 sec</strong><small>to detection</small><strong>87%</strong><small>RCA confidence</small></div></section>
        </div>
      </section>

      {analysisOpen && <div className="overlay" onMouseDown={(event) => { if (event.target === event.currentTarget) setAnalysisOpen(false); }}>
        <aside className="analysis-drawer" role="dialog" aria-modal="true" aria-labelledby="analysis-title">
          <div className="drawer-top"><div><p className="eyebrow">AI INCIDENT INVESTIGATION</p><h2 id="analysis-title">Root cause analysis</h2></div><button onClick={() => setAnalysisOpen(false)} aria-label="Close analysis">×</button></div>
          <div className="confidence"><div><span>87%</span><small>CONFIDENCE</small></div><p><strong>Database connection pool exhaustion</strong>The payment service began leaking connections after <code>pay-api@2.4.1</code>.</p></div>
          <div className="evidence"><h3>Correlated evidence</h3><ol><li><span>10:28:04</span><div><strong>Deployment completed</strong><small>pay-api@2.4.1 promoted to production</small></div></li><li><span>+ 02:11</span><div><strong>Pool utilization crossed 95%</strong><small>Connections were acquired but not released</small></div></li><li><span>+ 02:38</span><div><strong>Checkout latency spiked 733%</strong><small>Gateway propagated payment timeouts</small></div></li><li><span>+ 02:42</span><div><strong>Error budget began burning</strong><small>14.2% of payment requests failed</small></div></li></ol></div>
          <div className="recommendation"><span>RECOMMENDED ACTION</span><h3>Roll back pay-api@2.4.1</h3><p>Restores the last stable build while the connection lifecycle regression is investigated.</p><code>kubectl rollout undo deploy/payment-api</code></div>
          <div className="drawer-actions"><button className="ghost-button" onClick={() => setToast('Incident report copied to clipboard')}>Copy report</button><button className="primary-button" onClick={resolveIncident}>Run rollback</button></div>
          <p className="disclaimer">AI-generated analysis · Verify actions before executing in production.</p>
        </aside>
      </div>}

      {searchOpen && <div className="search-overlay" onMouseDown={(event) => { if (event.target === event.currentTarget) setSearchOpen(false); }}><div className="command" role="dialog" aria-modal="true" aria-label="Global search"><label><span>⌕</span><input autoFocus placeholder="Search services, traces, or incidents…" /></label><p>QUICK ACCESS</p>{['Payment Service · critical path', 'INC-1042 · Payment failures', 'Deployment pay-api@2.4.1'].map((result, index) => <button key={result} onClick={() => { setSearchOpen(false); if (index === 1 && isIncident) setAnalysisOpen(true); }}><span>{['◇', '!', '↗'][index]}</span>{result}<kbd>↵</kbd></button>)}<small>ESC to close · ⌘K to open anywhere</small></div></div>}
      {toast && <div className="toast" role="status"><span>✓</span>{toast}</div>}
    </main>
  );
}
