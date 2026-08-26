# IncidentIQ

IncidentIQ is an interactive application-observability command center that turns noisy production telemetry into an explainable incident narrative. It demonstrates how platform teams can detect a regression, connect it to a deployment, understand the blast radius, and take a safe recovery action from one focused interface.

![IncidentIQ social preview](public/og.png)

## Why this project exists

On-call engineers rarely suffer from too little data. They suffer from context spread across metrics, traces, logs, deployments, and service maps. IncidentIQ explores a product response to that problem: correlate the evidence first, then present the most likely root cause with confidence, chronology, and a reversible recommended action.

## Product walkthrough

1. Start from a healthy 12-service production environment.
2. Select **Inject incident** to simulate a payment-service regression.
3. Observe SLO burn, latency, throughput, logs, and dependent services change together.
4. Open **Investigate with AI** to see the correlated evidence timeline and 87% confidence root cause.
5. Run the proposed rollback to return the system to a recovering state.

Press `⌘ K` (or `Ctrl K`) anywhere to open global search. The dashboard is responsive and supports keyboard navigation, reduced-motion preferences, accessible dialog semantics, live status announcements, and mobile-friendly tables.

## Engineering highlights

- Event-driven incident simulation with coordinated state across metrics, services, navigation, charts, and logs.
- Explainable RCA drawer that connects deployment, saturation, latency, and error-budget evidence.
- Keyboard-first command palette and semantic, accessible interaction patterns.
- Responsive operations UI designed for desktop command centers and mobile on-call use.
- Cloudflare Worker-compatible React output through Vinext and the OpenAI Sites Vite integration.
- Share-ready Open Graph artwork and metadata for portfolio and recruiter links.

## Stack

- React 19 + TypeScript
- Next-compatible App Router
- Vinext + Vite
- Tailwind CSS 4 processing with a custom CSS design system
- OpenAI Sites / Cloudflare Workers deployment target

## Run locally

Requirements: Node.js 22.13 or newer.

```bash
npm install
npm run dev
```

Open `http://localhost:3000`. For a production check:

```bash
npm run build
npm run lint
```

## Architecture

The demo intentionally keeps the incident dataset local and deterministic, which makes the recruiter walkthrough instant and repeatable. The UI models the boundaries a production version would use:

- **Signal ingestion:** OpenTelemetry logs, metrics, and distributed traces.
- **Correlation:** time-window joins across deploy events and service dependencies.
- **Incident state:** severity, SLO impact, affected services, and lifecycle.
- **Investigation:** evidence ranking, root-cause confidence, and operator-approved actions.

A production extension would move event state behind an API, stream telemetry through a queue, store time-series data in an observability backend, and require authenticated approval plus audit logging for remediation.

## Resume-ready summary

> Built IncidentIQ, a responsive React/TypeScript observability dashboard that simulates production incidents, correlates deployment and telemetry signals, and presents explainable root-cause analysis with operator-approved remediation. Designed accessible keyboard workflows and deployed a Worker-compatible build.

## Interview talking points

- Why correlation and evidence chronology are more trustworthy than an unexplained AI answer.
- How error-budget burn provides better operational context than raw error rate.
- Why remediation remains human-approved even when diagnosis is automated.
- How deterministic demo data improves portfolio reliability while preserving realistic system boundaries.

## Disclaimer

IncidentIQ is a portfolio simulation. Its metrics and AI analysis are deterministic demo data, and the displayed rollback command is never executed against infrastructure.
