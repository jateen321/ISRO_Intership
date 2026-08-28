# SENTINEL-OPS: Autonomous Vision-Language Tactical Defense & Incident Investigation System

> **BSERC Defence Space Summer Internship & ISRO Project Submission**  
> **Domain**: Artificial Intelligence (AI/ML), Computer Vision & Autonomous Agent Systems  
> **Hardware Requirements**: Laptop, Webcam, Network Connection

---

## 📌 Project Overview
**SENTINEL-OPS** is an autonomous multi-modal AI defense intelligence and perimeter surveillance system. It integrates **real-time video perception (Laptop Camera / OpenCV)**, custom **Deep Learning CNN threat scoring (PyTorch `ThreatSeverityNet`)**, and an **Autonomous ReAct AI Agent loop** with dynamic tool execution.

The project is hosted in an interactive **Next.js 16 + React 19 + TailwindCSS** tactical operations dashboard.

---

## 🏛️ System Architecture

```
+--------------------------+     +-------------------------------+     +----------------------------------+
| Laptop Camera /          | --> | PyTorch CNN Classifier        | --> | Autonomous ReAct AI Agent        |
| Real-Time Video Feed     |     | (ThreatSeverityNet.py)        |     | (app/api/agent/route.ts)         |
+--------------------------+     +-------------------------------+     +----------------------------------+
                                                                                       |
                                                                                       | (Invokes Tactical Tools)
                                                                                       v
                                                                      +-----------------------------------+
                                                                      | Tool Execution Suite              |
                                                                      | - tool_run_pytorch_classifier     |
                                                                      | - tool_escalate_defcon            |
                                                                      | - tool_query_threat_intel_db      |
                                                                      | - tool_log_telemetry              |
                                                                      +-----------------------------------+
                                                                                       |
                                                                                       v
                                                                      +-----------------------------------+
                                                                      | Next.js 16 Tactical Ops UI        |
                                                                      | (http://localhost:3000)           |
                                                                      +-----------------------------------+
```

---

## 🚀 Key Technical Features

1. **Edge Computer Vision**:
   - Direct laptop webcam integration via browser `navigator.mediaDevices.getUserMedia`.
   - Real-time sector monitoring and passive perimeter anomaly triggers.

2. **Custom PyTorch Deep Learning Classifier (`ThreatSeverityNet.py`)**:
   - 3-layer Convolutional feature extractor (`nn.Conv2d`, `BatchNorm2d`, `MaxPool2d`, `AdaptiveAvgPool2d`).
   - Binary threat probability head (`Linear` + `Sigmoid`) outputting real-time confidence scores.

3. **Autonomous AI Defense Agent (ReAct Loop)**:
   - Evaluates multi-modal visual observations using ReAct (`Thought` $\rightarrow$ `Action` $\rightarrow$ `Tool Output`).
   - Dynamically manages security readiness (`DEFCON-5 Nominal`, `DEFCON-3 Elevated`, `DEFCON-1 Critical`).
   - Automatically generates structured military intelligence briefings and incident telemetry logs.

---

## 🛠️ Tech Stack

* **Frontend**: Next.js 16, React 19, TypeScript, TailwindCSS 4, Vite
* **Deep Learning Framework**: PyTorch (`torch`, `torchvision`, `torch.nn.Module`)
* **Computer Vision**: OpenCV, Browser MediaStream API
* **Backend**: Next.js Server App Router API (`/api/agent`)

---

## 💻 How to Run

### 1. Launch the Next.js Operations Dashboard
```bash
npm run dev
```
Open **`http://localhost:3000`** in your browser.
- Click **"Start Laptop Camera"** to initialize live video perception.
- Click **"Command Agent"** or enable **"Autonomous Sentinel Scan"** to observe the AI Agent's real-time Chain of Thought.

### 2. Run Standalone PyTorch Deep Learning Test
```bash
python3 ThreatSeverityNet.py
```

---

## 📋 BSERC Submission Checklist
- [x] Project Title & Domain Classification (AI/ML & Defence Space)
- [x] Functional Prototype with Laptop Webcam Integration
- [x] PyTorch Deep Learning Neural Network Architecture
- [x] Autonomous AI Agent Tool Calling Suite
- [x] Comprehensive Technical Documentation
