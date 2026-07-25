import os
import time
import json
import torch
import uvicorn
import urllib.request
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from slm_core.model_loader import generate
from slm_core.candidate_evaluator import CandidateEvaluator

app = FastAPI(
    title="Rez_SLM Local AI Engine API & Web UI",
    description="Local REST API Server serving Gemma 4 QLoRA Fine-Tuned Candidate Evaluator & SLM Engine.",
    version="1.2.0"
)

# Enable CORS for external web/desktop apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

evaluator = CandidateEvaluator(model_generate_fn=generate)

# Secret API Key to prevent unauthorized access from untrusted processes
API_KEY_SECRET = os.getenv("REZ_SLM_API_KEY", "rez_slm_secure_local_key_2026")

# Registered microservices for health monitoring
MICROSERVICES = {
    "main_server": "http://127.0.0.1:8000/health",
    "candidate_microservice": "http://127.0.0.1:8001/health",
    "recruiter_microservice": "http://127.0.0.1:8002/health",
    "prochat_frontend": "http://127.0.0.1:8080",
}

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized API Key")
    return x_api_key

# Request Schemas
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 256

class CandidateItem(BaseModel):
    name: str
    resume: str

class EvaluateCandidateRequest(BaseModel):
    job_description: str
    candidates: List[CandidateItem]

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def get_control_center_dashboard():
    """Serves the rich glassmorphic Web UI Control Center Dashboard"""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Rez_SLM Control Center & Cluster Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #070b14;
      --card-bg: rgba(15, 23, 42, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent-gold: #eab308;
      --accent-cyan: #06b6d4;
      --accent-green: #10b981;
      --accent-red: #ef4444;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', sans-serif; }
    body {
      background: var(--bg-dark);
      background-image: 
        radial-gradient(at 10% 20%, rgba(6, 182, 212, 0.12) 0px, transparent 50%),
        radial-gradient(at 90% 80%, rgba(234, 179, 8, 0.1) 0px, transparent 50%),
        radial-gradient(at 50% 50%, rgba(15, 23, 42, 0.9) 0px, transparent 100%);
      color: var(--text-main);
      min-height: 100vh;
      padding: 24px;
    }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--card-border);
    }
    .title-group h1 {
      font-size: 24px;
      font-weight: 800;
      background: linear-gradient(135deg, #fff 30%, var(--accent-cyan));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .title-group p { font-size: 13px; color: var(--text-muted); margin-top: 4px; }
    .badge-live {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: var(--accent-green);
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 600;
    }
    .pulse-dot {
      width: 8px; height: 8px; background: var(--accent-green); border-radius: 50%;
      box-shadow: 0 0 8px var(--accent-green);
      animation: pulse 1.8s infinite;
    }
    @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.85); } }
    
    .nav-links { display: flex; gap: 12px; }
    .btn-nav {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--card-border);
      color: var(--text-main);
      padding: 8px 16px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      text-decoration: none;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .btn-nav:hover {
      background: rgba(6, 182, 212, 0.15);
      border-color: rgba(6, 182, 212, 0.4);
      color: #fff;
    }

    .grid-dashboard {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 20px;
      margin-bottom: 24px;
    }
    .card {
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 20px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    .card-title {
      font-size: 14px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .stat-value { font-size: 28px; font-weight: 800; color: #fff; margin-bottom: 6px; }
    .stat-sub { font-size: 12px; color: var(--text-muted); }

    .meter-bar-container {
      width: 100%;
      height: 10px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 5px;
      overflow: hidden;
      margin: 12px 0 6px;
    }
    .meter-bar-fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--accent-cyan), var(--accent-gold));
      transition: width 0.5s ease;
    }

    .service-list { display: flex; flex-direction: column; gap: 10px; }
    .service-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px 14px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 8px;
    }
    .service-info { display: flex; flex-direction: column; }
    .service-name { font-size: 13px; font-weight: 600; color: #fff; }
    .service-url { font-size: 11px; font-family: 'JetBrains Mono', monospace; color: var(--text-muted); }
    .status-pill {
      font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 12px;
      display: flex; align-items: center; gap: 5px;
    }
    .status-online { background: rgba(16, 185, 129, 0.15); color: var(--accent-green); border: 1px solid rgba(16, 185, 129, 0.3); }
    .status-offline { background: rgba(239, 68, 68, 0.15); color: var(--accent-red); border: 1px solid rgba(239, 68, 68, 0.3); }

    .console-card { grid-column: 1 / -1; }
    .console-input {
      width: 100%;
      background: rgba(0, 0, 0, 0.4);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 12px;
      color: #fff;
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px;
      margin-bottom: 12px;
      outline: none;
    }
    .console-btn {
      background: linear-gradient(135deg, var(--accent-cyan), #0284c7);
      border: none;
      color: #fff;
      font-weight: 700;
      padding: 10px 20px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 13px;
      transition: transform 0.1s;
    }
    .console-btn:hover { transform: translateY(-1px); }
    .console-output {
      margin-top: 14px;
      background: #040711;
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 8px;
      padding: 14px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      color: #a5f3fc;
      max-height: 200px;
      overflow-y: auto;
      white-space: pre-wrap;
    }
  </style>
</head>
<body>
  <div class="header">
    <div class="title-group">
      <h1>🚀 Rez_SLM Control Center</h1>
      <p>Multi-Tenant Local Engine & Microservices Cluster Web UI Dashboard</p>
    </div>
    <div style="display: flex; align-items: center; gap: 16px;">
      <span class="badge-live"><span class="pulse-dot"></span> Cluster Online</span>
      <div class="nav-links">
        <a href="http://127.0.0.1:8080" target="_blank" class="btn-nav">💬 Open ProChat App</a>
        <a href="/docs" target="_blank" class="btn-nav">⚡ Swagger API Docs</a>
      </div>
    </div>
  </div>

  <div class="grid-dashboard">
    <!-- GPU VRAM Card -->
    <div class="card">
      <div class="card-title">
        <span>GPU Acceleration (RTX 4070 SUPER)</span>
        <span id="gpu-name-tag" style="font-size: 11px; color: var(--accent-cyan);">CUDA 12.4</span>
      </div>
      <div class="stat-value" id="vram-val">-- MB</div>
      <div class="stat-sub" id="vram-sub">Allocated VRAM Memory</div>
      <div class="meter-bar-container">
        <div class="meter-bar-fill" id="vram-fill"></div>
      </div>
      <div class="stat-sub" style="margin-top: 6px;" id="vram-total-str">Total VRAM: 12,281.5 MB</div>
    </div>

    <!-- Cluster Microservices Card -->
    <div class="card" style="grid-column: span 2;">
      <div class="card-title">
        <span>Microservice Cluster Health</span>
        <span id="cluster-count" style="font-size: 11px; color: var(--accent-green);">4/4 Online</span>
      </div>
      <div class="service-list" id="services-container">
        <div style="color: var(--text-muted); font-size: 13px;">Polling microservices...</div>
      </div>
    </div>
  </div>

  <!-- Interactive Test Console -->
  <div class="card console-card">
    <div class="card-title">⚡ Interactive SLM Generation Console</div>
    <input type="text" id="prompt-input" class="console-input" placeholder="Type a prompt to send to local Gemma 4 SLM engine..." value="Analyze candidate summary: High concurrency Go developer with PyTorch skills.">
    <button class="console-btn" onclick="sendPrompt()">Run Inference</button>
    <div class="console-output" id="console-output">Output will appear here...</div>
  </div>

  <script>
    async function updateDashboard() {
      try {
        // Fetch GPU / Engine status
        const statusRes = await fetch('/v1/status');
        const statusData = await statusRes.json();
        if (statusData.gpu_metrics) {
          const metrics = statusData.gpu_metrics;
          document.getElementById('gpu-name-tag').innerText = metrics.gpu_name || 'CUDA Device';
          document.getElementById('vram-val').innerText = metrics.vram_allocated_mb + ' MB';
          document.getElementById('vram-sub').innerText = `Reserved: ${metrics.vram_reserved_mb} MB | Free: ${metrics.vram_free_mb} MB`;
          document.getElementById('vram-fill').style.width = metrics.vram_utilization_pct + '%';
          document.getElementById('vram-total-str').innerText = `VRAM Utilization: ${metrics.vram_utilization_pct}% of ${metrics.vram_total_mb} MB`;
        }

        // Fetch Microservices state
        const svcsRes = await fetch('/v1/microservices');
        const svcsData = await svcsRes.json();
        document.getElementById('cluster-count').innerText = `${svcsData.online_services} Online (${svcsData.cluster_health.toUpperCase()})`;

        const container = document.getElementById('services-container');
        container.innerHTML = '';
        
        for (const [name, info] of Object.entries(svcsData.microservices)) {
          const isOnline = info.status === 'online';
          const item = document.createElement('div');
          item.className = 'service-item';
          item.innerHTML = `
            <div class="service-info">
              <div class="service-name">${name.replace('_', ' ').toUpperCase()}</div>
              <div class="service-url">${info.target_url}</div>
            </div>
            <div style="display: flex; align-items: center; gap: 12px;">
              <span style="font-size: 11px; font-family: monospace; color: var(--text-muted);">${info.latency_ms} ms</span>
              <span class="status-pill ${isOnline ? 'status-online' : 'status-offline'}">
                ${isOnline ? '● ONLINE' : '✖ OFFLINE'}
              </span>
            </div>
          `;
          container.appendChild(item);
        }
      } catch (err) {
        console.error('Dashboard refresh error:', err);
      }
    }

    async function sendPrompt() {
      const prompt = document.getElementById('prompt-input').value;
      const outputEl = document.getElementById('console-output');
      outputEl.innerText = 'Generating response from local Gemma 4 model...';
      try {
        const res = await fetch('/v1/generate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-API-Key': 'rez_slm_secure_local_key_2026'
          },
          body: JSON.stringify({ prompt: prompt, max_tokens: 200 })
        });
        const data = await res.json();
        outputEl.innerText = JSON.stringify(data, null, 2);
      } catch (err) {
        outputEl.innerText = 'Error: ' + err.message;
      }
    }

    // Initial load + interval poll
    updateDashboard();
    setInterval(updateDashboard, 3000);
  </script>
</body>
</html>"""
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/health")
def health_check():
    """Simple health check endpoint"""
    return {"status": "online", "engine": "Rez_SLM Gemma 4 RTX 4070 SUPER"}

@app.get("/status")
@app.get("/v1/status")
def get_server_status():
    """Detailed server status including CUDA VRAM memory usage and engine state"""
    is_cuda = torch.cuda.is_available()
    gpu_info = {}
    if is_cuda:
        device_idx = 0
        allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device_idx) / (1024 ** 2)
        total_vram = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 2)
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(device_idx),
            "vram_allocated_mb": round(allocated, 2),
            "vram_reserved_mb": round(reserved, 2),
            "vram_total_mb": round(total_vram, 2),
            "vram_free_mb": round(total_vram - reserved, 2),
            "vram_utilization_pct": round((reserved / total_vram) * 100, 1),
        }

    return {
        "status": "online",
        "service": "Rez_SLM Core Inference Engine",
        "version": "1.1.0",
        "cuda_available": is_cuda,
        "gpu_metrics": gpu_info,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

@app.get("/services")
@app.get("/v1/services")
@app.get("/v1/microservices")
def check_microservices_state():
    """Pings and reports the live state of all registered cluster microservices"""
    results = {}
    online_count = 0

    for service_name, url in MICROSERVICES.items():
        start_time = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Rez-SLM-Monitor/1.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                elapsed_ms = round((time.time() - start_time) * 1000, 2)
                code = resp.getcode()
                content_type = resp.headers.get("Content-Type", "")

                data = None
                if "application/json" in content_type:
                    try:
                        data = json.loads(resp.read().decode("utf-8"))
                    except Exception:
                        data = None

                online_count += 1
                results[service_name] = {
                    "status": "online",
                    "http_code": code,
                    "latency_ms": elapsed_ms,
                    "target_url": url,
                    "payload": data
                }
        except Exception as e:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            results[service_name] = {
                "status": "offline",
                "error": str(e),
                "latency_ms": elapsed_ms,
                "target_url": url,
            }

    total_services = len(MICROSERVICES)
    if online_count == total_services:
        cluster_health = "healthy"
    elif online_count > 0:
        cluster_health = "degraded"
    else:
        cluster_health = "critical"

    return {
        "cluster_health": cluster_health,
        "online_services": f"{online_count}/{total_services}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "microservices": results,
    }

@app.get("/system")
@app.get("/v1/system")
def get_system_info():
    """Returns underlying environment specifications"""
    return {
        "cpu_cores": os.cpu_count(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "python_pid": os.getpid(),
    }

@app.post("/v1/generate")
def api_generate(req: GenerateRequest, api_key: str = Depends(verify_api_key)):
    try:
        response = generate(req.prompt, max_tokens=req.max_tokens)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/evaluate_candidates")
def api_evaluate(req: EvaluateCandidateRequest, api_key: str = Depends(verify_api_key)):
    try:
        candidate_list = [{"name": c.name, "resume": c.resume} for c in req.candidates]
        ranked_results = evaluator.rank_candidates(req.job_description, candidate_list)
        return {"job_description": req.job_description, "results": ranked_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("[Rez_SLM API Engine] Starting Local API Server on http://127.0.0.1:5000 ...")
    uvicorn.run(app, host="127.0.0.1", port=5000)

