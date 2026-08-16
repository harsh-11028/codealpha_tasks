/**
 * One-command development orchestrator (npm run dev).
 * Spawns FastAPI backend & React frontend concurrently using safe non-concatenated argument arrays,
 * verifies ./.venv presence, synchronizes startup via HTTP health checks, and opens browser automatically.
 */

const fs = require('fs');
const path = require('path');
const {
  IS_WIN,
  ROOT_DIR,
  VENV_DIR,
  FRONTEND_DIR,
  colors,
  log,
  errorLog,
  getVenvBinaries,
  isPortOccupied,
  waitForBackendHealth,
  runSafeSync,
  spawnService
} = require('./utils');

async function launchDevelopmentStudio() {
  log("ORCHESTRATOR", "🚀 Booting AI Handwritten OCR Production Studio...", colors.magenta);

  // 1. Pre-flight Check: Ensure virtual environment ./.venv and frontend node_modules exist
  const venvBins = getVenvBinaries();
  const frontendModules = path.join(FRONTEND_DIR, 'node_modules');

  if (!fs.existsSync(VENV_DIR) || !fs.existsSync(venvBins.python) || !fs.existsSync(frontendModules)) {
    log("PRE-FLIGHT", "⚠️ Required virtual environment (.venv) or node_modules incomplete. Running automatic setup...", colors.yellow);
    runSafeSync("node", ["scripts/setup.js"]);
  }

  // 2. TCP Port Occupancy Verification
  const isBackendRunning = await isPortOccupied(8000);
  const isFrontendRunning = await isPortOccupied(5173);

  if (isBackendRunning) {
    log("PORT CHECK", "⚠️ Port 8000 is already active! Assuming an external FastAPI process is running.", colors.yellow);
  }
  if (isFrontendRunning) {
    log("PORT CHECK", "⚠️ Port 5173 is currently occupied by another development server.", colors.yellow);
  }

  const activeChildProcesses = [];

  // 3. Launch FastAPI Hot-Reload Backend (using direct .venv uvicorn executable without shell string parsing)
  if (!isBackendRunning) {
    log("FASTAPI", "⚡ Spawning FastAPI Hot-Reload Server from ./.venv on port 8000...", colors.blue);
    const uvicornArgs = ["backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"];
    
    const backendChild = spawnService("Backend", venvBins.uvicorn, uvicornArgs, {}, colors.blue);
    activeChildProcesses.push(backendChild);
  }

  // 4. Intelligent Health Synchronization (Waits for API & Neural Models to warm up before showing UI)
  log("HEALTH CHECK", "⏳ Awaiting backend API neural model warmup & SQLite schema readiness...", colors.cyan);
  try {
    const health = await waitForBackendHealth("http://127.0.0.1:8000/api/health", 45000);
    log("HEALTH CHECK", `✅ Backend Server Active [Status: ${health.status.toUpperCase()} | Compute Engine: ${health.device || 'AUTO'}]`, colors.green);
  } catch (err) {
    errorLog("HEALTH CHECK", "❌ Backend did not report ready within timeout period. Please check terminal logs above for Python stack traces.");
    log("HEALTH CHECK", "Proceeding to open Frontend UI so you can inspect interface...", colors.yellow);
  }

  // 5. Launch React Vite Dev Server & Automatically Open Default Web Browser
  log("VITE STUDIO", "🌐 Launching React TypeScript Studio & opening browser at http://localhost:5173 ...", colors.cyan);
  const npmCmd = IS_WIN ? "npm.cmd" : "npm";
  const viteArgs = ["run", "dev", "--", "--open", "http://localhost:5173"];

  const frontendChild = spawnService("React-UI", npmCmd, viteArgs, { cwd: FRONTEND_DIR }, colors.cyan);
  activeChildProcesses.push(frontendChild);

  console.log(`
${colors.green}${colors.bright}====================================================================
🌟 ENTERPRISE STUDIO LIVE & SYNCHRONIZED IN UNISON!
====================================================================${colors.reset}
👉 Interactive Studio UI:  ${colors.bright}http://localhost:5173${colors.reset}
👉 OpenAPI Documentation:  ${colors.bright}http://localhost:8000/docs${colors.reset}
👉 SQLite DB Repository:   ./ocr_system.db

${colors.dim}Press Ctrl+C in this terminal to gracefully stop both servers.${colors.reset}
====================================================================
  `);

  // 6. Graceful Process Termination Handler
  const shutdown = (signal) => {
    console.log(`\n${colors.yellow}[ORCHESTRATOR] Received ${signal} — performing clean shutdown of backend and frontend services...${colors.reset}`);
    for (const child of activeChildProcesses) {
      if (child && !child.killed) {
        try {
          if (IS_WIN) {
            runSafeSync("taskkill", ["/pid", child.pid.toString(), "/T", "/F"], { stdio: 'ignore' });
          } else {
            child.kill("SIGTERM");
          }
        } catch (e) {
          // Process already ended
        }
      }
    }
    process.exit(0);
  };

  process.on('SIGINT', () => shutdown('SIGINT'));
  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('exit', () => shutdown('EXIT'));
}

launchDevelopmentStudio().catch(err => {
  errorLog("FATAL", `Orchestrator error: ${err.message}`);
  process.exit(1);
});
