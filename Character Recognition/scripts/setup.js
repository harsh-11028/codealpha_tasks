/**
 * Automated One-Command Setup Script (triggered via 'npm install' or 'npm run setup').
 * Features dynamic PyTorch platform resolution, safe array process spawning without string concatenation,
 * and reliable creation of ./.venv strictly inside the project root directory.
 */

const fs = require('fs');
const path = require('path');
const os = require('os');
const {
  IS_WIN,
  ROOT_DIR,
  VENV_DIR,
  FRONTEND_DIR,
  colors,
  log,
  errorLog,
  getSystemPython,
  getVenvBinaries,
  runSafeSync
} = require('./utils');

async function configureWorkspace() {
  log("SYSTEM SETUP", "✨ Initializing automated environment detection & dynamic installation...", colors.magenta);
  log("WORKSPACE", `Absolute Root: ${ROOT_DIR}`, colors.dim);

  // 1. Detect and verify Node.js installation
  const nodeVersion = process.versions.node;
  const majorNode = parseInt(nodeVersion.split('.')[0], 10);
  if (majorNode < 18) {
    errorLog("NODE DETECT", `Node.js v18 or higher is required. Detected v${nodeVersion}. Please update Node.js.`);
    process.exit(1);
  }
  log("NODE DETECT", `✅ Node.js v${nodeVersion} verified`, colors.green);

  // 2. Detect System Python
  const sysPy = getSystemPython();
  if (!sysPy) {
    errorLog("PYTHON DETECT", "❌ Python 3.9+ was not found in PATH.");
    console.error(`
====================================================================
CRITICAL SETUP ERROR: Compatible Python not found.
Please install Python 3.10 or newer from https://www.python.org/downloads/
or via system package manager (brew install python3 / sudo apt install python3).
====================================================================
    `);
    process.exit(1);
  }
  log("PYTHON DETECT", `✅ Found ${sysPy.version} via command '${sysPy.command}'`, colors.green);

  // 3. Automated Virtual Environment (Venv) configuration at ./.venv
  const venvBins = getVenvBinaries();
  const venvRelativePath = "./" + path.relative(ROOT_DIR, VENV_DIR);

  if (!fs.existsSync(VENV_DIR) || !fs.existsSync(venvBins.python)) {
    log("VENV", `🐍 Virtual environment missing or incomplete. Creating at ${venvRelativePath}...`, colors.yellow);
    try {
      runSafeSync(sysPy.command, ["-m", "venv", ".venv"]);
      log("VENV", `✅ Virtual environment successfully generated at ${venvRelativePath}`, colors.green);
    } catch (err) {
      errorLog("VENV", `Failed to create virtual environment: ${err.message}`);
      console.error("If running Linux/Ubuntu, ensure 'python3-venv' package is installed (sudo apt install python3-venv).");
      process.exit(1);
    }
  } else {
    log("VENV", `✅ Active Python virtual environment detected at ${venvRelativePath}`, colors.green);
  }

  // 4. Verify Virtual Environment Pip & Upgrade
  if (!fs.existsSync(venvBins.pip)) {
    errorLog("PIP CHECK", `❌ Pip executable missing at ${venvBins.pip}. Please delete the .venv directory and re-run setup.`);
    process.exit(1);
  }

  log("PIP UPGRADE", "🔄 Upgrading virtual environment pip installer...", colors.cyan);
  try {
    runSafeSync(venvBins.pip, ["install", "--upgrade", "pip", "--quiet"]);
  } catch (err) {
    log("PIP UPGRADE", "⚠️ Non-critical warning: could not upgrade pip. Continuing with existing version...", colors.yellow);
  }

  // 5. Dynamic PyTorch & Deep Learning Installation
  const platform = os.platform();
  const arch = os.arch();
  log("PYTORCH RESOLUTION", `🎯 Analyzing host environment (Platform: ${platform.toUpperCase()} | Arch: ${arch} | Python: ${sysPy.major}.${sysPy.minor})...`, colors.cyan);
  
  try {
    log("PYTORCH INSTALL", "⚡ Dynamically fetching compatible PyTorch, TorchVision & TorchAudio wheels...", colors.blue);
    // Installing torch torchvision torchaudio directly allows pip to match the exact Python version (e.g. 3.13)
    // and architecture (such as macOS Apple Silicon arm64, Intel x64, or Windows/Linux x86_64) automatically.
    const torchPackages = ["torch", "torchvision", "torchaudio"];
    runSafeSync(venvBins.pip, ["install", ...torchPackages]);
    log("PYTORCH INSTALL", "✅ Native compatible PyTorch runtime installed successfully!", colors.green);
  } catch (err) {
    errorLog("PYTORCH INSTALL", `❌ Failed to install PyTorch runtime: ${err.message}`);
    process.exit(1);
  }

  // 6. Install Backend & OCR Requirements
  log("BACKEND INSTALL", "📦 Installing OpenCV, FastAPI, EasyOCR, Tesseract wrapper & support suites...", colors.cyan);
  try {
    runSafeSync(venvBins.pip, ["install", "-r", "requirements.txt"]);
    log("BACKEND INSTALL", "✅ Backend ML & Web frameworks configured cleanly!", colors.green);
  } catch (err) {
    errorLog("BACKEND INSTALL", `❌ Requirement installation failed: ${err.message}`);
    process.exit(1);
  }

  // 7. Install Frontend Node Dependencies
  const frontendModules = path.join(FRONTEND_DIR, 'node_modules');
  log("FRONTEND INSTALL", "🎨 Checking React TypeScript Studio packages in ./frontend...", colors.cyan);
  try {
    const npmCmd = IS_WIN ? "npm.cmd" : "npm";
    runSafeSync(npmCmd, ["install"], { cwd: FRONTEND_DIR });
    log("FRONTEND INSTALL", "✅ React Studio packages verified & ready!", colors.green);
  } catch (err) {
    errorLog("FRONTEND INSTALL", `❌ Frontend npm install error: ${err.message}`);
    process.exit(1);
  }

  // 8. Auto-generate environment settings file (.env) if absent
  const envPath = path.join(ROOT_DIR, '.env');
  if (!fs.existsSync(envPath)) {
    const defaultEnv = `APP_NAME="AI Handwritten OCR Studio"\nAPP_ENV=development\nDEBUG=True\nAPI_PORT=8000\nDATABASE_URL="sqlite:///./ocr_system.db"\nEASYOCR_ENABLED=True\nTESSERACT_ENABLED=True\n`;
    fs.writeFileSync(envPath, defaultEnv, 'utf8');
    log("ENV CONFIG", "✅ Auto-created initial project .env settings file", colors.green);
  }

  console.log(`
${colors.green}${colors.bright}====================================================================
🚀 ENTERPRISE SETUP COMPLETE — TRUE ONE-COMMAND WORKFLOW READY!
====================================================================${colors.reset}
Virtual Environment: ./ .venv (Isolated & ready)
PyTorch Architecture: Dynamic Native Match (${os.platform()} ${os.arch()})

To run both FastAPI backend and React frontend studio together:
👉 Run: ${colors.bright}${colors.cyan}npm run dev${colors.reset}
====================================================================
  `);
}

configureWorkspace().catch(err => {
  errorLog("FATAL", `Unexpected setup termination: ${err.message}`);
  process.exit(1);
});
