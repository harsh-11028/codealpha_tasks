/**
 * Safe, cross-platform automation utilities.
 * Uses child_process.spawn and spawnSync with structured argument arrays to prevent
 * string concatenation or path splitting errors in directories containing spaces.
 */

const fs = require('fs');
const path = require('path');
const os = require('os');
const { spawn, spawnSync } = require('child_process');
const net = require('net');
const http = require('http');

const IS_WIN = os.platform() === 'win32';
const ROOT_DIR = path.resolve(__dirname, '..');
// Virtual environment created strictly inside project directory as ./ .venv
const VENV_DIR = path.join(ROOT_DIR, '.venv');
const FRONTEND_DIR = path.join(ROOT_DIR, 'frontend');

// Terminal color codes
const colors = {
  reset: "\x1b[0m",
  bright: "\x1b[1m",
  dim: "\x1b[2m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  blue: "\x1b[34m",
  magenta: "\x1b[35m",
  cyan: "\x1b[36m",
  red: "\x1b[31m"
};

function log(prefix, msg, color = colors.cyan) {
  const time = new Date().toLocaleTimeString();
  console.log(`${color}[${time}] [${prefix}]${colors.reset} ${msg}`);
}

function errorLog(prefix, msg) {
  const time = new Date().toLocaleTimeString();
  console.error(`${colors.red}[${time}] [ERROR: ${prefix}]${colors.reset} ${msg}`);
}

/**
 * Safe synchronous command execution using argument arrays.
 * Never concatenates string command lines; immune to directory spaces.
 */
function runSafeSync(executable, args = [], options = {}) {
  const cmdDisplay = `${path.basename(executable)} ${args.join(' ')}`;
  // For Windows `.cmd` files (like npm.cmd), shell mode ensures Windows batch execution
  const isWindowsBatch = IS_WIN && (executable.endsWith('.cmd') || executable.endsWith('.bat') || executable === 'npm');
  
  const result = spawnSync(executable, args, {
    cwd: ROOT_DIR,
    stdio: 'inherit',
    shell: isWindowsBatch,
    env: { ...process.env, ...options.env },
    ...options
  });

  if (result.error) {
    throw new Error(`Execution failed (${cmdDisplay}): ${result.error.message}`);
  }
  if (result.status !== 0) {
    throw new Error(`Command exited with status ${result.status} (${cmdDisplay})`);
  }
  return result;
}

/**
 * Detect system Python command (python3 or python) and extract exact version string.
 */
function getSystemPython() {
  const candidates = IS_WIN ? ['python', 'py', 'python3'] : ['python3', 'python'];
  for (const cmd of candidates) {
    try {
      const res = spawnSync(cmd, ['--version'], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
      if (res.status === 0 && (res.stdout || res.stderr)) {
        const output = (res.stdout || res.stderr).trim();
        const match = output.match(/Python (\d+)\.(\d+)(\.\d+)?/i);
        if (match) {
          const major = parseInt(match[1], 10);
          const minor = parseInt(match[2], 10);
          if (major === 3 && minor >= 9) {
            return { command: cmd, version: output, major, minor };
          }
        }
      }
    } catch (e) {
      // Command candidate not accessible
    }
  }
  return null;
}

/**
 * Resolve exact executable paths inside the project ./.venv directory
 */
function getVenvBinaries() {
  if (IS_WIN) {
    return {
      python: path.join(VENV_DIR, 'Scripts', 'python.exe'),
      pip: path.join(VENV_DIR, 'Scripts', 'pip.exe'),
      uvicorn: path.join(VENV_DIR, 'Scripts', 'uvicorn.exe'),
      pytest: path.join(VENV_DIR, 'Scripts', 'pytest.exe')
    };
  } else {
    return {
      python: path.join(VENV_DIR, 'bin', 'python3'),
      pip: path.join(VENV_DIR, 'bin', 'pip'),
      uvicorn: path.join(VENV_DIR, 'bin', 'uvicorn'),
      pytest: path.join(VENV_DIR, 'bin', 'pytest')
    };
  }
}

/**
 * Check if a network port is currently active on localhost
 */
function isPortOccupied(port) {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.once('error', (err) => {
      if (err.code === 'EADDRINUSE') {
        resolve(true);
      } else {
        resolve(false);
      }
    });
    server.once('listening', () => {
      server.close();
      resolve(false);
    });
    server.listen(port, '127.0.0.1');
  });
}

/**
 * Actively poll FastAPI health endpoint until HTTP 200 or timeout occurs
 */
function waitForBackendHealth(url = 'http://127.0.0.1:8000/api/health', timeoutMs = 40000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const interval = setInterval(() => {
      if (Date.now() - start > timeoutMs) {
        clearInterval(interval);
        return reject(new Error(`Timeout waiting for backend service at ${url}`));
      }
      http.get(url, (res) => {
        if (res.statusCode === 200) {
          let data = '';
          res.on('data', chunk => { data += chunk; });
          res.on('end', () => {
            try {
              const json = JSON.parse(data);
              if (json.status === 'ok') {
                clearInterval(interval);
                resolve(json);
              }
            } catch (e) {
              // Waiting for JSON readiness
            }
          });
        }
      }).on('error', () => {
        // Backend socket not open yet; keep polling
      });
    }, 500);
  });
}

/**
 * Spawn long-running background service using safe array arguments & color-coded logs
 */
function spawnService(name, executable, args = [], options = {}, color = colors.green) {
  const isWindowsBatch = IS_WIN && (executable.endsWith('.cmd') || executable.endsWith('.bat') || executable === 'npm');

  const child = spawn(executable, args, {
    cwd: ROOT_DIR,
    shell: isWindowsBatch,
    env: { ...process.env, FORCE_COLOR: '1', ...options.env },
    ...options
  });

  child.stdout?.on('data', (data) => {
    const lines = data.toString().trim().split('\n');
    for (const line of lines) {
      if (line.trim()) {
        console.log(`${color}[${name}]${colors.reset} ${line.trim()}`);
      }
    }
  });

  child.stderr?.on('data', (data) => {
    const lines = data.toString().trim().split('\n');
    for (const line of lines) {
      if (line.trim()) {
        console.log(`${colors.yellow}[${name}]${colors.reset} ${line.trim()}`);
      }
    }
  });

  child.on('error', (err) => {
    errorLog(name, `Process spawn failure: ${err.message}`);
  });

  return child;
}

module.exports = {
  IS_WIN,
  ROOT_DIR,
  VENV_DIR,
  FRONTEND_DIR,
  colors,
  log,
  errorLog,
  runSafeSync,
  getSystemPython,
  getVenvBinaries,
  isPortOccupied,
  waitForBackendHealth,
  spawnService
};
