/**
 * Automated test suite execution wrapper using safe non-concatenated process spawning.
 */

const { IS_WIN, FRONTEND_DIR, colors, log, errorLog, getVenvBinaries, runSafeSync } = require('./utils');

function executeVerifiedSuites() {
  log("TEST SUITE", "🧪 Initiating automated software engineering verification suite...", colors.magenta);
  const venvBins = getVenvBinaries();

  try {
    log("PYTEST", "Executing Computer Vision, OCR reconstruction & FastAPI unit suites from ./.venv ...", colors.cyan);
    runSafeSync(venvBins.pytest, ["tests/", "-v"]);
    log("PYTEST", "✅ All Python deep learning and backend API test suites PASSED!", colors.green);
  } catch (err) {
    errorLog("PYTEST", `❌ Python test suite encountered failures: ${err.message}`);
    process.exit(1);
  }

  try {
    log("REACT BUILD", "Verifying React TypeScript production compiler asset bundling...", colors.cyan);
    const npmCmd = IS_WIN ? "npm.cmd" : "npm";
    runSafeSync(npmCmd, ["run", "build"], { cwd: FRONTEND_DIR });
    log("REACT BUILD", "✅ React static bundle compiled cleanly without TypeScript errors!", colors.green);
  } catch (err) {
    errorLog("REACT BUILD", "❌ Frontend static build verification failed.");
    process.exit(1);
  }

  console.log(`\n${colors.green}${colors.bright}✅ ALL SYSTEM AUTOMATED TESTS & BUILDS PASSED CLEANLY!${colors.reset}\n`);
}

executeVerifiedSuites();
