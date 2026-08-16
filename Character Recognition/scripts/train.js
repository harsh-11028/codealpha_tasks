/**
 * Automated deep learning model training orchestration using localized virtual environment.
 * Supports passing arbitrary arguments to the PyTorch trainer (e.g. --model crnn --epochs 20).
 */

const { colors, log, errorLog, getVenvBinaries, runSafeSync } = require('./utils');

function runModelTraining() {
  log("TRAINING ENGINE", "🔥 Initiating PyTorch Neural Network Training from localized ./.venv ...", colors.magenta);
  const venvBins = getVenvBinaries();
  
  // Parse user arguments passed via CLI (e.g. npm run train -- --model crnn --epochs 15)
  const userArgs = process.argv.slice(2);
  const trainArgs = ["-m", "models.training.train"];

  if (userArgs.length > 0) {
    log("TRAINING CONFIG", `Custom training parameters provided: ${userArgs.join(" ")}`, colors.cyan);
    trainArgs.push(...userArgs);
  } else {
    log("TRAINING CONFIG", "No arguments supplied. Defaulting to architecture: cnn_batchnorm (5 epochs)", colors.cyan);
    log("TRAINING TIP", "To customize training, run: npm run train -- --model <architecture> --epochs <N>", colors.yellow);
    trainArgs.push("--model", "cnn_batchnorm", "--epochs", "5");
  }

  log("DATASET LOADER", "Datasets (MNIST/EMNIST/IAM) will be verified or downloaded into ./models/datasets/raw/", colors.cyan);
  log("CHECKPOINTING", "Trained PyTorch weights will be saved directly to ./models/saved_models/", colors.cyan);

  try {
    runSafeSync(venvBins.python, trainArgs);
    console.log(`\n${colors.green}${colors.bright}✅ MODEL TRAINING SUCCESSFULLY COMPLETED! Checkpoints saved in ./models/saved_models/${colors.reset}\n`);
    log("NEXT STEPS", "Your Antigravity Studio will now automatically load and utilize these newly trained neural weights!", colors.green);
  } catch (err) {
    errorLog("TRAINING ERROR", `❌ Model training encountered an exception: ${err.message}`);
    process.exit(1);
  }
}

runModelTraining();
