# Training Guide — Dataset Download & Model Training

This guide walks you through downloading a free dataset and training your first emotion recognition model.

---

## Step 1: Download the RAVDESS Dataset (Free)

RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) contains 1,440 audio files across 8 emotions recorded by 24 professional actors.

1. Go to: https://zenodo.org/record/1188976
2. Download **Audio_Speech_Actors_01-24.zip** (around 215 MB)
3. Unzip it. You will get a folder structure like:
   ```
   Actor_01/
     03-01-01-01-01-01-01.wav
     03-01-02-01-01-01-01.wav
     ...
   Actor_02/
   ...
   ```
4. Move the extracted `Actor_01` through `Actor_24` folders into:
   ```
   backend/data/raw/ravdess/
   ```

---

## Step 2: Run the Dataset Pipeline

This script reads the raw audio files, maps them to emotion labels, and creates stratified train/validation/test splits.

```bash
cd "/Users/harshmac/Desktop/Emotion Recognition/speech-emotion-recognition/backend"
source venv/bin/activate

python -m dataset.pipeline --datasets ravdess --raw-dir data/raw --out-dir data/processed
```

After this runs, you will see:
```
backend/data/processed/
  metadata.csv     ← all samples with labels and splits
  features.h5      ← pre-extracted audio features
```

---

## Step 3: Train a Model

Start with the **CNN** model — it is fast to train and gives good accuracy:

```bash
python training/train.py \
  --model cnn \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.001
```

To train the best model (Wav2Vec 2.0 transfer learning — takes longer but gets 90%+ accuracy):

```bash
python training/train.py \
  --model wav2vec2 \
  --epochs 20 \
  --batch-size 16 \
  --lr 0.00005
```

Training artifacts are saved to:
- `backend/saved_models/best_model.pt` — best model checkpoint
- `backend/runs/` — TensorBoard logs
- `backend/logs/` — training CSV logs

---

## Step 4: Monitor Training with TensorBoard

```bash
pip install tensorboard
tensorboard --logdir backend/runs/
```

Open http://localhost:6006 to see loss and accuracy curves in real time.

---

## Step 5: Evaluate the Trained Model

```bash
python training/evaluate.py --model-path saved_models/best_model.pt --output-dir metrics/
```

This saves a **confusion matrix plot** to `metrics/confusion_matrix_cnn.png`.

---

## Step 6: Restart the Backend

Once training is complete, restart the backend and it will automatically load your trained model:

```bash
./run.sh
```

The Dashboard page will show **"Model Loaded"** and you can now use the live microphone and file upload features for real predictions!

---

## Tips

| Model | Training Time (CPU) | Expected Accuracy |
|-------|-------------------|-------------------|
| CNN | ~15 min (50 epochs) | ~75-80% |
| CNN+LSTM | ~25 min | ~80-85% |
| BiLSTM | ~20 min | ~72-77% |
| CNN+Attention | ~30 min | ~83-87% |
| Wav2Vec2 | ~2-3 hrs | ~88-92% |

- Use a **GPU** if available — training will be 10-20x faster
- Add more datasets (TESS, CREMA-D) for higher accuracy
- The best model checkpoint is automatically saved every epoch if validation F1 improves
