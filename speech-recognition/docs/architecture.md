# System Architecture — Speech Emotion Recognition

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Browser (Client)                         │
│                                                              │
│  React 18 + TypeScript + Vite                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ Landing  │ │Dashboard │ │Prediction│ │ History  │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│  ┌───────────────────────────────────────────────────┐       │
│  │  State: Zustand | Charts: Recharts | Anim: Framer │       │
│  └───────────────────────────────────────────────────┘       │
└──────────────────────┬───────────────────────────────────────┘
                       │ HTTPS / WSS
                       │
┌──────────────────────▼───────────────────────────────────────┐
│                    Nginx Reverse Proxy                        │
│  Static files served | /api/* → backend | /ws/* → backend   │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│                   FastAPI Backend                             │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │               API Layer (routes/)                    │    │
│  │  POST /predict   POST /predict-live   POST /upload   │    │
│  │  GET  /health    GET  /model-info     GET  /metrics  │    │
│  │  GET  /history   WS   /ws/stream                     │    │
│  └───────────────────┬─────────────────────────────────┘    │
│                      │                                       │
│  ┌───────────────────▼─────────────────────────────────┐    │
│  │           ML Pipeline (ml/)                          │    │
│  │                                                      │    │
│  │  ┌─────────────────────────────────────────────┐    │    │
│  │  │ 1. Audio Preprocessing                       │    │    │
│  │  │    Load → Resample → Denoise → Normalize      │    │    │
│  │  │    Trim silence → Pad/Trim to fixed length    │    │    │
│  │  └──────────────────┬──────────────────────────┘    │    │
│  │                     │                                │    │
│  │  ┌──────────────────▼──────────────────────────┐    │    │
│  │  │ 2. Feature Extraction                         │    │    │
│  │  │    MFCC + Δ + ΔΔ | Mel Spec | Chroma         │    │    │
│  │  │    ZCR | Spectral | Tonnetz | Pitch           │    │    │
│  │  └──────────────────┬──────────────────────────┘    │    │
│  │                     │                                │    │
│  │  ┌──────────────────▼──────────────────────────┐    │    │
│  │  │ 3. Model Inference                            │    │    │
│  │  │  ┌─────────┐ ┌──────────┐ ┌──────────────┐  │    │    │
│  │  │  │  CNN    │ │CNN+LSTM  │ │   BiLSTM     │  │    │    │
│  │  │  └─────────┘ └──────────┘ └──────────────┘  │    │    │
│  │  │  ┌────────────────┐ ┌────────────────────┐   │    │    │
│  │  │  │ CNN+Attention  │ │   Wav2Vec2 (TL)    │   │    │    │
│  │  │  └────────────────┘ └────────────────────┘   │    │    │
│  │  │         ↓ Best model auto-selected             │    │    │
│  │  └──────────────────┬──────────────────────────┘    │    │
│  │                     │                                │    │
│  │  ┌──────────────────▼──────────────────────────┐    │    │
│  │  │ 4. Prediction Engine + XAI                   │    │    │
│  │  │    Softmax | Confidence | Feature importance  │    │    │
│  │  └─────────────────────────────────────────────┘    │    │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Database (SQLAlchemy)                              │     │
│  │  SQLite (dev)  →  PostgreSQL (prod)                 │     │
│  │  Tables: predictions | uploaded_files | models      │     │
│  └────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

## Data Flow — Prediction Request

```
1. User records/uploads audio
        ↓
2. Frontend sends POST /predict (multipart/form-data)
        ↓
3. Security middleware validates file type, size, MIME
        ↓
4. AudioProcessor: load → resample to 22050Hz → denoise → trim silence → normalize → pad/crop to 6s
        ↓
5. FeatureExtractor: compute MFCC/Mel/Chroma/... → NumPy array
        ↓
6. PredictionEngine.predict(features) → Model forward pass → Softmax
        ↓
7. Return PredictionResponse: {emotion, confidence, probabilities, explanation, spectrogram_data}
        ↓
8. Save to DB (prediction_history)
        ↓
9. Frontend renders emotion card + charts
```

## Data Flow — Live Streaming

```
1. User clicks "Record"
        ↓
2. Browser WebAudio API captures mic → 3s chunks
        ↓
3. WebSocket send: binary PCM data
        ↓
4. Server: /ws/stream receives chunk → AudioProcessor → Features → Model
        ↓
5. Server sends back: {emotion, confidence, probabilities} JSON
        ↓
6. Frontend updates real-time emotion display
```

## Dataset Normalization

Different datasets use different emotion naming conventions. All are normalized to our 8-class schema:

| Dataset | Raw Label | Normalized |
|---------|-----------|-----------|
| RAVDESS | 01 | neutral |
| RAVDESS | 02 | calm |
| RAVDESS | 03 | happy |
| RAVDESS | 04 | sad |
| RAVDESS | 05 | angry |
| RAVDESS | 06 | fear |
| RAVDESS | 07 | disgust |
| RAVDESS | 08 | surprise |
| TESS | neutral | neutral |
| TESS | happy | happy |
| TESS | sad | sad |
| TESS | angry | angry |
| TESS | fear | fear |
| TESS | disgust | disgust |
| TESS | ps | surprise |
| SAVEE | n | neutral |
| SAVEE | h | happy |
| SAVEE | sa | sad |
| SAVEE | a | angry |
| SAVEE | f | fear |
| SAVEE | d | disgust |
| SAVEE | su | surprise |
| CREMA-D | NEU | neutral |
| CREMA-D | HAP | happy |
| CREMA-D | SAD | sad |
| CREMA-D | ANG | angry |
| CREMA-D | FEA | fear |
| CREMA-D | DIS | disgust |
| EMO-DB | N | neutral |
| EMO-DB | F | happy |
| EMO-DB | T | sad |
| EMO-DB | W | angry |
| EMO-DB | A | fear |
| EMO-DB | E | disgust |
| EMO-DB | L | calm |
