// src/types/index.ts — Shared TypeScript type definitions with runtime defaults

// ── Emotion metadata ───────────────────────────────────────────────────────────
export type EmotionName =
  | 'neutral'
  | 'calm'
  | 'happy'
  | 'sad'
  | 'angry'
  | 'fear'
  | 'disgust'
  | 'surprise';

export const EMOTION_META: Record<
  EmotionName,
  { emoji: string; color: string; label: string }
> = {
  neutral:  { emoji: '😐', color: '#94a3b8', label: 'Neutral' },
  calm:     { emoji: '😌', color: '#38bdf8', label: 'Calm' },
  happy:    { emoji: '😊', color: '#fbbf24', label: 'Happy' },
  sad:      { emoji: '😢', color: '#818cf8', label: 'Sad' },
  angry:    { emoji: '😠', color: '#ef4444', label: 'Angry' },
  fear:     { emoji: '😨', color: '#a855f7', label: 'Fear' },
  disgust:  { emoji: '🤢', color: '#22c55e', label: 'Disgust' },
  surprise: { emoji: '😲', color: '#f97316', label: 'Surprise' },
};

// ── API response interfaces (all optional fields — backend may omit any) ───────
export interface HealthResponse {
  status: string;
  version?: string;
  uptime_seconds: number;
  model_loaded?: boolean;
  model_name?: string;
}

export interface ModelInfo {
  model_name: string;
  model_version?: string;
  architecture?: string;
  num_classes?: number;
  emotions: string[];
  is_loaded: boolean;
  device?: string;
}

export interface EmotionStats {
  emotion: string;           // keep as string — backend may send unknown emotions
  count: number;
  avg_confidence: number | null;
}

export interface MetricsResponse {
  total_predictions: number;
  emotion_distribution: EmotionStats[];
  avg_inference_time_ms: number | null;
  avg_confidence: number | null;
}

export interface HistoryItem {
  id: number;
  filename: string;
  emotion: string;
  confidence: number;
  duration_seconds?: number | null;
  inference_time_ms?: number | null;
  created_at: string;
}

export interface PredictionResult {
  id?: number;
  predicted_emotion?: string;
  emotion?: string;           // fallback field name
  confidence: number;
  all_probabilities?: Array<{ emotion: string; probability: number }>;
  probabilities?: Record<string, number>;
  waveform_data?: number[] | null;
  mfcc_data?: number[][] | null;
  spectrogram_data?: number[][] | null;
  feature_importance?: Record<string, number> | null;
  audio_duration_seconds?: number | null;
  duration_seconds?: number | null;
  sample_rate?: number | null;
  inference_time_ms?: number | null;
  model_name?: string;
  model_version?: string;
}

export interface UploadResponse {
  prediction: PredictionResult;
  file_id: number;
  filename: string;
}

// ── Runtime defaults (safe fallbacks when API returns null/undefined) ──────────
export const DEFAULT_HEALTH: HealthResponse = {
  status: 'unknown',
  uptime_seconds: 0,
  model_loaded: false,
  model_name: '—',
};

export const DEFAULT_MODEL_INFO: ModelInfo = {
  model_name: '—',
  architecture: '—',
  emotions: [],
  is_loaded: false,
  device: '—',
};

export const DEFAULT_METRICS: MetricsResponse = {
  total_predictions: 0,
  emotion_distribution: [],
  avg_inference_time_ms: null,
  avg_confidence: null,
};
