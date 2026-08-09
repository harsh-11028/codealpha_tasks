// src/services/api.ts — Centralized Axios HTTP client with defensive defaults

import axios, { AxiosError } from 'axios';
import type {
  HealthResponse,
  ModelInfo,
  MetricsResponse,
  HistoryItem,
  PredictionResult,
} from '../types';
import {
  DEFAULT_HEALTH,
  DEFAULT_MODEL_INFO,
  DEFAULT_METRICS,
} from '../types';

// ── Axios instance ─────────────────────────────────────────────────────────────
const api = axios.create({
  baseURL: '/api',
  timeout: 30_000,
  headers: { 'Content-Type': 'application/json' },
});

// ── Response interceptor: normalise errors into readable messages ──────────────
api.interceptors.response.use(
  (res) => res,
  (err: AxiosError) => {
    const detail =
      (err.response?.data as any)?.detail ??
      (err.response?.data as any)?.message ??
      err.message ??
      'Unknown error';
    return Promise.reject(new Error(String(detail)));
  }
);

// ── Helper: validate that a value is a real array ─────────────────────────────
function ensureArray<T>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}

// ── Health ────────────────────────────────────────────────────────────────────
export const fetchHealth = async (): Promise<HealthResponse> => {
  try {
    const { data } = await api.get<HealthResponse>('/health');
    return {
      ...DEFAULT_HEALTH,
      ...data,
      status: data?.status ?? DEFAULT_HEALTH.status,
      uptime_seconds: Number(data?.uptime_seconds) || 0,
    };
  } catch {
    return { ...DEFAULT_HEALTH, status: 'offline' };
  }
};

// ── Model Info ────────────────────────────────────────────────────────────────
export const fetchModelInfo = async (): Promise<ModelInfo> => {
  try {
    const { data } = await api.get<ModelInfo>('/model-info');
    return {
      ...DEFAULT_MODEL_INFO,
      ...data,
      emotions: ensureArray<string>(data?.emotions),
    };
  } catch {
    return { ...DEFAULT_MODEL_INFO };
  }
};

// ── Metrics ───────────────────────────────────────────────────────────────────
export const fetchMetrics = async (): Promise<MetricsResponse> => {
  try {
    const { data } = await api.get<MetricsResponse>('/metrics');
    return {
      ...DEFAULT_METRICS,
      ...data,
      total_predictions: Number(data?.total_predictions) || 0,
      avg_inference_time_ms: Number(data?.avg_inference_time_ms) || 0,
      avg_confidence: Number(data?.avg_confidence) || 0,
      emotion_distribution: ensureArray(data?.emotion_distribution),
    };
  } catch {
    return { ...DEFAULT_METRICS };
  }
};

// ── History ───────────────────────────────────────────────────────────────────
export const fetchHistory = async (
  skip = 0,
  limit = 20
): Promise<HistoryItem[]> => {
  try {
    const { data } = await api.get<HistoryItem[]>(
      `/history?skip=${skip}&limit=${limit}`
    );
    return ensureArray<HistoryItem>(data);
  } catch {
    return [];
  }
};

// ── File Prediction ───────────────────────────────────────────────────────────
export const predictFile = async (file: File): Promise<any> => {
  const form = new FormData();
  form.append('file', file);
  // May throw — callers must catch; we don't swallow predict errors
  const { data } = await api.post('/predict', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
};

// ── Single Prediction by ID ───────────────────────────────────────────────────
export const fetchPrediction = async (id: number): Promise<PredictionResult | null> => {
  try {
    const { data } = await api.get<PredictionResult>(`/predict/${id}`);
    return data ?? null;
  } catch {
    return null;
  }
};

export default api;
