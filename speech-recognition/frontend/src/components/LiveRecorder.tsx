// src/components/LiveRecorder.tsx
// Complete 4-state recording workflow:
//   idle → recording → preview → result
// After stopping, shows an audio player with replay/delete/submit controls,
// then sends the recording through the same /predict pipeline as file uploads.

import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Mic,
  Square,
  Loader2,
  Play,
  Trash2,
  Send,
  RotateCcw,
  AlertTriangle,
} from 'lucide-react';
import { useAudioRecorder } from '../hooks/useAudioRecorder';
import { predictFile } from '../services/api';
import { EMOTION_META } from '../types';
import type { EmotionName } from '../types';
import { WaveformChart } from './charts/WaveformChart';
import { EmotionRadar } from './charts/EmotionRadar';
import { FeatureImportanceBar } from './charts/FeatureImportanceBar';

// ── Helpers ────────────────────────────────────────────────────────────────────
const fmt = (seconds: number) => {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
};

type Phase = 'idle' | 'recording' | 'preview' | 'loading' | 'result';

// ── Component ──────────────────────────────────────────────────────────────────
export const LiveRecorder = () => {
  const { duration, error: micError, audioBlob, audioUrl, startRecording, stopRecording, resetRecording } =
    useAudioRecorder();

  const [phase, setPhase] = useState<Phase>('idle');
  const [result, setResult] = useState<any | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // ── Start recording ──────────────────────────────────────────────────────────
  const handleStart = useCallback(async () => {
    setResult(null);
    setSubmitError(null);
    setPhase('recording');
    await startRecording();
  }, [startRecording]);

  // ── Stop recording → go to preview ──────────────────────────────────────────
  const handleStop = useCallback(() => {
    stopRecording();
    // onstop in the hook sets audioBlob/audioUrl; give a tick for state update
    setTimeout(() => setPhase('preview'), 100);
  }, [stopRecording]);

  // ── Discard and start fresh ──────────────────────────────────────────────────
  const handleReset = useCallback(() => {
    resetRecording();
    setResult(null);
    setSubmitError(null);
    setPhase('idle');
  }, [resetRecording]);

  // ── Submit recording to /predict ─────────────────────────────────────────────
  const handleSubmit = useCallback(async () => {
    if (!audioBlob) return;
    setPhase('loading');
    setSubmitError(null);
    try {
      // Convert blob to a File so the existing predictFile() works unchanged
      const ext = audioBlob.type.includes('ogg') ? 'ogg' : audioBlob.type.includes('mp4') ? 'mp4' : 'webm';
      const file = new File([audioBlob], `recording.${ext}`, { type: audioBlob.type });
      const res = await predictFile(file);
      setResult(res);
      setPhase('result');
    } catch (e: any) {
      const msg = e?.response?.data?.detail ?? 'Prediction failed. Make sure the backend is running.';
      setSubmitError(msg);
      setPhase('preview'); // go back to preview so user can retry
    }
  }, [audioBlob]);

  // ── Derived display values ──────────────────────────────────────────────────
  const emotion = result?.predicted_emotion as EmotionName | undefined;
  const meta = emotion ? EMOTION_META[emotion] : null;
  const probs: Record<string, number> =
    result?.all_probabilities?.reduce((acc: any, curr: any) => {
      acc[curr.emotion] = curr.probability;
      return acc;
    }, {}) ?? {};

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div className="w-full max-w-2xl mx-auto space-y-6">
      {/* ── Main Recorder Panel ── */}
      <div className="glass-panel p-6 sm:p-8 rounded-2xl w-full flex flex-col items-center justify-center space-y-8 relative overflow-hidden">
        {/* Recording glow */}
        {phase === 'recording' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.15 }}
            className="absolute inset-0 bg-gradient-to-tr from-primary to-accent pointer-events-none"
          />
        )}

        {/* Title */}
        <div className="text-center z-10">
          <h2 className="text-2xl font-semibold mb-1">Live Emotion Recognition</h2>
          <p className="text-muted-foreground text-sm">
            {phase === 'idle' && 'Speak into your microphone — we\'ll analyse the emotion.'}
            {phase === 'recording' && 'Recording in progress…'}
            {phase === 'preview' && 'Preview your recording, then submit or re-record.'}
            {phase === 'loading' && 'Analysing your audio with the AI model…'}
            {phase === 'result' && 'Emotion analysis complete!'}
          </p>
        </div>

        {/* ── IDLE / RECORDING control ── */}
        <AnimatePresence mode="wait">
          {(phase === 'idle' || phase === 'recording') && (
            <motion.div
              key="controls"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="flex flex-col items-center gap-4 z-10"
            >
              {/* Big mic / stop button */}
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={phase === 'recording' ? handleStop : handleStart}
                className={`relative flex items-center justify-center w-24 h-24 rounded-full shadow-2xl transition-all duration-300 ${
                  phase === 'recording'
                    ? 'bg-destructive/10 text-destructive border-2 border-destructive/50'
                    : 'bg-primary text-primary-foreground hover:bg-primary/90'
                }`}
              >
                {phase === 'recording' && (
                  <motion.div
                    className="absolute inset-0 rounded-full border-2 border-destructive opacity-50"
                    animate={{ scale: [1, 1.5, 1], opacity: [0.5, 0, 0.5] }}
                    transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
                  />
                )}
                {phase === 'recording' ? (
                  <Square className="w-8 h-8 fill-current" />
                ) : (
                  <Mic className="w-10 h-10" />
                )}
              </motion.button>

              {/* Timer */}
              <motion.div className="text-2xl font-mono tracking-widest font-light">
                {fmt(duration)}
              </motion.div>

              {phase === 'recording' && (
                <p className="text-xs text-muted-foreground animate-pulse">
                  Click the square to stop
                </p>
              )}
            </motion.div>
          )}

          {/* ── PREVIEW state ── */}
          {phase === 'preview' && audioUrl && (
            <motion.div
              key="preview"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="w-full space-y-5 z-10"
            >
              {/* Audio player */}
              <div className="rounded-xl bg-white/5 p-4 border border-white/10">
                <p className="text-xs text-muted-foreground mb-3 flex items-center gap-1.5">
                  <Play className="w-3 h-3" /> Preview your recording ({fmt(duration)})
                </p>
                {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
                <audio controls src={audioUrl} className="w-full h-10 accent-blue-500" />
              </div>

              {submitError && (
                <div className="bg-red-500/10 border border-red-500/30 text-red-400 text-sm rounded-xl p-3 flex items-start gap-2">
                  <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
                  {submitError}
                </div>
              )}

              {/* Action buttons */}
              <div className="flex gap-3">
                <button
                  onClick={handleReset}
                  className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-sm transition-colors"
                >
                  <Trash2 className="w-4 h-4 text-red-400" />
                  <span>Re-record</span>
                </button>
                <button
                  onClick={handleSubmit}
                  className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl bg-primary text-primary-foreground hover:bg-primary/90 text-sm font-medium transition-colors"
                >
                  <Send className="w-4 h-4" />
                  <span>Analyse Emotion</span>
                </button>
              </div>
            </motion.div>
          )}

          {/* ── LOADING state ── */}
          {phase === 'loading' && (
            <motion.div
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center gap-4 z-10 py-4"
            >
              <Loader2 className="w-12 h-12 text-blue-400 animate-spin" />
              <p className="text-muted-foreground text-sm">Running AI inference…</p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Mic permission / connection error */}
        {micError && (
          <div className="w-full bg-destructive/10 border border-destructive/30 text-destructive text-sm rounded-lg p-3 z-10 text-center">
            {micError}
          </div>
        )}
      </div>

      {/* ── RESULT panel (rendered below the recorder card) ── */}
      <AnimatePresence>
        {phase === 'result' && result && (
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="space-y-6"
          >
            {/* Emotion Hero */}
            <div className="glass-panel rounded-2xl p-6 flex items-center justify-between">
              <div className="flex items-center gap-5">
                <span className="text-6xl">{meta?.emoji}</span>
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Detected Emotion</p>
                  <h2
                    className="text-3xl font-outfit font-bold capitalize"
                    style={{ color: meta?.color }}
                  >
                    {emotion}
                  </h2>
                </div>
              </div>
              <div className="text-right">
                <p className="text-xs text-muted-foreground mb-1">Confidence</p>
                <p className="text-4xl font-bold text-white">
                  {Math.round((result.confidence ?? 0) * 100)}%
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  {result.inference_time_ms?.toFixed(1)}ms · {result.model_name}
                </p>
              </div>
            </div>

            {/* Charts */}
            <div className="grid md:grid-cols-2 gap-6">
              <div className="glass-panel rounded-2xl p-5">
                <h3 className="text-sm font-semibold mb-3 text-muted-foreground">
                  Emotion Probability Radar
                </h3>
                <EmotionRadar probabilities={probs} color={meta?.color ?? '#60a5fa'} />
              </div>
              <div className="glass-panel rounded-2xl p-5">
                <h3 className="text-sm font-semibold mb-3 text-muted-foreground">
                  XAI — Feature Importance
                </h3>
                <FeatureImportanceBar data={result.feature_importance} />
              </div>
            </div>

            {/* Waveform */}
            {result.waveform_data?.length > 0 && (
              <div className="glass-panel rounded-2xl p-5">
                <h3 className="text-sm font-semibold mb-3 text-muted-foreground">
                  Waveform · {result.audio_duration_seconds?.toFixed(2)}s @ {result.sample_rate}Hz
                </h3>
                <WaveformChart data={result.waveform_data} color={meta?.color} />
              </div>
            )}

            {/* All probabilities */}
            <div className="glass-panel rounded-2xl p-5">
              <h3 className="text-sm font-semibold mb-4 text-muted-foreground">
                All Emotion Probabilities
              </h3>
              <div className="space-y-2">
                {Object.entries(probs)
                  .sort(([, a], [, b]) => b - a)
                  .map(([em, prob]) => {
                    const m = EMOTION_META[em as EmotionName];
                    return (
                      <div key={em}>
                        <div className="flex justify-between text-sm mb-1">
                          <span className="flex items-center gap-2">
                            <span>{m?.emoji}</span>
                            <span className="capitalize">{em}</span>
                          </span>
                          <span>{(prob * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${prob * 100}%` }}
                            transition={{ duration: 0.6 }}
                            className="h-full rounded-full"
                            style={{ backgroundColor: m?.color ?? '#60a5fa' }}
                          />
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>

            {/* Record again */}
            <button
              onClick={handleReset}
              className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-sm transition-colors"
            >
              <RotateCcw className="w-4 h-4" />
              Record another clip
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
