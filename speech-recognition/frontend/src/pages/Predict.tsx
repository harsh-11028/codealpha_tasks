// src/pages/Predict.tsx — Drag & drop audio file upload with full result visualization
import { useState, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileAudio, CheckCircle2, Loader2, X } from 'lucide-react';
import { predictFile } from '../services/api';
import { EMOTION_META } from '../types';
import type { EmotionName } from '../types';
import { WaveformChart } from '../components/charts/WaveformChart';
import { EmotionRadar } from '../components/charts/EmotionRadar';
import { FeatureImportanceBar } from '../components/charts/FeatureImportanceBar';
import { SpectrogramHeatmap } from '../components/charts/SpectrogramHeatmap';

const ACCEPTED = ['audio/wav', 'audio/mpeg', 'audio/ogg', 'audio/mp4', 'audio/x-m4a'];

export const Predict = () => {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const processFile = useCallback(async (f: File) => {
    if (!ACCEPTED.includes(f.type)) {
      setError('Unsupported format. Please upload a WAV, MP3, OGG, or M4A file.');
      return;
    }
    setFile(f);
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const res = await predictFile(f);
      setResult(res);
    } catch (e: any) {
      setError(
        e?.response?.data?.detail ?? 'Prediction failed. Make sure the backend is running.'
      );
    } finally {
      setLoading(false);
    }
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const f = e.dataTransfer.files[0];
      if (f) processFile(f);
    },
    [processFile]
  );

  const emotion = result?.predicted_emotion as EmotionName | undefined;
  const meta = emotion ? EMOTION_META[emotion] : null;

  const probs: Record<string, number> = result?.all_probabilities?.reduce((acc: any, curr: any) => {
    acc[curr.emotion] = curr.probability;
    return acc;
  }, {}) || {};

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="absolute bottom-0 left-0 w-[40%] h-[40%] rounded-full bg-purple-600/10 blur-[120px] pointer-events-none" />

      <div className="container mx-auto px-4 py-10 relative z-10 max-w-4xl">
        <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-10">
          <h1 className="text-4xl font-outfit font-bold mb-2">Predict from File</h1>
          <p className="text-muted-foreground">Upload an audio file to analyze its emotional content.</p>
        </motion.div>

        {/* Drop Zone */}
        <motion.div
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          className={`relative glass-panel rounded-2xl p-10 flex flex-col items-center justify-center cursor-pointer transition-all duration-200 border-2 ${
            isDragging ? 'border-blue-500/60 bg-blue-500/5' : 'border-dashed border-white/10 hover:border-white/20'
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            className="hidden"
            accept="audio/*"
            onChange={(e) => e.target.files?.[0] && processFile(e.target.files[0])}
          />

          {loading ? (
            <div className="flex flex-col items-center gap-3">
              <Loader2 className="w-12 h-12 text-blue-400 animate-spin" />
              <p className="text-muted-foreground text-sm">Analyzing audio…</p>
            </div>
          ) : file && result ? (
            <div className="flex flex-col items-center gap-3">
              <CheckCircle2 className="w-12 h-12 text-green-400" />
              <p className="text-white font-medium">{file.name}</p>
              <button
                className="text-xs text-muted-foreground hover:text-white flex items-center gap-1"
                onClick={(e) => { e.stopPropagation(); setFile(null); setResult(null); }}
              >
                <X className="w-3 h-3" /> Clear
              </button>
            </div>
          ) : (
            <>
              <FileAudio className="w-12 h-12 text-muted-foreground mb-4" />
              <p className="text-white font-medium mb-1">Drop your audio file here</p>
              <p className="text-muted-foreground text-sm">or click to browse · WAV, MP3, OGG, M4A</p>
              {file && !loading && (
                <p className="mt-3 text-xs text-blue-400">{file.name}</p>
              )}
            </>
          )}

          {isDragging && (
            <div className="absolute inset-0 rounded-2xl bg-blue-500/10 border-2 border-blue-500 flex items-center justify-center">
              <div className="flex items-center gap-3 text-blue-400 font-medium">
                <Upload className="w-5 h-5" /> Drop to analyze
              </div>
            </div>
          )}
        </motion.div>

        {error && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            className="mt-4 bg-red-500/10 border border-red-500/30 text-red-400 text-sm rounded-xl p-4"
          >
            {error}
          </motion.div>
        )}

        {/* Results */}
        <AnimatePresence>
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="mt-8 space-y-6"
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

              {/* 2-col charts */}
              <div className="grid md:grid-cols-2 gap-6">
                <div className="glass-panel rounded-2xl p-5">
                  <h3 className="text-sm font-semibold mb-3 text-muted-foreground">Emotion Probability Radar</h3>
                  <EmotionRadar
                    probabilities={probs}
                    color={meta?.color ?? '#60a5fa'}
                  />
                </div>
                <div className="glass-panel rounded-2xl p-5">
                  <h3 className="text-sm font-semibold mb-3 text-muted-foreground">XAI — Feature Importance</h3>
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

              {/* Spectrogram */}
              {result.spectrogram_data?.length > 0 && (
                <div className="glass-panel rounded-2xl p-5">
                  <SpectrogramHeatmap data={result.spectrogram_data} />
                </div>
              )}

              {/* All probabilities table */}
              <div className="glass-panel rounded-2xl p-5">
                <h3 className="text-sm font-semibold mb-4 text-muted-foreground">All Emotion Probabilities</h3>
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
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};
