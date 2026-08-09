// src/pages/Dashboard.tsx — Model status & system health overview
// SAFETY: every value is guarded with optional chaining + nullish coalescing.
// The page never throws even if every API field is null/undefined.

import { useEffect, useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Activity, Cpu, Zap, Brain, Clock, TrendingUp, RefreshCw, WifiOff } from 'lucide-react';
import { fetchHealth, fetchModelInfo, fetchMetrics } from '../services/api';
import { EMOTION_META } from '../types';
import type { HealthResponse, ModelInfo, MetricsResponse, EmotionName } from '../types';

// ── Sub-components ─────────────────────────────────────────────────────────────

const StatCard = ({
  icon: Icon,
  label,
  value,
  color = 'text-blue-400',
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string | number;
  color?: string;
}) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    className="glass-panel rounded-2xl p-5 flex items-start gap-4"
  >
    <div className={`mt-1 ${color}`}>
      <Icon className="w-5 h-5" />
    </div>
    <div>
      <p className="text-muted-foreground text-xs mb-1">{label}</p>
      <p className="text-white font-bold text-xl leading-none">{value}</p>
    </div>
  </motion.div>
);

const SkeletonCard = () => (
  <div className="glass-panel rounded-2xl p-5 animate-pulse">
    <div className="h-3 bg-white/10 rounded w-1/2 mb-3" />
    <div className="h-6 bg-white/10 rounded w-3/4" />
  </div>
);

// ── Helpers ────────────────────────────────────────────────────────────────────

/** Safe toFixed: returns '—' for null/undefined/NaN */
const safePct = (v: number | null | undefined) =>
  v != null && isFinite(v) ? `${(v * 100).toFixed(1)}%` : '—';

const safeMs = (v: number | null | undefined) =>
  v != null && isFinite(v) ? `${v.toFixed(1)}ms` : '—';

const safeUptime = (seconds: number | null | undefined) => {
  if (seconds == null || !isFinite(seconds)) return '—';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}m ${s}s`;
};

// ── Page ───────────────────────────────────────────────────────────────────────
export const Dashboard = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [offline, setOffline] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setOffline(false);
    try {
      const [h, m, mt] = await Promise.all([
        fetchHealth(),
        fetchModelInfo(),
        fetchMetrics(),
      ]);
      // fetchHealth/ModelInfo/Metrics never throw — they return defaults on error.
      // Detect backend-offline by checking the sentinel value we set.
      if (h.status === 'offline') setOffline(true);
      setHealth(h);
      setModel(m);
      setMetrics(mt);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  // Guaranteed-safe distribution array
  const distribution = Array.isArray(metrics?.emotion_distribution)
    ? metrics!.emotion_distribution
    : [];

  return (
    <div className="min-h-screen bg-background text-foreground relative">
      <div className="absolute top-0 right-0 w-[40%] h-[40%] rounded-full bg-blue-600/10 blur-[120px] pointer-events-none" />

      <div className="container mx-auto px-4 py-10 relative z-10">

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 flex items-start justify-between"
        >
          <div>
            <h1 className="text-4xl font-outfit font-bold mb-2">Dashboard</h1>
            <p className="text-muted-foreground">System status and live performance metrics.</p>
          </div>
          <button
            onClick={load}
            disabled={loading}
            className="glass-panel p-2.5 rounded-xl text-muted-foreground hover:text-white transition-colors disabled:opacity-40"
            title="Refresh"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </motion.div>

        {/* ── Loading skeletons ── */}
        {loading && (
          <div className="grid grid-cols-2 lg:grid-cols-3 gap-4 mb-10">
            {Array.from({ length: 6 }).map((_, i) => <SkeletonCard key={i} />)}
          </div>
        )}

        {/* ── Offline banner ── */}
        {!loading && offline && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8 glass-panel rounded-2xl p-5 flex items-center gap-4 border border-yellow-500/20"
          >
            <WifiOff className="w-5 h-5 text-yellow-400 shrink-0" />
            <div>
              <p className="font-medium text-yellow-400">Backend Offline</p>
              <p className="text-sm text-muted-foreground">
                Start the server with <code className="text-xs bg-white/10 px-1.5 py-0.5 rounded">./run.sh</code> — showing cached data.
              </p>
            </div>
          </motion.div>
        )}

        {/* ── Status pills ── */}
        {!loading && (
          <>
            <div className="flex flex-wrap gap-3 mb-8">
              <div
                className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-sm border ${
                  health?.status === 'healthy'
                    ? 'border-green-500/30 bg-green-500/10 text-green-400'
                    : health?.status === 'offline'
                    ? 'border-red-500/30 bg-red-500/10 text-red-400'
                    : 'border-yellow-500/30 bg-yellow-500/10 text-yellow-400'
                }`}
              >
                <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" />
                {(health?.status ?? 'unknown').toUpperCase()}
              </div>
              <div
                className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-sm border ${
                  model?.is_loaded
                    ? 'border-blue-500/30 bg-blue-500/10 text-blue-400'
                    : 'border-red-500/30 bg-red-500/10 text-red-400'
                }`}
              >
                <Brain className="w-3.5 h-3.5" />
                {model?.is_loaded
                  ? `${model.model_name ?? 'Model'} Loaded`
                  : 'No Model Loaded'}
              </div>
            </div>

            {/* ── Stat grid ── */}
            <div className="grid grid-cols-2 lg:grid-cols-3 gap-4 mb-10">
              <StatCard icon={Activity}    label="Uptime"            value={safeUptime(health?.uptime_seconds)}                   color="text-green-400" />
              <StatCard icon={Cpu}         label="Device"            value={(model?.device ?? '—').toUpperCase()}                 color="text-blue-400" />
              <StatCard icon={Brain}       label="Architecture"      value={(model?.architecture ?? '—').toUpperCase()}           color="text-purple-400" />
              <StatCard icon={TrendingUp}  label="Total Predictions" value={metrics?.total_predictions ?? 0}                      color="text-amber-400" />
              <StatCard icon={Zap}         label="Avg Inference"     value={safeMs(metrics?.avg_inference_time_ms)}               color="text-cyan-400" />
              <StatCard icon={Clock}       label="Avg Confidence"    value={safePct(metrics?.avg_confidence)}                    color="text-rose-400" />
            </div>

            {/* ── Emotion distribution ── */}
            {distribution.length > 0 ? (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.15 }}
                className="glass-panel rounded-2xl p-6"
              >
                <h2 className="text-lg font-semibold mb-5">Emotion Distribution</h2>
                <div className="space-y-3">
                  {[...distribution]
                    .sort((a, b) => (b?.count ?? 0) - (a?.count ?? 0))
                    .map((item) => {
                      if (!item?.emotion) return null;
                      const meta = EMOTION_META[item.emotion as EmotionName];
                      const count = item.count ?? 0;
                      const total = metrics?.total_predictions || 1;
                      const pct = Math.round((count / total) * 100);
                      const avgConf = item.avg_confidence;
                      return (
                        <div key={item.emotion}>
                          <div className="flex justify-between items-center mb-1">
                            <span className="text-sm flex items-center gap-2">
                              <span>{meta?.emoji ?? '🎭'}</span>
                              <span className="capitalize">{item.emotion}</span>
                            </span>
                            <span className="text-xs text-muted-foreground">
                              {count} · {safePct(avgConf != null ? avgConf / 100 : null)} avg
                            </span>
                          </div>
                          <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${pct}%` }}
                              transition={{ duration: 0.8, delay: 0.1 }}
                              className="h-full rounded-full"
                              style={{ backgroundColor: meta?.color ?? '#60a5fa' }}
                            />
                          </div>
                        </div>
                      );
                    })}
                </div>
              </motion.div>
            ) : (
              /* ── Empty state ── */
              !loading && (
                <div className="glass-panel rounded-2xl p-10 text-center text-muted-foreground">
                  <p className="text-4xl mb-3">📊</p>
                  <p className="font-medium">No predictions yet</p>
                  <p className="text-sm mt-1">Upload or record audio on the Home or Predict page.</p>
                </div>
              )
            )}
          </>
        )}
      </div>
    </div>
  );
};
