// src/pages/Analytics.tsx — Recharts emotion distribution analytics
// SAFETY: all array operations guarded, all numeric fields nullable.

import { useEffect, useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  PieChart, Pie, Legend,
} from 'recharts';
import { RefreshCw } from 'lucide-react';
import { fetchMetrics } from '../services/api';
import type { MetricsResponse, EmotionName } from '../types';
import { EMOTION_META } from '../types';

const safeFixed = (v: number | null | undefined, decimals = 1) =>
  v != null && isFinite(v) ? v.toFixed(decimals) : '—';

export const Analytics = () => {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchMetrics(); // never throws — returns defaults
      setMetrics(data);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  // Always safe arrays
  const distribution = Array.isArray(metrics?.emotion_distribution)
    ? metrics!.emotion_distribution
    : [];

  const barData = distribution.map(({ emotion, count, avg_confidence }) => ({
    name: `${EMOTION_META[emotion as EmotionName]?.emoji ?? ''} ${emotion}`,
    count: count ?? 0,
    confidence: avg_confidence != null ? parseFloat((avg_confidence * 100).toFixed(1)) : 0,
    fill: EMOTION_META[emotion as EmotionName]?.color ?? '#60a5fa',
  }));

  const pieData = distribution
    .filter((e) => (e?.count ?? 0) > 0)
    .map(({ emotion, count }) => ({
      name: emotion,
      value: count ?? 0,
      fill: EMOTION_META[emotion as EmotionName]?.color ?? '#60a5fa',
    }));

  return (
    <div className="min-h-screen bg-background text-foreground relative">
      <div className="absolute top-20 right-0 w-[35%] h-[35%] rounded-full bg-emerald-600/10 blur-[120px] pointer-events-none" />

      <div className="container mx-auto px-4 py-10 relative z-10 max-w-5xl">

        {/* Header */}
        <div className="flex items-start justify-between mb-10">
          <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}>
            <h1 className="text-4xl font-outfit font-bold mb-2">Analytics</h1>
            <p className="text-muted-foreground">Aggregate emotion statistics across all predictions.</p>
          </motion.div>
          <button
            onClick={load}
            disabled={loading}
            className="glass-panel p-2.5 rounded-xl text-muted-foreground hover:text-white transition-colors disabled:opacity-40 mt-1"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>

        {/* ── Loading skeleton ── */}
        {loading && (
          <div className="space-y-6 animate-pulse">
            <div className="grid grid-cols-3 gap-4">
              {[0,1,2].map(i => (
                <div key={i} className="glass-panel rounded-2xl p-5">
                  <div className="h-7 bg-white/10 rounded w-1/2 mb-2 mx-auto" />
                  <div className="h-3 bg-white/10 rounded w-3/4 mx-auto" />
                </div>
              ))}
            </div>
            <div className="glass-panel rounded-2xl p-6 h-48 bg-white/5" />
          </div>
        )}

        {!loading && (
          <>
            {/* Summary pills */}
            <div className="grid grid-cols-3 gap-4 mb-10">
              {[
                { label: 'Total Predictions', value: metrics?.total_predictions ?? 0 },
                {
                  label: 'Avg Confidence',
                  value: metrics?.avg_confidence != null
                    ? `${(metrics.avg_confidence * 100).toFixed(1)}%`
                    : '—',
                },
                {
                  label: 'Avg Inference',
                  value: metrics?.avg_inference_time_ms != null
                    ? `${safeFixed(metrics.avg_inference_time_ms)}ms`
                    : '—',
                },
              ].map(({ label, value }) => (
                <motion.div
                  key={label}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="glass-panel rounded-2xl p-5 text-center"
                >
                  <p className="text-2xl font-bold text-white">{value}</p>
                  <p className="text-xs text-muted-foreground mt-1">{label}</p>
                </motion.div>
              ))}
            </div>

            {/* Bar Chart */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="glass-panel rounded-2xl p-6 mb-6"
            >
              <h2 className="text-lg font-semibold mb-5">Prediction Count by Emotion</h2>
              {barData.length === 0 ? (
                <div className="h-48 flex flex-col items-center justify-center text-muted-foreground gap-2">
                  <span className="text-4xl">📊</span>
                  <span className="text-sm">No predictions yet — upload some audio first!</span>
                </div>
              ) : (
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={barData} margin={{ bottom: 10 }}>
                      <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} interval={0} />
                      <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
                      <Tooltip
                        contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8 }}
                      />
                      <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                        {barData.map((entry, i) => (
                          <Cell key={i} fill={entry.fill} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </motion.div>

            {/* Pie Chart */}
            {pieData.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="glass-panel rounded-2xl p-6"
              >
                <h2 className="text-lg font-semibold mb-5">Emotion Share</h2>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={pieData}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        paddingAngle={3}
                        dataKey="value"
                      >
                        {pieData.map((entry, i) => (
                          <Cell key={i} fill={entry.fill} />
                        ))}
                      </Pie>
                      <Tooltip
                        contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8 }}
                      />
                      <Legend
                        formatter={(value) => (
                          <span className="text-xs text-muted-foreground capitalize">{value}</span>
                        )}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </motion.div>
            )}
          </>
        )}
      </div>
    </div>
  );
};
