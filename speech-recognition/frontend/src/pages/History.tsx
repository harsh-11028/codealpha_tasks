// src/pages/History.tsx — Paginated prediction history table
// SAFETY: fetchHistory returns [] on error; items array always safe.

import { useEffect, useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { ChevronLeft, ChevronRight, RefreshCw, Clock } from 'lucide-react';
import { fetchHistory } from '../services/api';
import { EMOTION_META } from '../types';
import type { HistoryItem, EmotionName } from '../types';

const PAGE_SIZE = 15;

const fmt = (iso: string) => {
  try {
    return new Date(iso).toLocaleString('en-IN', {
      dateStyle: 'medium',
      timeStyle: 'short',
    });
  } catch {
    return iso ?? '—';
  }
};

export const History = () => {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [page, setPage] = useState(0);
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(true);

  const load = useCallback(async (p: number) => {
    setLoading(true);
    try {
      const data = await fetchHistory(p * PAGE_SIZE, PAGE_SIZE); // returns [] on error
      const safeData = Array.isArray(data) ? data : [];
      setItems(safeData);
      setHasMore(safeData.length === PAGE_SIZE);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(page); }, [page, load]);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="container mx-auto px-4 py-10 max-w-5xl">

        {/* Header */}
        <div className="flex items-center justify-between mb-10">
          <div>
            <h1 className="text-4xl font-outfit font-bold mb-2">History</h1>
            <p className="text-muted-foreground">All previous emotion predictions.</p>
          </div>
          <button
            onClick={() => load(page)}
            className="glass-panel p-2 rounded-lg text-muted-foreground hover:text-white transition-colors"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>

        <div className="glass-panel rounded-2xl overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/5 text-muted-foreground text-xs uppercase tracking-wider">
                  <th className="text-left p-4">#</th>
                  <th className="text-left p-4">File</th>
                  <th className="text-left p-4">Emotion</th>
                  <th className="text-right p-4">Confidence</th>
                  <th className="text-right p-4">Duration</th>
                  <th className="text-right p-4">Inference</th>
                  <th className="text-right p-4">Date</th>
                </tr>
              </thead>
              <tbody>
                {/* Loading skeleton */}
                {loading && Array.from({ length: 6 }).map((_, i) => (
                  <tr key={i} className="border-b border-white/5 animate-pulse">
                    {Array.from({ length: 7 }).map((_, j) => (
                      <td key={j} className="p-4">
                        <div className="h-3 bg-white/5 rounded w-3/4" />
                      </td>
                    ))}
                  </tr>
                ))}

                {/* Empty state */}
                {!loading && items.length === 0 && (
                  <tr>
                    <td colSpan={7} className="p-12 text-center">
                      <div className="flex flex-col items-center gap-3 text-muted-foreground">
                        <Clock className="w-10 h-10 opacity-30" />
                        <p className="font-medium">No predictions yet</p>
                        <p className="text-sm">Go to <strong>Predict</strong> or <strong>Home</strong> to analyse an audio file.</p>
                      </div>
                    </td>
                  </tr>
                )}

                {/* Data rows */}
                {!loading && items.map((item, idx) => {
                  if (!item) return null;
                  const meta = EMOTION_META[item.emotion as EmotionName];
                  return (
                    <motion.tr
                      key={item.id ?? idx}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: idx * 0.02 }}
                      className="border-b border-white/5 hover:bg-white/[0.03] transition-colors"
                    >
                      <td className="p-4 text-muted-foreground">{item.id ?? '—'}</td>
                      <td className="p-4 font-medium truncate max-w-[180px]" title={item.filename}>
                        {item.filename ?? '—'}
                      </td>
                      <td className="p-4">
                        {item.emotion ? (
                          <span
                            className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium border"
                            style={{
                              color: meta?.color ?? '#94a3b8',
                              borderColor: `${meta?.color ?? '#94a3b8'}40`,
                              backgroundColor: `${meta?.color ?? '#94a3b8'}15`,
                            }}
                          >
                            {meta?.emoji ?? '🎭'} {item.emotion}
                          </span>
                        ) : '—'}
                      </td>
                      <td className="p-4 text-right">
                        {item.confidence != null ? `${(item.confidence * 100).toFixed(1)}%` : '—'}
                      </td>
                      <td className="p-4 text-right text-muted-foreground">
                        {item.duration_seconds != null ? `${item.duration_seconds.toFixed(1)}s` : '—'}
                      </td>
                      <td className="p-4 text-right text-muted-foreground">
                        {item.inference_time_ms != null ? `${item.inference_time_ms.toFixed(0)}ms` : '—'}
                      </td>
                      <td className="p-4 text-right text-muted-foreground text-xs">
                        {item.created_at ? fmt(item.created_at) : '—'}
                      </td>
                    </motion.tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-between p-4 border-t border-white/5">
            <span className="text-xs text-muted-foreground">Page {page + 1}</span>
            <div className="flex gap-2">
              <button
                disabled={page === 0 || loading}
                onClick={() => setPage((p) => p - 1)}
                className="p-1.5 rounded-lg disabled:opacity-30 hover:bg-white/5 transition-colors"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <button
                disabled={!hasMore || loading}
                onClick={() => setPage((p) => p + 1)}
                className="p-1.5 rounded-lg disabled:opacity-30 hover:bg-white/5 transition-colors"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
