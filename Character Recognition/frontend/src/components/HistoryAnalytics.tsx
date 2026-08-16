import React, { useEffect, useState, useCallback } from 'react';
import { History, Trash2, Download, RefreshCw, BarChart3, Clock } from 'lucide-react';
import ocrApi, { type PredictionRecord } from '../services/api';

export const HistoryAnalytics: React.FC = () => {
  const [history, setHistory] = useState<PredictionRecord[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [filterType, setFilterType] = useState<string>('');

  const fetchHistoryData = useCallback(async () => {
    setLoading(true);
    try {
      const records = await ocrApi.getHistory(50, filterType || undefined);
      setHistory(records);
      const st = await ocrApi.getHistoryStats();
      setStats(st);
    } catch (err) {
      console.error('Failed to load OCR history analytics:', err);
    } finally {
      setLoading(false);
    }
  }, [filterType]);

  useEffect(() => {
    fetchHistoryData();
  }, [fetchHistoryData]);

  const handleDelete = async (id: number) => {
    try {
      await ocrApi.deleteHistory(id);
      setHistory(history.filter(h => h.id !== id));
      if (stats) setStats({ ...stats, total_predictions: Math.max(0, stats.total_predictions - 1) });
    } catch (err) {
      console.error('Failed to delete history item:', err);
    }
  };

  const handleExportText = (text: string, id: number) => {
    ocrApi.exportText(text, 'txt', `ocr_record_${id}`);
  };

  return (
    <div className="animate-fade-in" style={{ width: '100%' }}>
      {/* Analytics KPI Banner */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '20px', marginBottom: '30px' }}>
        <div className="card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '13px', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>Total Predictions</span>
            <History size={20} color="var(--primary)" />
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '32px', fontWeight: 700, color: '#fff', marginTop: '10px' }}>
            {stats ? stats.total_predictions : '—'}
          </div>
        </div>

        <div className="card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '13px', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>Mean Confidence</span>
            <BarChart3 size={20} color="var(--success)" />
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '32px', fontWeight: 700, color: '#10b981', marginTop: '10px' }}>
            {stats && stats.mean_confidence ? `${Math.round(stats.mean_confidence * 100)}%` : '0%'}
          </div>
        </div>

        <div className="card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '13px', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>Avg Inference Latency</span>
            <Clock size={20} color="var(--secondary)" />
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '32px', fontWeight: 700, color: '#06b6d4', marginTop: '10px' }}>
            {stats ? `${stats.mean_processing_ms || 0} ms` : '0 ms'}
          </div>
        </div>

        <div className="card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '13px', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>Most Frequent Engine</span>
            <span className="chip" style={{ background: 'rgba(124, 58, 237, 0.15)', borderColor: 'rgba(124, 58, 237, 0.4)', color: 'var(--primary)' }}>HYBRID-CRNN</span>
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '24px', fontWeight: 700, color: '#fff', marginTop: '14px' }}>
            Multi-Model OCR
          </div>
        </div>
      </div>

      {/* Main Table Card */}
      <div className="card">
        <div className="card-title">
          <span><History size={18} style={{ verticalAlign: 'middle', marginRight: '8px' }} /> OCR Scan History & Logs</span>
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
            <select
              className="select-input"
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              style={{ padding: '8px 14px', fontSize: '13px' }}
            >
              <option value="">All Task Types</option>
              <option value="sentence">Sentence / Document</option>
              <option value="word">Single Words</option>
              <option value="character">Isolated Characters</option>
            </select>

            <button className="btn btn-secondary" style={{ padding: '8px 14px', fontSize: '13px' }} onClick={fetchHistoryData}>
              <RefreshCw size={15} className={loading ? 'animate-spin' : ''} /> Refresh Logs
            </button>
          </div>
        </div>

        {loading ? (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
            Retrieving database query logs...
          </div>
        ) : history.length === 0 ? (
          <div style={{ padding: '60px 20px', textAlign: 'center', color: 'var(--text-muted)' }}>
            No previous OCR scanning logs found in SQLite repository. Upload images in the Workspace to populate logs!
          </div>
        ) : (
          <div className="table-container">
            <table className="table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Task Type</th>
                  <th>Recognized Text Snippet</th>
                  <th>Confidence Score</th>
                  <th>Engine / Model</th>
                  <th>Latency</th>
                  <th>Timestamp</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {history.map((record) => (
                  <tr key={record.id}>
                    <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>#{record.id}</td>
                    <td>
                      <span className="chip" style={{ background: 'rgba(255,255,255,0.05)' }}>
                        {record.input_type.toUpperCase()}
                      </span>
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)', color: '#fff', maxWidth: '320px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {record.raw_text || '[No text detected]'}
                    </td>
                    <td>
                      <span style={{ fontWeight: 700, color: record.confidence >= 0.85 ? '#10b981' : record.confidence >= 0.65 ? '#f59e0b' : '#f43f5e' }}>
                        {Math.round(record.confidence * 100)}%
                      </span>
                    </td>
                    <td>
                      <span style={{ color: 'var(--secondary)', fontSize: '13px' }}>
                        {record.engine_used?.toUpperCase() || 'CUSTOM'}
                      </span>
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{record.processing_ms} ms</td>
                    <td style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
                      {new Date(record.created_at).toLocaleDateString()} {new Date(record.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </td>
                    <td>
                      <div style={{ display: 'flex', gap: '6px' }}>
                        <button
                          className="btn btn-secondary"
                          style={{ padding: '6px', minWidth: 0, height: '32px' }}
                          title="Download Text"
                          onClick={() => handleExportText(record.raw_text || '', record.id)}
                        >
                          <Download size={15} />
                        </button>
                        <button
                          className="btn btn-danger"
                          style={{ padding: '6px', minWidth: 0, height: '32px' }}
                          title="Delete Record"
                          onClick={() => handleDelete(record.id)}
                        >
                          <Trash2 size={15} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
