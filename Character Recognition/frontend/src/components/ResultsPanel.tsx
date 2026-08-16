import React, { useState } from 'react';
import { Copy, Download, Check, Layers, Clock, ShieldCheck, Cpu, Eye } from 'lucide-react';
import ocrApi, { type OCRResult } from '../services/api';

interface ResultsPanelProps {
  result: OCRResult | null;
  loading: boolean;
  error?: string | null;
}

export const ResultsPanel: React.FC<ResultsPanelProps> = ({ result, loading, error }) => {
  const [copied, setCopied] = useState<boolean>(false);
  const [viewMode, setViewMode] = useState<'text' | 'annotation' | 'tokens'>('text');
  const [exportFormat, setExportFormat] = useState<'txt' | 'pdf' | 'docx'>('txt');

  const handleCopy = () => {
    if (result && result.text) {
      navigator.clipboard.writeText(result.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleExport = () => {
    if (result && result.text) {
      ocrApi.exportText(result.text, exportFormat, 'ocr_analysis');
    }
  };

  const getConfColor = (score: number) => {
    if (score >= 0.85) return '#10b981'; // Green
    if (score >= 0.65) return '#f59e0b'; // Yellow
    return '#f43f5e'; // Red
  };

  if (loading) {
    return (
      <div className="card animate-fade-in" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '540px' }}>
        <div className="status-dot" style={{ width: '40px', height: '40px', marginBottom: '20px', background: 'var(--secondary)', boxShadow: '0 0 25px var(--secondary)' }} />
        <h3 style={{ fontSize: '18px', fontWeight: 600, color: '#fff', marginBottom: '8px' }}>Neural Engine Running...</h3>
        <p style={{ color: 'var(--text-muted)', fontSize: '14px', maxWidth: '280px', textAlign: 'center' }}>
          Applying CLAHE contrast enhancement, word segmentation, and vision model decoding.
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card animate-fade-in" style={{ borderColor: 'rgba(244, 63, 94, 0.4)', background: 'rgba(244, 63, 94, 0.05)' }}>
        <h3 style={{ color: '#f43f5e', fontSize: '18px', fontWeight: 600, marginBottom: '8px' }}>Pipeline Error / Alert</h3>
        <p style={{ color: 'var(--text-muted)', fontSize: '14px', lineHeight: 1.6 }}>{error}</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="card animate-fade-in" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '540px', opacity: 0.7 }}>
        <Layers size={48} color="var(--text-muted)" style={{ marginBottom: '16px' }} />
        <h3 style={{ fontSize: '18px', fontWeight: 600, color: 'var(--text-main)', marginBottom: '6px' }}>No Results Yet</h3>
        <p style={{ color: 'var(--text-muted)', fontSize: '14px', textAlign: 'center', maxWidth: '300px' }}>
          Upload an image or start live webcam studio on the left to see intelligent OCR reconstructions here.
        </p>
      </div>
    );
  }

  const confPercent = Math.round(result.confidence * 100);

  return (
    <div className="card animate-fade-in">
      <div className="card-title">
        <span>Extracted Text Reconstructions</span>
        <div style={{ display: 'flex', gap: '6px' }}>
          <button
            className={`btn ${viewMode === 'text' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ padding: '6px 12px', fontSize: '12px' }}
            onClick={() => setViewMode('text')}
          >
            Text Output
          </button>
          {result.annotated_image && (
            <button
              className={`btn ${viewMode === 'annotation' ? 'btn-primary' : 'btn-secondary'}`}
              style={{ padding: '6px 12px', fontSize: '12px' }}
              onClick={() => setViewMode('annotation')}
            >
              <Eye size={14} /> Bbox Visualizer
            </button>
          )}
        </div>
      </div>

      {viewMode === 'text' ? (
        <div className="result-box">
          {result.text || <span style={{ fontStyle: 'italic', color: 'var(--text-muted)' }}>[Empty text output detected]</span>}
        </div>
      ) : (
        <div className="preview-container" style={{ background: '#000', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)' }}>
          <img
            src={`data:image/png;base64,${result.annotated_image}`}
            alt="BBox Annotations"
            className="preview-img"
            style={{ maxHeight: '420px' }}
          />
        </div>
      )}

      {/* Action Toolbar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', margin: '16px 0 24px 0' }}>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button className="btn btn-secondary" style={{ padding: '8px 16px', fontSize: '13px' }} onClick={handleCopy}>
            {copied ? <><Check size={15} color="#10b981" /> Copied!</> : <><Copy size={15} /> Copy Text</>}
          </button>
        </div>

        <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
          <select
            className="select-input"
            style={{ padding: '8px 12px', fontSize: '13px' }}
            value={exportFormat}
            onChange={(e) => setExportFormat(e.target.value as any)}
          >
            <option value="txt">.TXT (Plain Text)</option>
            <option value="pdf">.PDF (Formatted Report)</option>
            <option value="docx">.DOCX (Word Document)</option>
          </select>
          <button className="btn btn-secondary" style={{ padding: '8px 14px', fontSize: '13px', background: 'rgba(6, 182, 212, 0.15)', borderColor: 'rgba(6, 182, 212, 0.35)', color: '#06b6d4' }} onClick={handleExport}>
            <Download size={15} /> Export File
          </button>
        </div>
      </div>

      {/* Confidence Bar */}
      <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <span style={{ fontSize: '13px', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '1px' }}>
            Model Confidence Score
          </span>
          <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: getConfColor(result.confidence), fontSize: '16px' }}>
            {confPercent}% ({result.confidence >= 0.85 ? 'High Accuracy' : 'Review Suggested'})
          </span>
        </div>
        <div className="progress-container">
          <div
            className="progress-bar"
            style={{
              width: `${confPercent}%`,
              background: getConfColor(result.confidence),
              boxShadow: `0 0 12px ${getConfColor(result.confidence)}`
            }}
          />
        </div>
      </div>

      {/* Granular Execution Stats */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-label"><Clock size={12} style={{ display: 'inline', marginRight: '4px' }} /> Latency Speed</div>
          <div className="stat-val">{result.processing_ms.toFixed(0)} ms</div>
        </div>
        <div className="stat-card">
          <div className="stat-label"><Cpu size={12} style={{ display: 'inline', marginRight: '4px' }} /> Engine Used</div>
          <div className="stat-val" style={{ fontSize: '16px' }}>{result.engine_used?.toUpperCase() || 'CUSTOM'}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label"><ShieldCheck size={12} style={{ display: 'inline', marginRight: '4px' }} /> Words / Chars</div>
          <div className="stat-val" style={{ fontSize: '18px' }}>
            {result.word_boxes?.length || result.text.split(' ').length} / {result.text.length}
          </div>
        </div>
      </div>
    </div>
  );
};
