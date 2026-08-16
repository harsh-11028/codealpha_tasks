import React, { useState, useRef } from 'react';
import { Upload, X, Sliders, Image as ImageIcon, Sparkles } from 'lucide-react';

interface FileDropzoneProps {
  onFileSelect: (file: File, task: 'character' | 'word' | 'sentence', engine: string, model: string) => void;
  loading: boolean;
}

export const FileDropzone: React.FC<FileDropzoneProps> = ({ onFileSelect, loading }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [taskMode, setTaskMode] = useState<'character' | 'word' | 'sentence'>('sentence');
  const [engine, setEngine] = useState<string>('auto');
  const [model, setModel] = useState<string>('auto');
  const [dragOver, setDragOver] = useState<boolean>(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (file: File) => {
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const clearSelection = () => {
    setSelectedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (selectedFile && !loading) {
      onFileSelect(selectedFile, taskMode, engine, model);
    }
  };

  return (
    <div className="card animate-fade-in">
      <div className="card-title">
        <span><ImageIcon size={18} style={{ verticalAlign: 'middle', marginRight: '8px' }} /> Upload Handwriting / Document</span>
        <span className="chip" style={{ background: 'rgba(6, 182, 212, 0.1)', borderColor: 'rgba(6, 182, 212, 0.3)', color: '#06b6d4' }}>
          Interactive Workspace
        </span>
      </div>

      <form onSubmit={handleSubmit}>
        {!previewUrl ? (
          <div
            className={`dropzone ${dragOver ? 'active' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload className="dropzone-icon" />
            <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '8px', color: '#fff' }}>
              Drag & Drop Image Here, or Click to Browse
            </h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '13px', maxWidth: '320px', margin: '0 auto' }}>
              Supports scanned notes, whiteboards, invoices, or isolated character boxes (PNG, JPG, TIFF, WEBP up to 10MB)
            </p>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              style={{ display: 'none' }}
              onChange={(e) => e.target.files?.[0] && handleFileChange(e.target.files[0])}
            />
          </div>
        ) : (
          <div className="preview-container" style={{ marginBottom: '16px' }}>
            <img src={previewUrl} alt="Preview" className="preview-img" />
            <button
              type="button"
              className="btn btn-danger"
              style={{ position: 'absolute', top: '12px', right: '12px', padding: '8px 12px', zIndex: 10 }}
              onClick={clearSelection}
              disabled={loading}
            >
              <X size={16} /> Remove
            </button>
          </div>
        )}

        <div style={{ background: 'rgba(0, 0, 0, 0.25)', padding: '18px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)', margin: '20px 0' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '14px', color: 'var(--text-main)', fontWeight: 600, fontSize: '14px' }}>
            <Sliders size={16} color="var(--secondary)" /> Recognition Pipeline Configuration
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '14px' }}>
            <div className="control-group" style={{ marginBottom: 0 }}>
              <label className="label">Task Mode</label>
              <select
                className="select-input"
                value={taskMode}
                onChange={(e) => setTaskMode(e.target.value as any)}
                disabled={loading}
              >
                <option value="sentence">Full Document / Sentence (Hierarchical Pipeline)</option>
                <option value="word">Single Handwritten Word (CRNN + CTC)</option>
                <option value="character">Isolated Character (ViT / ResNet / CNN)</option>
              </select>
            </div>

            <div className="control-group" style={{ marginBottom: 0 }}>
              <label className="label">OCR Engine Strategy</label>
              <select
                className="select-input"
                value={engine}
                onChange={(e) => setEngine(e.target.value)}
                disabled={loading}
              >
                <option value="auto">Smart Auto-Routing (Quality Assessment)</option>
                <option value="hybrid">Hybrid Engine Consensus</option>
                <option value="custom">Custom Deep Learning Engine</option>
                <option value="easyocr">EasyOCR (Natural Scene / Cursive)</option>
                <option value="tesseract">Tesseract 5.0 (Structured / Printed)</option>
              </select>
            </div>

            <div className="control-group" style={{ marginBottom: 0 }}>
              <label className="label">Neural Architecture</label>
              <select
                className="select-input"
                value={model}
                onChange={(e) => setModel(e.target.value)}
                disabled={loading}
              >
                <option value="auto">Auto (Top Benchmark Model)</option>
                <option value="vit">Vision Transformer (ViT-Patch4)</option>
                <option value="crnn">CRNN + BiLSTM + CTC (Sequence)</option>
                <option value="residual_cnn">Residual ResNet-CNN (Pre-Act)</option>
                <option value="cnn_batchnorm">5-Layer CNN + BatchNorm + SE-Attn</option>
                <option value="cnn_basic">Baseline 4-Block CNN</option>
              </select>
            </div>
          </div>
        </div>

        <button
          type="submit"
          className="btn btn-primary"
          style={{ width: '100%', fontSize: '16px', padding: '14px' }}
          disabled={!selectedFile || loading}
        >
          {loading ? (
            <span>Processing through Neural Layers...</span>
          ) : (
            <>
              <Sparkles size={20} /> Execute Intelligent OCR Analysis
            </>
          )}
        </button>
      </form>
    </div>
  );
};
