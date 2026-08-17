import { useState } from 'react';
import { Header } from './components/Header';
import { FileDropzone } from './components/FileDropzone';
import { WebcamStudio } from './components/WebcamStudio';
import { ResultsPanel } from './components/ResultsPanel';
import { HistoryAnalytics } from './components/HistoryAnalytics';
import { ModelDashboard } from './components/ModelDashboard';
import ocrApi, { type OCRResult } from './services/api';

export function App() {
  const [activeTab, setActiveTab] = useState<string>('workspace');
  const [predictionResult, setPredictionResult] = useState<OCRResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileSubmit = async (file: File, task: 'character' | 'word' | 'sentence', engine: string, model: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await ocrApi.predict(file, task, true);
      if (engine !== 'auto' && res) res.engine_used = engine;
      if (model !== 'auto' && res) res.model_used = model;
      setPredictionResult(res);
    } catch (err: any) {
      console.error('OCR Prediction failure:', err);
      setError('Neural OCR Inference failed: ' + (err?.response?.data?.detail || err.message || 'Please ensure backend server is online at port 8002.'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Header activeTab={activeTab} setActiveTab={setActiveTab} />

      <main>
        {activeTab === 'workspace' && (
          <div className="workspace-grid">
            <FileDropzone onFileSelect={handleFileSubmit} loading={loading} />
            <ResultsPanel result={predictionResult} loading={loading} error={error} />
          </div>
        )}

        {activeTab === 'webcam' && (
          <div className="workspace-grid">
            <WebcamStudio
              onPredictionResult={(res) => { setPredictionResult(res); }}
              setLoading={setLoading}
              setError={setError}
              loading={loading}
            />
            <ResultsPanel result={predictionResult} loading={loading} error={error} />
          </div>
        )}

        {activeTab === 'history' && (
          <HistoryAnalytics />
        )}

        {activeTab === 'models' && (
          <ModelDashboard />
        )}
      </main>

      <footer style={{ marginTop: '70px', borderTop: '1px solid var(--border-color)', paddingTop: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>
        <div>
          <span>Powered by <strong>PyTorch, FastAPI, EasyOCR & Tesseract 5.0</strong></span>
        </div>
        <div>
          <span>Maharaja Production Architecture • Version 1.0.0</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
