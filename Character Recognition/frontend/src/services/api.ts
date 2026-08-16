import axios from 'axios';

const BASE_URL = 'http://localhost:8000/api';

export interface BoundingBox {
  x: number;
  y: number;
  w: number;
  h: number;
  text?: string;
  char?: string;
}

export interface OCRResult {
  text: string;
  confidence: float_score;
  processing_ms: number;
  engine_used?: string;
  model_used?: string;
  word_boxes?: BoundingBox[];
  char_boxes?: BoundingBox[];
  line_boxes?: BoundingBox[];
  confidence_stats?: {
    overall?: number;
    min?: number;
    max?: number;
    low_conf_fraction?: number;
  };
  annotated_image?: string; // base64
}

export interface PredictionRecord {
  id: number;
  session_id: string;
  input_type: string;
  raw_text: string;
  confidence: number;
  model_used: string;
  engine_used: string;
  processing_ms: number;
  created_at: string;
}

export interface ModelInfo {
  name: string;
  parameters: number;
  metrics: Record<string, number>;
  device: string;
}

export interface ModelInfoResponse {
  active_model: string;
  all_models: ModelInfo[];
  total_predictions: number;
}

type float_score = number;

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
});

export const ocrApi = {
  async predict(file: File, task: 'character' | 'word' | 'sentence', annotate = true): Promise<OCRResult> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('annotate', annotate ? 'true' : 'false');

    const endpoint = `/predict-${task}`;
    const response = await api.post(endpoint, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    
    // Normalize character prediction response to uniform OCRResult structure
    if (task === 'character') {
      return {
        text: response.data.char || response.data.text || '?',
        confidence: response.data.confidence || 0,
        processing_ms: response.data.processing_ms || 0,
        engine_used: 'custom',
        model_used: 'cnn/vit'
      };
    }
    
    if (task === 'word') {
      return {
        text: response.data.word || response.data.text || '',
        confidence: response.data.confidence || 0,
        processing_ms: response.data.processing_ms || 0,
        engine_used: response.data.engine_used,
        model_used: response.data.model_used
      };
    }

    return response.data;
  },

  async uploadWebcam(base64Data: string, task = 'sentence'): Promise<OCRResult> {
    // Convert base64 data URL to Blob / File to pass through regular pipeline if needed,
    // or call /webcam endpoint
    const response = await api.post('/webcam', {
      image_base64: base64Data,
      task
    });
    
    // Once uploaded to webcam endpoint, fetch image as File to run prediction
    const res = await fetch(`http://localhost:8000${response.data.image_url}`);
    const blob = await res.blob();
    const file = new File([blob], 'webcam_capture.png', { type: 'image/png' });
    return this.predict(file, task as any);
  },

  async getHistory(limit = 50, input_type?: string): Promise<PredictionRecord[]> {
    const params: Record<string, any> = { limit };
    if (input_type) params.input_type = input_type;
    const response = await api.get('/history', { params });
    return response.data;
  },

  async deleteHistory(id: number): Promise<void> {
    await api.delete(`/history/${id}`);
  },

  async getHistoryStats(): Promise<any> {
    const response = await api.get('/history/stats');
    return response.data;
  },

  async getHealth(): Promise<{ status: string; model_loaded: boolean; device: string; uptime_seconds: number }> {
    const response = await api.get('/health');
    return response.data;
  },

  async getModelInfo(): Promise<ModelInfoResponse> {
    const response = await api.get('/model-info');
    return response.data;
  },

  async exportText(text: string, format: 'txt' | 'pdf' | 'docx', filename = 'ocr_export'): Promise<void> {
    const response = await api.post('/export', null, {
      params: { text, format, filename },
      responseType: 'blob',
    });
    
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `${filename}.${format}`);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
  }
};

export default ocrApi;
