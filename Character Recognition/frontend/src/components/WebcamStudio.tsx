import React, { useRef, useState, useCallback, useEffect } from 'react';
import { Camera, Radio } from 'lucide-react';
import ocrApi, { type OCRResult } from '../services/api';

interface WebcamStudioProps {
  onPredictionResult: (result: OCRResult) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  loading: boolean;
}

export const WebcamStudio: React.FC<WebcamStudioProps> = ({ onPredictionResult, setLoading, setError, loading }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  const [streamActive, setStreamActive] = useState<boolean>(false);
  const [liveLooping, setLiveLooping] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  const startCamera = async () => {
    try {
      setCameraError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: 'environment' }
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setStreamActive(true);
      }
    } catch (err) {
      setCameraError('Unable to access webcam. Please verify browser permissions or attach a camera device.');
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setStreamActive(false);
      setLiveLooping(false);
    }
  };

  const captureAndPredict = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || !streamActive) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const base64Image = canvas.toDataURL('image/png');

    setLoading(true);
    setError(null);
    try {
      const res = await ocrApi.uploadWebcam(base64Image, 'sentence');
      onPredictionResult(res);
    } catch (err: any) {
      setError('Webcam capture OCR failed: ' + (err?.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  }, [streamActive, setLoading, setError, onPredictionResult]);

  useEffect(() => {
    let intervalId: any = null;
    if (liveLooping && streamActive) {
      intervalId = setInterval(() => {
        if (!loading) {
          captureAndPredict();
        }
      }, 4000); // Sample every 4 seconds in live mode
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [liveLooping, streamActive, loading, captureAndPredict]);

  useEffect(() => {
    return () => stopCamera();
  }, []);

  return (
    <div className="card animate-fade-in">
      <div className="card-title">
        <span><Camera size={18} style={{ verticalAlign: 'middle', marginRight: '8px' }} /> Live Webcam Recognition Studio</span>
        <span className="chip" style={{ background: streamActive ? 'rgba(16, 185, 129, 0.15)' : 'rgba(255, 255, 255, 0.05)', borderColor: streamActive ? 'rgba(16, 185, 129, 0.4)' : 'var(--border-color)', color: streamActive ? '#10b981' : 'var(--text-muted)' }}>
          {streamActive ? '● Video Stream Active' : '● Camera Idle'}
        </span>
      </div>

      <div style={{ position: 'relative', borderRadius: 'var(--radius-sm)', overflow: 'hidden', background: '#07080e', minHeight: '360px', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px solid var(--border-color)' }}>
        {!streamActive ? (
          <div style={{ textAlign: 'center', padding: '40px 20px' }}>
            <Camera size={48} color="var(--primary)" style={{ marginBottom: '16px', filter: 'drop-shadow(0 0 10px rgba(124, 58, 237, 0.4))' }} />
            <h3 style={{ color: '#fff', fontSize: '18px', fontWeight: 600, marginBottom: '8px' }}>Webcam Device Offline</h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '14px', marginBottom: '24px', maxWidth: '300px', margin: '0 auto 24px auto' }}>
              Aim your camera at handwritten notes or physical papers to trigger real-time neural OCR transcription.
            </p>
            <button className="btn btn-primary" onClick={startCamera}>
              Enable Video Stream
            </button>
            {cameraError && (
              <p style={{ color: '#f43f5e', fontSize: '13px', marginTop: '16px' }}>{cameraError}</p>
            )}
          </div>
        ) : (
          <video ref={videoRef} autoPlay playsInline muted style={{ width: '100%', maxHeight: '420px', objectFit: 'cover' }} />
        )}
        <canvas ref={canvasRef} style={{ display: 'none' }} />
      </div>

      {streamActive && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '20px' }}>
          <div style={{ display: 'flex', gap: '12px' }}>
            <button
              className="btn btn-primary"
              onClick={captureAndPredict}
              disabled={loading}
              style={{ padding: '10px 20px' }}
            >
              <Camera size={16} /> {loading ? 'Scanning...' : 'Capture & Recognize Now'}
            </button>

            <button
              className={`btn ${liveLooping ? 'btn-danger' : 'btn-secondary'}`}
              style={{ padding: '10px 18px', borderColor: liveLooping ? '#f43f5e' : 'var(--border-color)' }}
              onClick={() => setLiveLooping(!liveLooping)}
            >
              <Radio size={16} color={liveLooping ? '#fff' : 'var(--secondary)'} />
              {liveLooping ? 'Stop Continuous Loop' : 'Enable Live Continuous Auto-Scan (4s intervals)'}
            </button>
          </div>

          <button className="btn btn-secondary" style={{ padding: '10px 14px', opacity: 0.8 }} onClick={stopCamera}>
            Disconnect Camera
          </button>
        </div>
      )}
    </div>
  );
};
