import { useState, useEffect, useRef, useCallback } from 'react';

export interface PredictionResult {
  emotion: string;
  confidence: number;
  confidence_pct: number;
  probabilities: Record<string, number>;
  emoji: string;
  color: string;
  inference_time_ms: number;
}

export const useWebSocketStream = (url: string) => {
  const [isConnected, setIsConnected] = useState(false);
  const [latestPrediction, setLatestPrediction] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  
  const connect = useCallback(() => {
    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = url.startsWith('ws') ? url : `${protocol}//${window.location.host}${url}`;
      
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        setIsConnected(true);
        setError(null);
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'prediction') {
            setLatestPrediction(data as PredictionResult);
          } else if (data.type === 'error') {
            setError(data.message);
          }
        } catch (err) {
          console.error("Failed to parse WS message", event.data);
        }
      };
      
      ws.onclose = () => {
        setIsConnected(false);
      };
      
      ws.onerror = () => {
        setError("WebSocket connection error");
        setIsConnected(false);
      };
      
      wsRef.current = ws;
    } catch (err: any) {
      setError(err.message || "Failed to establish WebSocket");
    }
  }, [url]);
  
  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
  }, []);
  
  const sendAudioChunk = useCallback((base64Data: string, sampleRate: number = 22050) => {
    if (wsRef.current && isConnected) {
      wsRef.current.send(JSON.stringify({
        type: 'audio_chunk',
        data: base64Data,
        sample_rate: sampleRate
      }));
    }
  }, [isConnected]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);
  
  return {
    isConnected,
    latestPrediction,
    error,
    connect,
    disconnect,
    sendAudioChunk
  };
};
