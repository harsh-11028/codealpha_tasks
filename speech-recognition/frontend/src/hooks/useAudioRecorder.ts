// src/hooks/useAudioRecorder.ts
// Collects all recorded chunks into a single Blob so the app can
// preview / replay / submit the recording after stopping.

import { useState, useRef, useCallback } from 'react';

export interface RecorderState {
  /** Recorder is actively capturing audio */
  isRecording: boolean;
  /** Elapsed seconds */
  duration: number;
  /** Any mic / browser error */
  error: string | null;
  /** Blob of the complete recording (available after stop) */
  audioBlob: Blob | null;
  /** Object-URL pointing at audioBlob – use as <audio src> */
  audioUrl: string | null;
}

const MIME_TYPES_PREFERRED = [
  'audio/webm;codecs=opus',
  'audio/webm',
  'audio/ogg;codecs=opus',
  'audio/ogg',
  'audio/mp4',
];

function getSupportedMime(): string {
  for (const type of MIME_TYPES_PREFERRED) {
    if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(type)) {
      return type;
    }
  }
  return ''; // let browser choose
}

export const useAudioRecorder = () => {
  const [state, setState] = useState<RecorderState>({
    isRecording: false,
    duration: 0,
    error: null,
    audioBlob: null,
    audioUrl: null,
  });

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  /** Revoke any old object-URL to avoid memory leaks */
  const revokeUrl = useCallback(() => {
    setState((prev) => {
      if (prev.audioUrl) URL.revokeObjectURL(prev.audioUrl);
      return { ...prev, audioUrl: null, audioBlob: null };
    });
  }, []);

  /**
   * Start recording. Optional `onChunk` callback receives each base64-encoded
   * audio chunk (for WebSocket streaming use-cases).
   */
  const startRecording = useCallback(
    async (onChunk?: (base64: string) => void) => {
      // Clean up any previous recording
      revokeUrl();
      chunksRef.current = [];

      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        streamRef.current = stream;

        const mimeType = getSupportedMime();
        const options = mimeType ? { mimeType } : {};
        const recorder = new MediaRecorder(stream, options);
        mediaRecorderRef.current = recorder;

        recorder.ondataavailable = async (e) => {
          if (e.data && e.data.size > 0) {
            // Accumulate full-recording chunks
            chunksRef.current.push(e.data);

            // Also stream to WebSocket if caller cares
            if (onChunk) {
              const buffer = await e.data.arrayBuffer();
              const base64 = btoa(
                new Uint8Array(buffer).reduce(
                  (data, byte) => data + String.fromCharCode(byte),
                  ''
                )
              );
              onChunk(base64);
            }
          }
        };

        recorder.onstop = () => {
          const mimeForBlob = mimeType || 'audio/webm';
          const blob = new Blob(chunksRef.current, { type: mimeForBlob });
          const url = URL.createObjectURL(blob);
          setState((prev) => ({
            ...prev,
            isRecording: false,
            audioBlob: blob,
            audioUrl: url,
          }));
        };

        recorder.start(250); // emit chunks every 250 ms

        setState((prev) => ({
          ...prev,
          isRecording: true,
          error: null,
          duration: 0,
          audioBlob: null,
          audioUrl: null,
        }));

        timerRef.current = window.setInterval(() => {
          setState((prev) => ({ ...prev, duration: prev.duration + 1 }));
        }, 1000);
      } catch (err: any) {
        const msg =
          err?.name === 'NotAllowedError'
            ? 'Microphone permission denied. Please allow mic access in your browser settings.'
            : err?.name === 'NotFoundError'
            ? 'No microphone found. Please connect a microphone and try again.'
            : err?.message ?? 'Could not access microphone.';
        setState((prev) => ({ ...prev, error: msg }));
      }
    },
    [revokeUrl]
  );

  const stopRecording = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop(); // triggers onstop → sets audioBlob/audioUrl
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  }, []);

  /** Discard current recording so the user can start fresh */
  const resetRecording = useCallback(() => {
    stopRecording();
    revokeUrl();
    chunksRef.current = [];
    setState({
      isRecording: false,
      duration: 0,
      error: null,
      audioBlob: null,
      audioUrl: null,
    });
  }, [stopRecording, revokeUrl]);

  return {
    ...state,
    startRecording,
    stopRecording,
    resetRecording,
  };
};
