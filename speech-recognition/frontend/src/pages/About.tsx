// src/pages/About.tsx — Project architecture and tech stack
import { motion } from 'framer-motion';
import { ExternalLink, BookOpen, Layers, Brain, Server, Monitor } from 'lucide-react';

const techStack = [
  {
    icon: Monitor,
    layer: 'Frontend',
    color: 'text-cyan-400',
    bg: 'bg-cyan-400/10',
    border: 'border-cyan-400/20',
    items: ['React 18', 'TypeScript 5', 'Vite', 'Tailwind CSS', 'Framer Motion', 'Recharts'],
  },
  {
    icon: Server,
    layer: 'Backend',
    color: 'text-blue-400',
    bg: 'bg-blue-400/10',
    border: 'border-blue-400/20',
    items: ['FastAPI', 'Uvicorn', 'SQLAlchemy', 'WebSockets', 'Pydantic', 'SQLite'],
  },
  {
    icon: Brain,
    layer: 'Machine Learning',
    color: 'text-purple-400',
    bg: 'bg-purple-400/10',
    border: 'border-purple-400/20',
    items: ['PyTorch 2.x', 'Wav2Vec 2.0', 'HuggingFace', 'CNN', 'BiLSTM', 'CNN+Attention'],
  },
  {
    icon: Layers,
    layer: 'Audio Processing',
    color: 'text-emerald-400',
    bg: 'bg-emerald-400/10',
    border: 'border-emerald-400/20',
    items: ['Librosa', 'SoundFile', 'SciPy', 'MFCC', 'Mel Spectrogram', 'Chroma'],
  },
];

const steps = [
  { n: '01', title: 'Audio Input', desc: 'Live microphone stream or uploaded audio file (WAV, MP3, OGG).' },
  { n: '02', title: 'Preprocessing', desc: 'Resampling, silence trimming, spectral gating noise reduction.' },
  { n: '03', title: 'Feature Extraction', desc: '11 acoustic features: MFCC, Δ-MFCC, Mel Spectrogram, Chroma, ZCR, RMS, Spectral Centroid, Rolloff, Bandwidth, Contrast, Tonnetz.' },
  { n: '04', title: 'Deep Learning', desc: 'Feature tensor fed into the active PyTorch model (CNN, BiLSTM, Wav2Vec2, etc.).' },
  { n: '05', title: 'Explainability', desc: 'Gradient-based saliency maps (XAI) highlight which audio features drove the prediction.' },
  { n: '06', title: 'Results', desc: 'Emotion, confidence scores, waveform, spectrogram, and feature importance returned in real-time.' },
];

export const About = () => (
  <div className="min-h-screen bg-background text-foreground">
    <div className="absolute top-10 left-0 w-[35%] h-[35%] rounded-full bg-amber-600/10 blur-[120px] pointer-events-none" />

    <div className="container mx-auto px-4 py-10 relative z-10 max-w-5xl">
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-10">
        <h1 className="text-4xl font-outfit font-bold mb-2">About</h1>
        <p className="text-muted-foreground max-w-2xl">
          Speech Emotion Recognition (SER) is a deep learning system that classifies human emotions
          from audio speech signals in real-time. Built as a major AI/ML project.
        </p>
      </motion.div>

      {/* How it works */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="glass-panel rounded-2xl p-6 mb-8"
      >
        <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-blue-400" /> How It Works
        </h2>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {steps.map(({ n, title, desc }) => (
            <div key={n} className="flex gap-3">
              <div className="w-8 h-8 shrink-0 rounded-lg bg-blue-500/20 text-blue-400 text-xs font-bold flex items-center justify-center">
                {n}
              </div>
              <div>
                <p className="font-medium text-sm text-white mb-1">{title}</p>
                <p className="text-xs text-muted-foreground leading-relaxed">{desc}</p>
              </div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Tech Stack */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="mb-8"
      >
        <h2 className="text-xl font-semibold mb-5">Tech Stack</h2>
        <div className="grid sm:grid-cols-2 gap-4">
          {techStack.map(({ icon: Icon, layer, color, bg, border, items }) => (
            <div
              key={layer}
              className={`glass-panel rounded-2xl p-5 border ${border}`}
            >
              <div className="flex items-center gap-3 mb-4">
                <div className={`p-2 rounded-lg ${bg}`}>
                  <Icon className={`w-4 h-4 ${color}`} />
                </div>
                <h3 className="font-semibold">{layer}</h3>
              </div>
              <div className="flex flex-wrap gap-2">
                {items.map((item) => (
                  <span
                    key={item}
                    className="text-xs px-2 py-1 rounded-md bg-white/5 text-muted-foreground"
                  >
                    {item}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* GitHub */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="glass-panel rounded-2xl p-6 flex items-center justify-between"
      >
        <div>
          <h3 className="font-semibold mb-1">Open Source</h3>
          <p className="text-sm text-muted-foreground">View the full source code on GitHub</p>
        </div>
        <a
          href="https://github.com/yourusername/speech-emotion-recognition"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-white/10 hover:bg-white/15 transition-colors text-sm font-medium"
        >
          <ExternalLink className="w-4 h-4" /> GitHub
        </a>
      </motion.div>
    </div>
  </div>
);
