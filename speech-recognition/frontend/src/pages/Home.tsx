import { LiveRecorder } from '../components/LiveRecorder';

export const Home = () => {
  return (
    <div className="min-h-screen bg-background text-foreground relative overflow-hidden">
      {/* Decorative ambient background */}
      <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-primary/20 blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] rounded-full bg-accent/20 blur-[120px] pointer-events-none" />
      
      <main className="container mx-auto px-4 py-12 lg:py-24 relative z-10 flex flex-col items-center min-h-screen">
        
        {/* Header section */}
        <div className="text-center mb-16 space-y-4 max-w-3xl">
          <div className="inline-flex items-center rounded-full border border-white/10 bg-white/5 px-3 py-1 text-sm font-medium backdrop-blur-sm mb-4">
            <span className="flex h-2 w-2 rounded-full bg-green-500 mr-2 animate-pulse"></span>
            System Online • Wav2Vec 2.0 Active
          </div>
          <h1 className="text-5xl md:text-7xl font-outfit font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white via-white/90 to-white/50">
            Understand Emotion <br className="hidden md:block"/> in Every Voice.
          </h1>
          <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto font-light leading-relaxed">
            State-of-the-art Speech Emotion Recognition powered by Deep Learning. 
            Speak naturally and watch the AI decode the emotional spectrum of your voice in real-time.
          </p>
        </div>

        {/* Main interactive section */}
        <div className="w-full">
          <LiveRecorder />
        </div>

        {/* Features banner */}
        <div className="mt-24 grid grid-cols-1 md:grid-cols-3 gap-8 w-full max-w-5xl opacity-80">
          <div className="glass-panel p-6 rounded-2xl flex flex-col items-center text-center">
            <div className="h-12 w-12 rounded-full bg-primary/20 flex items-center justify-center mb-4 text-primary">⚡</div>
            <h3 className="font-semibold text-lg mb-2">Real-time Processing</h3>
            <p className="text-sm text-muted-foreground">Streaming WebSocket inference with sub-100ms latency.</p>
          </div>
          <div className="glass-panel p-6 rounded-2xl flex flex-col items-center text-center">
            <div className="h-12 w-12 rounded-full bg-purple-500/20 flex items-center justify-center mb-4 text-purple-400">🧠</div>
            <h3 className="font-semibold text-lg mb-2">Transformer Powered</h3>
            <p className="text-sm text-muted-foreground">Utilizing HuggingFace Wav2Vec 2.0 for superior acoustic modeling.</p>
          </div>
          <div className="glass-panel p-6 rounded-2xl flex flex-col items-center text-center">
            <div className="h-12 w-12 rounded-full bg-emerald-500/20 flex items-center justify-center mb-4 text-emerald-400">🔍</div>
            <h3 className="font-semibold text-lg mb-2">Explainable AI</h3>
            <p className="text-sm text-muted-foreground">Gradient-based saliency mapping to highlight emotional triggers.</p>
          </div>
        </div>

      </main>
    </div>
  );
};
