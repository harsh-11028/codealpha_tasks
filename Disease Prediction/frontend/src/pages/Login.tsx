import { useState, useEffect } from 'react';
import { Link, Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { HeartPulse, Activity, ShieldCheck, Stethoscope } from 'lucide-react';
import { cn } from '../utils/cn';

export default function Login() {
  const { login, isAuthenticated } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await login({ email, password });
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to login. Please check your credentials.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex w-full bg-slate-50 font-sans">
      {/* Left Panel - Visuals */}
      <div className="hidden lg:flex w-[55%] relative overflow-hidden bg-slate-900">
        {/* Animated Gradient Background */}
        <div className="absolute inset-0 bg-gradient-to-br from-blue-900 via-slate-900 to-slate-800 opacity-90" />
        
        {/* Decorative Circles */}
        <div className="absolute top-[-10%] left-[-10%] w-[500px] h-[500px] rounded-full bg-blue-600/20 blur-[100px] animate-pulse" style={{ animationDuration: '4s' }} />
        <div className="absolute bottom-[-10%] right-[-10%] w-[600px] h-[600px] rounded-full bg-blue-800/30 blur-[120px] animate-pulse" style={{ animationDuration: '6s', animationDelay: '1s' }} />

        {/* Content */}
        <div className="relative z-10 flex flex-col justify-between h-full p-16 text-white w-full">
          <div className={cn("flex items-center space-x-3 transition-all duration-1000 transform", mounted ? "translate-y-0 opacity-100" : "-translate-y-8 opacity-0")}>
            <div className="p-3 bg-blue-500/20 rounded-2xl backdrop-blur-md border border-white/10">
              <HeartPulse className="h-8 w-8 text-blue-400" />
            </div>
            <span className="text-3xl font-bold tracking-tight text-white">MedPredict <span className="text-blue-400">AI</span></span>
          </div>

          <div className={cn("max-w-xl transition-all duration-1000 delay-300 transform", mounted ? "translate-y-0 opacity-100" : "translate-y-8 opacity-0")}>
            <h1 className="text-5xl font-extrabold tracking-tight leading-[1.1] mb-6 text-white">
              The future of <br/>
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-blue-200">predictive healthcare.</span>
            </h1>
            <p className="text-slate-300 text-xl leading-relaxed mb-12 font-medium">
              Empowering medical professionals with state-of-the-art machine learning models for accurate, early disease detection.
            </p>

            <div className="grid grid-cols-1 gap-6">
              {[
                { icon: Activity, title: 'Multi-disease Diagnostics', desc: 'Heart disease, diabetes, and breast cancer prediction.' },
                { icon: ShieldCheck, title: 'Secure Processing', desc: 'End-to-end encrypted data handling and storage.' },
                { icon: Stethoscope, title: 'Clinical Precision', desc: 'Evaluated on standard clinical datasets with high accuracy.' }
              ].map((item, idx) => (
                <div key={idx} className="flex items-start space-x-4 p-4 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-md hover:bg-white/10 transition-colors duration-300">
                  <div className="p-2 bg-blue-500/20 rounded-lg text-blue-300">
                    <item.icon className="w-6 h-6" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-white text-lg">{item.title}</h3>
                    <p className="text-slate-400 text-sm mt-1">{item.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className={cn("text-slate-400 text-sm font-medium transition-all duration-1000 delay-500", mounted ? "opacity-100" : "opacity-0")}>
            © {new Date().getFullYear()} MedPredict AI. Secure System Access.
          </div>
        </div>
      </div>

      {/* Right Panel - Login Form */}
      <div className="w-full lg:w-[45%] flex items-center justify-center p-8 sm:p-12 lg:p-24 bg-white relative">
        <div className={cn("w-full max-w-md space-y-10 transition-all duration-700 transform", mounted ? "translate-y-0 opacity-100" : "translate-y-4 opacity-0")}>
          
          <div className="lg:hidden flex items-center space-x-3 mb-12">
             <div className="p-2 bg-blue-50 rounded-xl">
              <HeartPulse className="h-6 w-6 text-blue-600" />
            </div>
            <span className="text-2xl font-bold tracking-tight text-slate-900">MedPredict AI</span>
          </div>

          <div>
            <h2 className="text-4xl font-extrabold text-slate-900 tracking-tight">Welcome back</h2>
            <p className="mt-3 text-lg text-slate-500">Enter your credentials to access the system.</p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="space-y-5">
              <div className="space-y-1">
                <Input
                  label="Email address"
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="doctor@hospital.com"
                  className="h-12 text-base px-4 bg-slate-50 border-slate-200 focus:bg-white focus:ring-blue-500 focus:border-blue-500 transition-all"
                />
              </div>
              <div className="space-y-1">
                <div className="flex items-center justify-between mb-1">
                  <label className="block text-sm font-medium text-slate-700">Password</label>
                  <a href="#" className="text-sm font-medium text-blue-600 hover:text-blue-500 transition-colors">Forgot password?</a>
                </div>
                {/* Notice that Input internally already renders a label if passed, but since we wanted a custom layout with 'Forgot password', we skip the label prop and use our own above, and override the input styles. Wait, the custom Input component might have its own label. Let me check the Input component. For now I will omit 'label' prop on Password to avoid double labels. */}
                <Input
                  type="password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="h-12 text-base px-4 bg-slate-50 border-slate-200 focus:bg-white focus:ring-blue-500 focus:border-blue-500 transition-all"
                />
              </div>
            </div>

            {error && (
              <div className="p-4 rounded-xl bg-red-50 border border-red-100 flex items-start space-x-3 animate-in fade-in slide-in-from-top-2 duration-300">
                <div className="text-red-500 mt-0.5">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                </div>
                <p className="text-sm font-medium text-red-800">{error}</p>
              </div>
            )}

            <Button 
              type="submit" 
              className="w-full h-12 text-base font-semibold shadow-sm hover:shadow-md transition-all duration-300 bg-blue-600 hover:bg-blue-700" 
              isLoading={loading}
            >
              Sign in to Dashboard
            </Button>
          </form>

          <p className="text-center text-base text-slate-600">
            Don't have an account?{' '}
            <Link to="/register" className="font-semibold text-blue-600 hover:text-blue-700 transition-colors">
              Create an account
            </Link>
          </p>

          <div className="mt-12 p-4 bg-amber-50/50 border border-amber-100/50 rounded-xl text-xs text-amber-800/80 leading-relaxed text-center">
            <strong className="font-semibold text-amber-900">Medical Disclaimer:</strong> This system is for educational and demonstrative purposes only. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment.
          </div>
        </div>
      </div>
    </div>
  );
}
