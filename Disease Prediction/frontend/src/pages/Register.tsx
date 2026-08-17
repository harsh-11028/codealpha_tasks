import { useState } from 'react';
import { Link, Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { Heart, Activity, ShieldCheck } from 'lucide-react';

export default function Register() {
  const { register, isAuthenticated } = useAuth();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  if (isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }
    setLoading(true);
    try {
      await register({ name, email, password });
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to register');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex bg-slate-50">
      <div className="hidden lg:flex lg:w-1/2 bg-blue-600 text-white p-12 flex-col justify-between">
        <div>
          <div className="flex items-center space-x-2 text-2xl font-bold mb-12">
            <Heart className="h-8 w-8" />
            <span>MedPredict AI</span>
          </div>
          <h1 className="text-4xl font-bold leading-tight mb-6">Join the Future of Healthcare</h1>
          <p className="text-blue-100 text-lg mb-8 max-w-md">Create an account to access advanced predictive models and assist in early disease detection.</p>
          <ul className="space-y-4">
            <li className="flex items-center space-x-3">
              <Activity className="h-6 w-6 text-blue-200" />
              <span>State-of-the-art algorithms</span>
            </li>
            <li className="flex items-center space-x-3">
              <ShieldCheck className="h-6 w-6 text-blue-200" />
              <span>HIPAA compliant architecture</span>
            </li>
          </ul>
        </div>
        <div className="text-blue-200 text-sm">
          <p>© {new Date().getFullYear()} MedPredict AI. All rights reserved.</p>
        </div>
      </div>

      <div className="w-full lg:w-1/2 flex items-center justify-center p-8">
        <div className="w-full max-w-md space-y-8">
          <div className="text-center">
            <h2 className="text-3xl font-bold text-slate-900">Create an account</h2>
            <p className="mt-2 text-slate-600">Enter your details to get started</p>
          </div>

          <form onSubmit={handleSubmit} className="mt-8 space-y-6">
            <div className="space-y-4">
              <Input
                label="Full Name"
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Dr. Jane Doe"
              />
              <Input
                label="Email address"
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="doctor@hospital.com"
              />
              <Input
                label="Password"
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
              />
              <Input
                label="Confirm Password"
                type="password"
                required
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder="••••••••"
              />
            </div>

            {error && <div className="text-red-500 text-sm font-medium text-center bg-red-50 p-2 rounded">{error}</div>}

            <Button type="submit" className="w-full" isLoading={loading}>
              Register
            </Button>
          </form>

          <p className="text-center text-sm text-slate-600">
            Already have an account?{' '}
            <Link to="/login" className="font-semibold text-blue-600 hover:text-blue-500">
              Sign in
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
