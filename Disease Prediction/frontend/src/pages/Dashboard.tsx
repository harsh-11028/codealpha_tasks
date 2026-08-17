import { useEffect, useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { dashboardApi } from '../services/api';
import type { DashboardStats } from '../types';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { LoadingSpinner } from '../components/ui/LoadingSpinner';
import { Activity, Heart, Droplets, Zap, AlertCircle } from 'lucide-react';
import { PieChart, Pie, Cell, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Legend } from 'recharts';
import { Link } from 'react-router-dom';
import { cn } from '../utils/cn';

const COLORS = ['#2563EB', '#16A34A', '#DC2626', '#F59E0B'];

export default function Dashboard() {
  const { user } = useAuth();
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    dashboardApi.getStats()
      .then(data => setStats(data.data))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="flex h-64 items-center justify-center"><LoadingSpinner size="lg" /></div>;
  if (!stats) return <div>Failed to load dashboard statistics.</div>;

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-2xl font-bold text-slate-900">Welcome back, {user?.name}</h2>
          <p className="text-slate-500">Here's an overview of your disease prediction activity.</p>
        </div>
        <div className="flex space-x-2">
          <Link to="/predict" className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 font-medium text-sm transition-colors">
            New Prediction
          </Link>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardContent className="p-6 flex items-center space-x-4">
            <div className="p-3 bg-blue-100 text-blue-600 rounded-full"><Activity className="h-6 w-6" /></div>
            <div>
              <p className="text-sm font-medium text-slate-500">Total Predictions</p>
              <h3 className="text-2xl font-bold text-slate-900">{stats.total_predictions}</h3>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 flex items-center space-x-4">
            <div className="p-3 bg-red-100 text-red-600 rounded-full"><Heart className="h-6 w-6" /></div>
            <div>
              <p className="text-sm font-medium text-slate-500">Heart Disease</p>
              <h3 className="text-2xl font-bold text-slate-900">{stats.heart_predictions}</h3>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 flex items-center space-x-4">
            <div className="p-3 bg-blue-100 text-blue-500 rounded-full"><Droplets className="h-6 w-6" /></div>
            <div>
              <p className="text-sm font-medium text-slate-500">Diabetes</p>
              <h3 className="text-2xl font-bold text-slate-900">{stats.diabetes_predictions}</h3>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 flex items-center space-x-4">
            <div className="p-3 bg-purple-100 text-purple-600 rounded-full"><Zap className="h-6 w-6" /></div>
            <div>
              <p className="text-sm font-medium text-slate-500">Breast Cancer</p>
              <h3 className="text-2xl font-bold text-slate-900">{stats.breast_cancer_predictions}</h3>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Disease Distribution</CardTitle>
          </CardHeader>
          <CardContent className="h-[300px]">
            {stats.disease_distribution.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={stats.disease_distribution} cx="50%" cy="50%" innerRadius={60} outerRadius={80} paddingAngle={5} dataKey="value">
                    {stats.disease_distribution.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <RechartsTooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-slate-500">No data available</div>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Prediction Trend (Last 30 Days)</CardTitle>
          </CardHeader>
          <CardContent className="h-[300px]">
            {stats.prediction_trend.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={stats.prediction_trend} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                  <Line type="monotone" dataKey="count" stroke="#2563EB" strokeWidth={2} />
                  <CartesianGrid stroke="#ccc" strokeDasharray="5 5" vertical={false} />
                  <XAxis dataKey="date" tick={{fontSize: 12}} />
                  <YAxis tick={{fontSize: 12}} />
                  <RechartsTooltip />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-slate-500">No data available</div>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Link to="/predict/heart" className="group block h-full">
          <Card className="h-full transition-all hover:shadow-md hover:border-blue-200">
            <CardContent className="p-6 text-center space-y-3">
              <div className="mx-auto w-12 h-12 bg-red-50 text-red-500 rounded-full flex items-center justify-center group-hover:scale-110 transition-transform"><Heart /></div>
              <h3 className="font-semibold text-slate-900">Heart Disease</h3>
              <p className="text-sm text-slate-500">Analyze 13 clinical features to predict presence of heart disease.</p>
            </CardContent>
          </Card>
        </Link>
        <Link to="/predict/diabetes" className="group block h-full">
          <Card className="h-full transition-all hover:shadow-md hover:border-blue-200">
            <CardContent className="p-6 text-center space-y-3">
              <div className="mx-auto w-12 h-12 bg-blue-50 text-blue-500 rounded-full flex items-center justify-center group-hover:scale-110 transition-transform"><Droplets /></div>
              <h3 className="font-semibold text-slate-900">Diabetes</h3>
              <p className="text-sm text-slate-500">Predict onset of diabetes based on diagnostic measurements.</p>
            </CardContent>
          </Card>
        </Link>
        <Link to="/predict/breast-cancer" className="group block h-full">
          <Card className="h-full transition-all hover:shadow-md hover:border-blue-200">
            <CardContent className="p-6 text-center space-y-3">
              <div className="mx-auto w-12 h-12 bg-purple-50 text-purple-600 rounded-full flex items-center justify-center group-hover:scale-110 transition-transform"><Zap /></div>
              <h3 className="font-semibold text-slate-900">Breast Cancer</h3>
              <p className="text-sm text-slate-500">Classify tumors as malignant or benign using cell nuclei features.</p>
            </CardContent>
          </Card>
        </Link>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Recent Predictions</CardTitle>
        </CardHeader>
        <CardContent>
          {stats.recent_predictions.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left text-slate-500">
                <thead className="text-xs text-slate-700 uppercase bg-slate-50">
                  <tr>
                    <th className="px-4 py-3 rounded-tl-lg">Date</th>
                    <th className="px-4 py-3">Disease</th>
                    <th className="px-4 py-3">Result</th>
                    <th className="px-4 py-3">Probability</th>
                    <th className="px-4 py-3 rounded-tr-lg">Model</th>
                  </tr>
                </thead>
                <tbody>
                  {stats.recent_predictions.map((p) => (
                    <tr key={p.id} className="border-b border-slate-100 last:border-0 hover:bg-slate-50">
                      <td className="px-4 py-3 font-medium text-slate-900">{new Date(p.created_at).toLocaleString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false })}</td>
                      <td className="px-4 py-3 capitalize">{p.disease.replace('_', ' ')}</td>
                      <td className="px-4 py-3">
                        <Badge variant={p.prediction === 1 ? 'danger' : 'success'}>
                          {p.prediction === 1 ? 'Positive' : 'Negative'}
                        </Badge>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-2 bg-slate-200 rounded-full overflow-hidden">
                            <div 
                              className={cn("h-full", p.prediction === 1 ? "bg-red-500" : "bg-green-500")} 
                              style={{ width: `${p.probability * 100}%` }} 
                            />
                          </div>
                          <span>{(p.probability * 100).toFixed(1)}%</span>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-xs text-slate-500">{p.model_used}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-slate-500 text-center py-4">No recent predictions found.</p>
          )}
        </CardContent>
      </Card>

      <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4 text-sm text-yellow-800 flex items-start">
        <AlertCircle className="h-5 w-5 mr-2 shrink-0 mt-0.5" />
        <p><strong>Medical Disclaimer:</strong> The predictions and analysis provided by this application are powered by machine learning algorithms for research and educational purposes only. They do not constitute professional medical advice, diagnosis, or treatment.</p>
      </div>
    </div>
  );
}
