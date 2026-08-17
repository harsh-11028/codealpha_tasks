import { useState, useEffect } from 'react';
import { modelsApi } from '../services/api';
import type { ModelMetrics } from '../types';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/Card';
import { LoadingSpinner } from '../components/ui/LoadingSpinner';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { cn } from '../utils/cn';

export default function Models() {
  const [activeTab, setActiveTab] = useState('heart');
  const [metrics, setMetrics] = useState<ModelMetrics[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    modelsApi.getPerformance(activeTab)
      .then(res => setMetrics(res.data))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [activeTab]);

  const tabs = [
    { id: 'heart', label: 'Heart Disease' },
    { id: 'diabetes', label: 'Diabetes' },
    { id: 'breast_cancer', label: 'Breast Cancer' },
  ];

  const bestModel = metrics.length > 0 ? metrics.reduce((prev, current) => (prev.accuracy > current.accuracy) ? prev : current) : null;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">Model Performance Comparison</h2>
        <p className="text-slate-500">Analyze and compare different machine learning algorithms used for prediction.</p>
      </div>

      <div className="flex border-b border-slate-200 space-x-8">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={cn("py-4 text-sm font-medium border-b-2 transition-colors", activeTab === tab.id ? "border-blue-600 text-blue-600" : "border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300")}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="flex justify-center py-20"><LoadingSpinner size="lg" /></div>
      ) : (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Metrics Overview ({activeTab.replace('_', ' ')})</CardTitle>
                <CardDescription>Comparison of accuracy, precision, recall and F1-score.</CardDescription>
              </CardHeader>
              <CardContent className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={metrics} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="algorithm" />
                    <YAxis domain={[0, 1]} />
                    <Tooltip cursor={{fill: '#f1f5f9'}} />
                    <Legend />
                    <Bar dataKey="accuracy" fill="#2563EB" name="Accuracy" />
                    <Bar dataKey="precision" fill="#16A34A" name="Precision" />
                    <Bar dataKey="recall" fill="#F59E0B" name="Recall" />
                    <Bar dataKey="f1_score" fill="#9333EA" name="F1 Score" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Detailed Metrics Table</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm text-left">
                    <thead className="text-xs uppercase bg-slate-50 text-slate-500 border-b">
                      <tr>
                        <th className="px-4 py-3">Algorithm</th>
                        <th className="px-4 py-3">Accuracy</th>
                        <th className="px-4 py-3">Precision</th>
                        <th className="px-4 py-3">Recall</th>
                        <th className="px-4 py-3">F1 Score</th>
                        <th className="px-4 py-3">ROC-AUC</th>
                      </tr>
                    </thead>
                    <tbody>
                      {metrics.map(m => (
                        <tr key={m.algorithm} className={cn("border-b", m.algorithm === bestModel?.algorithm ? "bg-blue-50/50" : "")}>
                          <td className="px-4 py-3 font-medium text-slate-900">
                            {m.algorithm}
                            {m.algorithm === bestModel?.algorithm && <span className="ml-2 text-[10px] bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded-full">BEST</span>}
                          </td>
                          <td className="px-4 py-3 text-slate-600">{(m.accuracy * 100).toFixed(2)}%</td>
                          <td className="px-4 py-3 text-slate-600">{(m.precision * 100).toFixed(2)}%</td>
                          <td className="px-4 py-3 text-slate-600">{(m.recall * 100).toFixed(2)}%</td>
                          <td className="px-4 py-3 text-slate-600">{(m.f1_score * 100).toFixed(2)}%</td>
                          <td className="px-4 py-3 text-slate-600">{(m.roc_auc * 100).toFixed(2)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </div>
          
          <Card className="bg-slate-50">
            <CardContent className="p-6">
              <h3 className="font-semibold text-slate-900 mb-3">Understanding the Metrics</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-slate-600">
                <div><strong>Accuracy:</strong> The proportion of correctly predicted cases out of all predictions.</div>
                <div><strong>Precision:</strong> Out of all positive predictions, how many were actually positive.</div>
                <div><strong>Recall (Sensitivity):</strong> Out of all actual positives, how many were correctly predicted.</div>
                <div><strong>F1 Score:</strong> The harmonic mean of Precision and Recall. Better for imbalanced datasets.</div>
                <div><strong>ROC-AUC:</strong> Measures the model's ability to distinguish between classes. 1.0 is perfect.</div>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
