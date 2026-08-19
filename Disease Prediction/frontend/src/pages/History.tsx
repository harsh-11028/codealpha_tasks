import { useState, useEffect } from 'react';
import { historyApi } from '../services/api';
import type { PredictionHistory } from '../types';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { Select } from '../components/ui/Select';
import { Input } from '../components/ui/Input';
import { LoadingSpinner } from '../components/ui/LoadingSpinner';
import { Trash2 } from 'lucide-react';
import { cn } from '../utils/cn';

export default function History() {
  const [history, setHistory] = useState<PredictionHistory[]>([]);
  const [loading, setLoading] = useState(true);
  const [filterDisease, setFilterDisease] = useState('');
  const [search, setSearch] = useState('');

  useEffect(() => {
    fetchHistory();
  }, [filterDisease]);

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const res = await historyApi.getHistory({ disease: filterDisease });
      setHistory(res.data?.predictions || []);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (confirm('Are you sure you want to delete this prediction record?')) {
      try {
        await historyApi.delete(id);
        setHistory(prev => prev.filter(h => h.id !== id));
      } catch (err) {
        console.error(err);
      }
    }
  };

  const filteredHistory = history.filter(h => h.id.includes(search) || h.model_used.toLowerCase().includes(search.toLowerCase()));

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-2xl font-bold text-slate-900">Prediction History</h2>
          <p className="text-slate-500">View and manage your past diagnostic predictions.</p>
        </div>
      </div>

      <Card>
        <CardHeader className="flex flex-col md:flex-row justify-between md:items-center gap-4 space-y-0 pb-4">
          <CardTitle>All Records</CardTitle>
          <div className="flex flex-col sm:flex-row gap-4 w-full md:w-auto">
            <Input placeholder="Search records..." value={search} onChange={e => setSearch(e.target.value)} className="w-full sm:w-64" />
            <Select 
              value={filterDisease} 
              onChange={e => setFilterDisease(e.target.value)} 
              options={[
                {label: 'All Diseases', value: ''},
                {label: 'Heart Disease', value: 'heart'},
                {label: 'Diabetes', value: 'diabetes'},
                {label: 'Breast Cancer', value: 'breast_cancer'}
              ]} 
              className="w-full sm:w-48"
            />
          </div>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="py-12 flex justify-center"><LoadingSpinner /></div>
          ) : filteredHistory.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left text-slate-500">
                <thead className="text-xs text-slate-700 uppercase bg-slate-50">
                  <tr>
                    <th className="px-4 py-3 rounded-tl-lg">Date</th>
                    <th className="px-4 py-3">Disease</th>
                    <th className="px-4 py-3">Result</th>
                    <th className="px-4 py-3">Probability</th>
                    <th className="px-4 py-3">Model</th>
                    <th className="px-4 py-3 rounded-tr-lg text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredHistory.map((item) => (
                    <tr key={item.id} className="border-b border-slate-100 hover:bg-slate-50">
                      <td className="px-4 py-4 text-slate-900">{new Date(item.created_at).toLocaleString('en-US', { dateStyle: 'medium', timeStyle: 'short' })}</td>
                      <td className="px-4 py-4 capitalize">{item.disease.replace('_', ' ')}</td>
                      <td className="px-4 py-4">
                        <Badge variant={item.prediction === 1 ? 'danger' : 'success'}>
                          {item.prediction === 1 ? 'Positive' : 'Negative'}
                        </Badge>
                      </td>
                      <td className="px-4 py-4">
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-2 bg-slate-200 rounded-full overflow-hidden">
                            <div 
                              className={cn("h-full", item.prediction === 1 ? "bg-red-500" : "bg-green-500")} 
                              style={{ width: `${item.probability * 100}%` }} 
                            />
                          </div>
                          <span className="text-xs font-medium">{(item.probability * 100).toFixed(1)}%</span>
                        </div>
                      </td>
                      <td className="px-4 py-4 text-xs">{item.model_used}</td>
                      <td className="px-4 py-4 text-right">
                        <Button variant="ghost" size="sm" onClick={() => handleDelete(item.id)} className="text-red-500 hover:text-red-700 hover:bg-red-50 px-2">
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-center py-12 text-slate-500">
              <p>No prediction history found.</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
