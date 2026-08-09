// src/components/charts/FeatureImportanceBar.tsx
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';

const COLORS = ['#60a5fa', '#a78bfa', '#34d399', '#fb923c'];

interface Props {
  data: Record<string, number>;
}

export const FeatureImportanceBar = ({ data }: Props) => {
  const chartData = Object.entries(data)
    .sort(([, a], [, b]) => b - a)
    .map(([name, value]) => ({ name, value: parseFloat(value.toFixed(4)) }));

  return (
    <div className="w-full h-48">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} layout="vertical" margin={{ left: 20 }}>
          <XAxis type="number" domain={[0, 'auto']} tick={{ fill: '#64748b', fontSize: 11 }} />
          <YAxis
            dataKey="name"
            type="category"
            width={130}
            tick={{ fill: '#94a3b8', fontSize: 11 }}
          />
          <Tooltip
            contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8 }}
            formatter={(v: unknown) => [(v as number).toFixed(4), 'Saliency']}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {chartData.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
