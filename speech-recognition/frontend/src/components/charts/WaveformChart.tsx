// src/components/charts/WaveformChart.tsx
import {
  LineChart,
  Line,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';

interface Props {
  data: number[];
  color?: string;
}

export const WaveformChart = ({ data, color = '#60a5fa' }: Props) => {
  const chartData = data.map((v, i) => ({ i, v }));

  return (
    <div className="w-full h-28">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData}>
          <Line
            type="monotone"
            dataKey="v"
            stroke={color}
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
          <Tooltip
            contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8 }}
            labelFormatter={() => ''}
            formatter={(v: unknown) => [(v as number).toFixed(4), 'Amplitude']}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};
