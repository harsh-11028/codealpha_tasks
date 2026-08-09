// src/components/charts/EmotionRadar.tsx
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  Radar,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';
import { EMOTION_META } from '../../types';
import type { EmotionName } from '../../types';

interface Props {
  probabilities: Record<string, number>;
  color?: string;
}

export const EmotionRadar = ({ probabilities, color = '#60a5fa' }: Props) => {
  const data = Object.entries(probabilities).map(([emotion, value]) => ({
    emotion: EMOTION_META[emotion as EmotionName]?.emoji
      ? `${EMOTION_META[emotion as EmotionName].emoji} ${emotion}`
      : emotion,
    value: Math.round(value * 100),
  }));

  return (
    <div className="w-full h-64">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart data={data}>
          <PolarGrid stroke="#1e293b" />
          <PolarAngleAxis
            dataKey="emotion"
            tick={{ fill: '#94a3b8', fontSize: 11 }}
          />
          <Radar
            dataKey="value"
            stroke={color}
            fill={color}
            fillOpacity={0.25}
            strokeWidth={2}
          />
          <Tooltip
            contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8 }}
            formatter={(v: unknown) => [`${v}%`, 'Confidence']}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
};
