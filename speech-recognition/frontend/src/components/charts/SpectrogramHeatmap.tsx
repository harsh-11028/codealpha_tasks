// src/components/charts/SpectrogramHeatmap.tsx
// CSS-grid based heatmap — efficient for large 2D arrays

interface Props {
  data: number[][];  // [n_mels][time_frames]  values 0–1
  label?: string;
}

const lerp = (t: number) => {
  // Deep blue → cyan → yellow colour ramp (viridis-like)
  if (t < 0.25) {
    const s = t / 0.25;
    return `rgb(${Math.round(s * 32)}, ${Math.round(s * 144)}, ${Math.round(200 + s * 55)})`;
  } else if (t < 0.5) {
    const s = (t - 0.25) / 0.25;
    return `rgb(${Math.round(32 + s * 100)}, ${Math.round(144 + s * 96)}, ${Math.round(255 - s * 80)})`;
  } else if (t < 0.75) {
    const s = (t - 0.5) / 0.25;
    return `rgb(${Math.round(132 + s * 100)}, ${Math.round(240 - s * 60)}, ${Math.round(175 - s * 100)})`;
  } else {
    const s = (t - 0.75) / 0.25;
    return `rgb(${Math.round(232 + s * 23)}, ${Math.round(180 + s * 75)}, ${Math.round(75 - s * 75)})`;
  }
};

export const SpectrogramHeatmap = ({ data, label = 'Mel Spectrogram' }: Props) => {
  if (!data || data.length === 0) {
    return (
      <div className="w-full h-32 flex items-center justify-center text-muted-foreground text-sm">
        No spectrogram data
      </div>
    );
  }

  // Downsample for performance: max 80 rows, 200 cols
  const stepRow = Math.max(1, Math.floor(data.length / 80));
  const stepCol = Math.max(1, Math.floor((data[0]?.length ?? 1) / 200));
  const rows = data.filter((_, i) => i % stepRow === 0);
  const cols = rows[0]?.length ?? 0;

  return (
    <div>
      <p className="text-xs text-muted-foreground mb-1">{label}</p>
      <div
        className="w-full rounded overflow-hidden"
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${Math.ceil(cols / stepCol)}, 1fr)`,
          gridTemplateRows: `repeat(${rows.length}, 3px)`,
          height: `${rows.length * 3}px`,
        }}
      >
        {[...rows].reverse().map((row, ri) =>
          row
            .filter((_, ci) => ci % stepCol === 0)
            .map((val, ci) => (
              <div
                key={`${ri}-${ci}`}
                style={{ backgroundColor: lerp(val) }}
              />
            ))
        )}
      </div>
    </div>
  );
};
