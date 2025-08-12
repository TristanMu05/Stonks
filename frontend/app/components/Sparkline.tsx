'use client';

import { useId, useMemo } from 'react';

type Props = {
  data: number[];
  width?: number;
  height?: number;
  stroke?: string;
  strokeWidth?: number;
  fillFrom?: string; // rgba or hex with alpha
  fillTo?: string;
};

// Convert a list of points into a smooth cubic-bezier SVG path using Catmull-Rom splines
function toSmoothPath(points: Array<[number, number]>): string {
  if (points.length < 2) return '';
  const d: string[] = [];
  const p = points;
  d.push(`M ${p[0][0]} ${p[0][1]}`);
  const crp = (i: number) => p[Math.max(0, Math.min(points.length - 1, i))];
  for (let i = 0; i < p.length - 1; i++) {
    const p0 = crp(i - 1);
    const p1 = crp(i);
    const p2 = crp(i + 1);
    const p3 = crp(i + 2);
    const c1x = p1[0] + (p2[0] - p0[0]) / 6;
    const c1y = p1[1] + (p2[1] - p0[1]) / 6;
    const c2x = p2[0] - (p3[0] - p1[0]) / 6;
    const c2y = p2[1] - (p3[1] - p1[1]) / 6;
    d.push(`C ${c1x} ${c1y}, ${c2x} ${c2y}, ${p2[0]} ${p2[1]}`);
  }
  return d.join(' ');
}

export default function Sparkline({
  data,
  width = 100,
  height = 40,
  stroke = 'rgba(99,102,241,0.95)', // indigo-ish
  strokeWidth = 1.5,
  fillFrom = 'rgba(99,102,241,0.20)',
  fillTo = 'rgba(99,102,241,0.00)',
}: Props) {
  const id = useId().replace(/[:]/g, '');

  const { path, area } = useMemo(() => {
    if (!data || data.length < 2) return { path: '', area: '' };
    const min = Math.min(...data);
    const max = Math.max(...data);
    const dx = Math.max(data.length - 1, 1);
    const pts: Array<[number, number]> = data.map((v, i) => {
      const x = (i / dx) * width;
      const y = max === min ? height / 2 : height - ((v - min) / (max - min)) * height;
      return [x, y];
    });
    const path = toSmoothPath(pts);
    const area = `${path} L ${width} ${height} L 0 ${height} Z`;
    return { path, area };
  }, [data, width, height]);

  if (!path) return null;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" className="h-full w-full">
      <defs>
        <linearGradient id={`g${id}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={fillFrom} />
          <stop offset="100%" stopColor={fillTo} />
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#g${id})`} stroke="none" />
      <path d={path} fill="none" stroke={stroke} strokeWidth={strokeWidth} />
    </svg>
  );
}


