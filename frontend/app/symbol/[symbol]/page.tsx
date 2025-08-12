'use client';

import SymbolDetail from '../../components/SymbolDetail';

export default function Page({ params }: { params: { symbol: string } }) {
  const symbol = decodeURIComponent(params.symbol).toUpperCase();
  return (
    <main className="min-h-dvh p-4 md:p-8">
      <a href="/" className="text-sm text-neutral-400 hover:text-neutral-200">← Back</a>
      <h1 className="mt-2 text-2xl md:text-3xl font-bold">{symbol}</h1>
      <SymbolDetail symbol={symbol} />
    </main>
  );
}


