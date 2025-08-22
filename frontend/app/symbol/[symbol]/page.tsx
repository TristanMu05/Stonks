'use client';

import SymbolDataProvider from '../../components/SymbolDataProvider';

export default function Page({ params }: { params: { symbol: string } }) {
  const symbol = decodeURIComponent(params.symbol).toUpperCase();
  
  return (
    <main className="min-h-dvh p-4 md:p-8">
      <div className="mb-4">
        <a 
          href="/" 
          className="text-sm text-neutral-400 hover:text-neutral-200 transition-colors"
        >
          ← Back to Dashboard
        </a>
      </div>
      
      <header className="mb-2">
        <h1 className="text-2xl md:text-3xl font-bold">{symbol}</h1>
        <p className="text-neutral-400 text-sm">Real-time trading data and analysis</p>
      </header>
      
      <SymbolDataProvider symbol={symbol} />
    </main>
  );
}